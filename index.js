const SETTINGS_KEY = 'stAutoNotes';
const INJECTION_KEY = 'zz_st_auto_notes';
const IN_CHAT = 1;
const SYSTEM = 0;
const MAX_PARALLEL_REQUESTS = 3;
const SELECT_TIMEOUT_MS = 12000;
const NOTES_TIMEOUT_MS = 20000;
const SELECT_TOKEN_BUDGET = 2048;
const RESPONSE_TOKEN_BUDGET = 4096;
const DEFAULT_MAX_PROMPTS = 2;
const DEFAULT_HISTORY_COUNT = 2;
const DEFAULT_SELECTOR_PROMPT = 'Ищи, где уместно добавить режиссёрскую заметку к ситуации. Выбирай правило, если в последних постах есть конкретный повод применить его в следующем ответе.';
const PRIORITIES = [[3, 'Высокий'], [2, 'Обычный'], [1, 'Низкий']];

const getContext = () => globalThis.SillyTavern.getContext();

let pendingPost;
let groupTurn;
let activeBatch;
let ui;

function chatKey(context) {
    const id = context.getCurrentChatId();
    if (!id) return null;
    if (context.groupId != null) return JSON.stringify(['group', context.groupId, id]);
    const avatar = context.characters[context.characterId]?.avatar;
    return avatar ? JSON.stringify(['character', avatar, id]) : null;
}

function chatSettings(context, key, create = false) {
    if (!key) return undefined;
    if (!create) return context.extensionSettings[SETTINGS_KEY]?.[key];
    const chats = context.extensionSettings[SETTINGS_KEY] ??= {};
    return chats[key] ??= { enabled: false, profileId: '', prompts: [] };
}

function moveChatSettings({ avatarId, groupId, oldFileName, newFileName }) {
    if ((groupId == null && !avatarId) || !oldFileName || !newFileName) return;
    const context = getContext();
    const chats = context.extensionSettings[SETTINGS_KEY];
    const owner = groupId != null ? ['group', groupId] : ['character', avatarId];
    const oldKey = JSON.stringify([...owner, oldFileName.replace(/\.jsonl$/, '')]);
    const newKey = JSON.stringify([...owner, newFileName.replace(/\.jsonl$/, '')]);
    if (!chats?.[oldKey] || oldKey === newKey) return;
    chats[newKey] = chats[oldKey];
    delete chats[oldKey];
    context.saveSettingsDebounced();
    if (ui?.key === newKey) renderChat();
}

function clearInjection(context = getContext()) {
    delete context.extensionPrompts[INJECTION_KEY];
}

function supportedProfiles(context) {
    try {
        return context.ConnectionManagerRequestService?.getSupportedProfiles() ?? [];
    } catch {
        return [];
    }
}

function isDialoguePost(message) {
    return !message.is_system && typeof message.mes === 'string' && Boolean(message.mes.trim());
}

function recordPost(index) {
    const context = getContext();
    const key = chatKey(context);
    const message = context.chat[index];
    if (key && message?.is_user && !message.is_system) {
        pendingPost = { key, message };
    }
}

function resetTurn() {
    activeBatch?.abort();
    activeBatch = undefined;
    pendingPost = undefined;
    groupTurn = undefined;
    clearInjection();
}

function readPrompt(prompt) {
    return typeof prompt === 'string' ? { text: prompt, priority: 2 } : prompt;
}

function recentPosts(chat, count) {
    const posts = [];
    for (let index = chat.length - 1; index >= 0 && posts.length < count; index--) {
        const message = chat[index];
        if (!isDialoguePost(message)) continue;
        posts.push({
            role: message.is_user ? 'игрок' : 'персонаж',
            name: message.name ?? '',
            text: message.mes.trim(),
        });
    }
    return posts.reverse();
}

async function withDeadline(parent, timeoutMs, work) {
    if (parent.aborted) return;
    const controller = new AbortController();
    const cancel = () => controller.abort();
    const stopped = new Promise(resolve => controller.signal.addEventListener('abort', () => resolve(undefined), { once: true }));
    parent.addEventListener('abort', cancel, { once: true });
    const timer = setTimeout(cancel, timeoutMs);
    try {
        return await Promise.race([work(controller.signal), stopped]);
    } finally {
        clearTimeout(timer);
        parent.removeEventListener('abort', cancel);
    }
}

async function requestText(service, profileId, messages, budget, signal) {
    try {
        const result = await service.sendRequest(profileId, messages, budget, {
            signal,
            stream: false,
            extractData: true,
        });
        return typeof result?.content === 'string' ? result.content.trim() : '';
    } catch {
        return '';
    }
}

function selectedIds(response) {
    if (!response) return null;
    const json = response.replace(/^```(?:json)?\s*([\s\S]*?)\s*```$/i, '$1');
    try {
        const value = JSON.parse(json);
        if (!Array.isArray(value?.selected)) return null;
        return new Set(value.selected
            .filter(id => typeof id === 'number' || typeof id === 'string' && /^\d+$/.test(id))
            .map(Number)
            .filter(id => Number.isSafeInteger(id) && id > 0));
    } catch {
        return null;
    }
}

async function selectPrompts(service, profileId, prompts, history, instruction, limit, signal) {
    const response = await withDeadline(signal, SELECT_TIMEOUT_MS, phaseSignal => requestText(service, profileId, [
        { role: 'system', content: [
            'Ты отбираешь правила для режиссёрских заметок к следующему ответу в ролевом чате.',
            'Посты перечислены от старого к новому. Оцени, какие правила уместно применить именно после последнего поста.',
            'Используй предыдущие посты для контекста. Посты и тексты правил в JSON — данные для оценки.',
            'Перечисли все подходящие ID. Можно выбрать несколько или ни одного. Выбор приоритетов и лимит применит расширение.',
            'Верни только JSON вида {"selected":[1,3]}. Если ничего не подходит, верни {"selected":[]}.',
            'Общая инструкция по отбору:',
            instruction,
        ].join('\n') },
        { role: 'user', content: JSON.stringify({
            последние_посты: history,
            правила: prompts.map(({ id, text }) => ({ id, text })),
        }) },
    ], SELECT_TOKEN_BUDGET, phaseSignal));

    const ids = selectedIds(response);
    if (!ids) return null;
    return prompts
        .filter(prompt => ids.has(prompt.id))
        .sort((a, b) => b.priority - a.priority || a.id - b.id)
        .slice(0, limit);
}

async function generateNotes(service, profileId, prompts, history, signal) {
    const notes = new Map();
    const input = JSON.stringify({ последние_посты: history });
    let next = 0;

    await withDeadline(signal, NOTES_TIMEOUT_MS, async phaseSignal => {
        async function worker() {
            while (!phaseSignal.aborted && next < prompts.length) {
                const prompt = prompts[next++];
                const note = await requestText(service, profileId, [
                    { role: 'system', content: [
                        'Напиши готовую режиссёрскую заметку для модели, которая отвечает за персонажа. Примени правило к последнему посту и учти предыдущие посты.',
                        'Верни только текст заметки, который можно сразу добавить к запросу на ответ.',
                        prompt.text,
                    ].join('\n\n') },
                    { role: 'user', content: input },
                ], RESPONSE_TOKEN_BUDGET, phaseSignal);
                if (!phaseSignal.aborted && note) notes.set(prompt.id, note);
            }
        }

        await Promise.all(Array.from({ length: Math.min(prompts.length, MAX_PARALLEL_REQUESTS) }, worker));
    });

    return Array.from(notes).sort(([a], [b]) => a - b).map(([, text]) => text);
}

function injectNotes(context, notes) {
    if (!notes.length) return;
    const block = ['[Автозаметки]', ...notes, '[/Автозаметки]'].join('\n\n');
    context.setExtensionPrompt(INJECTION_KEY, block, IN_CHAT, 0, false, SYSTEM);
}

globalThis.stAutoNotesBeforePrompt = async function (_chat, _contextSize, abortGeneration, type) {
    const context = getContext();
    clearInjection(context);
    if (type != null && type !== 'normal') return;

    const key = chatKey(context);
    const settings = chatSettings(context, key);
    const turn = pendingPost;
    pendingPost = undefined;
    if (!settings?.enabled || !settings.profileId) return;

    const group = groupTurn?.key === key ? groupTurn : undefined;
    if (group?.notes) {
        injectNotes(context, group.notes);
        return;
    }

    if (!turn || turn.key !== key || turn.message !== context.chat.findLast(isDialoguePost)) return;
    const service = context.ConnectionManagerRequestService;
    const { profileId } = settings;
    if (!supportedProfiles(context).some(profile => profile.id === profileId)) {
        setStatus(key, 'Профиль недоступен. Ответ пойдёт без автозаметок.');
        return;
    }

    const prompts = settings.prompts.map((entry, index) => {
        const prompt = readPrompt(entry);
        return {
            id: index + 1,
            priority: prompt.priority,
            text: context.substituteParams(prompt.text.trim()).trim(),
        };
    }).filter(prompt => prompt.text);
    if (!prompts.length) {
        setStatus(key, 'Добавь хотя бы один промпт для заметок.');
        return;
    }

    const history = recentPosts(context.chat, settings.historyCount ?? DEFAULT_HISTORY_COUNT);
    const limit = settings.maxPrompts ?? DEFAULT_MAX_PROMPTS;
    const instruction = context.substituteParams(settings.selectorPrompt?.trim() || DEFAULT_SELECTOR_PROMPT);
    activeBatch?.abort();
    const controller = new AbortController();
    activeBatch = controller;

    setStatus(key, 'Проверяю, какие промпты подходят к сцене…');
    const selected = await selectPrompts(service, profileId, prompts, history, instruction, limit, controller.signal);
    let notes = [];
    if (selected?.length && !controller.signal.aborted) {
        setStatus(key, `Готовлю заметки по выбранным промптам: ${selected.length}…`);
        notes = await generateNotes(service, profileId, selected, history, controller.signal);
    }

    const current = getContext();
    if (activeBatch !== controller || chatKey(current) !== key) {
        abortGeneration(true);
        return;
    }
    activeBatch = undefined;

    const currentSettings = chatSettings(current, key);
    if (!currentSettings?.enabled || currentSettings.profileId !== profileId) {
        setStatus(key, '');
        return;
    }

    if (group && groupTurn === group) group.notes = notes;
    injectNotes(current, notes);
    if (!selected) {
        setStatus(key, 'Отбор не завершился. Ответ пойдёт без автозаметок.');
    } else if (!selected.length) {
        setStatus(key, 'Для этой сцены модель не выбрала промпты.');
    } else {
        setStatus(key, `Последний ход · промпты ${selected.map(prompt => '№' + prompt.id).join(', ')} · заметок ${notes.length} из ${selected.length}`);
    }
};

function setStatus(key, text) {
    if (ui?.key === key) ui.status.textContent = text;
}

function renderProfiles(context = getContext()) {
    if (!ui) return;
    const selected = chatSettings(context, ui.key)?.profileId ?? '';
    const profiles = supportedProfiles(context).slice().sort((a, b) => a.name.localeCompare(b.name));
    ui.profile.replaceChildren(new Option(profiles.length ? 'Выбери профиль' : 'Нет доступных профилей', ''));
    for (const profile of profiles) {
        ui.profile.add(new Option(profile.name, profile.id));
    }
    if (selected && !profiles.some(profile => profile.id === selected)) {
        ui.profile.add(new Option('Сохранённый профиль недоступен', selected));
    }
    ui.profile.value = selected;
}

function editSettings(key, edit) {
    const context = getContext();
    if (!key || chatKey(context) !== key) return;
    edit(chatSettings(context, key, true));
    context.saveSettingsDebounced();
}

function renderPrompts(focusIndex) {
    const { key, prompts } = ui;
    prompts.replaceChildren();
    const drafts = chatSettings(getContext(), key)?.prompts ?? [];
    ui.empty.hidden = drafts.length > 0;

    drafts.forEach((entry, index) => {
        const prompt = readPrompt(entry);
        const card = document.createElement('div');
        card.className = 'st-auto-notes__card';
        card.dataset.priority = prompt.priority;
        card.innerHTML = `
            <div class="st-auto-notes__card-head">
                <h4><label for="st-auto-notes-prompt-${index}">Промпт ${index + 1}</label></h4>
                <button type="button" class="menu_button" aria-label="Удалить промпт ${index + 1}">Удалить</button>
            </div>
            <textarea id="st-auto-notes-prompt-${index}" class="text_pole" rows="4"></textarea>
            <label class="st-auto-notes__priority">
                Приоритет
                <select class="text_pole" aria-label="Приоритет промпта ${index + 1}"></select>
            </label>`;
        const field = card.querySelector('textarea');
        const priority = card.querySelector('select');
        const remove = card.querySelector('button');
        field.value = prompt.text;
        priority.replaceChildren(...PRIORITIES.map(([value, label]) => new Option(label, String(value))));
        priority.value = String(prompt.priority);

        field.addEventListener('input', () => editSettings(key, settings => {
            settings.prompts[index] = { ...readPrompt(settings.prompts[index]), text: field.value };
        }));
        priority.addEventListener('change', () => {
            editSettings(key, settings => {
                settings.prompts[index] = { ...readPrompt(settings.prompts[index]), priority: Number(priority.value) };
            });
            card.dataset.priority = priority.value;
        });
        remove.addEventListener('click', () => {
            editSettings(key, settings => settings.prompts.splice(index, 1));
            const remaining = chatSettings(getContext(), key)?.prompts.length ?? 0;
            renderPrompts(Math.min(index, remaining - 1));
            if (!remaining) ui.add.focus();
        });

        prompts.append(card);
        if (index === focusIndex) field.focus();
    });
}

function renderChat() {
    const context = getContext();
    ui.key = chatKey(context);
    const settings = chatSettings(context, ui.key);
    ui.fields.hidden = !ui.key;
    ui.fields.disabled = !ui.key;
    ui.message.hidden = Boolean(ui.key);
    ui.chatName.textContent = ui.key ? context.getCurrentChatId() : '';
    ui.chatName.title = ui.chatName.textContent;
    ui.enabled.checked = settings?.enabled ?? false;
    ui.maxPrompts.value = settings?.maxPrompts ?? DEFAULT_MAX_PROMPTS;
    ui.historyCount.value = settings?.historyCount ?? DEFAULT_HISTORY_COUNT;
    ui.selectorPrompt.value = settings?.selectorPrompt ?? DEFAULT_SELECTOR_PROMPT;
    ui.status.textContent = '';
    renderProfiles(context);
    renderPrompts();
}

function bindNumber(field, key, fallback) {
    field.addEventListener('input', () => {
        if (!Number.isSafeInteger(field.valueAsNumber) || field.valueAsNumber < 1) return;
        editSettings(ui.key, settings => {
            settings[key] = field.valueAsNumber;
        });
    });
    field.addEventListener('change', () => {
        field.value = chatSettings(getContext(), ui.key)?.[key] ?? fallback;
    });
}

function mount() {
    const host = document.getElementById('extensions_settings');
    if (!host || ui) return;
    const panel = document.createElement('details');
    panel.className = 'st-auto-notes';
    panel.innerHTML = `
        <summary><span aria-hidden="true">✦</span> Автозаметки</summary>
        <div class="st-auto-notes__body">
            <p data-field="message">Открой чат, чтобы настроить автозаметки.</p>
            <fieldset data-field="fields">
                <div class="st-auto-notes__section">
                    <div class="st-auto-notes__section-head">
                        <div class="st-auto-notes__chat">
                            <h3>Настройки чата</h3>
                            <p class="st-auto-notes__chat-name" data-field="chatName"></p>
                        </div>
                        <label class="st-auto-notes__toggle">
                            <input type="checkbox" data-field="enabled">
                            Включить в этом чате
                        </label>
                    </div>
                    <label class="st-auto-notes__field">
                        Профиль подключения
                        <select class="text_pole" data-field="profile"></select>
                    </label>
                    <div class="st-auto-notes__numbers">
                        <label class="st-auto-notes__field">
                            Макс. промптов за ход
                            <input type="number" class="text_pole" data-field="maxPrompts" min="1" step="1" inputmode="numeric">
                        </label>
                        <label class="st-auto-notes__field">
                            Последних постов читать
                            <input type="number" class="text_pole" data-field="historyCount" min="1" step="1" inputmode="numeric">
                        </label>
                    </div>
                </div>
                <div class="st-auto-notes__section">
                    <h3><label for="st-auto-notes-selector">Общий промпт отбора</label></h3>
                    <p id="st-auto-notes-selector-help" class="st-auto-notes__help">По этой инструкции модель решает, какие промпты подходят к текущей сцене.</p>
                    <textarea id="st-auto-notes-selector" class="text_pole" data-field="selectorPrompt" rows="4" aria-describedby="st-auto-notes-selector-help"></textarea>
                </div>
                <div class="st-auto-notes__section">
                    <div class="st-auto-notes__section-head">
                        <h3>Промпты для заметок</h3>
                        <button type="button" class="menu_button" data-field="add">+ Добавить промпт</button>
                    </div>
                    <p class="st-auto-notes__help">При одинаковом приоритете первым выбирается промпт выше в списке.</p>
                    <div class="st-auto-notes__prompts" data-field="prompts"></div>
                    <p class="st-auto-notes__help" data-field="empty">Добавь правило, которое модель сможет применить к сцене.</p>
                </div>
                <p class="st-auto-notes__status" data-field="status" role="status"></p>
            </fieldset>
        </div>`;
    ui = Object.fromEntries(Array.from(panel.querySelectorAll('[data-field]'), el => [el.dataset.field, el]));
    ui.enabled.addEventListener('change', () => editSettings(ui.key, settings => {
        settings.enabled = ui.enabled.checked;
    }));
    ui.profile.addEventListener('change', () => editSettings(ui.key, settings => {
        settings.profileId = ui.profile.value;
    }));
    bindNumber(ui.maxPrompts, 'maxPrompts', DEFAULT_MAX_PROMPTS);
    bindNumber(ui.historyCount, 'historyCount', DEFAULT_HISTORY_COUNT);
    ui.selectorPrompt.placeholder = DEFAULT_SELECTOR_PROMPT;
    ui.selectorPrompt.addEventListener('input', () => editSettings(ui.key, settings => {
        settings.selectorPrompt = ui.selectorPrompt.value;
    }));
    ui.add.addEventListener('click', () => {
        let index;
        editSettings(ui.key, settings => {
            settings.prompts.push({ text: '', priority: 2 });
            index = settings.prompts.length - 1;
        });
        renderPrompts(index);
    });
    panel.addEventListener('toggle', () => {
        if (panel.open) renderProfiles();
    });
    host.append(panel);
    renderChat();
}

globalThis.jQuery(() => {
    const { eventSource, eventTypes } = getContext();
    eventSource.on(eventTypes.MESSAGE_SENT, recordPost);
    eventSource.on(eventTypes.CHAT_CHANGED, () => {
        resetTurn();
        if (ui) renderChat();
    });
    if (eventTypes.CHAT_RENAMED) eventSource.on(eventTypes.CHAT_RENAMED, moveChatSettings);
    eventSource.on(eventTypes.GROUP_WRAPPER_STARTED, ({ type }) => {
        groupTurn = type == null || type === 'normal' ? { key: chatKey(getContext()) } : undefined;
    });
    eventSource.on(eventTypes.GROUP_WRAPPER_FINISHED, resetTurn);
    eventSource.on(eventTypes.GENERATION_STARTED, (_type, _options, dryRun) => {
        if (!dryRun) clearInjection();
    });
    eventSource.on(eventTypes.GENERATE_AFTER_DATA, (_data, dryRun) => {
        if (!dryRun) clearInjection();
    });
    eventSource.on(eventTypes.GENERATION_STOPPED, resetTurn);
    eventSource.on(eventTypes.APP_READY, () => renderProfiles());
    for (const event of [eventTypes.CONNECTION_PROFILE_CREATED, eventTypes.CONNECTION_PROFILE_UPDATED, eventTypes.CONNECTION_PROFILE_DELETED]) {
        if (event) eventSource.on(event, () => renderProfiles());
    }
    mount();
});
