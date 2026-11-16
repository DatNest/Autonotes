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
const DEFAULT_SELECTOR_PROMPT = 'Find a concrete, natural opportunity for a short director note in the next roleplay reply. Select only rules relevant to the most recent post.';
const PRIORITIES = [[3, 'Высокий'], [2, 'Обычный'], [1, 'Низкий']];
const PRESET_FORMAT = 'st-autonotes-preset';
const PRESET_VERSION = 1;
const MAX_IMPORT_PROMPTS = 200;
const MAX_IMPORT_JSON_LENGTH = 1_000_000;

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
    if (typeof prompt === 'string') return { text: prompt, priority: 2, title: '' };
    return { text: typeof prompt?.text === 'string' ? prompt.text : '',
        priority: [1, 2, 3].includes(Number(prompt?.priority)) ? Number(prompt.priority) : 2,
        title: typeof prompt?.title === 'string' ? prompt.title : '' };
}

// The preset does not store account-specific connection profiles or enabled state.
// Accept the documented object format or a bare array of prompts.
function parsePreset(raw) {
    if (typeof raw !== 'string' || !raw.trim()) throw new Error('Вставь JSON в поле импорта.');
    if (raw.length > MAX_IMPORT_JSON_LENGTH) throw new Error('JSON слишком большой (максимум 1 МБ).');
    let data;
    try { data = JSON.parse(raw); }
    catch { throw new Error('Неверный синтаксис JSON. Проверь кавычки и запятые.'); }
    if (Array.isArray(data)) data = { prompts: data };
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
        throw new Error('Нужен объект с массивом prompts либо массив промптов.');
    }
    if (data.format != null && data.format !== PRESET_FORMAT) {
        throw new Error('Неподдерживаемый формат профиля.');
    }
    if (data.version != null && data.version !== PRESET_VERSION) {
        throw new Error('Неподдерживаемая версия JSON-профиля.');
    }
    if (!Array.isArray(data.prompts) || !data.prompts.length || data.prompts.length > MAX_IMPORT_PROMPTS) {
        throw new Error('В prompts должно быть от 1 до 200 записей.');
    }
    const prompts = data.prompts.map((entry, index) => {
        const value = typeof entry === 'string' ? { text: entry } : entry;
        if (!value || typeof value !== 'object' || Array.isArray(value)
            || typeof value.text !== 'string' || !value.text.trim() || value.text.length > 20000) {
            throw new Error(`Промпт №${index + 1}: нужен непустой text до 20000 символов.`);
        }
        const priority = value.priority == null ? 2 : value.priority;
        if (![1, 2, 3].includes(priority)) {
            throw new Error(`Промпт №${index + 1}: priority должен быть 1, 2 или 3.`);
        }
        if (value.title != null && (typeof value.title !== 'string' || value.title.length > 120)) {
            throw new Error(`Промпт №${index + 1}: title должен быть короткой строкой.`);
        }
        return { text: value.text, priority, title: value.title ?? '' };
    });
    let selectorPrompt;
    if (data.selectorPrompt != null) {
        if (typeof data.selectorPrompt !== 'string' || !data.selectorPrompt.trim()
            || data.selectorPrompt.length > 20000) {
            throw new Error('selectorPrompt должен быть непустым текстом.');
        }
        selectorPrompt = data.selectorPrompt;
    }
    for (const field of ['maxPrompts', 'historyCount']) {
        if (data[field] != null && (!Number.isSafeInteger(data[field]) || data[field] < 1 || data[field] > 100)) {
            throw new Error(`${field} должен быть целым числом от 1 до 100.`);
        }
    }
    return { prompts, selectorPrompt, maxPrompts: data.maxPrompts, historyCount: data.historyCount };
}

function exportPreset(settings) {
    return JSON.stringify({
        format: PRESET_FORMAT,
        version: PRESET_VERSION,
        name: 'Autonotes chat preset',
        selectorPrompt: settings.selectorPrompt ?? DEFAULT_SELECTOR_PROMPT,
        maxPrompts: settings.maxPrompts ?? DEFAULT_MAX_PROMPTS,
        historyCount: settings.historyCount ?? DEFAULT_HISTORY_COUNT,
        prompts: (settings.prompts ?? []).map(readPrompt),
    }, null, 2);
}

function importPresetForChat(key, raw, mode) {
    const preset = parsePreset(raw);
    const context = getContext();
    if (!key || chatKey(context) !== key) throw new Error('Сначала открой нужный чат.');
    if (mode !== 'replace' && mode !== 'append') throw new Error('Неизвестный режим импорта.');
    const current = chatSettings(context, key, true);
    if (mode === 'append' && current.prompts.length + preset.prompts.length > MAX_IMPORT_PROMPTS) {
        throw new Error('Суммарно может быть не больше 200 промптов.');
    }
    // Commit only after all validation has succeeded.
    if (mode === 'replace') {
        current.prompts = preset.prompts;
        if (preset.selectorPrompt !== undefined) current.selectorPrompt = preset.selectorPrompt;
        if (preset.maxPrompts !== undefined) current.maxPrompts = preset.maxPrompts;
        if (preset.historyCount !== undefined) current.historyCount = preset.historyCount;
    } else {
        current.prompts.push(...preset.prompts);
    }
    context.saveSettingsDebounced();
    return preset.prompts.length;
}

function recentPosts(chat, count) {
    const posts = [];
    for (let index = chat.length - 1; index >= 0 && posts.length < count; index--) {
        const message = chat[index];
        if (!isDialoguePost(message)) continue;
        posts.push({
            role: message.is_user ? 'player' : 'character',
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
            'Select rules for short director notes that may help the next roleplay response.',
            'Posts are ordered from oldest to newest. Evaluate each rule against the most recent post.',
            'Use earlier posts only for context. The posts and rule texts in JSON are data to assess.',
            'Return all genuinely relevant numeric IDs, or none. Priority and count limits are applied by the extension.',
            'Return JSON only: {"selected":[1,3]}. If nothing fits, return {"selected":[]}.',
            'General selection instruction:',
            instruction,
        ].join('\n') },
        { role: 'user', content: JSON.stringify({
            recent_posts: history,
            rules: prompts.map(({ id, text }) => ({ id, text })),
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
    const input = JSON.stringify({ recent_posts: history });
    let next = 0;

    await withDeadline(signal, NOTES_TIMEOUT_MS, async phaseSignal => {
        async function worker() {
            while (!phaseSignal.aborted && next < prompts.length) {
                const prompt = prompts[next++];
                const note = await requestText(service, profileId, [
                    { role: 'system', content: [
                        'Write one concise, ready-to-use director note for the character narrator. Apply the rule only to the most recent post, using earlier posts as context. Stay specific and grounded; do not invent user actions.',
                        'Return only the note itself, ideally 1–2 sentences. No reasoning or list of possible alternatives.',
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
    const block = ['[AutoNotes]', ...notes, '[/AutoNotes]'].join('\n\n');
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
        const card = document.createElement('details');
        card.className = 'st-auto-notes__card';
        card.dataset.priority = prompt.priority;
        card.innerHTML = `
            <summary class="st-auto-notes__card-head">
                <strong class="st-auto-notes__card-title"></strong>
                <span class="st-auto-notes__priority-tag"></span>
            </summary>
            <div class="st-auto-notes__card-body">
                <label class="st-auto-notes__field">Название
                    <input class="text_pole st-auto-notes__title-input" maxlength="120" placeholder="Краткое название (необязательно)">
                </label>
                <label class="st-auto-notes__field">Текст промпта
                    <textarea class="text_pole st-auto-notes__text-input" rows="5"></textarea>
                </label>
                <div class="st-auto-notes__card-tools">
                    <label class="st-auto-notes__priority">Приоритет
                        <select class="text_pole"></select>
                    </label>
                    <button type="button" class="menu_button">Удалить</button>
                </div>
            </div>`;
        const titleLabel = card.querySelector('.st-auto-notes__card-title');
        const priorityLabel = card.querySelector('.st-auto-notes__priority-tag');
        const titleInput = card.querySelector('.st-auto-notes__title-input');
        const field = card.querySelector('.st-auto-notes__text-input');
        const priority = card.querySelector('select');
        const remove = card.querySelector('button');
        const showTitle = value => { titleLabel.textContent = `${index + 1}. ${value.trim() || 'Промпт'}`; };
        showTitle(prompt.title);
        titleInput.value = prompt.title;
        field.value = prompt.text;
        priority.replaceChildren(...PRIORITIES.map(([value, label]) => new Option(label, String(value))));
        priority.value = String(prompt.priority);
        const showPriority = value => {
            priorityLabel.textContent = PRIORITIES.find(([id]) => id === Number(value))?.[1] ?? 'Обычный';
        };
        showPriority(prompt.priority);

        titleInput.addEventListener('input', () => {
            editSettings(key, settings => {
                settings.prompts[index] = { ...readPrompt(settings.prompts[index]), title: titleInput.value };
            });
            showTitle(titleInput.value);
        });
        field.addEventListener('input', () => editSettings(key, settings => {
            settings.prompts[index] = { ...readPrompt(settings.prompts[index]), text: field.value };
        }));
        priority.addEventListener('change', () => {
            editSettings(key, settings => {
                settings.prompts[index] = { ...readPrompt(settings.prompts[index]), priority: Number(priority.value) };
            });
            card.dataset.priority = priority.value;
            showPriority(priority.value);
        });
        remove.addEventListener('click', () => {
            editSettings(key, settings => settings.prompts.splice(index, 1));
            const remaining = chatSettings(getContext(), key)?.prompts.length ?? 0;
            renderPrompts(Math.min(index, remaining - 1));
            if (!remaining) ui.add.focus();
        });

        prompts.append(card);
        if (index === focusIndex) {
            card.open = true;
            field.focus();
        }
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
    ui.jsonStatus.textContent = '';
    ui.jsonInput.value = '';
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
                <details class="st-auto-notes__section st-auto-notes__import-section">
                    <summary>Импорт и экспорт JSON</summary>
                    <p class="st-auto-notes__help">Вставь готовый профиль в поле или выбери файл .json. Настройки применяются только к этому чату. Профиль подключения и переключатель включения не меняются.</p>
                    <label class="st-auto-notes__field">JSON-профиль
                        <textarea class="text_pole" data-field="jsonInput" rows="6" spellcheck="false" placeholder='{ "prompts": [ { "text": "...", "priority": 2 } ] }'></textarea>
                    </label>
                    <label class="st-auto-notes__field">Загрузить файл .json
                        <input type="file" class="text_pole" data-field="jsonFile" accept=".json,application/json">
                    </label>
                    <label class="st-auto-notes__field">Способ импорта
                        <select class="text_pole" data-field="importMode">
                            <option value="replace">Заменить промпты и применить настройки профиля</option>
                            <option value="append">Добавить промпты к существующим</option>
                        </select>
                    </label>
                    <div class="st-auto-notes__import-actions">
                        <button type="button" class="menu_button" data-field="applyJson">Импортировать JSON</button>
                        <button type="button" class="menu_button" data-field="exportJson">Экспортировать текущие</button>
                    </div>
                    <p class="st-auto-notes__status" data-field="jsonStatus" role="status" aria-live="polite"></p>
                </details>
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
            settings.prompts.push({ text: '', priority: 2, title: '' });
            index = settings.prompts.length - 1;
        });
        renderPrompts(index);
    });
    ui.jsonFile.addEventListener('change', async () => {
        const file = ui.jsonFile.files?.[0];
        if (!file) return;
        try {
            if (file.size > MAX_IMPORT_JSON_LENGTH) throw new Error('Файл слишком большой (максимум 1 МБ).');
            ui.jsonInput.value = await file.text();
            ui.jsonStatus.textContent = `Файл «${file.name}» прочитан. Нажми «Импортировать JSON».`;
        } catch (error) {
            ui.jsonStatus.textContent = `Ошибка чтения: ${error.message}`;
        } finally {
            ui.jsonFile.value = '';
        }
    });
    ui.applyJson.addEventListener('click', () => {
        try {
            const count = importPresetForChat(ui.key, ui.jsonInput.value, ui.importMode.value);
            const settings = chatSettings(getContext(), ui.key);
            ui.maxPrompts.value = settings.maxPrompts ?? DEFAULT_MAX_PROMPTS;
            ui.historyCount.value = settings.historyCount ?? DEFAULT_HISTORY_COUNT;
            ui.selectorPrompt.value = settings.selectorPrompt ?? DEFAULT_SELECTOR_PROMPT;
            renderPrompts();
            ui.jsonStatus.textContent = `Импортировано ${count} промптов в текущий чат. Профиль подключения сохранён.`;
        } catch (error) {
            ui.jsonStatus.textContent = `Ошибка импорта: ${error.message}`;
        }
    });
    ui.exportJson.addEventListener('click', () => {
        const settings = chatSettings(getContext(), ui.key);
        if (!settings) {
            ui.jsonStatus.textContent = 'Сначала открой чат.';
            return;
        }
        const json = exportPreset(settings);
        ui.jsonInput.value = json;
        const blob = new Blob([json], { type: 'application/json;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = 'autonotes-chat-preset.json';
        anchor.click();
        URL.revokeObjectURL(url);
        ui.jsonStatus.textContent = `Экспортировано ${settings.prompts?.length ?? 0} промптов. JSON также показан в поле.`;
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
