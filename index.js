const SETTINGS_KEY = 'stAutoNotes';
const INJECTION_KEY = 'zz_st_auto_notes';
const IN_CHAT = 1;
const SYSTEM = 0;
const MAX_PARALLEL_REQUESTS = 3;
const SELECT_TIMEOUT_MS = 12000;
const VECTOR_TIMEOUT_MS = 20000;
const NOTES_TIMEOUT_MS = 20000;
const CONNECTION_TEST_TIMEOUT_MS = 30000;
const CONNECTION_TEST_TOKEN_BUDGET = 512;
const SELECT_TOKEN_BUDGET = 2048;
const RESPONSE_TOKEN_BUDGET = 4096;
const DEFAULT_MAX_PROMPTS = 2;
const DEFAULT_HISTORY_COUNT = 2;
const DEFAULT_ACTIVATION_EVERY = 1;
const DEFAULT_SELECTOR_PROMPT = 'Выбирай только те правила, которые действительно полезны для следующего ответа. Оцени конкретную ситуацию последних сообщений, а не отдельные совпадающие слова.';
const PRIORITIES = [[3, 'Высокий'], [2, 'Обычный'], [1, 'Низкий']];

const DEFAULT_VECTOR_THRESHOLD = 0.45;
const VECTOR_MODES = { vector: 'По смыслу · векторы', model: 'Через общий промпт' };
const VECTOR_MODELS = {
    openai: ['openai_model', 'text-embedding-3-small'],
    openrouter: ['openrouter_model', 'openai/text-embedding-3-large'],
    togetherai: ['togetherai_model', 'togethercomputer/m2-bert-80M-32k-retrieval'],
    electronhub: ['electronhub_model', 'text-embedding-3-small'],
    cohere: ['cohere_model', 'embed-english-v3.0'],
    palm: ['google_model', 'text-embedding-005'],
    vertexai: ['google_model', 'text-embedding-005'],
    ollama: ['ollama_model', 'mxbai-embed-large'],
    vllm: ['vllm_model', ''],
    chutes: ['chutes_model', 'chutes-qwen-qwen3-embedding-8b'],
    nanogpt: ['nanogpt_model', 'text-embedding-3-small'],
    siliconflow: ['siliconflow_model', 'Qwen/Qwen3-Embedding-0.6B'],
    workers_ai: ['workers_ai_model', '@cf/baai/bge-m3'],
};
const VECTOR_BATCH_SIZE = 5;
const VECTOR_CACHE_SIZE = 12;
const vectorCache = new Map();

function vectorSource(context) {
    const settings = context.extensionSettings.vectors ?? {};
    const source = settings.source || 'transformers';
    const body = { source };

    if (Object.hasOwn(VECTOR_MODELS, source)) {
        const [field, fallback] = VECTOR_MODELS[source];
        body.model = settings[field] || fallback;
    } else if (!['transformers', 'mistral', 'nomicai', 'llamacpp'].includes(source)) {
        return null;
    }

    if (['ollama', 'llamacpp', 'vllm'].includes(source)) {
        body.apiUrl = settings.use_alt_endpoint
            ? settings.alt_endpoint_url
            : context.textCompletionSettings?.server_urls?.[source];
        if (!body.apiUrl) return null;
    }

    const completion = context.chatCompletionSettings ?? {};
    if (source === 'vertexai') {
        body.vertexai_auth_mode = completion.vertexai_auth_mode;
        body.vertexai_region = completion.vertexai_region;
        body.vertexai_express_project_id = completion.vertexai_express_project_id;
    }
    if (source === 'ollama') body.keep = Boolean(settings.ollama_keep);
    if (source === 'siliconflow') body.siliconflow_endpoint = completion.siliconflow_endpoint;
    if (source === 'workers_ai') body.workers_ai_account_id = completion.workers_ai_account_id;
    return body;
}

function vectorHash(text) {
    let low = 0xdeadbeef;
    let high = 0x41c6ce57;
    for (let index = 0; index < text.length; index++) {
        const code = text.charCodeAt(index);
        low = Math.imul(low ^ code, 2654435761);
        high = Math.imul(high ^ code, 1597334677);
    }
    low = Math.imul(low ^ (low >>> 16), 2246822507) ^ Math.imul(high ^ (high >>> 13), 3266489909);
    high = Math.imul(high ^ (high >>> 16), 2246822507) ^ Math.imul(low ^ (low >>> 13), 3266489909);
    return 4294967296 * (high & 0x1fffff) + (low >>> 0);
}

async function vectorRequest(context, operation, body, signal) {
    const response = await fetch(`/api/vector/${operation}`, {
        method: 'POST',
        headers: context.getRequestHeaders(),
        body: JSON.stringify(body),
        signal,
    });
    if (!response.ok) throw new Error(`Vector Storage: ${operation} HTTP ${response.status}`);
    if (operation === 'list' || operation === 'query') return response.json();
}

function vectorProviderHash(provider) {
    const scope = { ...provider };
    delete scope.keep;
    return vectorHash(JSON.stringify(scope));
}

function rankPrompts(prompts, predicate, limit) {
    return prompts
        .filter(predicate)
        .sort((a, b) => b.priority - a.priority || a.id - b.id)
        .slice(0, limit);
}

async function selectVectorPrompts(context, provider, key, prompts, history, threshold, limit, signal) {
    const collectionId = `st-autonotes-v1-${vectorHash(key)}-${vectorProviderHash(provider)}`;
    const body = { ...provider, collectionId };
    const entries = new Map(prompts.map(({ trigger, hash, id }) => [hash, { text: trigger, hash, index: id }]));
    const searchText = entries.size ? history.slice().reverse().map(post => post.text).join('\n\n') : '';
    const signature = JSON.stringify([threshold, searchText, [...entries.keys()].sort()]);
    const cached = vectorCache.get(collectionId);
    let matches = cached?.signature === signature ? cached.matches : undefined;

    if (!matches) {
        try {
            matches = await withDeadline(signal, VECTOR_TIMEOUT_MS, async phaseSignal => {
                const saved = await vectorRequest(context, 'list', body, phaseSignal);
                if (!Array.isArray(saved)) throw new Error('Invalid vector index response');
                const savedHashes = new Set(saved);
                const obsolete = [...savedHashes].filter(hash => !entries.has(hash));
                const missing = Array.from(entries.values()).filter(entry => !savedHashes.has(entry.hash));

                if (obsolete.length) {
                    await vectorRequest(context, 'delete', { ...body, hashes: obsolete }, phaseSignal);
                }
                for (let offset = 0; offset < missing.length; offset += VECTOR_BATCH_SIZE) {
                    const items = missing.slice(offset, offset + VECTOR_BATCH_SIZE);
                    await vectorRequest(context, 'insert', { ...body, items }, phaseSignal);
                }

                if (!searchText) return new Set();
                const result = await vectorRequest(context, 'query', {
                    ...body,
                    searchText,
                    topK: entries.size + saved.length - savedHashes.size,
                    threshold,
                }, phaseSignal);
                if (!Array.isArray(result?.metadata)) throw new Error('Invalid vector search response');

                // Only metadata is threshold-filtered by the SillyTavern query endpoint.
                return new Set(result.metadata
                    .filter(item => item && entries.get(item.hash)?.text === item.text)
                    .map(item => item.hash));
            });
        } catch {
            return null;
        }

        if (!matches || signal.aborted) return null;
        vectorCache.delete(collectionId);
        if (vectorCache.size >= VECTOR_CACHE_SIZE) vectorCache.delete(vectorCache.keys().next().value);
        vectorCache.set(collectionId, { signature, matches });
    }

    return rankPrompts(prompts, prompt => matches.has(prompt.hash), limit);
}

const getContext = () => globalThis.SillyTavern.getContext();

let pendingPost;
let groupTurn;
let activeBatch;
let activeConnectionTest;
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

function countPlayerPosts(chat) {
    return chat.reduce((count, message) => message.is_user && isDialoguePost(message) ? count + 1 : count, 0);
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
            role: message.is_user ? 'игрок' : 'ответ персонажа или ведущего',
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
            'Ты отбираешь правила для коротких контекстных заметок к следующему ответу в ролевом чате.',
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
    return rankPrompts(prompts, prompt => ids.has(prompt.id), limit);
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
                        'Напиши краткую контекстную заметку для модели, продолжающей ролевой чат: она может отыгрывать одного персонажа, группу персонажей или вести повествование. Примени правило к последнему посту и учти предыдущие посты.',
                        'Сохраняй установленные факты и ограничения, но не предрешай исход сцены без прямого требования правила.',
                        'Пиши заметку на языке последнего поста, если правило не требует другого языка.',
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
    const interval = settings.activationEvery ?? DEFAULT_ACTIVATION_EVERY;
    if (interval > 1) {
        const elapsed = Math.max(0, countPlayerPosts(context.chat) - (settings.activationStart ?? 0));
        const progress = elapsed % interval;
        if (!elapsed || progress) {
            setStatus(key, `До проверки правил · постов ${progress} из ${interval}`);
            return;
        }
    }
    const service = context.ConnectionManagerRequestService;
    const { profileId } = settings;
    if (!supportedProfiles(context).some(profile => profile.id === profileId)) {
        setStatus(key, 'Профиль недоступен. Ответ пойдёт без автозаметок.');
        return;
    }

    const mode = settings.mode ?? 'vector';
    const provider = mode === 'vector' ? vectorSource(context) : null;
    if (mode === 'vector' && !provider) {
        setStatus(key, 'Источник Vector Storage недоступен для автозаметок. Проверь настройки векторов.');
        return;
    }

    const prompts = settings.prompts.map((entry, index) => {
        const prompt = readPrompt(entry);
        const trigger = prompt.trigger?.trim() || prompt.text.trim();
        return {
            id: index + 1,
            priority: prompt.priority,
            trigger,
            hash: vectorHash(trigger),
            text: prompt.text.trim(),
        };
    }).filter(prompt => prompt.text);
    if (!prompts.length && !provider) {
        setStatus(key, 'Добавь хотя бы один промпт для заметок.');
        return;
    }

    const history = recentPosts(context.chat, settings.historyCount ?? DEFAULT_HISTORY_COUNT);
    const limit = settings.maxPrompts ?? DEFAULT_MAX_PROMPTS;
    activeBatch?.abort();
    const controller = new AbortController();
    activeBatch = controller;

    setStatus(key, provider ? 'Ищу подходящие правила по смыслу…' : 'Проверяю, какие промпты подходят к сцене…');
    let selected;
    if (provider) {
        selected = await selectVectorPrompts(context, provider, key, prompts, history,
            settings.vectorThreshold ?? DEFAULT_VECTOR_THRESHOLD, limit, controller.signal);
    } else {
        const instruction = context.substituteParams(settings.selectorPrompt?.trim() || DEFAULT_SELECTOR_PROMPT);
        const candidates = prompts.map(prompt => ({ ...prompt, text: context.substituteParams(prompt.text).trim() }))
            .filter(prompt => prompt.text);
        selected = await selectPrompts(service, profileId, candidates, history, instruction, limit, controller.signal);
    }
    let notes = [];
    if (selected?.length && !controller.signal.aborted) {
        setStatus(key, `Готовлю заметки по выбранным промптам: ${selected.length}…`);
        const ready = provider
            ? selected.map(prompt => ({ ...prompt, text: context.substituteParams(prompt.text).trim() })).filter(prompt => prompt.text)
            : selected;
        notes = await generateNotes(service, profileId, ready, history, controller.signal);
    }

    const current = getContext();
    if (activeBatch !== controller || chatKey(current) !== key) {
        abortGeneration(true);
        return;
    }
    activeBatch = undefined;

    const currentSettings = chatSettings(current, key);
    if (!currentSettings?.enabled || currentSettings.profileId !== profileId || (currentSettings.mode ?? 'vector') !== mode) {
        setStatus(key, '');
        return;
    }

    if (group && groupTurn === group) group.notes = notes;
    injectNotes(current, notes);
    if (!selected) {
        setStatus(key, provider
            ? 'Векторный поиск не завершился. Проверь Vector Storage. Ответ пойдёт без автозаметок.'
            : 'Отбор не завершился. Ответ пойдёт без автозаметок.');
    } else if (!prompts.length) {
        setStatus(key, 'Добавь хотя бы один промпт для заметок.');
    } else if (!selected.length) {
        setStatus(key, 'Для этой сцены подходящих правил не нашлось.');
    } else {
        setStatus(key, `Последний ход · промпты ${selected.map(prompt => '№' + prompt.id).join(', ')} · заметок ${notes.length} из ${selected.length}`);
    }
};

function setStatus(key, text) {
    if (ui?.key === key) ui.status.textContent = text;
}

async function checkVectorConnection(context, signal) {
    const provider = vectorSource(context);
    if (!provider) return 'Векторы · проверь источник и URL в Vector Storage.';
    const label = `Векторы (${provider.source})`;
    if (provider.source === 'vllm' && !provider.model) {
        return `${label} · укажи Vectorization Model.`;
    }

    const text = 'The lighthouse shines above the quiet harbor.';
    const hash = vectorHash(text);
    const body = { ...provider, collectionId: `st-autonotes-connection-test-${vectorProviderHash(provider)}` };
    try {
        const found = await withDeadline(signal, CONNECTION_TEST_TIMEOUT_MS, async phaseSignal => {
            const saved = await vectorRequest(context, 'list', body, phaseSignal);
            if (!Array.isArray(saved)) throw new Error('Invalid vector index response');
            if (!saved.includes(hash)) {
                await vectorRequest(context, 'insert', { ...body, items: [{ hash, text, index: 0 }] }, phaseSignal);
            }
            const result = await vectorRequest(context, 'query', {
                ...body, searchText: text, topK: 1, threshold: 0,
            }, phaseSignal);
            return result?.metadata?.some(item => item?.hash === hash && item.text === text) ?? false;
        });
        if (found === undefined) return `${label} · истекло время ожидания.`;
        return found ? `${label} · работают.` : `${label} · тестовая запись не найдена.`;
    } catch (error) {
        const code = /\bHTTP (\d{3})$/.exec(error?.message)?.[1];
        return `${label} · ошибка${code ? ' HTTP ' + code : ' запроса'}. Проверь URL, ключ и модель.`;
    }
}

async function checkNotesConnection(context, profileId, signal) {
    const profile = supportedProfiles(context).find(item => item.id === profileId);
    if (!profileId) return 'Подсказки · выбери профиль подключения.';
    if (!profile) return 'Подсказки · выбранный профиль недоступен.';

    const label = `Подсказки (${profile.name})`;
    const reply = await withDeadline(signal, CONNECTION_TEST_TIMEOUT_MS, phaseSignal => requestText(
        context.ConnectionManagerRequestService,
        profileId,
        [{ role: 'user', content: 'Reply with OK.' }],
        CONNECTION_TEST_TOKEN_BUDGET,
        phaseSignal,
    ));
    if (reply === undefined) return `${label} · истекло время ожидания.`;
    return reply ? `${label} · модель ответила.` : `${label} · текст ответа не получен. Проверь профиль.`;
}

function clearConnectionTest() {
    activeConnectionTest?.abort();
    activeConnectionTest = undefined;
    ui.testButton.disabled = false;
    ui.testStatus.textContent = '';
}

async function testConnection() {
    const context = getContext();
    const key = chatKey(context);
    if (!key || key !== ui.key || activeConnectionTest) return;
    const settings = chatSettings(context, key);
    const controller = new AbortController();
    activeConnectionTest = controller;
    ui.testButton.disabled = true;
    ui.testStatus.textContent = 'Проверяю подключение…';

    const checks = [checkNotesConnection(context, settings?.profileId, controller.signal)];
    if ((settings?.mode ?? 'vector') === 'vector') {
        checks.unshift(checkVectorConnection(context, controller.signal));
    }
    const results = await Promise.all(checks);
    if (activeConnectionTest !== controller) return;
    activeConnectionTest = undefined;
    ui.testButton.disabled = false;
    ui.testStatus.textContent = results.join('\n');
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
    const isVector = ui.mode.value === 'vector';

    drafts.forEach((entry, index) => {
        const prompt = readPrompt(entry);
        const card = document.createElement('div');
        card.className = 'st-auto-notes__card';
        card.dataset.priority = prompt.priority;
        card.innerHTML = `
            <div class="st-auto-notes__card-head">
                <h4>Промпт ${index + 1}</h4>
                <button type="button" class="menu_button" aria-label="Удалить промпт ${index + 1}">Удалить</button>
            </div>
            <label class="st-auto-notes__field st-auto-notes__trigger" ${isVector ? '' : 'hidden'}>
                Когда срабатывает
                <textarea class="text_pole" rows="2" placeholder="Например: персонаж пытается бежать после недавнего ранения"></textarea>
            </label>
            <label class="st-auto-notes__field">
                Что подсказать
                <textarea id="st-auto-notes-prompt-${index}" class="text_pole" rows="4"></textarea>
            </label>
            <label class="st-auto-notes__priority">
                Приоритет
                <select class="text_pole" aria-label="Приоритет промпта ${index + 1}"></select>
            </label>`;
        const field = card.querySelector('textarea[id]');
        const trigger = card.querySelector('.st-auto-notes__trigger textarea');
        const priority = card.querySelector('select');
        const remove = card.querySelector('button');
        field.value = prompt.text;
        trigger.value = prompt.trigger ?? '';
        priority.replaceChildren(...PRIORITIES.map(([value, label]) => new Option(label, String(value))));
        priority.value = String(prompt.priority);

        trigger.addEventListener('input', () => editSettings(key, settings => {
            settings.prompts[index] = { ...readPrompt(settings.prompts[index]), trigger: trigger.value };
        }));
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

function renderMode() {
    const isVector = ui.mode.value === 'vector';
    ui.vectorOptions.hidden = !isVector;
    ui.selectorOptions.hidden = isVector;
    renderPrompts();
}

function renderChat() {
    const context = getContext();
    clearConnectionTest();
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
    ui.activationEvery.value = settings?.activationEvery ?? DEFAULT_ACTIVATION_EVERY;
    ui.mode.value = settings?.mode ?? 'vector';
    ui.vectorThreshold.value = settings?.vectorThreshold ?? DEFAULT_VECTOR_THRESHOLD;
    ui.selectorPrompt.value = settings?.selectorPrompt ?? DEFAULT_SELECTOR_PROMPT;
    ui.status.textContent = '';
    renderProfiles(context);
    renderMode();
}

function bindNumber(field, key, fallback, update = (settings, value) => { settings[key] = value; }, event = 'input') {
    field.addEventListener(event, () => {
        if (!Number.isSafeInteger(field.valueAsNumber) || field.valueAsNumber < 1) return;
        editSettings(ui.key, settings => update(settings, field.valueAsNumber));
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
                        <label class="st-auto-notes__field">
                            Проверять раз в N постов
                            <input type="number" class="text_pole" data-field="activationEvery" min="1" step="1" inputmode="numeric" aria-describedby="st-auto-notes-activation-help">
                        </label>
                    </div>
                    <p class="st-auto-notes__help" id="st-auto-notes-activation-help">Считаются новые посты игрока. 1 — проверять каждый пост.</p>
                </div>
                <div class="st-auto-notes__section">
                    <label class="st-auto-notes__field">
                        Как отбирать правила
                        <select class="text_pole" data-field="mode"></select>
                    </label>
                    <div data-field="vectorOptions">
                        <p class="st-auto-notes__help">Поиск использует источник из Vector Storage. Если поле «Когда срабатывает» пустое, поиск идёт по тексту правила.</p>
                        <label class="st-auto-notes__field">
                            Порог совпадения
                            <input type="number" class="text_pole" data-field="vectorThreshold" min="0" max="1" step="0.01" inputmode="decimal">
                        </label>
                        <p class="st-auto-notes__help">Выше порог — меньше совпадений.</p>
                    </div>
                    <div data-field="selectorOptions">
                        <h3><label for="st-auto-notes-selector">Общий промпт отбора</label></h3>
                        <p id="st-auto-notes-selector-help" class="st-auto-notes__help">По этой инструкции модель решает, какие промпты подходят к текущей сцене.</p>
                        <textarea id="st-auto-notes-selector" class="text_pole" data-field="selectorPrompt" rows="4" aria-describedby="st-auto-notes-selector-help"></textarea>
                    </div>
                    <div class="st-auto-notes__connection-test">
                        <button type="button" class="menu_button" data-field="testButton">Проверить подключение</button>
                        <p class="st-auto-notes__status" data-field="testStatus" role="status"></p>
                    </div>
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
    ui.profile.addEventListener('change', () => {
        editSettings(ui.key, settings => { settings.profileId = ui.profile.value; });
        clearConnectionTest();
    });
    ui.testButton.addEventListener('click', testConnection);
    ui.mode.replaceChildren(...Object.entries(VECTOR_MODES).map(([value, label]) => new Option(label, value)));
    ui.mode.addEventListener('change', () => {
        editSettings(ui.key, settings => { settings.mode = ui.mode.value; });
        clearConnectionTest();
        ui.status.textContent = '';
        renderMode();
    });
    ui.vectorThreshold.addEventListener('input', () => {
        const value = ui.vectorThreshold.valueAsNumber;
        if (!Number.isFinite(value) || value < 0 || value > 1) return;
        editSettings(ui.key, settings => { settings.vectorThreshold = value; });
    });
    ui.vectorThreshold.addEventListener('change', () => {
        ui.vectorThreshold.value = chatSettings(getContext(), ui.key)?.vectorThreshold ?? DEFAULT_VECTOR_THRESHOLD;
    });
    bindNumber(ui.maxPrompts, 'maxPrompts', DEFAULT_MAX_PROMPTS);
    bindNumber(ui.historyCount, 'historyCount', DEFAULT_HISTORY_COUNT);
    bindNumber(ui.activationEvery, 'activationEvery', DEFAULT_ACTIVATION_EVERY, (settings, value) => {
        if ((settings.activationEvery ?? DEFAULT_ACTIVATION_EVERY) === value) return;
        settings.activationEvery = value;
        settings.activationStart = countPlayerPosts(getContext().chat);
        ui.status.textContent = '';
    }, 'change');
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
