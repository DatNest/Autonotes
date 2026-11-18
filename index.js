const SETTINGS_KEY = 'stAutoNotes';
const EXTENSION_VERSION = '1.5.1';
const UPDATE_SEEN_KEY = 'stAutoNotesUpdateSeen';
const RELEASE_NOTES = [
    'Все разделы закрыты после обновления страницы. Нажми на нужный заголовок, чтобы раскрыть его.',
    'Настройки разбиты на группы: подключение, частота и контекст, отбор правил, перегенерация.',
    'Импорт и экспорт находятся под «Добавить промпт». Карандаш ✏️ справа от тегов включает редактирование: название, цвет, порядок стрелками и удаление крестиком с подтверждением.',
    'Добавляй теги кнопкой «+» на отдельной второй строке под лентой.',
    'Теги расположены одной лентой. Прокручивай её колесом мыши на ПК или свайпом на телефоне.',
    'Список промптов занимает небольшой блок примерно на пять закрытых записей. Остальные записи доступны при прокрутке внутри списка.',
];
let updateNoticeOpen = false;
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
const DEFAULT_SELECTOR_PROMPT = 'Ищи, где уместно добавить режиссёрскую заметку к ситуации. Выбирай правило, если в последних постах есть конкретный повод применить его в следующем ответе.';
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
    pendingInjection = undefined;
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

    return Array.from(notes).sort(([a], [b]) => a - b).map(([id, text]) => ({ id, text }));
}

const debugRuns = new Map();
let pendingInjection;

function noteAnchor(context) {
    const lastUser = context.chat.findLastIndex(message => message.is_user && isDialoguePost(message));
    if (lastUser < 0) return '';
    return String(vectorHash(JSON.stringify(context.chat.slice(0, lastUser + 1)
        .filter(isDialoguePost).map(message => [Boolean(message.is_user), message.mes]))));
}

function updateDebug(key, values) {
    debugRuns.set(key, { ...debugRuns.get(key), ...values });
    if (ui?.key === key) renderDebug();
}

function renderDebug() {
    if (!ui?.debugBody) return;
    const run = debugRuns.get(ui.key);
    ui.debugBody.replaceChildren();
    if (!run) {
        ui.debugBody.textContent = 'Проверок в этом чате ещё не было.';
        return;
    }
    const heading = document.createElement('p');
    heading.textContent = new Date(run.time).toLocaleString() + (run.reused ? ' · сохранённые подсказки' : '');
    const status = document.createElement('p');
    status.textContent = run.status ?? 'Подготовка…';
    const delivery = document.createElement('p');
    delivery.textContent = run.delivery ?? 'Включение в запрос ещё не проверено.';
    ui.debugBody.append(heading, status, delivery);
    for (const prompt of run.selected ?? []) {
        const item = document.createElement('details');
        item.className = 'st-auto-notes__debug-item';
        const title = document.createElement('summary');
        title.textContent = 'Промпт ' + prompt.id;
        const rule = document.createElement('p');
        rule.className = 'st-auto-notes__debug-text';
        rule.textContent = prompt.text;
        const note = document.createElement('p');
        note.className = 'st-auto-notes__debug-text';
        note.textContent = run.notes?.find(note => note.id === prompt.id)?.text ?? 'Подсказка не получена.';
        item.append(title, rule, note);
        ui.debugBody.append(item);
    }
}

function requestContainsNotes(data, notes) {
    const strings = [];
    const pending = [data];
    const seen = new Set();
    while (pending.length) {
        const value = pending.pop();
        if (typeof value === 'string') strings.push(value);
        else if (value && typeof value === 'object' && !seen.has(value)) {
            seen.add(value);
            pending.push(...Object.values(value));
        }
    }
    return notes.every(note => {
        const text = getContext().substituteParams(note.text);
        return strings.some(value => value.includes(text));
    });
}

function recordPreparedNotes(data, dryRun) {
    if (dryRun || !pendingInjection) return;
    const { key, notes } = pendingInjection;
    if (chatKey(getContext()) !== key) return;
    updateDebug(key, {
        delivery: requestContainsNotes(data, notes)
            ? 'Включение в собранный запрос подтверждено.'
            : 'Подсказки не найдены в собранном запросе.',
    });
}

function injectNotes(context, notes) {
    if (!notes.length) return;
    const block = ['[Автозаметки]', ...notes.map(note => note.text), '[/Автозаметки]'].join('\n\n');
    context.setExtensionPrompt(INJECTION_KEY, block, IN_CHAT, 0, false, SYSTEM);
    pendingInjection = { key: chatKey(context), notes };
}

globalThis.stAutoNotesBeforePrompt = async function (_chat, _contextSize, abortGeneration, type) {
    const context = getContext();
    clearInjection(context);
    if (type != null && !['normal', 'swipe', 'regenerate'].includes(type)) return;

    const key = chatKey(context);
    const settings = chatSettings(context, key);
    const turn = pendingPost;
    pendingPost = undefined;
    if (!settings?.enabled || !settings.profileId) return;

    if (type === 'swipe' || type === 'regenerate' || groupTurn?.reroll) {
        const saved = settings.lastNotes;
        if (settings.reuseNotes !== false && saved?.anchor === noteAnchor(context) && saved.notes?.length) {
            updateDebug(key, {
                time: Date.now(), selected: saved.selected, notes: saved.notes, reused: true,
                status: 'Используются подсказки предыдущего ответа.',
                delivery: 'Ожидается сборка запроса.',
            });
            injectNotes(context, saved.notes);
        } else {
            setStatus(key, 'Перегенерация без сохранённых подсказок.');
            if (debugRuns.has(key)) updateDebug(key, { delivery: 'Текущая перегенерация идёт без подсказок.' });
        }
        return;
    }

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
    updateDebug(key, {
        time: Date.now(), selected: [], notes: [], reused: false,
        status: 'Отбор правил…', delivery: 'Ожидается генерация подсказок.',
    });
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
    if (activeBatch === controller && chatKey(getContext()) === key) {
        updateDebug(key, { selected: (selected ?? []).map(({ id, text }) => ({ id, text })) });
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

    const activated = (selected ?? []).map(({ id, text }) => ({ id, text }));
    currentSettings.lastNotes = { time: Date.now(), anchor: noteAnchor(current), selected: activated, notes };
    current.saveSettingsDebounced();
    updateDebug(key, {
        notes,
        delivery: notes.length ? 'Ожидается сборка запроса.' : 'Подсказки для запроса отсутствуют.',
    });
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
    const run = debugRuns.get(key);
    if (run && !text.startsWith('До проверки правил')) {
        updateDebug(key, { status: text });
    }
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

const DEFAULT_TAG_COLOR = '#b78bb3';
let selectedTag = '';
let editingTags = false;
let openPromptIndexes = new Set();

function recordTags(entry, tags) {
    const prompt = readPrompt(entry);
    const ids = Array.isArray(prompt.tagIds) ? prompt.tagIds : [prompt.tagId];
    return [...new Set(ids)].filter(id => tags.some(tag => tag.id === id));
}

function withRecordTags(entry, tagIds) {
    const { tagId, ...prompt } = readPrompt(entry);
    return { ...prompt, tagIds };
}

function tagButton(label, count, id, color, onClick, pressed = selectedTag === id) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'st-auto-notes__tag';
    button.dataset.tag = id;
    button.setAttribute('aria-pressed', String(pressed));
    if (color) button.style.setProperty('--an-tag-color', color);
    const icon = document.createElement('i');
    icon.className = 'fa-solid fa-tag';
    icon.setAttribute('aria-hidden', 'true');
    const name = document.createElement('span');
    name.className = 'st-auto-notes__tag-name';
    name.textContent = label;
    const badge = document.createElement('span');
    badge.className = 'st-auto-notes__tag-count';
    badge.textContent = String(count);
    button.append(icon, name);
    if (count !== null) button.append(badge);
    button.addEventListener('click', () => {
        if (onClick) { onClick(); return; }
        selectedTag = id;
        renderPrompts();
    });
    return button;
}

function renderTags(drafts, tags) {
    const scrollLeft = ui.tagFilters.scrollLeft;
    ui.tagFilters.replaceChildren();
    ui.tagFilters.append(tagButton('Все', drafts.length, ''));
    ui.tagFilters.append(tagButton('Без тега', drafts.filter(prompt => !recordTags(prompt, tags).length).length, '__untagged'));
    for (const tag of tags) {
        const count = drafts.filter(prompt => recordTags(prompt, tags).includes(tag.id)).length;
        ui.tagFilters.append(editingTags ? tagEditor(tag, tags) : tagButton(tag.name, count, tag.id, tag.color));
    }
    ui.tagFilters.scrollLeft = scrollLeft;
}

function tagEditor(tag, tags) {
    const key = ui.key;
    const row = document.createElement('div');
    row.className = 'st-auto-notes__tag-edit';
    row.dataset.editTag = tag.id;
    const name = document.createElement('input');
    name.type = 'text';
    name.className = 'text_pole';
    name.value = tag.name;
    name.setAttribute('aria-label', 'Название тега');
    const color = document.createElement('input');
    color.type = 'color';
    color.value = tag.color;
    color.setAttribute('aria-label', 'Цвет тега ' + tag.name);
    name.addEventListener('change', () => {
        const value = name.value.trim();
        if (!value) { name.value = tag.name; return; }
        editSettings(key, settings => {
            const target = settings.tags.find(item => item.id === tag.id);
            if (target) target.name = value;
        });
        renderPrompts();
    });
    color.addEventListener('change', () => {
        editSettings(key, settings => {
            const target = settings.tags.find(item => item.id === tag.id);
            if (target) target.color = color.value;
        });
        renderPrompts();
    });
    row.append(color, name);
    for (const [offset, label, title] of [[-1, '←', 'Переместить тег влево'], [1, '→', 'Переместить тег вправо']]) {
        const move = document.createElement('button');
        move.type = 'button';
        move.textContent = label;
        move.title = title;
        move.setAttribute('aria-label', title + ': ' + tag.name);
        const index = tags.findIndex(item => item.id === tag.id);
        move.disabled = index + offset < 0 || index + offset >= tags.length;
        move.addEventListener('click', () => {
            editSettings(key, settings => {
                const index = settings.tags.findIndex(item => item.id === tag.id);
                const next = index + offset;
                if (index < 0 || next < 0 || next >= settings.tags.length) return;
                [settings.tags[index], settings.tags[next]] = [settings.tags[next], settings.tags[index]];
            });
            renderPrompts();
            [...ui.tagFilters.children].find(item => item.dataset.editTag === tag.id)?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        });
        row.append(move);
    }
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '×';
    remove.setAttribute('aria-label', 'Удалить тег ' + tag.name);
    remove.addEventListener('click', async () => {
        const answer = await showNotice('Удалить тег?', ['Хотите удалить этот тег: «' + tag.name + '»? Записи сохранятся.'], [['no', 'Нет'], ['yes', 'Да']]);
        if (answer !== 'yes' || ui.key !== key) return;
        editSettings(key, settings => {
            settings.prompts = settings.prompts.map(entry => {
                const ids = recordTags(entry, settings.tags);
                return ids.includes(tag.id) ? withRecordTags(entry, ids.filter(id => id !== tag.id)) : entry;
            });
            settings.tags = settings.tags.filter(item => item.id !== tag.id);
        });
        if (selectedTag === tag.id) selectedTag = '';
        renderPrompts();
    });
    row.append(remove);
    return row;
}

function renderPrompts(focusIndex) {
    const { key, prompts } = ui;
    prompts.replaceChildren();
    const settings = chatSettings(getContext(), key);
    const drafts = settings?.prompts ?? [];
    const tags = settings?.tags ?? [];
    if (selectedTag && selectedTag !== '__untagged' && !tags.some(tag => tag.id === selectedTag)) selectedTag = '';
    renderTags(drafts, tags);
    ui.empty.hidden = drafts.length > 0;
    ui.filterEmpty.hidden = true;
    const isVector = ui.mode.value === 'vector';

    drafts.forEach((entry, index) => {
        const prompt = readPrompt(entry);
        const tagIds = recordTags(entry, tags);
        if (selectedTag && (selectedTag === '__untagged' ? tagIds.length > 0 : !tagIds.includes(selectedTag))) return;
        const card = document.createElement('details');
        card.className = 'st-auto-notes__card';
        card.dataset.priority = prompt.priority;
        card.innerHTML = `
            <summary class="st-auto-notes__prompt-summary">
                <span class="st-auto-notes__prompt-heading">Промпт ${index + 1}</span>
                <span class="st-auto-notes__summary-tags"></span>
            </summary>
            <div class="st-auto-notes__prompt-content">
            <div class="st-auto-notes__card-head">
                <button type="button" class="menu_button" aria-label="Удалить промпт ${index + 1}">Удалить</button>
            </div>
            <label class="st-auto-notes__field st-auto-notes__trigger" ${isVector ? '' : 'hidden'}>
                Когда срабатывает
                <textarea class="text_pole" rows="2" placeholder="Например, кто-то заметил проявление сил Кларка"></textarea>
            </label>
            <label class="st-auto-notes__field">
                Что подсказать
                <textarea id="st-auto-notes-prompt-${index}" class="text_pole" rows="4"></textarea>
            </label>
            <label class="st-auto-notes__priority">
                Приоритет
                <select class="text_pole" aria-label="Приоритет промпта ${index + 1}"></select>
            </label>
            <div class="st-auto-notes__record-tags">
                <span>Теги</span>
                <div class="st-auto-notes__tag-filters" role="group" aria-label="Теги промпта ${index + 1}"></div>
            </div>
            </div>`;
        card.open = openPromptIndexes.has(index) || index === focusIndex;
        if (card.open) openPromptIndexes.add(index);
        card.addEventListener('toggle', event => {
            if (event.target !== card || !card.isConnected) return;
            if (card.open) openPromptIndexes.add(index);
            else openPromptIndexes.delete(index);
        });
        const summaryTags = card.querySelector('.st-auto-notes__summary-tags');
        for (const tag of tags.filter(tag => tagIds.includes(tag.id))) {
            const badge = document.createElement('span');
            badge.className = 'st-auto-notes__tag';
            badge.style.setProperty('--an-tag-color', tag.color);
            const icon = document.createElement('i');
            icon.className = 'fa-solid fa-tag';
            icon.setAttribute('aria-hidden', 'true');
            badge.append(icon, document.createTextNode(tag.name));
            summaryTags.append(badge);
        }
        const field = card.querySelector('textarea[id]');
        const trigger = card.querySelector('.st-auto-notes__trigger textarea');
        const priority = card.querySelector('select');
        const remove = card.querySelector('button');
        const tagChoices = card.querySelector('.st-auto-notes__record-tags .st-auto-notes__tag-filters');
        for (const tag of tags) {
            const choice = tagButton(tag.name, null, tag.id, tag.color, () => {
                editSettings(key, settings => {
                    const ids = recordTags(settings.prompts[index], settings.tags ?? []);
                    const next = ids.includes(tag.id) ? ids.filter(id => id !== tag.id) : [...ids, tag.id];
                    settings.prompts[index] = withRecordTags(settings.prompts[index], next);
                });
                renderPrompts();
                ui.prompts.querySelector(`[aria-label="Теги промпта ${index + 1}"] [data-tag="${tag.id}"]`)?.focus();
            }, tagIds.includes(tag.id));
            tagChoices.append(choice);
        }
        if (!tags.length) {
            const hint = document.createElement('span');
            hint.className = 'st-auto-notes__help';
            hint.textContent = 'Создай теги кнопкой «+» под лентой тегов.';
            tagChoices.append(hint);
        }
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
            openPromptIndexes = new Set([...openPromptIndexes]
                .filter(value => value !== index)
                .map(value => value > index ? value - 1 : value));
            const remaining = chatSettings(getContext(), key)?.prompts.length ?? 0;
            renderPrompts();
            ui.prompts.querySelector('summary')?.focus();
            if (!remaining) ui.add.focus();
        });

        prompts.append(card);
        if (index === focusIndex) field.focus();
    });
    ui.filterEmpty.hidden = !drafts.length || prompts.childElementCount > 0;
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
    addTagDialog?.close();
    ui.key = chatKey(context);
    selectedTag = '';
    openPromptIndexes.clear();
    editingTags = false;
    ui.tagSettingsButton.setAttribute('aria-pressed', 'false');
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
    ui.reuseNotes.checked = settings?.reuseNotes !== false;
    ui.mode.value = settings?.mode ?? 'vector';
    ui.vectorThreshold.value = settings?.vectorThreshold ?? DEFAULT_VECTOR_THRESHOLD;
    ui.selectorPrompt.value = settings?.selectorPrompt ?? DEFAULT_SELECTOR_PROMPT;
    ui.status.textContent = '';
    renderProfiles(context);
    if (!debugRuns.has(ui.key) && settings?.lastNotes) {
        updateDebug(ui.key, {
            time: settings.lastNotes.time ?? Date.now(), selected: settings.lastNotes.selected,
            notes: settings.lastNotes.notes, status: 'Сохранённые подсказки последнего хода.',
            delivery: 'Статус отправки до перезагрузки недоступен.',
        });
    }
    renderDebug();
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

function parseProfileJson(text) {
    const value = JSON.parse(text);
    const source = Array.isArray(value) ? { prompts: value } : value?.profile ?? value;
    if (!source || typeof source !== 'object' || !Array.isArray(source.prompts)) {
        throw new Error('В JSON нужен список prompts или массив записей.');
    }
    if (source.tags !== undefined && !Array.isArray(source.tags)) throw new Error('Поле tags должно быть массивом.');

    const tags = (source.tags ?? []).map(tag => {
        if (!tag || typeof tag.id !== 'string' || !tag.id || typeof tag.name !== 'string' || !tag.name.trim()) {
            throw new Error('У каждого тега нужны id и название.');
        }
        return { id: tag.id, name: tag.name.trim(), color: /^#[0-9a-f]{6}$/i.test(tag.color) ? tag.color : DEFAULT_TAG_COLOR };
    });
    if (new Set(tags.map(tag => tag.id)).size !== tags.length) throw new Error('В JSON повторяются id тегов.');
    const prompts = source.prompts.map(entry => {
        const prompt = readPrompt(entry);
        if (!prompt || typeof prompt.text !== 'string') throw new Error('У каждой записи нужен текст text.');
        if (prompt.trigger !== undefined && typeof prompt.trigger !== 'string') throw new Error('Условие trigger должно быть текстом.');
        if (prompt.tagIds !== undefined && (!Array.isArray(prompt.tagIds) || prompt.tagIds.some(id => typeof id !== 'string'))) {
            throw new Error('Теги записи tagIds должны быть массивом строк.');
        }
        return {
            text: prompt.text,
            trigger: prompt.trigger ?? '',
            priority: [1, 2, 3].includes(prompt.priority) ? prompt.priority : 2,
            tagIds: recordTags(prompt, tags),
        };
    });
    const profile = { enabled: false, profileId: '', prompts, tags };
    for (const name of ['profileId', 'selectorPrompt']) {
        if (source[name] !== undefined) {
            if (typeof source[name] !== 'string') throw new Error('Некорректное поле ' + name + '.');
            profile[name] = source[name];
        }
    }
    if (source.reuseNotes !== undefined) {
        if (typeof source.reuseNotes !== 'boolean') throw new Error('Некорректное поле reuseNotes.');
        profile.reuseNotes = source.reuseNotes;
    }
    if (source.enabled !== undefined) {
        if (typeof source.enabled !== 'boolean') throw new Error('Некорректное поле enabled.');
        profile.enabled = source.enabled;
    }
    if (source.mode !== undefined) {
        if (!Object.hasOwn(VECTOR_MODES, source.mode)) throw new Error('Неизвестный режим отбора.');
        profile.mode = source.mode;
    }
    for (const name of ['maxPrompts', 'historyCount', 'activationEvery']) {
        if (source[name] === undefined) continue;
        if (!Number.isSafeInteger(source[name]) || source[name] < 1) throw new Error('Некорректное поле ' + name + '.');
        profile[name] = source[name];
    }
    if (source.vectorThreshold !== undefined) {
        if (!Number.isFinite(source.vectorThreshold) || source.vectorThreshold < 0 || source.vectorThreshold > 1) {
            throw new Error('Порог совпадения должен быть от 0 до 1.');
        }
        profile.vectorThreshold = source.vectorThreshold;
    }
    return profile;
}

function appendProfile(settings, incoming) {
    settings.tags ??= [];
    const ids = new Map();
    for (const tag of incoming.tags) {
        const existing = settings.tags.find(item => item.name === tag.name && item.color === tag.color);
        if (existing) {
            ids.set(tag.id, existing.id);
        } else {
            const id = globalThis.crypto.randomUUID();
            settings.tags.push({ ...tag, id });
            ids.set(tag.id, id);
        }
    }
    settings.prompts.push(...incoming.prompts.map(prompt => ({
        ...prompt,
        tagIds: prompt.tagIds.map(id => ids.get(id)),
    })));
}

function createDialog(title) {
    const previousFocus = document.activeElement;
    const dialog = document.createElement('dialog');
    dialog.className = 'st-auto-notes st-auto-notes__confirm';
    // Keep modal interactions inside this dialog instead of reaching ST's drawer handlers.
    for (const event of ['click', 'pointerdown', 'pointerup', 'mousedown', 'mouseup', 'touchstart', 'touchend', 'keydown', 'keyup']) {
        dialog.addEventListener(event, event => event.stopPropagation());
    }
    const heading = document.createElement('h3');
    heading.id = 'st-auto-notes-dialog-' + globalThis.crypto.randomUUID();
    heading.textContent = title;
    dialog.setAttribute('aria-labelledby', heading.id);
    dialog.append(heading);
    const closed = new Promise(resolve => dialog.addEventListener('close', event => {
        event.stopPropagation();
        const result = dialog.returnValue;
        dialog.remove();
        if (previousFocus?.isConnected) previousFocus.focus({ preventScroll: true });
        resolve(result);
    }, { once: true }));
    document.body.append(dialog);
    return { dialog, closed };
}

function showNotice(title, paragraphs, buttons) {
    const { dialog, closed } = createDialog(title);
    for (const text of paragraphs) {
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        dialog.append(paragraph);
    }
    const actions = document.createElement('div');
    actions.className = 'st-auto-notes__dialog-actions';
    for (const [value, label] of buttons) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'menu_button';
        button.textContent = label;
        button.addEventListener('click', () => dialog.close(value));
        actions.append(button);
    }
    actions.firstElementChild.autofocus = true;
    dialog.append(actions);
    dialog.showModal();
    return closed;
}

let addTagDialog;

function showAddTagDialog() {
    if (addTagDialog || !ui.key) return;
    const key = ui.key;
    const { dialog, closed } = createDialog('Добавить тег');
    addTagDialog = dialog;
    const form = document.createElement('form');
    form.className = 'st-auto-notes__add-tag-form';
    const label = document.createElement('label');
    label.textContent = 'Название тега';
    const name = document.createElement('input');
    name.className = 'text_pole';
    name.type = 'text';
    name.required = true;
    name.autofocus = true;
    label.append(name);
    const colorLabel = document.createElement('label');
    colorLabel.textContent = 'Цвет тега';
    const color = document.createElement('input');
    color.type = 'color';
    color.value = DEFAULT_TAG_COLOR;
    colorLabel.append(color);
    const actions = document.createElement('div');
    actions.className = 'st-auto-notes__dialog-actions';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'menu_button';
    cancel.textContent = 'Отмена';
    cancel.addEventListener('click', () => dialog.close());
    const add = document.createElement('button');
    add.type = 'submit';
    add.className = 'menu_button';
    add.textContent = 'Добавить';
    actions.append(cancel, add);
    form.append(label, colorLabel, actions);
    form.addEventListener('submit', event => {
        event.preventDefault();
        const value = name.value.trim();
        if (!value || ui.key !== key) { name.focus(); return; }
        editSettings(key, settings => {
            settings.tags ??= [];
            settings.tags.push({ id: globalThis.crypto.randomUUID(), name: value, color: color.value });
        });
        renderPrompts();
        dialog.close();
    });
    dialog.append(form);
    closed.then(() => { if (addTagDialog === dialog) addTagDialog = undefined; });
    dialog.showModal();
}

function showImportDialog() {
    const key = ui.key;
    if (!key) return;
    const { dialog } = createDialog('Импорт JSON');
    const content = document.createElement('div');
    content.className = 'st-auto-notes__import-content';
    content.innerHTML = `
        <div class="st-auto-notes__file-row">
            <button type="button" class="menu_button" data-import="choose">Выбрать файл</button>
            <span data-import="filename">Файл не выбран</span>
            <input type="file" accept=".json,application/json" data-import="file" hidden>
        </div>
        <div class="st-auto-notes__import-mode">
            <span data-import="replaceLabel">Заменить</span>
            <label class="st-auto-notes__switch">
                <input type="checkbox" role="switch" aria-label="Добавить записи вместо замены" data-import="append" checked>
                <span class="st-auto-notes__switch-track" aria-hidden="true"></span>
            </label>
            <span data-import="appendLabel">Добавить</span>
        </div>
        <p class="st-auto-notes__import-description" data-import="description" role="status"></p>
        <p class="st-auto-notes__import-error" data-import="error" role="alert"></p>
        <div class="st-auto-notes__dialog-actions">
            <button type="button" class="menu_button" data-import="cancel">Отмена</button>
            <button type="button" class="menu_button" data-import="submit" disabled>Импортировать</button>
        </div>`;
    dialog.append(content);
    const controls = Object.fromEntries(Array.from(content.querySelectorAll('[data-import]'),
        element => [element.dataset.import, element]));
    let busy = false;

    function describeMode() {
        const append = controls.append.checked;
        controls.description.textContent = append
            ? 'Записи и теги из файла добавятся к текущим. Настройки текущего профиля сохранятся.'
            : 'Записи, теги и настройки текущего профиля будут заменены содержимым JSON.';
        controls.replaceLabel.dataset.active = String(!append);
        controls.appendLabel.dataset.active = String(append);
    }

    controls.choose.autofocus = true;
    controls.choose.addEventListener('click', () => controls.file.click());
    controls.file.addEventListener('change', () => {
        controls.filename.textContent = controls.file.files?.[0]?.name ?? 'Файл не выбран';
        controls.submit.disabled = !controls.file.files?.length;
        controls.error.textContent = '';
    });
    controls.append.addEventListener('change', describeMode);
    controls.cancel.addEventListener('click', () => dialog.close('cancel'));
    controls.submit.addEventListener('click', async () => {
        const file = controls.file.files?.[0];
        if (!file || busy) return;
        if (chatKey(getContext()) !== key) {
            controls.error.textContent = 'Чат изменился. Открой импорт в нужном чате заново.';
            return;
        }
        busy = true;
        controls.submit.disabled = true;
        controls.choose.disabled = true;
        controls.append.disabled = true;
        controls.error.textContent = '';
        try {
            const imported = await importProfile(file, controls.append.checked ? 'append' : 'replace', key,
                message => { controls.error.textContent = message; }, () => dialog.isConnected);
            if (imported && dialog.isConnected) dialog.close('imported');
        } finally {
            busy = false;
            controls.submit.disabled = !controls.file.files?.length;
            controls.choose.disabled = false;
            controls.append.disabled = false;
        }
    });
    describeMode();
    dialog.showModal();
}

function confirmProfileReplacement() {
    return showNotice('Переписать профиль?',
        ['Удалить старые записи и переписать новым JSON полностью?'],
        [['no', 'Нет'], ['yes', 'Да']]).then(result => result === 'yes');
}

async function showUpdateNotice() {
    const context = getContext();
    if (updateNoticeOpen || context.extensionSettings[UPDATE_SEEN_KEY] === EXTENSION_VERSION) return;
    updateNoticeOpen = true;
    try {
        const result = await showNotice('Что нового · Автозаметки ' + EXTENSION_VERSION,
            RELEASE_NOTES, [['understood', 'Понятно']]);
        if (result === 'understood') {
            const current = getContext();
            current.extensionSettings[UPDATE_SEEN_KEY] = EXTENSION_VERSION;
            current.saveSettingsDebounced();
        }
    } finally {
        updateNoticeOpen = false;
    }
}

async function importProfile(file, mode, key, reportError = message => setStatus(key, message), isActive = () => true) {
    try {
        const incoming = parseProfileJson(await file.text());
        if (!isActive() || chatKey(getContext()) !== key) return false;
        const existing = chatSettings(getContext(), key)?.prompts ?? [];
        if (mode === 'replace' && existing.length > 0 && !await confirmProfileReplacement()) return false;
        const context = getContext();
        if (!isActive() || chatKey(context) !== key) return false;
        resetTurn();
        vectorCache.clear();
        debugRuns.delete(key);
        if (mode === 'replace') {
            incoming.activationStart = countPlayerPosts(context.chat);
            context.extensionSettings[SETTINGS_KEY] ??= {};
            context.extensionSettings[SETTINGS_KEY][key] = incoming;
            context.saveSettingsDebounced();
        } else {
            editSettings(key, settings => appendProfile(settings, incoming));
        }
        renderChat();
        setStatus(key, `JSON загружен · записей ${incoming.prompts.length}.`);
        return true;
    } catch (error) {
        reportError(error instanceof SyntaxError ? 'Не удалось прочитать JSON. Проверь формат файла.' : error.message);
        return false;
    }
}

function exportProfile() {
    const settings = chatSettings(getContext(), ui.key);
    if (!settings) return;
    const { activationStart, lastNotes, ...profile } = settings;
    const blob = new Blob([JSON.stringify({ version: 1, profile }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'AutoNotes-profile.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
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
                <details class="st-auto-notes__section">
                    <summary class="st-auto-notes__section-summary">
                        <span class="st-auto-notes__chat">
                            <span class="st-auto-notes__section-title">Настройки</span>
                            <span class="st-auto-notes__chat-name" data-field="chatName"></span>
                        </span>
                    </summary>
                    <div class="st-auto-notes__section-content st-auto-notes__settings-groups">
                        <label class="st-auto-notes__toggle">
                            <input type="checkbox" data-field="enabled">
                            Включить в этом чате
                        </label>
                        <details class="st-auto-notes__settings-group">
                            <summary>Подключение</summary>
                            <div class="st-auto-notes__settings-content">
                                <label class="st-auto-notes__field">
                                    Профиль для подсказок
                                    <select class="text_pole" data-field="profile"></select>
                                </label>
                                <p class="st-auto-notes__help">Модель из этого профиля пишет подсказки. Источник векторов задаётся в Vector Storage.</p>
                                <button type="button" class="menu_button" data-field="testButton">Проверить подключение</button>
                                <p class="st-auto-notes__status" data-field="testStatus" role="status"></p>
                            </div>
                        </details>
                        <details class="st-auto-notes__settings-group">
                            <summary>Частота и контекст</summary>
                            <div class="st-auto-notes__settings-content">
                                <label class="st-auto-notes__field">
                                    Проверять раз в N постов
                                    <input type="number" class="text_pole" data-field="activationEvery" min="1" step="1" inputmode="numeric" aria-describedby="st-auto-notes-activation-help">
                                </label>
                                <p class="st-auto-notes__help" id="st-auto-notes-activation-help">Считаются новые посты игрока. 1 — проверять каждый пост.</p>
                                <label class="st-auto-notes__field">
                                    Последних постов читать
                                    <input type="number" class="text_pole" data-field="historyCount" min="1" step="1" inputmode="numeric">
                                </label>
                            </div>
                        </details>
                        <details class="st-auto-notes__settings-group">
                            <summary>Отбор правил</summary>
                            <div class="st-auto-notes__settings-content">
                                <label class="st-auto-notes__field">
                                    Способ отбора
                                    <select class="text_pole" data-field="mode"></select>
                                </label>
                                <label class="st-auto-notes__field">
                                    Макс. промптов за ход
                                    <input type="number" class="text_pole" data-field="maxPrompts" min="1" step="1" inputmode="numeric">
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
                                    <label class="st-auto-notes__field" for="st-auto-notes-selector">Общий промпт отбора</label>
                                    <p id="st-auto-notes-selector-help" class="st-auto-notes__help">По этой инструкции модель решает, какие промпты подходят к текущей сцене.</p>
                                    <textarea id="st-auto-notes-selector" class="text_pole" data-field="selectorPrompt" rows="4" aria-describedby="st-auto-notes-selector-help"></textarea>
                                </div>
                            </div>
                        </details>
                        <details class="st-auto-notes__settings-group">
                            <summary>Перегенерация</summary>
                            <div class="st-auto-notes__settings-content">
                                <label class="st-auto-notes__toggle">
                                    <input type="checkbox" data-field="reuseNotes">
                                    Сохранять подсказки при перегенерации
                                </label>
                                <p class="st-auto-notes__help">Готовые подсказки используются повторно на свайпах и при перегенерации того же хода. При выключении перегенерация идёт без подсказок.</p>
                            </div>
                        </details>
                    </div>
                </details>
                <details class="st-auto-notes__section">
                    <summary class="st-auto-notes__section-summary"><span class="st-auto-notes__section-title">Промпты для заметок</span></summary>
                    <div class="st-auto-notes__section-content">
                    <button type="button" class="menu_button" data-field="add">+ Добавить промпт</button>
                                <div class="st-auto-notes__json-actions">
                                    <button type="button" class="menu_button" data-field="importJson">Импорт</button>
                                    <button type="button" class="menu_button" data-field="exportJson">Экспорт</button>
                                </div>

                    <p class="st-auto-notes__help">При одинаковом приоритете первым выбирается промпт выше в списке.</p>
                    <div class="st-auto-notes__tag-bar">
                        <div class="st-auto-notes__tag-filters" data-field="tagFilters" role="group" aria-label="Фильтр по тегам" tabindex="0"></div>
                        <button type="button" class="st-auto-notes__tag-settings-button" data-field="tagSettingsButton" aria-label="Редактировать теги" title="Редактировать теги" aria-pressed="false">✏️</button>
                    </div>
                    <div class="st-auto-notes__tag-add-row">
                        <button type="button" class="st-auto-notes__tag-add-button" data-field="addTag" aria-label="Добавить тег" title="Добавить тег">+</button>
                    </div>

                    <div class="st-auto-notes__prompts" data-field="prompts" tabindex="0" role="region" aria-label="Список промптов"></div>
                    <p class="st-auto-notes__help" data-field="filterEmpty" hidden>С этим тегом пока нет записей.</p>
                    <p class="st-auto-notes__help" data-field="empty">Добавь правило, которое модель сможет применить к сцене.</p>
                    </div>
                </details>
                <details class="st-auto-notes__section">
                    <summary class="st-auto-notes__section-summary"><span class="st-auto-notes__section-title">Отладка</span></summary>
                    <div class="st-auto-notes__section-content" data-field="debugBody" aria-live="polite"></div>
                </details>
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
    ui.reuseNotes.addEventListener('change', () => editSettings(ui.key, settings => {
        settings.reuseNotes = ui.reuseNotes.checked;
    }));
    ui.testButton.addEventListener('click', testConnection);
    ui.importJson.addEventListener('click', showImportDialog);
    ui.tagSettingsButton.addEventListener('click', () => {
        editingTags = !editingTags;
        ui.tagSettingsButton.setAttribute('aria-pressed', String(editingTags));
        ui.tagSettingsButton.title = editingTags ? 'Завершить редактирование тегов' : 'Редактировать теги';
        renderPrompts();
    });
    ui.tagFilters.addEventListener('wheel', event => {
        const strip = ui.tagFilters;
        if (strip.scrollWidth <= strip.clientWidth || Math.abs(event.deltaX) >= Math.abs(event.deltaY)) return;
        const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? strip.clientWidth : 1;
        const before = strip.scrollLeft;
        strip.scrollLeft += event.deltaY * unit;
        if (strip.scrollLeft !== before) event.preventDefault();
    }, { passive: false });
    ui.exportJson.addEventListener('click', exportProfile);
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
    ui.addTag.addEventListener('click', showAddTagDialog);
    ui.add.addEventListener('click', () => {
        let index;
        editSettings(ui.key, settings => {
            settings.prompts.push({ text: '', priority: 2, tagIds: selectedTag && selectedTag !== '__untagged' ? [selectedTag] : [] });
            index = settings.prompts.length - 1;
        });
        renderPrompts(index);
    });
    panel.addEventListener('toggle', event => {
        if (event.target === panel && panel.open) {
            renderProfiles();
            void showUpdateNotice();
        }
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
        groupTurn = [undefined, null, 'normal', 'regenerate', 'swipe'].includes(type)
            ? { key: chatKey(getContext()), reroll: type === 'regenerate' || type === 'swipe' }
            : undefined;
    });
    eventSource.on(eventTypes.GROUP_WRAPPER_FINISHED, resetTurn);
    eventSource.on(eventTypes.GENERATION_STARTED, (_type, _options, dryRun) => {
        if (!dryRun) clearInjection();
    });
    eventSource.on(eventTypes.GENERATE_AFTER_DATA, (data, dryRun) => {
        recordPreparedNotes(data, dryRun);
        if (!dryRun) clearInjection();
    });
    eventSource.on(eventTypes.GENERATION_STOPPED, resetTurn);
    eventSource.on(eventTypes.APP_READY, () => renderProfiles());
    for (const event of [eventTypes.CONNECTION_PROFILE_CREATED, eventTypes.CONNECTION_PROFILE_UPDATED, eventTypes.CONNECTION_PROFILE_DELETED]) {
        if (event) eventSource.on(event, () => renderProfiles());
    }
    mount();
});
