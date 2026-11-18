const SETTINGS_KEY = 'stAutoNotes';
const EXTENSION_VERSION = '1.4.0';
const UPDATE_SEEN_KEY = 'stAutoNotesUpdateSeen';
const RELEASE_NOTES = [
    'Два режима загрузки JSON. В «Промптах для заметок» нажми «Загрузить JSON · Дополнить», чтобы добавить записи к текущим.',
    'Полная замена профиля. Нажми «Загрузить JSON · Переписать», выбери файл и подтверди замену кнопкой «Да». «Нет» отменяет загрузку.',
    'Перенос профиля. Кнопка «Скачать JSON» сохраняет записи, теги и настройки в файл для загрузки в другом чате.',
    'Окно обновлений. После обновления открой «Автозаметки», чтобы увидеть новые функции. Нажми «Понятно», чтобы отметить сводку прочитанной.',
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

const DEFAULT_TAG_COLOR = '#b78bb3';
let selectedTag = '';

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
    ui.tagFilters.replaceChildren();
    ui.tagFilters.append(tagButton('Все', drafts.length, ''));
    ui.tagFilters.append(tagButton('Без тега', drafts.filter(prompt => !recordTags(prompt, tags).length).length, '__untagged'));
    for (const tag of tags) {
        const count = drafts.filter(prompt => recordTags(prompt, tags).includes(tag.id)).length;
        ui.tagFilters.append(tagButton(tag.name, count, tag.id, tag.color));
    }
}

function renderTagEditor() {
    const key = ui.key;
    const tags = chatSettings(getContext(), key)?.tags ?? [];
    ui.tagEditor.replaceChildren();
    for (const tag of tags) {
        const row = document.createElement('div');
        row.className = 'st-auto-notes__tag-edit';
        const name = document.createElement('input');
        name.type = 'text';
        name.className = 'text_pole';
        name.value = tag.name;
        name.setAttribute('aria-label', 'Название тега');
        const color = document.createElement('input');
        color.type = 'color';
        color.value = tag.color;
        color.setAttribute('aria-label', 'Цвет тега');
        const remove = document.createElement('button');
        remove.className = 'menu_button';
        remove.type = 'button';
        remove.textContent = 'Удалить';
        remove.setAttribute('aria-label', 'Удалить тег ' + tag.name);
        name.addEventListener('change', () => {
            const value = name.value.trim();
            if (!value) { name.value = tag.name; return; }
            editSettings(key, settings => {
                settings.tags.find(item => item.id === tag.id).name = value;
            });
            renderPrompts();
        });
        color.addEventListener('input', () => {
            editSettings(key, settings => {
                settings.tags.find(item => item.id === tag.id).color = color.value;
            });
            renderPrompts();
        });
        remove.addEventListener('click', () => {
            editSettings(key, settings => {
                settings.prompts = settings.prompts.map(entry => {
                    const ids = recordTags(entry, settings.tags);
                    return ids.includes(tag.id)
                        ? withRecordTags(entry, ids.filter(id => id !== tag.id))
                        : entry;
                });
                settings.tags = settings.tags.filter(item => item.id !== tag.id);
            });
            if (selectedTag === tag.id) selectedTag = '';
            renderTagEditor();
            renderPrompts();
        });
        row.append(color, name, remove);
        ui.tagEditor.append(row);
    }
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
            </div>`;
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
            hint.textContent = 'Создай теги в «Настроить теги».';
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
            const remaining = chatSettings(getContext(), key)?.prompts.length ?? 0;
            renderPrompts(Math.min(index, remaining - 1));
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
    ui.key = chatKey(context);
    selectedTag = '';
    ui.tagName.value = '';
    ui.tagColor.value = DEFAULT_TAG_COLOR;
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
    renderTagEditor();
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

function showNotice(title, paragraphs, buttons) {
    return new Promise(resolve => {
        const dialog = document.createElement('dialog');
        dialog.className = 'st-auto-notes st-auto-notes__confirm';
        const heading = document.createElement('h3');
        heading.id = 'st-auto-notes-dialog-title';
        heading.textContent = title;
        dialog.setAttribute('aria-labelledby', heading.id);
        dialog.append(heading);
        for (const text of paragraphs) {
            const paragraph = document.createElement('p');
            paragraph.textContent = text;
            dialog.append(paragraph);
        }
        const form = document.createElement('form');
        form.method = 'dialog';
        for (const [value, label] of buttons) {
            const button = document.createElement('button');
            button.className = 'menu_button';
            button.value = value;
            button.textContent = label;
            form.append(button);
        }
        form.firstElementChild.autofocus = true;
        dialog.append(form);
        dialog.addEventListener('close', () => {
            const result = dialog.returnValue;
            dialog.remove();
            resolve(result);
        }, { once: true });
        document.body.append(dialog);
        dialog.showModal();
    });
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

async function importProfile(file, mode, key) {
    try {
        const incoming = parseProfileJson(await file.text());
        if (chatKey(getContext()) !== key) return;
        if (mode === 'replace' && !await confirmProfileReplacement()) return;
        const context = getContext();
        if (chatKey(context) !== key) return;
        resetTurn();
        vectorCache.clear();
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
    } catch (error) {
        setStatus(key, error instanceof SyntaxError ? 'Не удалось прочитать JSON. Проверь формат файла.' : error.message);
    }
}

function exportProfile() {
    const settings = chatSettings(getContext(), ui.key);
    if (!settings) return;
    const { activationStart, ...profile } = settings;
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
                    <div class="st-auto-notes__json-actions">
                        <button type="button" class="menu_button" data-field="importAppend">Загрузить JSON · Дополнить</button>
                        <button type="button" class="menu_button" data-field="importReplace">Загрузить JSON · Переписать</button>
                        <button type="button" class="menu_button" data-field="exportJson">Скачать JSON</button>
                    </div>
                    <p class="st-auto-notes__help">При одинаковом приоритете первым выбирается промпт выше в списке.</p>
                    <div class="st-auto-notes__tag-filters" data-field="tagFilters" role="group" aria-label="Фильтр по тегам"></div>
                    <details class="st-auto-notes__tag-manager">
                        <summary>Настроить теги</summary>
                        <div data-field="tagEditor"></div>
                        <div class="st-auto-notes__tag-edit">
                            <input type="color" data-field="tagColor" value="#b78bb3" aria-label="Цвет нового тега">
                            <input type="text" class="text_pole" data-field="tagName" placeholder="Название тега" aria-label="Название нового тега">
                            <button type="button" class="menu_button" data-field="addTag">+ Тег</button>
                        </div>
                    </details>
                    <div class="st-auto-notes__prompts" data-field="prompts"></div>
                    <p class="st-auto-notes__help" data-field="filterEmpty" hidden>С этим тегом пока нет записей.</p>
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
    for (const [button, mode] of [[ui.importAppend, 'append'], [ui.importReplace, 'replace']]) {
        button.addEventListener('click', () => {
            const key = ui.key;
            const picker = document.createElement('input');
            picker.type = 'file';
            picker.accept = '.json,application/json';
            picker.addEventListener('change', () => {
                const file = picker.files?.[0];
                if (file) void importProfile(file, mode, key);
            }, { once: true });
            picker.click();
        });
    }
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
    ui.addTag.addEventListener('click', () => {
        const name = ui.tagName.value.trim();
        if (!name) { ui.tagName.focus(); return; }
        editSettings(ui.key, settings => {
            settings.tags ??= [];
            settings.tags.push({ id: globalThis.crypto.randomUUID(), name, color: ui.tagColor.value });
        });
        ui.tagName.value = '';
        renderTagEditor();
        renderPrompts();
    });
    ui.tagName.addEventListener('keydown', event => {
        if (event.key === 'Enter') { event.preventDefault(); ui.addTag.click(); }
    });
    ui.add.addEventListener('click', () => {
        let index;
        editSettings(ui.key, settings => {
            settings.prompts.push({ text: '', priority: 2, tagIds: selectedTag && selectedTag !== '__untagged' ? [selectedTag] : [] });
            index = settings.prompts.length - 1;
        });
        renderPrompts(index);
    });
    panel.addEventListener('toggle', () => {
        if (panel.open) {
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
