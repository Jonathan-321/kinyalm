const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const elements = {
  composer: $("#composer"),
  input: $("#message-input"),
  send: $("#send-button"),
  conversation: $("#conversation"),
  empty: $("#empty-state"),
  messages: $("#message-list"),
  title: $("#conversation-title"),
  modeLabel: $("#topbar-mode"),
  language: $("#language-select"),
  level: $("#level-select"),
  notice: $("#connection-notice"),
  status: $("#runtime-status"),
  location: $("#runtime-location"),
  statusDot: $("#status-dot"),
  details: $("#details-panel"),
  detailsList: $("#details-list"),
  lastRun: $("#last-run"),
  lastRunMetrics: $("#last-run-metrics"),
  sidebar: $("#sidebar"),
  sidebarScrim: $("#sidebar-scrim"),
  feedbackDialog: $("#feedback-dialog"),
  feedbackForm: $("#feedback-form"),
  correction: $("#correction-input"),
  toast: $("#toast"),
};

const state = {
  conversationId: crypto.randomUUID(),
  messages: [],
  mode: "converse",
  busy: false,
  controller: null,
  health: null,
  pendingFeedback: null,
  toastTimer: null,
  activeRequest: null,
};

function runtimeReady() {
  return state.health?.status === "ready";
}

function icon(paths) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  paths.forEach(([tag, attributes]) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    svg.append(node);
  });
  return svg;
}

const icons = {
  copy: () => icon([
    ["rect", { width: "14", height: "14", x: "8", y: "8", rx: "2" }],
    ["path", { d: "M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" }],
  ]),
  thumbsUp: () => icon([
    ["path", { d: "M7 10v12" }],
    ["path", { d: "M15 5.9 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2h0a3.13 3.13 0 0 1 3 3.9Z" }],
  ]),
  thumbsDown: () => icon([
    ["path", { d: "M17 14V2" }],
    ["path", { d: "M9 18.1 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22h0a3.13 3.13 0 0 1-3-3.9Z" }],
  ]),
  retry: () => icon([
    ["path", { d: "M3 12a9 9 0 1 0 3-6.7L3 8" }],
    ["path", { d: "M3 3v5h5" }],
  ]),
};

function setMode(mode) {
  state.mode = mode;
  $$(".mode-button").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  const active = $(`.mode-button[data-mode="${mode}"]`);
  elements.modeLabel.textContent = active?.textContent || "Converse";
}

function autoResize() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 142)}px`;
  elements.send.disabled = state.busy
    ? false
    : !elements.input.value.trim() || !runtimeReady();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
  });
}

function showToast(message) {
  clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  state.toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 2200);
}

function setBusy(busy) {
  state.busy = busy;
  elements.send.classList.toggle("busy", busy);
  elements.send.disabled = busy
    ? false
    : !elements.input.value.trim() || !runtimeReady();
  elements.send.setAttribute("aria-label", busy ? "Stop response" : "Send message");
  elements.send.title = busy ? "Stop" : "Send";
}

function messageFooter(message) {
  const footer = document.createElement("div");
  footer.className = "message-footer";

  const actions = [
    ["copy", "Copy response", icons.copy],
    ["up", "Good response", icons.thumbsUp],
    ["down", "Needs correction", icons.thumbsDown],
    ["retry", "Regenerate response", icons.retry],
  ];
  actions.forEach(([action, label, iconFactory]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "icon-button";
    button.dataset.action = action;
    button.dataset.messageId = message.id;
    button.setAttribute("aria-label", label);
    button.title = label;
    button.append(iconFactory());
    footer.append(button);
  });

  if (message.metrics) {
    const metrics = document.createElement("span");
    metrics.className = "message-metrics";
    const speed = Number(message.metrics.tokens_per_second || 0).toFixed(1);
    const first = Number(message.metrics.first_token_seconds || 0).toFixed(1);
    metrics.textContent = `${speed} tok/s · ${first}s first token`;
    footer.append(metrics);
  }
  return footer;
}

function renderMessage(message, streaming = false) {
  const article = document.createElement("article");
  article.className = `message ${message.role}${streaming ? " streaming" : ""}`;
  article.dataset.messageId = message.id;

  if (message.role === "user") {
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.content;
    article.append(content);
    return article;
  }

  const avatar = document.createElement("img");
  avatar.className = "message-avatar";
  avatar.src = "/assets/kinyalm-mark.png";
  avatar.alt = "";
  avatar.width = 36;
  avatar.height = 36;

  const body = document.createElement("div");
  body.className = "message-body";
  const author = document.createElement("div");
  author.className = "message-author";
  author.textContent = "KinyaLM";
  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = message.content;
  body.append(author, content);
  if (!streaming) body.append(messageFooter(message));
  article.append(avatar, body);
  return article;
}

function appendMessage(message, streaming = false) {
  elements.empty.classList.add("hidden");
  const node = renderMessage(message, streaming);
  elements.messages.append(node);
  scrollToBottom();
  return node;
}

function updateConversationTitle(content) {
  if (state.messages.length > 1) return;
  const title = content.replace(/\s+/g, " ").trim();
  elements.title.textContent = title.length > 32 ? `${title.slice(0, 31)}…` : title;
}

function resetConversation() {
  state.controller?.abort();
  state.controller = null;
  state.activeRequest = null;
  state.conversationId = crypto.randomUUID();
  state.messages = [];
  state.pendingFeedback = null;
  elements.messages.replaceChildren();
  elements.empty.classList.remove("hidden");
  elements.title.textContent = "New conversation";
  elements.input.value = "";
  elements.notice.classList.add("hidden");
  delete elements.notice.dataset.source;
  elements.lastRun.classList.add("hidden");
  elements.lastRunMetrics.replaceChildren();
  setBusy(false);
  autoResize();
  elements.input.focus();
}

async function readNdjson(response, onEvent) {
  if (!response.body) throw new Error("Streaming is unavailable in this browser");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

async function sendMessage(content) {
  if (state.busy || !content.trim()) return;
  if (!runtimeReady()) {
    elements.notice.textContent = "KinyaLM is still loading. Your message is ready to send when the model is ready.";
    elements.notice.dataset.source = "loading";
    elements.notice.classList.remove("hidden");
    return;
  }
  const requestKey = crypto.randomUUID();
  const conversationId = state.conversationId;
  const requestSettings = {
    mode: state.mode,
    language: elements.language.value,
    level: elements.level.value,
  };
  state.activeRequest = requestKey;
  const userMessage = {
    id: crypto.randomUUID(),
    role: "user",
    content: content.trim(),
  };
  state.messages.push(userMessage);
  appendMessage(userMessage);
  updateConversationTitle(userMessage.content);
  elements.input.value = "";
  autoResize();

  const assistantMessage = {
    id: crypto.randomUUID(),
    role: "assistant",
    content: "",
    metrics: null,
    settings: requestSettings,
  };
  const assistantNode = appendMessage(assistantMessage, true);
  const contentNode = assistantNode.querySelector(".message-content");
  state.controller = new AbortController();
  setBusy(true);
  elements.notice.classList.add("hidden");
  delete elements.notice.dataset.source;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: conversationId,
        ...requestSettings,
        messages: state.messages.map(({ role, content: text }) => ({ role, content: text })),
      }),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `Request failed (${response.status})`);
    }

    await readNdjson(response, (event) => {
      if (state.activeRequest !== requestKey) return;
      if (event.type === "delta") {
        assistantMessage.content += event.text;
        contentNode.textContent = assistantMessage.content;
        scrollToBottom();
      } else if (event.type === "done") {
        assistantMessage.content = event.response;
        assistantMessage.metrics = event.metrics || null;
        contentNode.textContent = assistantMessage.content;
      } else if (event.type === "error") {
        throw new Error(event.error || "Generation failed");
      }
    });

    if (state.activeRequest !== requestKey) return;

    assistantNode.classList.remove("streaming");
    assistantNode.querySelector(".message-body").append(messageFooter(assistantMessage));
    state.messages.push(assistantMessage);
    showLastRun(assistantMessage.metrics);
  } catch (error) {
    if (state.activeRequest !== requestKey) return;
    assistantNode.classList.remove("streaming");
    if (error.name === "AbortError") {
      if (assistantMessage.content) {
        assistantMessage.content = assistantMessage.content.trim();
        contentNode.textContent = assistantMessage.content;
        assistantNode.querySelector(".message-body").append(messageFooter(assistantMessage));
        state.messages.push(assistantMessage);
      } else {
        assistantNode.remove();
        if (state.messages.at(-1)?.id === userMessage.id) state.messages.pop();
      }
      showToast("Response stopped");
    } else {
      assistantNode.remove();
      if (state.messages.at(-1)?.id === userMessage.id) state.messages.pop();
      elements.notice.textContent = error.message;
      elements.notice.dataset.source = "request";
      elements.notice.classList.remove("hidden");
    }
  } finally {
    if (state.activeRequest === requestKey) {
      state.activeRequest = null;
      state.controller = null;
      setBusy(false);
      elements.input.focus();
      scrollToBottom();
    }
  }
}

function showLastRun(metrics) {
  if (!metrics) return;
  const rows = [
    ["First token", `${Number(metrics.first_token_seconds || 0).toFixed(2)} s`],
    ["Generation", `${Number(metrics.tokens_per_second || 0).toFixed(2)} tok/s`],
    ["Prompt cache", `${metrics.cached_input_tokens || 0} / ${metrics.input_tokens || 0} tokens`],
    ["Output", `${metrics.output_tokens || 0} tokens`],
    ["Total time", `${Number(metrics.latency_seconds || 0).toFixed(2)} s`],
    ["Peak memory", `${Number(metrics.peak_unified_memory_gb || 0).toFixed(2)} GB`],
  ];
  elements.lastRunMetrics.replaceChildren();
  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    row.append(dt, dd);
    elements.lastRunMetrics.append(row);
  });
  elements.lastRun.classList.remove("hidden");
}

function updateHealth(health) {
  state.health = health;
  const status = health.status || "error";
  elements.statusDot.className = `status-dot ${status}`;
  elements.status.textContent = status === "ready" ? "Ready locally" : status === "loading" ? "Loading model" : "Runtime error";
  elements.location.textContent = health.runtime?.location || "On this Mac";

  const runtime = health.runtime || {};
  const rows = [
    ["Status", elements.status.textContent],
    ["Model", runtime.base_model || "Loading"],
    ["Checkpoint", runtime.checkpoint || "—"],
    ["Backend", runtime.backend || "—"],
    ["Quantization", runtime.quantization || "—"],
    ["Location", runtime.location || "On this Mac"],
  ];
  elements.detailsList.replaceChildren();
  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    row.append(dt, dd);
    elements.detailsList.append(row);
  });
  if (status === "error" && health.error) {
    elements.notice.textContent = health.error;
    elements.notice.dataset.source = "health";
    elements.notice.classList.remove("hidden");
  } else if (
    status === "ready"
    && ["health", "loading"].includes(elements.notice.dataset.source)
  ) {
    elements.notice.classList.add("hidden");
    delete elements.notice.dataset.source;
  }
  autoResize();
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Local service unavailable");
    updateHealth(await response.json());
  } catch (error) {
    updateHealth({ status: "error", error: error.message, runtime: {} });
  }
}

function feedbackPayload(message, rating, correction = "") {
  const index = state.messages.findIndex((item) => item.id === message.id);
  const userPrompt = [...state.messages.slice(0, index)].reverse().find((item) => item.role === "user");
  return {
    conversation_id: state.conversationId,
    message_id: message.id,
    user_prompt: userPrompt?.content || "Unknown prompt",
    response: message.content,
    correction,
    rating,
    mode: message.settings?.mode || state.mode,
    language: message.settings?.language || elements.language.value,
    level: message.settings?.level || elements.level.value,
  };
}

async function saveFeedback(message, rating, correction = "") {
  const response = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(feedbackPayload(message, rating, correction)),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || "Could not save review");
  }
  showToast("Review saved locally");
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) {
    state.controller?.abort();
    return;
  }
  sendMessage(elements.input.value);
});

elements.input.addEventListener("input", autoResize);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

$$(".mode-button").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
$$("[data-prompt]").forEach((button) => button.addEventListener("click", () => {
  if (button.dataset.modeTarget) setMode(button.dataset.modeTarget);
  sendMessage(button.dataset.prompt);
}));

$("#new-chat").addEventListener("click", resetConversation);
$("#clear-chat").addEventListener("click", resetConversation);
$("#runtime-details-button").addEventListener("click", () => {
  elements.details.classList.add("open");
  elements.details.setAttribute("aria-hidden", "false");
});
$("#runtime-summary").addEventListener("click", () => $("#runtime-details-button").click());
$("#details-close").addEventListener("click", () => {
  elements.details.classList.remove("open");
  elements.details.setAttribute("aria-hidden", "true");
});

function setSidebar(open) {
  elements.sidebar.classList.toggle("open", open);
  elements.sidebarScrim.classList.toggle("open", open);
}
$("#mobile-menu").addEventListener("click", () => setSidebar(true));
$("#sidebar-close").addEventListener("click", () => setSidebar(false));
elements.sidebarScrim.addEventListener("click", () => setSidebar(false));

elements.messages.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-action]");
  if (!button) return;
  const message = state.messages.find((item) => item.id === button.dataset.messageId);
  if (!message) return;
  const action = button.dataset.action;
  try {
    if (action === "copy") {
      await navigator.clipboard.writeText(message.content);
      showToast("Response copied");
    } else if (action === "up") {
      await saveFeedback(message, "up");
    } else if (action === "down") {
      state.pendingFeedback = message;
      elements.correction.value = "";
      elements.feedbackDialog.showModal();
      elements.correction.focus();
    } else if (action === "retry") {
      const index = state.messages.findIndex((item) => item.id === message.id);
      const prompt = [...state.messages.slice(0, index)].reverse().find((item) => item.role === "user");
      state.messages = state.messages.slice(0, Math.max(0, index - 1));
      button.closest(".message").previousElementSibling?.remove();
      button.closest(".message").remove();
      if (message.settings) {
        setMode(message.settings.mode);
        elements.language.value = message.settings.language;
        elements.level.value = message.settings.level;
      }
      if (prompt) sendMessage(prompt.content);
    }
  } catch (error) {
    showToast(error.message);
  }
});

elements.feedbackForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.pendingFeedback) return;
  try {
    await saveFeedback(state.pendingFeedback, "down", elements.correction.value);
    elements.feedbackDialog.close();
    state.pendingFeedback = null;
  } catch (error) {
    showToast(error.message);
  }
});
$("#feedback-cancel").addEventListener("click", () => elements.feedbackDialog.close());
$("#feedback-close").addEventListener("click", () => elements.feedbackDialog.close());

setMode("converse");
autoResize();
refreshHealth();
setInterval(refreshHealth, 5000);
