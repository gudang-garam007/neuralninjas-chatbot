(function () {
  // CHANGE THIS to your deployed backend URL
  const API_URL = "https://neuralninjas-chatbot.onrender.com/chat";

  const sessionId = (() => {
    let id = localStorage.getItem("nn_chat_session");
    if (!id) {
      id = "s_" + Math.random().toString(36).slice(2) + Date.now();
      localStorage.setItem("nn_chat_session", id);
    }
    return id;
  })();

  // Ninja-star / circuit-node agent logo, reused for launcher + avatars
  const NN_LOGO_SVG = `
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="nnGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#22d3ee"/>
          <stop offset="100%" stop-color="#7b2cbf"/>
        </linearGradient>
      </defs>
      <g fill="none" stroke="url(#nnGrad)" stroke-width="1.6" stroke-linejoin="round">
        <path d="M24 3 L28 20 L45 24 L28 28 L24 45 L20 28 L3 24 L20 20 Z" fill="rgba(34,211,238,0.08)"/>
      </g>
      <circle cx="24" cy="24" r="5" fill="#0c0c12" stroke="url(#nnGrad)" stroke-width="1.6"/>
      <circle cx="24" cy="24" r="2" fill="#22d3ee"/>
      <g stroke="url(#nnGrad)" stroke-width="1" opacity="0.8">
        <line x1="24" y1="14" x2="24" y2="19"/>
        <line x1="24" y1="29" x2="24" y2="34"/>
        <line x1="14" y1="24" x2="19" y2="24"/>
        <line x1="29" y1="24" x2="34" y2="24"/>
      </g>
    </svg>`;

  const CLOSE_SVG = `
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M6 6L18 18M6 18L18 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    </svg>`;

  const SEND_SVG = `
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M3 11L21 3L13 21L11 13L3 11Z" fill="#04070d"/>
    </svg>`;

  const root = document.createElement("div");
  root.id = "nn-chat-root";
  document.body.appendChild(root);

  const launcher = document.createElement("button");
  launcher.id = "nn-chat-launcher";
  launcher.setAttribute("aria-label", "Open Neural Ninja AI assistant");
  launcher.innerHTML = NN_LOGO_SVG;
  root.appendChild(launcher);

  const win = document.createElement("div");
  win.id = "nn-chat-window";
  win.innerHTML = `
    <div id="nn-chat-header">
      <div class="nn-avatar">${NN_LOGO_SVG}</div>
      <div id="nn-chat-header-text">
        <div id="nn-chat-header-eyebrow" class="nn-mono">// AI ASSISTANT</div>
        <div id="nn-chat-header-title" class="nn-display">Neural Ninja AI</div>
      </div>
      <button id="nn-chat-close" aria-label="Close chat">${CLOSE_SVG}</button>
    </div>
    <div id="nn-chat-messages"></div>
    <div id="nn-chat-input-row">
      <input id="nn-chat-input" type="text" placeholder="Ask me anything..." autocomplete="off" />
      <button id="nn-chat-send" aria-label="Send message">${SEND_SVG}</button>
    </div>
  `;
  root.appendChild(win);

  const messagesEl = win.querySelector("#nn-chat-messages");
  const inputEl = win.querySelector("#nn-chat-input");
  const sendBtn = win.querySelector("#nn-chat-send");

  function openWindow() {
    win.classList.add("nn-open");
    launcher.classList.add("nn-hidden");
    if (!messagesEl.dataset.greeted) {
      addBotMessage("Hey! I'm your Neural Ninja AI buddy — ask me about our labs, blog articles, or pretty much anything AI/ML. I'll do my best to make it fun.");
      messagesEl.dataset.greeted = "1";
    }
    inputEl.focus();
  }
  function closeWindow() {
    win.classList.remove("nn-open");
    launcher.classList.remove("nn-hidden");
  }

  launcher.addEventListener("click", openWindow);
  win.querySelector("#nn-chat-close").addEventListener("click", closeWindow);

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addUserMessage(text) {
    const row = document.createElement("div");
    row.className = "nn-row nn-row-user";
    row.innerHTML = `<div class="nn-bubble nn-bubble-user"></div>`;
    row.querySelector(".nn-bubble").textContent = text;
    messagesEl.appendChild(row);
    scrollToBottom();
  }

  function addBotMessage(text, sources) {
    const row = document.createElement("div");
    row.className = "nn-row nn-row-bot";

    const avatar = document.createElement("div");
    avatar.className = "nn-row-avatar";
    avatar.innerHTML = NN_LOGO_SVG;

    const bubble = document.createElement("div");
    bubble.className = "nn-bubble nn-bubble-bot";
    bubble.textContent = text;

    if (sources && sources.length) {
      const src = document.createElement("div");
      src.className = "nn-sources";
      src.textContent = "Source: " + sources.join(", ");
      bubble.appendChild(src);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();
  }

  function addTypingIndicator() {
    const row = document.createElement("div");
    row.className = "nn-row nn-row-bot";
    row.id = "nn-typing-row";

    const avatar = document.createElement("div");
    avatar.className = "nn-row-avatar";
    avatar.innerHTML = NN_LOGO_SVG;

    const bubble = document.createElement("div");
    bubble.className = "nn-bubble nn-bubble-bot nn-typing";
    bubble.innerHTML = "<span></span><span></span><span></span>";

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();
    return row;
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    addUserMessage(text);
    inputEl.value = "";
    sendBtn.disabled = true;

    const typingRow = addTypingIndicator();

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      typingRow.remove();

      if (res.status === 429) {
        addBotMessage("You're sending messages too fast. Please wait a moment and try again.");
      } else if (!res.ok) {
        addBotMessage("Something went wrong on my end. Please try again shortly.");
      } else {
        const data = await res.json();
        addBotMessage(data.answer, data.sources);
      }
    } catch (e) {
      typingRow.remove();
      addBotMessage("Connection issue — please check your internet and try again.");
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });
})();
