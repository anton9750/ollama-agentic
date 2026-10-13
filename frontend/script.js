
"use strict";

const API_BASE = "http://127.0.0.1:8765";
const TOKEN_PREFIX = "#local-companion-token=";

const elements = {
  sidebar: document.getElementById("sidebar"),
  openSidebar: document.getElementById("openSidebar"),
  closeSidebar: document.getElementById("closeSidebar"),
  newChatButton: document.getElementById("newChatButton"),

  welcome: document.getElementById("welcomeScreen"),
  messages: document.getElementById("messages"),

  chatForm: document.getElementById("chatForm"),
  messageInput: document.getElementById("messageInput"),
  sendButton: document.getElementById("sendButton"),

  connectionDot: document.getElementById("connectionDot"),
  connectionText: document.getElementById("connectionText"),
  sidebarStatusDot: document.getElementById("sidebarStatusDot"),
  sidebarStatusText: document.getElementById("sidebarStatusText"),

  errorBanner: document.getElementById("errorBanner"),
  errorMessage: document.getElementById("errorMessage"),
  dismissError: document.getElementById("dismissError")
};

let isSending = false;
let conversation = [];

/*
 * Read the temporary token passed by launcher.py.
 * The launcher opens:
 * website-url/#local-companion-token=TOKEN
 */
function getCompanionToken() {
  const hash = window.location.hash;

  if (hash.startsWith(TOKEN_PREFIX)) {
    const token = hash.slice(TOKEN_PREFIX.length);

    if (token.length >= 32) {
      sessionStorage.setItem("localCompanionToken", token);

      // Remove the token from the visible URL.
      history.replaceState(
        null,
        "",
        window.location.pathname + window.location.search
      );

      return token;
    }
  }

  return sessionStorage.getItem("localCompanionToken");
}

function setConnectionStatus(status, message) {
  const connected = status === "connected";
  const disconnected = status === "disconnected";

  for (const dot of [
    elements.connectionDot,
    elements.sidebarStatusDot
  ]) {
    dot.classList.toggle("connected", connected);
    dot.classList.toggle("disconnected", disconnected);
  }

  elements.connectionText.textContent = message;
  elements.sidebarStatusText.textContent = message;
}

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorBanner.hidden = false;
}

function hideError() {
  elements.errorBanner.hidden = true;
  elements.errorMessage.textContent = "";
}

function setSending(sending) {
  isSending = sending;
  elements.sendButton.disabled = sending;
  elements.messageInput.disabled = sending;
  elements.sendButton.textContent = sending ? "…" : "↑";
  elements.sendButton.setAttribute(
    "aria-label",
    sending ? "Sending message" : "Send message"
  );
}

function showConversation() {
  elements.welcome.hidden = true;
  elements.messages.hidden = false;
}

function scrollToLatestMessage() {
  elements.messages.lastElementChild?.scrollIntoView({
    behavior: "smooth",
    block: "end"
  });
}

function addMessage(role, text) {
  const message = document.createElement("article");
  message.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "message-meta";
  label.textContent = role === "user" ? "You" : "Agentic AI";

  const content = document.createElement("div");
  content.className = "message-content";

  // Use textContent instead of innerHTML to avoid executing model output.
  content.textContent = text;

  message.append(label, content);
  elements.messages.appendChild(message);

  showConversation();
  scrollToLatestMessage();

  return content;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

async function checkConnection() {
  setConnectionStatus("checking", "Checking connection…");

  try {
    const response = await fetchWithTimeout(
      `${API_BASE}/health`,
      { method: "GET" },
      4000
    );

    if (!response.ok) {
      throw new Error("The local API returned an error.");
    }

    const data = await response.json();

    if (data.status !== "ok") {
      throw new Error("The local API is not ready.");
    }

    if (!getCompanionToken()) {
      setConnectionStatus(
        "disconnected",
        "Launch companion to connect"
      );
      return false;
    }

    setConnectionStatus("connected", "Companion connected");
    return true;
  } catch (error) {
    setConnectionStatus("disconnected", "Companion offline");
    return false;
  }
}

async function sendToLocalAI(message) {
  const token = getCompanionToken();

  if (!token) {
    throw new Error(
      "No companion token found. Launch the Windows companion and use the browser window it opens."
    );
  }

  let response;

  try {
    response = await fetchWithTimeout(
      `${API_BASE}/chat`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ message })
      },
      300000
    );
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(
        "The request took too long. Your local model may be busy."
      );
    }

    throw new Error(
      "Could not reach the local companion. Check that it is running and that your browser permits the local connection."
    );
  }

  if (response.status === 401) {
    sessionStorage.removeItem("localCompanionToken");
    setConnectionStatus("disconnected", "Authentication expired");

    throw new Error(
      "The companion token is invalid or expired. Restart the companion and reopen its browser window."
    );
  }

  if (!response.ok) {
    let detail = "";

    try {
      const data = await response.json();
      detail = data.detail || "";
    } catch {
      // The server did not return JSON.
    }

    throw new Error(
      detail || `The local API returned HTTP ${response.status}.`
    );
  }

  const data = await response.json();

  if (typeof data.reply !== "string") {
    throw new Error("The local API returned an unexpected response.");
  }

  return data.reply;
}

async function handleSend(message) {
  const cleanedMessage = message.trim();

  if (!cleanedMessage || isSending) {
    return;
  }

  hideError();

  addMessage("user", cleanedMessage);
  conversation.push({ role: "user", content: cleanedMessage });

  elements.messageInput.value = "";
  elements.messageInput.style.height = "auto";

  setSending(true);

  const replyElement = addMessage("assistant", "Thinking…");

  try {
    const reply = await sendToLocalAI(cleanedMessage);

    replyElement.textContent = reply;
    conversation.push({ role: "assistant", content: reply });

    setConnectionStatus("connected", "Companion connected");
  } catch (error) {
    replyElement.textContent = "I couldn't get a response.";

    showError(error.message || "An unexpected error occurred.");

    if (
      error.message.includes("reach the local companion") ||
      error.message.includes("token") ||
      error.message.includes("expired")
    ) {
      setConnectionStatus("disconnected", "Check companion");
    }
  } finally {
    setSending(false);
    elements.messageInput.focus();
    scrollToLatestMessage();
  }
}

function startNewChat() {
  if (isSending) {
    return;
  }

  conversation = [];
  elements.messages.replaceChildren();
  elements.messages.hidden = true;
  elements.welcome.hidden = false;

  hideError();
  elements.messageInput.value = "";
  elements.messageInput.focus();
}

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await handleSend(elements.messageInput.value);
});

elements.messageInput.addEventListener("keydown", (event) => {
  if (
    event.key === "Enter" &&
    !event.shiftKey &&
    !event.isComposing
  ) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

elements.messageInput.addEventListener("input", () => {
  elements.messageInput.style.height = "auto";
  elements.messageInput.style.height =
    `${Math.min(elements.messageInput.scrollHeight, 220)}px`;
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.messageInput.value = button.dataset.prompt || "";
    elements.messageInput.focus();
    elements.messageInput.style.height = "auto";
    elements.messageInput.style.height =
      `${Math.min(elements.messageInput.scrollHeight, 220)}px`;
  });
});

elements.newChatButton.addEventListener("click", startNewChat);

elements.dismissError.addEventListener("click", hideError);

elements.openSidebar.addEventListener("click", () => {
  elements.sidebar.classList.add("open");
});

elements.closeSidebar.addEventListener("click", () => {
  elements.sidebar.classList.remove("open");
});

document.querySelectorAll(".sidebar-link, .footer-link").forEach((link) => {
  link.addEventListener("click", () => {
    elements.sidebar.classList.remove("open");
  });
});

async function initialize() {
  await checkConnection();

  // Refresh the status periodically while the page is open.
  window.setInterval(() => {
    if (!isSending) {
      checkConnection();
    }
  }, 10000);
}

initialize();
