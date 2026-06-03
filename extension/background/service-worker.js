/**
 * service-worker.js — Friday's heartbeat.
 *
 * Responsibilities:
 *   1. Forward navigation events to the backend (/activity)
 *   2. Run the proactive check loop via chrome.alarms (/notify/pending)
 *   3. Route messages from content scripts and the side panel
 *   4. Manage the badge counter for pending items
 *
 * Everything here is offline-capable: the fetch calls fail gracefully
 * if the local server isn't running.
 */

import { post, get } from "../utils/api.js";

const ALARM_NAME = "friday-check";
const CHECK_INTERVAL_MINUTES = 10;

// ── Browser channel config ────────────────────────────────────────────────────

const WS_URL_BASE = "ws://localhost:8000/ws/browser";
// The secret must match EXTENSION_SECRET in the server's .env.
// In production, store this in chrome.storage.local after user setup.
const EXTENSION_SECRET = "change-me-in-production";

let ws = null;
let reconnectDelay = 1000; // ms — doubles on each failure, capped at 30s
let wsConnected = false;

// ── Startup ───────────────────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(async () => {
  console.log("[Friday] Extension installed / updated.");

  await chrome.alarms.create(ALARM_NAME, {
    delayInMinutes: 1,
    periodInMinutes: CHECK_INTERVAL_MINUTES,
  });

  chrome.sidePanel.setOptions({ enabled: true });
  connectBrowserChannel();
});

// ── Navigation observation ────────────────────────────────────────────────────

chrome.webNavigation.onCompleted.addListener(
  async (details) => {
    // Only track top-level frames, ignore extension pages
    if (details.frameId !== 0) return;
    if (details.url.startsWith("chrome")) return;
    if (details.url.startsWith("chrome-extension")) return;

    try {
      await post("/activity", {
        type: "navigation",
        url: details.url,
        timestamp: Date.now(),
      });
    } catch (err) {
      // Backend not running — log quietly, don't crash
      console.debug("[Friday] Backend unreachable, activity not logged.", err.message);
    }
  },
  { url: [{ schemes: ["http", "https"] }] }
);

// Track tab activation (helps working memory know what user is looking at)
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  try {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    if (!tab.url || tab.url.startsWith("chrome")) return;

    await post("/activity", {
      type: "tab_switch",
      url: tab.url,
      timestamp: Date.now(),
    });
  } catch (_) {}
});

// ── Proactive alarm loop ──────────────────────────────────────────────────────

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== ALARM_NAME) return;

  try {
    const data = await get("/notify/pending");
    const items = data?.items ?? [];

    // Update badge
    const count = items.length;
    await chrome.action.setBadgeText({ text: count > 0 ? String(count) : "" });
    await chrome.action.setBadgeBackgroundColor({ color: "#7C3AED" });

    // Fire Chrome notifications for high-urgency items
    for (const item of items.filter((i) => i.urgency === "high")) {
      chrome.notifications.create(`friday-${item.task_id}-${Date.now()}`, {
        type: "basic",
        iconUrl: "../icons/icon48.png",
        title: "Friday",
        message: item.message,
        priority: 1,
      });
    }

    // Relay all items to the side panel
    chrome.runtime.sendMessage({ type: "PENDING_ITEMS", items }).catch(() => {});
  } catch (err) {
    console.debug("[Friday] Check loop failed:", err.message);
  }
});

// ── Message routing ───────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender).then(sendResponse).catch((err) => {
    sendResponse({ error: err.message });
  });
  return true; // keep channel open for async
});

async function handleMessage(message, sender) {
  switch (message.type) {
    case "CREATE_TASK": {
      return await post("/tasks", message.payload);
    }

    case "GET_TASKS": {
      return await get(`/tasks?status=${message.status ?? "pending"}`);
    }

    case "UPDATE_TASK": {
      const { task_id, ...body } = message.payload;
      return await post(`/tasks/${task_id}`, body, "PATCH");
    }

    case "MARK_DONE": {
      return await post(`/tasks/${message.task_id}/done`, {});
    }

    case "SNOOZE_TASK": {
      return await post(`/tasks/${message.task_id}/snooze`, {});
    }

    case "GET_CONTEXT": {
      return await get("/context");
    }

    case "CONTENT_EVENT": {
      // Forwarded from content script
      return await post("/activity", message.payload);
    }

    default:
      return { error: `Unknown message type: ${message.type}` };
  }
}

// ── Browser channel (WebSocket) ───────────────────────────────────────────────

/**
 * Open a persistent WebSocket to the Friday backend.
 * The server can then request browser data on behalf of the agent.
 */
function connectBrowserChannel() {
  const url = `${WS_URL_BASE}?secret=${encodeURIComponent(EXTENSION_SECRET)}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    wsConnected = true;
    reconnectDelay = 1000;
    console.log("[Friday] Browser channel connected.");
    // Notify popup/sidepanel if they're open
    chrome.runtime.sendMessage({ type: "WS_STATUS", connected: true }).catch(() => {});
  };

  ws.onmessage = async (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      console.warn("[Friday] Received non-JSON WS message.");
      return;
    }
    const { id, action, params = {} } = msg;
    try {
      const result = await dispatchBrowserAction(action, params);
      ws.send(JSON.stringify({ id, result }));
    } catch (err) {
      ws.send(JSON.stringify({ id, error: err.message }));
    }
  };

  ws.onclose = () => {
    wsConnected = false;
    console.debug(`[Friday] Browser channel closed. Reconnecting in ${reconnectDelay}ms…`);
    chrome.runtime.sendMessage({ type: "WS_STATUS", connected: false }).catch(() => {});
    setTimeout(() => {
      reconnectDelay = Math.min(reconnectDelay * 2, 30_000);
      connectBrowserChannel();
    }, reconnectDelay);
  };

  ws.onerror = (err) => {
    console.debug("[Friday] Browser channel error:", err.message ?? err);
    // onclose fires after onerror — reconnect handled there
  };
}

// Start the browser channel when the service worker loads
connectBrowserChannel();

/**
 * Dispatch an action from the server to the appropriate Chrome API
 * or content script handler.
 */
async function dispatchBrowserAction(action, params) {
  switch (action) {
    // ── State tools — instant Chrome API reads ──
    case "get_active_tab": {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      return tab ? { id: tab.id, url: tab.url, title: tab.title } : null;
    }
    case "get_page_title": {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      return { title: tab?.title ?? null };
    }

    // ── Read tools ──
    case "get_tabs": {
      const tabs = await chrome.tabs.query({});
      return tabs.map((t) => ({ id: t.id, url: t.url, title: t.title, active: t.active }));
    }
    case "get_cookies": {
      const cookies = await chrome.cookies.getAll({ domain: params.domain ?? undefined });
      return cookies.map((c) => ({ name: c.name, value: c.value, domain: c.domain, path: c.path }));
    }
    case "get_dom":       return sendToBridge("GET_DOM",      params);
    case "get_selection": return sendToBridge("GET_SELECTION", params);
    case "get_meta":      return sendToBridge("GET_META",      params);

    // ── Write tools ──
    case "navigate": {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      await chrome.tabs.update(tab.id, { url: params.url });
      return { ok: true, tab_id: tab.id };
    }
    case "open_tab": {
      const tab = await chrome.tabs.create({ url: params.url, active: params.active ?? true });
      return { ok: true, tab_id: tab.id };
    }
    case "close_tab": {
      await chrome.tabs.remove(params.tab_id);
      return { ok: true };
    }
    case "click":      return sendToBridge("CLICK",      params);
    case "fill_input": return sendToBridge("FILL_INPUT", params);
    case "scrape_url": {
      // Open a new tab, wait for it to load, grab the DOM, then close it
      const tab = await chrome.tabs.create({ url: params.url, active: false });
      await waitForTabLoad(tab.id);
      const result = await chrome.tabs.sendMessage(tab.id, { type: "GET_DOM", payload: {} });
      await chrome.tabs.remove(tab.id);
      return result;
    }

    default:
      throw new Error(`Unknown action: ${action}`);
  }
}

/** Inject a message into the active tab's bridge.js content script. */
async function sendToBridge(type, payload = {}) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error("No active tab found.");
  return chrome.tabs.sendMessage(tab.id, { type, payload });
}

/** Promise that resolves when a tab finishes loading. */
function waitForTabLoad(tabId) {
  return new Promise((resolve) => {
    function onUpdated(id, changeInfo) {
      if (id === tabId && changeInfo.status === "complete") {
        chrome.tabs.onUpdated.removeListener(onUpdated);
        resolve();
      }
    }
    chrome.tabs.onUpdated.addListener(onUpdated);
  });
}
