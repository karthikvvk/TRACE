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

// ── Startup ───────────────────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(async () => {
  console.log("[Friday] Extension installed / updated.");

  // Schedule the proactive check loop
  await chrome.alarms.create(ALARM_NAME, {
    delayInMinutes: 1,
    periodInMinutes: CHECK_INTERVAL_MINUTES,
  });

  // Open the side panel on install so the user sees it immediately
  chrome.sidePanel.setOptions({ enabled: true });
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
