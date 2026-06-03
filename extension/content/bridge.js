/**
 * bridge.js — DOM-level handler for server-initiated browser actions.
 *
 * Injected into every page alongside observer.js.
 * Listens for messages from the service worker (via chrome.tabs.sendMessage)
 * and executes the requested DOM operation, sending the result back
 * via sendResponse.
 *
 * All actions are synchronous or fast DOM reads — heavy work (scraping a
 * remote URL) is done by opening a new tab from the service worker side,
 * not by this script.
 */

(function () {
  "use strict";

  // Guard against double-injection in frames
  if (window.__fridgeBridgeLoaded) return;
  window.__fridgeBridgeLoaded = true;

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    const { type, payload = {} } = msg;

    switch (type) {
      // ── Read: full HTML DOM ──────────────────────────────────────────────
      case "GET_DOM": {
        sendResponse({ html: document.documentElement.outerHTML });
        break;
      }

      // ── Read: user's current text selection ─────────────────────────────
      case "GET_SELECTION": {
        sendResponse({ text: window.getSelection().toString() });
        break;
      }

      // ── Read: page metadata + scroll position ───────────────────────────
      case "GET_META": {
        const desc =
          document.querySelector('meta[name="description"]')?.content ??
          document.querySelector('meta[property="og:description"]')?.content ??
          "";
        sendResponse({
          title: document.title,
          url: location.href,
          description: desc,
          scroll_y: window.scrollY,
          scroll_x: window.scrollX,
          viewport_height: window.innerHeight,
          body_height: document.body.scrollHeight,
        });
        break;
      }

      // ── Write: click an element by CSS selector ──────────────────────────
      case "CLICK": {
        const el = document.querySelector(payload.selector);
        if (!el) {
          sendResponse({ ok: false, error: `Selector not found: ${payload.selector}` });
          break;
        }
        el.click();
        sendResponse({ ok: true, tag: el.tagName.toLowerCase() });
        break;
      }

      // ── Write: fill an input / textarea ─────────────────────────────────
      case "FILL_INPUT": {
        const el = document.querySelector(payload.selector);
        if (!el) {
          sendResponse({ ok: false, error: `Selector not found: ${payload.selector}` });
          break;
        }
        // Set value using native input value setter so React/Vue detect the change
        const nativeInput = Object.getOwnPropertyDescriptor(
          el.tagName === "TEXTAREA"
            ? window.HTMLTextAreaElement.prototype
            : window.HTMLInputElement.prototype,
          "value"
        );
        nativeInput?.set?.call(el, payload.value);
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        sendResponse({ ok: true, selector: payload.selector });
        break;
      }

      // ── Write: execute arbitrary JS (gated by ALLOW_SCRIPT_EXECUTION) ───
      case "EXECUTE_SCRIPT": {
        try {
          // eslint-disable-next-line no-new-func
          const fn = new Function(payload.code);
          const result = fn();
          sendResponse({ ok: true, result: String(result ?? "") });
        } catch (err) {
          sendResponse({ ok: false, error: err.message });
        }
        break;
      }

      default:
        // Not our message — let other listeners handle it
        return false;
    }

    // Return true to keep sendResponse channel open for async cases
    return true;
  });
})();
