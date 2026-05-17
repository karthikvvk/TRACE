/**
 * observer.js — Content script: page-level activity capture.
 *
 * Runs in the context of every page (see manifest content_scripts config).
 * Only captures semantic signals, not raw DOM content or personal text.
 *
 * Signals captured:
 *   - Page visibility changes (tab focus/blur)
 *   - Form submit events (type only, no values)
 *   - Time-on-page (session length)
 *   - Inferred reading intent from document title
 *
 * Nothing here sends raw page content to the backend.
 * All forwarding goes through the service worker via chrome.runtime.sendMessage.
 */

(function () {
  "use strict";

  const SESSION_START = Date.now();

  // ── Page visibility ─────────────────────────────────────────────────────────

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      const sessionLengthS = Math.round((Date.now() - SESSION_START) / 1000);
      send({
        type: "tab_blur",
        url: location.href,
        title: document.title,
        session_length_s: sessionLengthS,
        timestamp: Date.now(),
      });
    }
  });

  // ── Form submit (type only, no values) ─────────────────────────────────────

  document.addEventListener(
    "submit",
    (e) => {
      const form = e.target;
      send({
        type: "form_submit",
        url: location.href,
        form_id: form.id || null,
        form_action: form.action || null,
        timestamp: Date.now(),
      });
    },
    { passive: true }
  );

  // ── Page load signal ────────────────────────────────────────────────────────

  window.addEventListener("load", () => {
    send({
      type: "page_load",
      url: location.href,
      title: document.title,
      timestamp: Date.now(),
    });
  });

  // ── Send helper ─────────────────────────────────────────────────────────────

  function send(payload) {
    chrome.runtime.sendMessage({ type: "CONTENT_EVENT", payload }).catch(() => {
      // Service worker may be sleeping — this is expected, ignore.
    });
  }
})();
