/**
 * api.js — Lightweight HTTP client for the Friday extension.
 *
 * All requests go to the local FastAPI backend.
 * BACKEND_URL is configurable for Colab/ngrok deployments.
 */

const BACKEND_URL = "http://127.0.0.1:8000";

/**
 * POST (or PATCH/PUT/DELETE) to the backend.
 * @param {string} path - e.g. "/tasks"
 * @param {object} body
 * @param {string} method - defaults to "POST"
 */
export async function post(path, body = {}, method = "POST") {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`[Friday API] ${method} ${path} → ${res.status}: ${err}`);
  }

  return res.json();
}

/**
 * GET from the backend.
 * @param {string} path - e.g. "/tasks?status=pending"
 */
export async function get(path) {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!res.ok) {
    throw new Error(`[Friday API] GET ${path} → ${res.status}`);
  }

  return res.json();
}

/**
 * Check if the backend is reachable.
 * @returns {Promise<boolean>}
 */
export async function isBackendAlive() {
  try {
    const data = await get("/");
    return data?.status === "running";
  } catch {
    return false;
  }
}

export { BACKEND_URL };
