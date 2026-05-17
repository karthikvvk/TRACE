/**
 * storage.js — IndexedDB wrapper for the extension's local storage.
 *
 * Used for caching tasks in the side panel so it remains usable offline.
 * Wraps the IndexedDB API in Promise-based helpers.
 *
 * Stores:
 *   - tasks     — local mirror of tasks from the backend
 *   - settings  — user preferences stored client-side
 */

const DB_NAME = "friday-extension";
const DB_VERSION = 1;

let _db = null;

async function openDB() {
  if (_db) return _db;

  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = (e) => {
      const db = e.target.result;

      if (!db.objectStoreNames.contains("tasks")) {
        const tasks = db.createObjectStore("tasks", { keyPath: "id" });
        tasks.createIndex("status", "status", { unique: false });
        tasks.createIndex("due", "due", { unique: false });
      }

      if (!db.objectStoreNames.contains("settings")) {
        db.createObjectStore("settings", { keyPath: "key" });
      }
    };

    req.onsuccess = (e) => {
      _db = e.target.result;
      resolve(_db);
    };

    req.onerror = () => reject(req.error);
  });
}

// ── Tasks ─────────────────────────────────────────────────────────────────────

export async function cacheTasks(tasks) {
  const db = await openDB();
  const tx = db.transaction("tasks", "readwrite");
  const store = tx.objectStore("tasks");
  for (const task of tasks) {
    store.put(task);
  }
  return new Promise((res, rej) => {
    tx.oncomplete = () => res(true);
    tx.onerror = () => rej(tx.error);
  });
}

export async function getCachedTasks(status = "pending") {
  const db = await openDB();
  const tx = db.transaction("tasks", "readonly");
  const idx = tx.objectStore("tasks").index("status");
  const req = idx.getAll(status);
  return new Promise((res, rej) => {
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
}

export async function clearCachedTasks() {
  const db = await openDB();
  const tx = db.transaction("tasks", "readwrite");
  tx.objectStore("tasks").clear();
  return new Promise((res, rej) => {
    tx.oncomplete = () => res(true);
    tx.onerror = () => rej(tx.error);
  });
}

// ── Settings ──────────────────────────────────────────────────────────────────

export async function setSetting(key, value) {
  const db = await openDB();
  const tx = db.transaction("settings", "readwrite");
  tx.objectStore("settings").put({ key, value });
  return new Promise((res, rej) => {
    tx.oncomplete = () => res(true);
    tx.onerror = () => rej(tx.error);
  });
}

export async function getSetting(key, defaultValue = null) {
  const db = await openDB();
  const tx = db.transaction("settings", "readonly");
  const req = tx.objectStore("settings").get(key);
  return new Promise((res, rej) => {
    req.onsuccess = () => res(req.result ? req.result.value : defaultValue);
    req.onerror = () => rej(req.error);
  });
}
