import { API_BASE } from "./config";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }
  return response.json();
}

export function getHealth() {
  return request("/api/v1/health");
}

export function getReplayStatus() {
  return request("/api/v1/replay/status");
}

export function startReplay(payload) {
  return request("/api/v1/replay/start", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function stopReplay() {
  return request("/api/v1/replay/stop", {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function getReplayEvents(limit = 100, since = 0) {
  return request(`/api/v1/replay/events?limit=${limit}&since=${since}`);
}
