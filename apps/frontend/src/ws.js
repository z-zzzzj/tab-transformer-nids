import { WS_BASE } from "./config";

export function createAlertSocket(onMessage, onOpen, onClose, onError) {
  const socket = new WebSocket(`${WS_BASE}/ws/alerts`);
  socket.onopen = () => {
    if (onOpen) {
      onOpen();
    }
  };
  socket.onclose = () => {
    if (onClose) {
      onClose();
    }
  };
  socket.onerror = (error) => {
    if (onError) {
      onError(error);
    }
  };
  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      onMessage(payload);
    } catch (error) {
      if (onError) {
        onError(error);
      }
    }
  };
  return socket;
}

