from __future__ import annotations

import json

from channels.generic.websocket import AsyncWebsocketConsumer

from streaming.runtime import get_runtime


class AlertsConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.channel_layer.group_add("alerts", self.channel_name)
        await self.accept()
        runtime = get_runtime()
        await self.send(
            text_data=json.dumps(
                {
                    "type": "status",
                    "status": runtime.replay_controller.status(),
                }
            )
        )

    async def disconnect(self, close_code: int) -> None:
        await self.channel_layer.group_discard("alerts", self.channel_name)

    async def receive(self, text_data: str | None = None, bytes_data: bytes | None = None) -> None:
        if text_data:
            try:
                payload = json.loads(text_data)
            except json.JSONDecodeError:
                return
            if payload.get("type") == "ping":
                await self.send(text_data=json.dumps({"type": "pong"}))

    async def alert_event(self, event: dict) -> None:
        await self.send(text_data=json.dumps({"type": "alert", "event": event.get("event", {})}))

