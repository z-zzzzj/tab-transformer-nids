from __future__ import annotations

from django.urls import path

from streaming.consumers import AlertsConsumer

websocket_urlpatterns = [
    path("ws/alerts", AlertsConsumer.as_asgi()),
]

