from __future__ import annotations

from django.urls import path

from api import views

urlpatterns = [
    path("health", views.health, name="health"),
    path("model/info", views.model_info, name="model_info"),
    path("infer", views.infer, name="infer"),
    path("infer/batch", views.infer_batch, name="infer_batch"),
    path("replay/start", views.replay_start, name="replay_start"),
    path("replay/stop", views.replay_stop, name="replay_stop"),
    path("replay/status", views.replay_status, name="replay_status"),
    path("replay/events", views.replay_events, name="replay_events"),
]

