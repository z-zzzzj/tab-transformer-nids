from __future__ import annotations

import json
from typing import Any

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from streaming.runtime import get_runtime


def _json_body(request: HttpRequest) -> dict[str, Any]:
    if not request.body:
        return {}
    return json.loads(request.body.decode("utf-8"))


def _error(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"ok": False, "error": message}, status=status)


def health(request: HttpRequest) -> JsonResponse:
    if request.method != "GET":
        return _error("Method not allowed", status=405)
    runtime = get_runtime()
    return JsonResponse(
        {
            "ok": True,
            "service": "tab-transformer-nids-backend",
            "model_loaded": runtime.model_runtime.model_loaded,
            "replay_running": runtime.replay_controller.running,
        }
    )


def model_info(request: HttpRequest) -> JsonResponse:
    if request.method != "GET":
        return _error("Method not allowed", status=405)
    runtime = get_runtime()
    return JsonResponse({"ok": True, "model": runtime.model_runtime.info()})


@csrf_exempt
def infer(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return _error("Method not allowed", status=405)
    try:
        payload = _json_body(request)
    except json.JSONDecodeError:
        return _error("Invalid JSON body")

    if not isinstance(payload, dict):
        return _error("Payload must be a JSON object")

    runtime = get_runtime()
    response = runtime.model_runtime.infer_payload(payload)
    return JsonResponse(response.model_dump())


@csrf_exempt
def infer_batch(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return _error("Method not allowed", status=405)

    try:
        payload = _json_body(request)
    except json.JSONDecodeError:
        return _error("Invalid JSON body")

    items: list[dict[str, Any]] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    else:
        return _error("Payload must be an array or {\"items\": [...]}")

    runtime = get_runtime()
    results = []
    for item in items:
        if not isinstance(item, dict):
            return _error("All batch items must be JSON objects")
        results.append(runtime.model_runtime.infer_payload(item).model_dump())

    return JsonResponse({"ok": True, "count": len(results), "items": results})


@csrf_exempt
def replay_start(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return _error("Method not allowed", status=405)
    try:
        payload = _json_body(request)
    except json.JSONDecodeError:
        return _error("Invalid JSON body")
    runtime = get_runtime()
    started, message = runtime.replay_controller.start(payload or {})
    return JsonResponse(
        {
            "ok": started,
            "message": message,
            "status": runtime.replay_controller.status(),
        },
        status=200 if started else 409,
    )


@csrf_exempt
def replay_stop(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return _error("Method not allowed", status=405)
    runtime = get_runtime()
    stopped, message = runtime.replay_controller.stop()
    return JsonResponse(
        {"ok": stopped, "message": message, "status": runtime.replay_controller.status()},
        status=200 if stopped else 409,
    )


def replay_status(request: HttpRequest) -> JsonResponse:
    if request.method != "GET":
        return _error("Method not allowed", status=405)
    runtime = get_runtime()
    return JsonResponse({"ok": True, "status": runtime.replay_controller.status()})


def replay_events(request: HttpRequest) -> JsonResponse:
    if request.method != "GET":
        return _error("Method not allowed", status=405)
    runtime = get_runtime()
    try:
        since = int(request.GET.get("since", "0"))
        limit = int(request.GET.get("limit", "200"))
    except ValueError:
        return _error("Query params since/limit must be integers")
    if since < 0:
        since = 0
    limit = max(1, min(limit, 500))
    events, next_cursor = runtime.event_buffer.list_since(since=since, limit=limit)
    return JsonResponse({"ok": True, "events": events, "next_cursor": next_cursor})

