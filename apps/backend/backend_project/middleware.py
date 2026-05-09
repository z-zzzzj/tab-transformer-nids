from __future__ import annotations

from typing import Iterable

from django.http import HttpRequest, HttpResponse


class SimpleCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.allowed_origins = {
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        }
        self.allowed_methods = ["GET", "POST", "OPTIONS"]
        self.allowed_headers = ["Content-Type", "Authorization", "X-Requested-With"]

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if request.method == "OPTIONS":
            response = HttpResponse(status=204)
        else:
            response = self.get_response(request)
        return self._apply_headers(request, response)

    def _apply_headers(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        origin = request.headers.get("Origin")
        if origin and self._origin_allowed(origin):
            response["Access-Control-Allow-Origin"] = origin
            response["Vary"] = self._merge_vary(response.get("Vary"), "Origin")
            response["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            response["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
            response["Access-Control-Max-Age"] = "86400"
        return response

    def _origin_allowed(self, origin: str) -> bool:
        return origin in self.allowed_origins

    @staticmethod
    def _merge_vary(existing: str | None, item: str) -> str:
        values = [value.strip() for value in (existing or "").split(",") if value.strip()]
        if item not in values:
            values.append(item)
        return ", ".join(values)
