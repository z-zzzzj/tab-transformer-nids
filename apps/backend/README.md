# Backend Service

## Run

```powershell
cd apps\backend
python -m daphne -b 127.0.0.1 -p 8000 backend_project.asgi:application
```

ASGI (websocket enabled):

```powershell
python -m daphne -b 127.0.0.1 -p 8000 backend_project.asgi:application
```

## Endpoints

- `GET /api/v1/health`
- `GET /api/v1/model/info`
- `POST /api/v1/infer`
- `POST /api/v1/infer/batch`
- `POST /api/v1/replay/start`
- `POST /api/v1/replay/stop`
- `GET /api/v1/replay/status`
- `GET /api/v1/replay/events`
- `WS /ws/alerts`
