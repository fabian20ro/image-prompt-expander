# Server + UI Codemap

## Backend Topology

- App bootstrap/lifespan: `src/server/app.py`
- HTTP routes: `src/server/routes.py`
- Request/response/task schemas: `src/server/models.py`
- Queue persistence + events: `src/server/queue_manager.py`
- Worker process orchestration: `src/server/worker.py`
- Task execution subprocess: `src/server/worker_subprocess.py`

## Request Path (Typical)

1. Browser hits route in `src/server/routes.py`.
2. Route validates with Pydantic models in `src/server/models.py`.
3. Route enqueues task via `QueueManager.add_task(...)`.
4. `Worker.run()` pops next task and spawns `worker_subprocess.py`.
5. Subprocess emits JSON lines for progress/result/image-ready.
6. Worker forwards updates to `QueueManager`.
7. SSE endpoint (`/api/events`) streams `queue_*`, `task_*`, `image_ready`.
8. UI JS in generated pages updates progress/logs/cards.

## UI Generation Layers

- Gallery page HTML/JS: `src/gallery.py`
- Index page HTML/JS: `src/gallery_index.py`
- Shared CSS/JS snippets: `src/html_components.py`

## URL Surface

- Index page: `/index`
- Gallery page: `/gallery/{run_id}`
- Archive page: `/archive/{run_id}`
- Queue/status/events: `/api/status`, `/api/events`, `/api/queue/clear`, `/api/worker/kill`
- Generation operations: `/api/generate`, `/api/gallery/{run_id}/...`

## Persistence

- Queue state: `generated/queue.json`
- Runs: `generated/prompts/<run_id>/...`
- Saved images/archives: `generated/saved/`
