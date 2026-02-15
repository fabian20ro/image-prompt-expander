# Server + UI Codemap

## Serve Entry Path

1. `python src/cli.py --serve [--port N]`
2. CLI starts `uvicorn` with `src/server/app.py:app`
3. FastAPI lifespan initializes:
   - `generated/` directories
   - `QueueManager` (disk-backed queue in `generated/queue.json`)
   - background `Worker` loop

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

## Queue + Worker Notes

- Tasks are persisted to disk, so pending/completed history survives server restarts.
- Heavy operations run in a subprocess (`worker_subprocess.py`) to isolate crashes and stream progress safely.
- Worker stderr is forwarded as `task_log` SSE events and shown in UI log panels.
- `POST /api/worker/kill` sends terminate/kill to the active subprocess and marks the task cancelled.

## UI Generation Layers

- Gallery page HTML/JS: `src/gallery.py`
- Index page HTML/JS: `src/gallery_index.py`
- Shared CSS/JS snippets: `src/html_components.py`

## URL Surface

- Root redirect: `/` -> `/index`
- Index page: `/index`
- Gallery page: `/gallery/{run_id}`
- Gallery static files: `/gallery/{run_id}/{filename:path}`
- Saved image file (flat archives): `/saved/{filename}`
- Legacy archive page/files: `/archive/{run_id}`, `/archive/{run_id}/{filename:path}`
- Queue/status/events: `/api/status`, `/api/events`, `/api/queue/clear`, `/api/worker/kill`
- Generation operations:
  - `POST /api/generate`
  - `POST /api/gallery/{run_id}/regenerate`
  - `POST /api/gallery/{run_id}/generate-all`
  - `POST /api/gallery/{run_id}/enhance-all`
  - `POST /api/gallery/{run_id}/image/{prompt_idx}/generate`
  - `POST /api/gallery/{run_id}/image/{prompt_idx}/enhance`
- Gallery metadata/utilities:
  - `GET /api/gallery/{run_id}`
  - `GET /api/gallery/{run_id}/grammar`
  - `PUT /api/gallery/{run_id}/grammar`
  - `GET /api/gallery/{run_id}/logs`
  - `POST /api/gallery/{run_id}/archive`
  - `DELETE /api/gallery/{run_id}`

## Persistence

- Queue state: `generated/queue.json`
- Runs: `generated/prompts/<run_id>/...`
- Saved images/archives: `generated/saved/` (flat PNG archives with embedded metadata)
