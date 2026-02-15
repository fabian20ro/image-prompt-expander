# LESSONS_LEARNED.md

Brief operational lessons discovered while working on this project.

## 2026-02-07

- Validate unsupported image model names before importing `mflux`; importing first can trigger native MLX aborts in test environments.
- Emit `queue_updated` alongside `queue_cleared` so existing SSE clients update immediately after queue clears.
- Open worker log files before long-running pipeline execution; opening after execution loses early progress logs.
- Keep docs and signatures aligned (`tiled_vae` defaults were documented as `True` while implemented as `False`).
- Use non-blocking toasts/modal confirmations instead of `alert/confirm` for better UX and less disruptive task flow.
- In tests, patching deep `mflux.*` import paths can still initialize native MLX/Metal; prefer `patch.dict(sys.modules, ...)` with fake modules to avoid native crashes.
- For worker logging tests, assert sequencing/intent (startup log appears before execution) instead of exact call counts, because success/failure paths add additional log entries.

## 2026-02-15

- In FastAPI apps that already use lifespan hooks, avoid `@app.on_event("startup")`; mounting static files directly with `check_dir=False` removes deprecation warnings and keeps startup behavior clean.
