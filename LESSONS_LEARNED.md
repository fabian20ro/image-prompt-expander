# Lessons Learned

> This file is maintained by AI agents working on this project.
> It captures validated, reusable insights discovered during development.
> **Read this file at the start of every task. Update it at the end of every iteration.**

## How to Use This File

### Reading (Start of Every Task)
Before starting any work, read this file to avoid repeating known mistakes
and to leverage proven approaches.

### Writing (End of Every Iteration)
After completing a task or iteration, evaluate whether any new insight was
gained that would be valuable for future sessions. If yes, add it to the
appropriate category below.

### Promotion from Iteration Log
Patterns that appear 2+ times in `ITERATION_LOG.md` should be promoted
here as a validated lesson.

### Pruning
If a lesson becomes obsolete (e.g., a dependency was removed, an API changed),
move it to the Archive section at the bottom with a date and reason.

---

## Architecture & Design Decisions

**[2026-02-15]** FastAPI lifespan + static mounting - In apps already using lifespan hooks, mount static files directly with `check_dir=False` instead of `@app.on_event("startup")` to avoid deprecation warnings and keep startup logic unified.

## Code Patterns & Pitfalls

**[2026-02-07]** Validate model names before importing `mflux` - Importing `mflux` before rejecting unsupported models can trigger native MLX aborts in unsupported test paths.
**[2026-02-07]** Keep docs and signatures aligned - Drift between documented defaults and implementation defaults (`tiled_vae`) causes repeated operator mistake
**[2026-02-24]** Prefer MetadataManager for metadata operations — Direct JSON manipulation of run metadata bypasses validation and path logic; use MetadataManager consistently.s.

## Testing & Quality

**[2026-02-07]** Prefer intent assertions for worker logs - Worker success/failure paths emit variable extra lines, so tests should assert sequencing/intent rather than exact call counts.
**[2026-02-07]** Use module fakes for MLX-heavy imports - Patching deep `mflux.*` paths can still initialize native MLX/Metal; prefer `patch.dict(sys.modules, ...)` with fake modules.
**[2026-02-24]** Mock external services in tests — Always mock LM Studio and mflux; requiring live GPU or inference server makes tests environment-dependent and fragile.

## Performance & Infrastructure

**[2026-02-07]** Open worker logs before long tasks - If log files are opened after pipeline execution starts, early progress information is permanently lost.

## Dependencies & External Services

<!-- Version constraints, API quirks, integration lessons -->
<!-- Format: **[YYYY-MM-DD]** Brief title — Explanation -->

## Process & Workflow

**[2026-02-07]** Emit queue state updates after clear - Emit `queue_updated` together with `queue_cleared` so existing SSE clients refresh immediately.
**[2026-02-07]** Prefer non-blocking UI confirmations - Replacing `alert/confirm` with toast/modal flows reduces disruptive UI pauses and improves task continuity.

---

## Archive

<!-- Lessons that are no longer applicable. Keep for historical context. -->
<!-- Format: **[YYYY-MM-DD] Archived [YYYY-MM-DD]** Title — Reason for archival -->
