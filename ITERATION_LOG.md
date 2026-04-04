# Iteration Log

> Append-only journal of AI agent work sessions on this project.
> **Add an entry at the end of every iteration.**
> When patterns emerge (same issue 2+ times), promote to `LESSONS_LEARNED.md`.

## Format

Each entry should follow this structure:

---

### [YYYY-MM-DD] Brief Description of Work Done

**Context:** What was the goal / what triggered this work
**What happened:** Key actions taken, decisions made
**Outcome:** Result - success, partial, or failure
**Insight:** (optional) What would you tell the next agent about this?
**Promoted to Lessons Learned:** Yes/No

---

### [2026-04-04] Added revision-safe grammar regeneration, layout persistence, and grammar-import galleries

**Context:** User reported that regenerating prompts after pasting a new grammar kept stale images attached to changed prompts, and that gallery canvas size could drift from selected `images per prompt` / `max prompts`. User also requested grammar undo/history and a way to create a gallery directly from pasted Tracery grammar.
**What happened:** Added persisted `gallery_layout` metadata with fallback for older runs; changed gallery rendering to respect persisted layout instead of hardcoded form defaults; added per-run persisted grammar history (`*_grammar_history.json`) plus gallery UI controls for undo/redo and revision restore; added local draft persistence so layout-triggered reloads do not drop unsaved grammar edits; changed regenerate flow to auto-save posted grammar, archive existing PNGs, delete active PNGs only after successful backup, and rebuild the gallery from the new prompts/layout; added `/api/generate-from-grammar` for grammar-first gallery creation; updated index UI with a grammar-import form; extended worker/task plumbing and route models; added regression tests for layout persistence, history, safe regeneration, and grammar-import queueing.
**Outcome:** Success — targeted suite passed (`158 passed`) and full suite passed (`324 passed`).
**Insight:** If a route both persists a file and records a revision snapshot, revision capture must happen before overwriting the file or the history bootstrap path will misclassify the first saved revision as the current baseline.
**Promoted to Lessons Learned:** No

---

### [2026-04-03] Migrated pip/venv workflow to uv

**Context:** User requested full migration from `pip` + `venv` to `uv`, with the legacy setup removed rather than kept for compatibility.
**What happened:** Added `pyproject.toml` and committed `uv.lock`; moved runtime/dev deps into `uv` metadata; made `mflux` an optional `images` extra; deleted `requirements.txt` and `requirements-dev.txt`; removed the local `venv/` directory; updated `.gitignore` to `.venv/`; rewrote README, AGENTS, codemaps, repo-local Claude skills/settings, runtime error messages, and CLI help text to use `uv sync` / `uv run`; refreshed the README test-count note after verification.
**Outcome:** Success — `uv lock`, `uv sync --group dev`, and `uv sync --group dev --extra images` all completed successfully; `uv run python src/cli.py --help` worked; full test suite passed under `uv` (`313 passed`).
**Insight:** For this repo’s current flat-import layout, `uv` works cleanly with `tool.uv.package = false`; switching to package/module entrypoints would be a separate refactor.
**Promoted to Lessons Learned:** Yes

---

### [2026-02-24] AI Agent Configuration Restructuring

**Context:** Applied standardized AI agent config guide (informed by "Evaluating AGENTS.md" and "SkillsBench" research) to restructure all agent configuration files.
**What happened:** Rewrote AGENTS.md to remove discoverable content (Quick Start, Project Maps, general coding rules), keeping only non-discoverable constraints (LM Studio, mflux, venv). Migrated two project-specific patterns ("mock external services", "prefer MetadataManager") from AGENTS.md Core Rules to LESSONS_LEARNED.md. Created four sub-agent files in `.claude/agents/` (architect, planner, agent-creator, ux-expert). Simplified CLAUDE.md pointer. Added SETUP_AI_AGENT_CONFIG.md for periodic maintenance.
**Outcome:** Success — AGENTS.md is now lean bootstrap context (~40 lines vs ~70), sub-agents are defined, learning system references are consolidated.
**Insight:** Keeping AGENTS.md small and non-discoverable makes it age better; discoverable instructions drift as code changes but non-discoverable constraints (external services, venv) remain stable.
**Promoted to Lessons Learned:** No

---

### [2026-02-15] Merged CLAUDE instructions into AGENTS and reversed link direction

**Context:** User requested AGENTS as the single instruction source, with CLAUDE reduced to a pointer.
**What happened:** Moved actionable guidance from `CLAUDE.md` into `AGENTS.md` (quick start, core rules, session checklist, dependencies, doc map), and replaced `CLAUDE.md` contents with a one-line instruction to read `AGENTS.md`.
**Outcome:** Success - AGENTS is now the canonical instruction file and CLAUDE is a redirect.
**Insight:** Keeping one canonical instruction file reduces drift between agent entrypoints.
**Promoted to Lessons Learned:** No

---

### [2026-02-15] Added project memory system files and wiring

**Context:** User requested a persistent AI-agent memory system with curated lessons, an append-only iteration log, and AGENTS workflow wiring.
**What happened:** Replaced `LESSONS_LEARNED.md` with a structured curated template, migrated existing validated lessons into categories, created `ITERATION_LOG.md` with required format, and updated `AGENTS.md` with mandatory read/write workflow rules.
**Outcome:** Success - memory system is now present and referenced from `AGENTS.md`.
**Insight:** Keep `LESSONS_LEARNED.md` high-signal and concise; put session detail in the log and only promote repeated/validated patterns.
**Promoted to Lessons Learned:** No

---

<!-- New entries go above this line, most recent first -->
