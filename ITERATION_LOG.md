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

### [2026-05-15] Added the shared HTML components test file to the test codemap

**Context:** The test codemap and README coverage summary listed most major test surfaces, but they did not call out `tests/test_html_components.py`, which covers the shared CSS/JS building blocks used by the gallery and index pages.
**What happened:** Added `tests/test_html_components.py` to `docs/codemaps/testing.md` and the README's test coverage list so the shared UI component tests are easier to find.
**Outcome:** Success — docs now reflect the full test surface visible in the current checkout.
**Insight:** Small codemap omissions are easiest to catch when the test file list is compared directly against the current collected suite.
**Promoted to Lessons Learned:** No

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

### [2026-05-08] Clarified dry-run CLI help and added regression coverage

**Context:** The CLI `--dry-run` option had stale help text and lacked a direct regression test.
**What happened:** Reworded the `--dry-run` help text to say it previews grammar without generating images, and added a CLI test that mocks grammar generation, verifies the preview output, and confirms the full pipeline is not constructed in dry-run mode.
**Outcome:** Success — focused CLI test file passed and the full suite stayed green.
**Insight:** Small CLI text tweaks are worth pinning with a behavior test when they describe a distinct execution path.
**Promoted to Lessons Learned:** No

---

### [2026-05-11] Corrected LM Studio base URL docs

**Context:** The project README and generate skill still pointed at the LM Studio host without the `/v1` API suffix, even though the code and other instructions use the full base URL.
**What happened:** Updated `README.md` and `.claude/skills/generate/SKILL.md` so the documented default matches `http://localhost:1234/v1`.
**Outcome:** Success — documentation now aligns with the configured default base URL.
**Insight:** When a tool-specific skill mirrors README setup instructions, keep both in sync with the actual configured API path.
**Promoted to Lessons Learned:** No

---

### [2026-05-12] Synced README test-suite count with collected tests

**Context:** The README’s development section listed an outdated test-suite count.
**What happened:** Ran `uv run pytest --collect-only -q` to verify the current collection count, then updated the README to say the suite currently collects 330 tests.
**Outcome:** Success — documentation now matches the observed test collection output.
**Insight:** Small maintenance docs can drift quietly; a quick collect-only check is enough to confirm the current count before editing.
**Promoted to Lessons Learned:** No

---
### [2026-05-13] Clarified zero-value prompt-only layout in CLI help

**Context:** The CLI already accepted `--images-per-prompt 0` as a prompt-only layout, but the `--help` text only mentioned the default and did not surface that contract.
**What happened:** Updated `src/cli.py` so the `--images-per-prompt` help string now says `0 = prompt-only layout`, and added a focused CLI test that asserts `--help` includes that wording.
**Outcome:** Success — `uv run pytest tests/test_cli.py -q` passed with 14 tests.
**Insight:** When the code treats `0` as a real sentinel, the CLI help should name that behavior explicitly so users do not assume it is invalid.
**Promoted to Lessons Learned:** No

---

### [2026-05-13] Synced README test-suite count with current collection

**Context:** The README’s development section had drifted from the live test collection count.
**What happened:** Ran `uv run pytest --collect-only -q` and confirmed the suite currently collects 331 tests, then updated the README to match.
**Outcome:** Success — documentation now matches the observed collection output.
**Insight:** Collect-only verification is the cheapest way to keep count-based docs honest.
**Promoted to Lessons Learned:** No

---

### [2026-05-13] Synced stale image-generator codemap note

**Context:** The testing codemap still claimed `tests/test_image_generator.py` aborted because of native MLX/mflux import behavior, but the current checkout had already passed the full suite.
**What happened:** Ran `uv run pytest -q` and `uv run pytest tests/test_image_generator.py -q` to verify the current state, then updated `docs/codemaps/testing.md` to say the image-generator test file runs cleanly and the suite passes.
**Outcome:** Success — the codemap now reflects the live verification result.
**Insight:** When a maintenance note mentions an environment-specific failure, re-run the narrow test before preserving the warning; stale caveats can survive long after the underlying issue is fixed.
**Promoted to Lessons Learned:** Yes

---

### [2026-05-14] Synced prompt-only gallery labels across UI surfaces

**Context:** The interactive gallery and index forms still used the shorter `Images/Prompt (0 = prompt-only)` label, while the CLI help and README already spelled out `0 = prompt-only layout`.
**What happened:** Updated the gallery and gallery index form labels to say `Images/Prompt (0 = prompt-only layout)`, and aligned the focused gallery/index tests with the new wording.
**Outcome:** Success — `uv run pytest tests/test_gallery.py tests/test_gallery_index.py -q` passed (11 tests).
**Insight:** Small copy syncs are easier to keep consistent when the exact runtime label is asserted in the tests that render the surface.
**Promoted to Lessons Learned:** No

---

### [2026-05-14] Clarified shared quantize behavior in CLI help and README

**Context:** The CLI `--quantize` flag already defaulted to 8 and was used by both prompt generation and standalone image enhancement, but the user-facing docs did not say that clearly.
**What happened:** Updated `src/cli.py` help text, the README options table, and the standalone enhancement section to note that `--quantize` applies to generation and enhancement and defaults to 8 when omitted; added a CLI help regression that asserts the new wording appears in `--help` output.
**Outcome:** Success — `uv run pytest tests/test_cli.py -q` passed (14 tests).
**Insight:** When one flag feeds multiple execution paths, the help text should name every path so users do not assume it is generation-only.
**Promoted to Lessons Learned:** Yes

---

### [2026-05-14] Exposed LM Studio base URL default in CLI help

**Context:** The CLI `--base-url` option already defaulted to `http://localhost:1234/v1`, but the help text did not surface that default for users scanning `--help` output.
**What happened:** Updated the `--base-url` help string to include the default URL and added a CLI help regression that asserts the base URL default is visible alongside the existing prompt-only layout/help coverage.
**Outcome:** Success — focused CLI tests passed (`14 passed`).
**Insight:** Click help output can wrap long option descriptions, so tests should assert stable substrings rather than a single exact line when a default is embedded in a long help string.
**Promoted to Lessons Learned:** Yes

### [2026-05-15] Noted benign uv environment warning during test-count verification

**Context:** While checking the live test collection count to keep docs honest, `uv run` emitted a `VIRTUAL_ENV` mismatch warning from the Hermes environment.
**What happened:** Ran `uv run pytest --collect-only -q`, confirmed the suite still collects 331 tests, and recorded that the warning did not block the repo-local command.
**Outcome:** Success — the docs count remains current and the environment quirk is now captured for future runs.
**Insight:** When `uv run` warns about `VIRTUAL_ENV` drift in this checkout, the repo command can still be trusted if the focused verification completes cleanly.
**Promoted to Lessons Learned:** Yes

### [2026-05-15] Documented grammar revision history in the gallery README

**Context:** The gallery now persists grammar revision history and exposes it in the UI, but the README only described the edit/regenerate path and did not mention the restore surface or the on-disk history file.
**What happened:** Updated the README gallery section to mention grammar-history review/restore and added `dragon_grammar_history.json` to the example output structure so the persisted revision file is discoverable.
**Outcome:** Success — user-facing docs now describe the grammar history capability that already exists in the app.
**Insight:** Small persistence features are easier to find later when the README names both the UI affordance and the file written to disk.
**Promoted to Lessons Learned:** No

### [2026-05-15] Documented benign Hermes uv warning in the README test notes

**Context:** `uv run pytest --collect-only -q` still completed successfully, but the shell emitted a recurring `VIRTUAL_ENV` mismatch warning under Hermes that can distract future maintainers during test runs.
**What happened:** Added a short note in the README's testing section explaining that the warning is benign when `uv run` finishes successfully and the repo environment is still used.
**Outcome:** Success — the warning is now documented alongside the test commands that can surface it.
**Insight:** Small environment quirks are easier to remember when they live next to the command that triggers them.
**Promoted to Lessons Learned:** No

### [2026-05-16] Documented prompt-only gallery layout in the README

**Context:** The gallery already supported the `Images/Prompt (0 = prompt-only layout)` control on per-gallery forms, but the README only said image settings were configured per-gallery and did not name the prompt-only affordance.
**What happened:** Added one README sentence under the Web UI section to call out the prompt-only layout control explicitly.
**Outcome:** Success — the user-facing docs now mention the shipped prompt-only gallery behavior in the place readers look for gallery setup details.
**Insight:** When a form exposes a useful zero-value sentinel, the README should name the exact control label so users can find it without hunting through the UI.
**Promoted to Lessons Learned:** No

---

<!-- New entries go above this line, most recent first -->
