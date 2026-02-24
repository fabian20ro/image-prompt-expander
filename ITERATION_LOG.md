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
