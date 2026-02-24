# AI Agent Configuration Setup Guide

> **Purpose:** This document serves two roles:
> 1. **Setup guide** — step-by-step instructions for creating all config files from scratch
> 2. **Maintenance protocol** — hand this document to an agent periodically to audit and clean all files, preserving only what's still useful
>
> Apply when: starting a new project, onboarding to an existing one, or running a periodic hygiene pass (weekly/monthly/yearly).

---

## Research Context — Why This Guide Exists

This guide is informed by two key studies and extensive practitioner experience:

- **[Evaluating AGENTS.md](https://arxiv.org/abs/2602.11988)** — Found that LLM-generated context files (via `/init`) **reduced** task success rates by ~3% on average while **increasing inference cost by 20%+**. Developer-provided files only improved performance by ~4% — marginal at best. Context files encouraged broader but less focused exploration.
- **[SkillsBench](https://arxiv.org/abs/2602.12670)** — Found that **curated, focused skills** (2–3 modules) outperform comprehensive documentation. Self-generated skills provide no benefit on average. Smaller models with good skills can match larger models without them.

**Core principle:** Help the model. Don't distract it. If the info is already in the codebase, it probably doesn't need to be in the config file.

---

## How the Files Work Together

There are four file types, each with a distinct role. Understanding how they synchronize is essential — overlap between them is a bug, not a feature.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENT STARTS TASK                            │
│                                                                     │
│  1. Reads AGENTS.md (always in context — bootstrap only)            │
│     → Non-discoverable constraints, legacy traps, file references   │
│     → Tells agent: read LESSONS_LEARNED.md + which sub-agents exist │
│                                                                     │
│  2. Reads LESSONS_LEARNED.md (curated wisdom)                       │
│     → Validated corrections and patterns from past sessions         │
│     → This is where "the agent keeps doing X wrong" lives           │
│                                                                     │
│  3. If task is complex → delegates to sub-agent                     │
│     → .claude/agents/architect.md, planner.md, etc.                 │
│     → Focused procedural knowledge for specific domains             │
│                                                                     │
│  4. Does the work                                                   │
│                                                                     │
│  5. End of iteration:                                               │
│     → ALWAYS appends to ITERATION_LOG.md (raw, what happened)       │
│     → If reusable insight → adds to LESSONS_LEARNED.md              │
│     → If something was surprising → flags to developer              │
│                                                                     │
│  Developer decides:                                                 │
│     → Fix the codebase? (preferred)                                 │
│     → Add to LESSONS_LEARNED.md? (if codebase fix isn't possible)   │
│     → Add to AGENTS.md? (only if it's a non-discoverable constraint │
│       that must be known BEFORE reading any other file)              │
│     → Create a new sub-agent? (invoke agent-creator)                │
│                                                                     │
│  PERIODIC MAINTENANCE (weekly/monthly/yearly/new model):            │
│     → Hand this document to an agent as a standalone task            │
│     → Agent audits ALL files using the Maintenance Protocol          │
│     → Stale entries archived, patterns promoted, overlaps removed    │
│     → Files get leaner over time — never fatter                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What lives where — the boundary rules

| Question | → File |
|----------|--------|
| Can the model discover this from the codebase? | **Nowhere.** Don't write it down. |
| Is this a constraint the model needs BEFORE it starts exploring? | **AGENTS.md** (e.g., "use pnpm not npm", "dev server already running") |
| Is this a correction for a repeated mistake? | **LESSONS_LEARNED.md** (not AGENTS.md) |
| Is this a raw observation from a single session? | **ITERATION_LOG.md** |
| Is this focused procedural knowledge for a recurring task domain? | **Sub-agent** in `.claude/agents/` |
| Does something in the codebase keep confusing agents? | **Fix the codebase first.** Then LESSONS_LEARNED if needed. |

### Promotion flow

```
Observation (single session)
  → ITERATION_LOG.md (always)

Same issue appears 2+ times in ITERATION_LOG
  → Promote to LESSONS_LEARNED.md

Lesson becomes obsolete (dependency removed, API changed, model improved)
  → Move to Archive section in LESSONS_LEARNED.md (never delete)

New recurring task domain emerges
  → Invoke agent-creator → new sub-agent in .claude/agents/

New model release
  → Delete AGENTS.md entirely. Test. Re-add only what's still needed.
  → Review LESSONS_LEARNED.md — many entries may be obsolete.

Periodic maintenance (weekly/monthly/yearly)
  → Hand this document to an agent: "Run the Periodic Maintenance Protocol"
  → Agent audits ALL files, removes stale info, promotes unhandled patterns
  → Agent produces a maintenance report for developer review
  → Files get leaner over time, never fatter
```

---

## File Structure Overview

After applying this guide, your project root should contain:

```
project-root/
├── AGENTS.md                 # Bootstrap context (minimal, non-discoverable constraints only)
├── CLAUDE.md                 # Redirect → AGENTS.md
├── LESSONS_LEARNED.md        # Curated corrections and validated wisdom
├── ITERATION_LOG.md          # Append-only session journal
└── .claude/
    └── agents/               # Specialized sub-agents
        ├── architect.md      # System design, ADRs
        ├── planner.md        # Multi-step implementation plans
        ├── agent-creator.md  # Meta-agent: creates new specialized agents
        └── ux-expert.md      # UI/UX decisions (frontend projects only)
```

---

## Periodic Maintenance Protocol

This section is designed as a **standalone task**. Hand this entire document to an agent with the instruction: *"Run the maintenance protocol on this project."* The agent should be able to audit and clean all config files without further guidance.

### When to Run

| Frequency | Trigger |
|-----------|---------|
| **Weekly** (active projects) | High iteration velocity, many ITERATION_LOG entries accumulating |
| **Monthly** (steady projects) | Default cadence for most projects |
| **Per model release** | New model may handle things differently — major cleanup opportunity |
| **Yearly** (dormant projects) | Before resuming work after a long pause |

### The Audit — Step by Step

The agent performing maintenance should execute these steps in order and report findings to the developer.

#### Phase 1: Audit AGENTS.md

Goal: AGENTS.md should be **as small as possible**. Every line must earn its place.

```
For each entry in AGENTS.md "Constraints":
  1. Try to discover this information from the codebase alone
     (check package.json, Makefile, tsconfig, CI config, etc.)
  2. If discoverable → REMOVE from AGENTS.md (it's dead weight)
  3. If not discoverable → KEEP, but check if it's still accurate
     (does the constraint still apply? did the tooling change?)
  4. If inaccurate → FIX or REMOVE

For each entry in AGENTS.md "Legacy & Deprecated":
  1. Check if the legacy code/routes/files still exist in the codebase
  2. If removed → REMOVE from AGENTS.md (warning no longer needed)
  3. If still present → KEEP

Check: Does AGENTS.md contain any corrections/patterns?
  → If yes, MOVE them to LESSONS_LEARNED.md (they don't belong here)

Check: Does AGENTS.md sub-agents table match actual files in .claude/agents/?
  → Remove rows for agents that no longer exist
  → Add rows for agents that exist but aren't listed
```

**Success metric:** AGENTS.md should be shorter after this audit, never longer.

#### Phase 2: Audit LESSONS_LEARNED.md

Goal: Every lesson must be **still relevant** and **not duplicated** elsewhere.

```
For each lesson in every category:
  1. Is this lesson still accurate?
     (Did the dependency change? Did the API update? Did a model fix this?)
     → If obsolete → MOVE to Archive section with date and reason
  2. Is this lesson now enforced by the codebase itself?
     (Was a linter rule added? A test? A type check?)
     → If enforced → MOVE to Archive ("Now enforced by [mechanism]")
  3. Is this lesson duplicated in AGENTS.md?
     → If yes → REMOVE from AGENTS.md (LESSONS_LEARNED is the canonical location)
  4. Is this lesson too verbose?
     → CONDENSE to essential signal. Future agents scanning 100 entries need brevity.
  5. Are there multiple lessons that say the same thing differently?
     → MERGE into one concise entry

Check category balance:
  → Any category with 20+ entries? Consider if some are too granular.
  → Any empty categories after 3+ months of active development?
     Not necessarily a problem, but worth flagging.
```

**Success metric:** Every remaining lesson passes the test: *"Would a new agent working on this project benefit from knowing this TODAY?"*

#### Phase 3: Audit ITERATION_LOG.md

Goal: The log is append-only — never delete entries. But patterns should be **promoted**.

```
Scan all entries since last maintenance:
  1. Identify repeated issues (same problem in 2+ entries)
     → If not yet in LESSONS_LEARNED → PROMOTE (add to appropriate category)
     → Mark entries as "Promoted to Lessons Learned: Yes"
  2. Identify entries where the insight was valuable but never promoted
     → Propose promotion to developer

Check log size:
  → Over 200 entries? Consider archiving older entries (>6 months) to
    ITERATION_LOG_ARCHIVE.md to keep the active log scannable.
    The archive is still append-only and tracked in git.
```

**Success metric:** Zero unhandled patterns sitting in the log for 2+ cycles.

#### Phase 4: Audit Sub-Agents (.claude/agents/)

Goal: Each agent must be **focused, current, and earning its keep**.

```
For each agent file:
  1. Is this agent still being invoked? (Check ITERATION_LOG for references)
     → If never/rarely used in the last 3 months → FLAG for developer review
     → Developer decides: delete, merge into another agent, or keep
  2. Does the agent reference tools/patterns that no longer exist?
     → UPDATE or REMOVE stale references
  3. Is the agent over 100 lines?
     → SPLIT or CONDENSE — it's probably too broad
  4. Does the agent overlap with another agent?
     → MERGE or clarify boundaries

Check: Are there recurring tasks in ITERATION_LOG that no agent covers?
  → Propose a new agent to the developer (don't auto-create)
```

#### Phase 5: Cross-File Consistency Check

Goal: Zero overlap, zero contradictions between files.

```
Check for content that appears in multiple files:
  → Corrections in both AGENTS.md and LESSONS_LEARNED? → Remove from AGENTS.md
  → Same constraint in AGENTS.md and a sub-agent? → Keep in one place only
  → Sub-agent principle contradicts a lesson? → FLAG for developer

Check references:
  → AGENTS.md sub-agents table matches .claude/agents/ directory?
  → AGENTS.md mentions LESSONS_LEARNED.md and ITERATION_LOG.md?
  → All file paths in all documents are still valid?
```

### Maintenance Report Format

After completing the audit, the agent produces a report:

```markdown
# Maintenance Report — [YYYY-MM-DD]

## Summary
- AGENTS.md: [N] entries removed, [N] kept, [N] corrected
- LESSONS_LEARNED.md: [N] archived, [N] condensed, [N] merged, [N] kept
- ITERATION_LOG.md: [N] patterns promoted, [N] entries since last maintenance
- Sub-agents: [N] updated, [N] flagged for review, [N] unchanged

## Changes Made
<!-- List each change with brief rationale -->

## Flagged for Developer Decision
<!-- Things the agent couldn't decide autonomously -->

## Health Score
- AGENTS.md size: [N] lines (target: <30)
- LESSONS_LEARNED.md active entries: [N] (target: <50 per category)
- ITERATION_LOG.md unprocessed entries: [N] (target: 0 patterns unhandled)
- Sub-agents count: [N] (warning if >8 — likely overlap)
- Cross-file duplicates found: [N] (target: 0)
```

### The Core Invariant

After every maintenance pass, this must be true:

> **An agent reading AGENTS.md → LESSONS_LEARNED.md → relevant sub-agent has exactly the context it needs for any task in this project. No more, no less. Nothing duplicated. Nothing stale. Nothing the codebase already tells it.**

If the maintenance agent cannot confirm this, it should flag specific violations for the developer.

---

## Quick Reference: Decision Flowchart

```
Agent keeps making the same mistake?
  └─ Can I fix the codebase to prevent it?
      ├─ YES → Fix the code (better tests, clearer naming, linting)
      └─ NO  → Log in ITERATION_LOG → if 2+ times → promote to LESSONS_LEARNED

Agent needs to know something BEFORE it starts exploring?
  └─ Is it discoverable from the codebase?
      ├─ YES → Don't add it anywhere
      └─ NO  → AGENTS.md "Constraints" section (keep it one line)

Complex multi-step task?
  └─ Invoke planner agent BEFORE writing code

Architecture decision?
  └─ Invoke architect agent → record decision as ADR

Frontend component design?
  └─ Invoke ux-expert agent

Recurring task domain with no specialized agent?
  └─ Invoke agent-creator → it will design one following SkillsBench constraints

New model release?
  └─ Delete AGENTS.md. Test. Re-add only what breaks.
  └─ Review LESSONS_LEARNED — archive anything the new model handles correctly.

AGENTS.md getting long?
  └─ Something is wrong. It should shrink over time, not grow.
  └─ Are corrections in AGENTS.md? Move them to LESSONS_LEARNED.
  └─ Is discoverable info in AGENTS.md? Delete it.

Time for maintenance? (weekly/monthly/yearly/new model)
  └─ Hand this entire document to an agent with:
     "Run the Periodic Maintenance Protocol on this project."
  └─ Review the maintenance report. Approve or adjust flagged items.
```

---

## References

- Röttger et al. (2026). *Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?* [arxiv.org/abs/2602.11988](https://arxiv.org/abs/2602.11988)
- Li et al. (2026). *SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks.* [arxiv.org/abs/2602.12670](https://arxiv.org/abs/2602.12670)
