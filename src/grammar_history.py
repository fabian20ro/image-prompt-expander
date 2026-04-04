"""Persisted grammar revision history helpers."""

import json
from datetime import datetime
from pathlib import Path


def _history_path(run_dir: Path, prefix: str) -> Path:
    return run_dir / f"{prefix}_grammar_history.json"


def load_grammar_history(run_dir: Path, prefix: str, current_grammar: str | None = None) -> list[dict]:
    """Load grammar history, falling back to a single current revision."""
    history_path = _history_path(run_dir, prefix)
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    if current_grammar is None:
        grammar_path = run_dir / f"{prefix}_grammar.json"
        current_grammar = grammar_path.read_text() if grammar_path.exists() else ""

    if not current_grammar:
        return []

    return [{
        "id": "initial",
        "created_at": datetime.now().isoformat(),
        "action": "initial",
        "grammar": current_grammar,
    }]


def save_grammar_history(run_dir: Path, prefix: str, history: list[dict]) -> Path:
    """Write grammar history to disk."""
    history_path = _history_path(run_dir, prefix)
    history_path.write_text(json.dumps(history, indent=2))
    return history_path


def append_grammar_revision(
    run_dir: Path,
    prefix: str,
    grammar: str,
    action: str,
) -> list[dict]:
    """Append a revision if the grammar changed or action is new."""
    history_path = _history_path(run_dir, prefix)
    history_exists = history_path.exists()
    history = load_grammar_history(run_dir, prefix)
    last = history[-1] if history else None
    if last and last.get("grammar") == grammar and last.get("action") == action:
        if not history_exists:
            save_grammar_history(run_dir, prefix, history)
        return history

    if last and last.get("grammar") == grammar:
        if not history_exists:
            save_grammar_history(run_dir, prefix, history)
        return history

    history.append({
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "created_at": datetime.now().isoformat(),
        "action": action,
        "grammar": grammar,
    })
    save_grammar_history(run_dir, prefix, history)
    return history
