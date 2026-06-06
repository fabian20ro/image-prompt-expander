"""Tests for src/grammar_history — persisted grammar revision history helpers."""

import json
from pathlib import Path

import pytest

# Add src directory to path for imports
sys_path = __import__("sys").path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys_path:
    sys_path.insert(0, str(src_dir))

from grammar_history import (
    _history_path,
    append_grammar_revision,
    load_grammar_history,
    save_grammar_history,
)


class TestHistoryPath:
    def test_returns_correct_path(self, run_dir):
        result = _history_path(run_dir, "test")
        assert result == run_dir / "test_grammar_history.json"


class TestLoadGrammarHistory:
    def test_loads_existing_valid_json(self, run_dir):
        history = [
            {"id": "rev1", "created_at": "2024-01-01T00:00:00", "action": "initial", "grammar": "rule1"},
            {"id": "rev2", "created_at": "2024-01-02T00:00:00", "action": "update", "grammar": "rule2"},
        ]
        path = _history_path(run_dir, "test")
        path.write_text(json.dumps(history))

        result = load_grammar_history(run_dir, "test")
        assert len(result) == 2
        assert result[0]["id"] == "rev1"
        assert result[1]["action"] == "update"

    def test_returns_empty_when_no_file_and_no_grammar(self, run_dir):
        result = load_grammar_history(run_dir, "missing")
        assert result == []

    def test_falls_back_to_current_grammar_when_no_history_file(self, run_dir):
        grammar_text = '{"origin": ["hello"]}'
        result = load_grammar_history(run_dir, "test", current_grammar=grammar_text)
        assert len(result) == 1
        assert result[0]["id"] == "initial"
        assert result[0]["grammar"] == grammar_text

    def test_falls_back_to_current_grammar_on_corrupted_json(self, run_dir):
        path = _history_path(run_dir, "test")
        path.write_text("not valid json{{{")

        grammar_text = '{"origin": ["fallback"]}'
        result = load_grammar_history(run_dir, "test", current_grammar=grammar_text)
        assert len(result) == 1
        assert result[0]["id"] == "initial"

    def test_returns_empty_when_current_grammar_is_empty_string(self, run_dir):
        result = load_grammar_history(run_dir, "test", current_grammar="")
        assert result == []

    def test_reads_from_disk_when_no_current_grammar_provided(self, run_dir):
        grammar_text = '{"origin": ["from_disk"]}'
        (run_dir / "test_grammar.json").write_text(grammar_text)

        result = load_grammar_history(run_dir, "test")
        assert len(result) == 1
        assert result[0]["grammar"] == grammar_text


class TestSaveGrammarHistory:
    def test_writes_and_returns_path(self, run_dir):
        history = [{"id": "x", "created_at": "2024-01-01T00:00:00", "action": "test", "grammar": "g"}]
        result_path = save_grammar_history(run_dir, "test", history)

        assert result_path.exists()
        loaded = json.loads(result_path.read_text())
        assert len(loaded) == 1
        assert loaded[0]["id"] == "x"


class TestAppendGrammarRevision:
    def test_appends_new_revision(self, run_dir):
        history = append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        assert len(history) == 1
        assert history[0]["grammar"] == "rule_a"

    def test_skips_when_grammar_and_action_unchanged(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        history = append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        assert len(history) == 1

    def test_skips_when_only_grammar_unchanged(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        history = append_grammar_revision(run_dir, "test", grammar="rule_a", action="update")
        assert len(history) == 1

    def test_appends_when_grammar_changes(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        history = append_grammar_revision(run_dir, "test", grammar="rule_b", action="update")
        assert len(history) == 2
        assert history[1]["grammar"] == "rule_b"

    def test_creates_history_file_on_first_append(self, run_dir):
        history = append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        path = _history_path(run_dir, "test")
        assert path.exists()
