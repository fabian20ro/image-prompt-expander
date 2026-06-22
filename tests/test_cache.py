import pytest
import json
from pathlib import Path
from src.grammar_generator import cache_grammar, get_cached_grammar, get_cached_raw_response, hash_prompt, clean_grammar_output
from config import settings

def test_grammar_cache_lifecycle(tmp_path, monkeypatch):
    # Setup: mock CACHE_DIR to a temp directory
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Data
    prompt_hash = "abc123def456"
    grammar_content = '{"prompt": "test"}'
    raw_response = "```json\n" + grammar_content + "\n```"
    user_prompt = "test prompt"

    # Act: Cache it
    cache_grammar(
        prompt_hash=prompt_hash,
        grammar=grammar_content,
        raw_response=raw_response,
        user_prompt=user_prompt,
    )

    # Assert: File exists
    expected_file = mock_cache_dir / f"{prompt_hash}.tracery.json"
    assert expected_file.exists()

    # Act: Retrieve it
    retrieved_grammar = get_cached_grammar(prompt_hash)

    # Assert: Matches
    assert retrieved_grammar == grammar_content

def test_get_cached_grammar_miss(tmp_path, monkeypatch):
    # Setup
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Act
    retrieved = get_cached_grammar("nonexistent")

    # Assert
    assert retrieved is None

def test_get_cached_raw_response_miss(tmp_path, monkeypatch):
    # Setup
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Act
    retrieved = get_cached_raw_response("nonexistent")

    # Assert
    assert retrieved is None

def test_cache_full_integrity(tmp_path, monkeypatch):
    # Setup
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Data
    prompt_hash = "full_integrity_test"
    grammar_content = '{"origin": {"a": ["#b#"], "b": ["1", "2", "3", "4", "5", "6"]}}'
    raw_response = "```json\n" + grammar_content + "\n```"
    user_prompt = "Integrity test prompt"

    # Act: Cache it
    returned_grammar, was_cached, returned_raw = cache_grammar(
        prompt_hash=prompt_hash,
        grammar=grammar_content,
        raw_response=raw_response,
        user_prompt=user_prompt,
    )

    # Assert: Returned values
    assert returned_grammar == grammar_content
    assert was_cached is False
    assert returned_raw == raw_response

    # Assert: Grammar
    assert get_cached_grammar(prompt_hash) == grammar_content

    # Assert: Raw response
    assert get_cached_raw_response(prompt_hash) == raw_response

    # Assert: Metadata
    metadata_file = mock_cache_dir / f"{prompt_hash}.metaprompt.json"
    assert metadata_file.exists()
    metadata = json.loads(metadata_file.read_text())
    assert metadata["user_prompt"] == user_prompt
    assert metadata["hash"] == prompt_hash
    assert metadata["prompt_schema"] == "ernie-v2"
    assert metadata["lm_model"] == settings.lm_studio.model
    assert "created_at" in metadata

def test_hash_prompt_deterministic():
    # Test that hash_prompt is deterministic
    prompt = "a beautiful sunset over the mountains"
    hash1 = hash_prompt(prompt)
    hash2 = hash_prompt(prompt)
    assert hash1 == hash2
    assert len(hash1) == 12
    assert len(hash1) > 0

def test_hash_prompt_different_prompts():
    # Test that different prompts result in different hashes
    prompt1 = "a beautiful sunset over the mountains"
    prompt2 = "a beautiful sunrise over the mountains"
    assert hash_prompt(prompt1) != hash_prompt(prompt2)

def test_hash_prompt_emojis():
    # Test that hash_prompt handles emojis correctly
    prompt = "a beautiful sunset 🌅 over the mountains 🏔️"
    hash1 = hash_prompt(prompt)
    hash2 = hash_prompt(prompt)
    assert hash1 == hash2
    assert len(hash1) == 12

def test_clean_grammar_output():
    # Basic
    assert clean_grammar_output('{"a": 1}') == '{"a": 1}'
    # Markdown code blocks
    assert clean_grammar_output('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert clean_grammar_output('```tracery\n{"a": 1}\n```') == '{"a": 1}'
    # Thinking blocks
    assert clean_grammar_output('<think>some thought</think>{"a": 1}') == '{"a": 1}'
    # Smart quotes
    assert clean_grammar_output('{\u201c\u201d: 1}') == '{"": 1}'
    # Extra text
    assert clean_grammar_output('Here is the json: {"a": 1} end of message') == '{"a": 1}'
