import pytest
import json
from pathlib import Path
import src.grammar_generator as grammar_gen
from src.grammar_generator import cache_grammar, get_cached_grammar, get_cached_raw_response, hash_prompt, clean_grammar_output, validate_grammar_structure
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


def test_hash_prompt_schema_versioned():
    # Changing schema version should produce a different hash for the same prompt,
    # ensuring cache invalidation when grammar schema changes.
    prompt = "test prompt"
    with pytest.MonkeyPatch.context() as mp:
        original_schema = grammar_gen.PROMPT_SCHEMA_VERSION
        hash_original = hash_prompt(prompt)

        mp.setattr("src.grammar_generator.PROMPT_SCHEMA_VERSION", "ernie-v3")
        hash_new = hash_prompt(prompt)

    assert hash_new != hash_original

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

def test_clean_grammar_output_robustness():
    # Multiple thinking blocks and code blocks
    assert clean_grammar_output('<think>1</think>```json\n{"a": 1}\n```<think>2</think>') == '{"a": 1}'
    # Array extraction
    assert clean_grammar_output('Result: [1, 2, 3] end') == '[1, 2, 3]'
    # Unbalanced braces - should return original after strip
    assert clean_grammar_output('{ "unbalanced": 1 ') == '{ "unbalanced": 1'

def test_cache_directory_creation(tmp_path, monkeypatch):
    # Setup: mock CACHE_DIR to a non-existent path
    mock_cache_dir = tmp_path / "nested" / "cache"
    # We don't call .mkdir() here
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Act: Cache something
    prompt_hash = "dir_creation_test"
    grammar_content = '{"a": 1}'
    raw_response = '```json\n{"a": 1}\n```'
    user_prompt = "test"
    cache_grammar(prompt_hash, grammar_content, raw_response, user_prompt)

    # Assert
    assert mock_cache_dir.exists()

def test_clean_grammar_output_extra_whitespace():
    assert clean_grammar_output("   {\"a\": 1}   ") == "{\"a\": 1}"

def test_clean_grammar_output_nested_json_in_text():
    assert clean_grammar_output('The JSON is: {"a": 1, "b": [1, 2]} and more') == '{"a": 1, "b": [1, 2]}'

def test_clean_grammar_output_smart_quotes_in_json():
    assert clean_grammar_output('{"key": \u201cvalue\u201d}') == '{"key": "value"}'

def test_clean_grammar_output_multiple_json_objects():
    assert clean_grammar_output('{"first": 1} {"second": 2}') == '{"first": 1}'

def test_clean_grammar_output_markdown_variations():
    # Test markdown block with no language identifier
    assert clean_grammar_output('```\n{"a": 1}\n```') == '{"a": 1}'
    # Test markdown block with unknown language
    assert clean_grammar_output('```xml\n{"a": 1}\n```') == '{"a": 1}'

def test_clean_grammar_output_unclosed_markdown():
    assert clean_grammar_output('```json\n{"a": 1}') == '{"a": 1}'

def test_clean_grammar_output_multiple_thinking_blocks():
    assert clean_grammar_output('text{"a": 1}') == '{"a": 1}'

def test_validate_grammar_structure():
    # Valid grammar (flat)
    valid_grammar = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": ["1"]
    }
    # This should not raise an exception
    validate_grammar_structure(valid_grammar)

    # Invalid: no varying rule (all rules have exactly 1 alternative)
    no_varying = {
        "origin": ["#a#"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Grammar must contain at least one varying rule"):
        validate_grammar_structure(no_varying)

    # Invalid: missing origin
    with pytest.raises(ValueError, match='Grammar must be a JSON object containing an "origin" rule'):
        validate_grammar_structure({"a": ["1"]})

    # Invalid: too many rules
    too_many_rules = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": ["1"],
        "b": ["1"],
        "c": ["1"],
        "d": ["1"],
        "e": ["1"],
        "f": ["1"],
        "g": ["1"],
        "h": ["1"]
    }
    with pytest.raises(ValueError, match='Grammar must contain at most 8 rules'):
        validate_grammar_structure(too_many_rules)

    # Invalid: duplicate alternatives
    duplicates = {
        "origin": ["1", "1", "3", "4", "5", "6"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* contains duplicate alternatives"):
        validate_grammar_structure(duplicates)

    # Invalid: rule with too few/many alternatives (not varying)
    too_few_alternatives = {
        "origin": ["1", "2"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Varying grammar rule .* must contain 5–7 alternatives"):
        validate_grammar_structure(too_few_alternatives)

    # Invalid: too many alternatives (above 7)
    too_many_alternatives = {
        "origin": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Varying grammar rule .* must contain 5–7 alternatives"):
        validate_grammar_structure(too_many_alternatives)

    # Edge case: exactly 5 and exactly 7 should be valid (no raise)
    five_alt = {"origin": ["1", "2", "3", "#a#", "5"], "a": ["1"]}
    validate_grammar_structure(five_alt)
    seven_alt = {"origin": ["1", "2", "3", "4", "5", "6", "#a#"], "a": ["1"]}
    validate_grammar_structure(seven_alt)

    # Edge case: exactly 8 should fail (one past the upper bound)
    eight_alt = {
        "origin": ["1", "2", "3", "4", "5", "6", "7", "#a#"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Varying grammar rule .* must contain 5–7 alternatives"):
        validate_grammar_structure(eight_alt)

    # Invalid: missing references
    missing_ref = {
        "origin": ["#missing#", "2", "3", "4", "5", "6"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Grammar references missing rules: missing"):
        validate_grammar_structure(missing_ref)

    # Invalid: non-string alternatives
    non_string = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": [1]
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* must contain non-empty strings"):
        validate_grammar_structure(non_string)

    # Invalid: rule value is not a list (e.g. a bare string from malformed JSON)
    not_a_list = {
        "origin": "#a#",
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* must be a non-empty array"):
        validate_grammar_structure(not_a_list)

    # Invalid: empty list alternative (no options)
    empty_rule = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": []
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* must be a non-empty array"):
        validate_grammar_structure(empty_rule)

    # Invalid: rule value is None instead of a list
    none_rule = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": None
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* must be a non-empty array"):
        validate_grammar_structure(none_rule)


from unittest.mock import patch, MagicMock

def test_get_cached_raw_response_hit():
    """Cache hit: raw response file exists and can be read back exactly."""
    prompt_hash = "raw_hit_test"
    raw_content = "<think>thought</think>```json\n{\"origin\": [\"#a#\"]}\n```"
    with patch("src.grammar_generator.CACHE_DIR") as mock_dir:
        mock_file = MagicMock(spec=Path)
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = raw_content
        (mock_dir.__truediv__).return_value = mock_file

        retrieved = get_cached_raw_response(prompt_hash)

    assert retrieved == raw_content


def test_cache_grammar_overwrite(tmp_path, monkeypatch):
    """Cache update: re-caching the same hash must overwrite both grammar and raw files."""
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "overwrite_test"
    initial_grammar = '{"origin": ["#a#"], "a": ["1"]}'
    initial_raw = '```json\ninitial content\n```'
    user_prompt = "original prompt"

    cache_grammar(prompt_hash, initial_grammar, initial_raw, user_prompt)

    # First retrieval matches what was written
    assert get_cached_grammar(prompt_hash) == initial_grammar
    assert get_cached_raw_response(prompt_hash) == initial_raw

    # Now re-cache with new content (simulates regeneration with different params)
    updated_grammar = '{"origin": ["#b#", "x"], "a": ["1"], "b": ["hello"]}'
    updated_raw = '```json\nupdated thinking\n```'

    cache_grammar(prompt_hash, updated_grammar, updated_raw, user_prompt)

    # Overwritten content is now retrievable and old content is gone
    assert get_cached_grammar(prompt_hash) == updated_grammar
    assert get_cached_raw_response(prompt_hash) == updated_raw


def test_cache_persists_all_three_files(tmp_path, monkeypatch):
    """cache_grammar must write all three files: grammar, raw response, and metadata."""
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "triple_file_test"
    grammar_content = '{"origin": ["#a#"], "a": ["1"]}'
    raw_response = '```json\n{"origin":["#a#"],"a":["1"]}\n```'
    user_prompt = "multi-file test"

    cache_grammar(prompt_hash, grammar_content, raw_response, user_prompt)

    # All three files must exist for this hash
    assert (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()
    assert (mock_cache_dir / f"{prompt_hash}.raw.txt").exists()
    assert (mock_cache_dir / f"{prompt_hash}.metaprompt.json").exists()

    # Each file must contain the correct content exactly as written
    assert (mock_cache_dir / f"{prompt_hash}.tracery.json").read_text() == grammar_content
    assert (mock_cache_dir / f"{prompt_hash}.raw.txt").read_text() == raw_response