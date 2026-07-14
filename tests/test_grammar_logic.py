"""Tests for grammar_generator logic."""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grammar_generator import (
    _api_root,
    get_system_prompt,
    hash_prompt,
    clean_grammar_output,
)


def test_get_system_prompt_default(tmp_path):
    # Given
    generic_content = "generic content"
    (tmp_path / "system_prompt.txt").write_text(generic_content)
    
    # When
    result = get_system_prompt(templates_dir=tmp_path)
    
    # Then
    assert result == generic_content


def test_hash_prompt():
    # Given
    prompt = "a cat"
    # When
    h1 = hash_prompt(prompt)
    h2 = hash_prompt(prompt)
    h3 = hash_prompt("a dog")
    
    # Then
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 12


def test_clean_grammar_output_multiple_blocks():
    # Given
    raw = "Some text\n```json\n{\"a\": 1}\n```\nMore text\n```tracery\n{\"b\": 2}\n```"
    # When
    result = clean_grammar_output(raw)
    # Then
    assert result == '{"a": 1}'


def test_handles_json_array():
    # Verifies that if the LLM returns a JSON array, it is preserved
    # Given
    input_text = '```json\n[1, 2, 3]\n```'
    # When
    result = clean_grammar_output(input_text)
    # Then
    assert result == '[1, 2, 3]'


def test_handles_json_object_with_extra_content():
    # Verifies that content after the JSON object is discarded
    # Given
    input_text = '{"a": 1} extra'
    # When
    result = clean_grammar_output(input_text)
    # Then
    assert result == '{"a": 1}'


def test_handles_json_in_text_array():
    # Verifies extraction of JSON array from within text
    # Given
    input_text = 'Here is an array: [1, 2, 3] and more'
    # When
    result = clean_grammar_output(input_text)
    # Then
    assert result == '[1, 2, 3]'


def test_clean_grammar_output_no_blocks():
    # Given
    raw = "Just some text with {\"a\": 1} inside"
    # When
    result = clean_grammar_output(raw)
    # Then
    assert result == '{"a": 1}'


def test_clean_grammar_output_normalizes_smart_quotes():
    """Smart (curly) double quotes in LLM output must be replaced with straight ASCII quotes."""
    raw = "Here is the grammar: {\u201ckolors\u201d: [\u201cA sepia tone, soft grainy texture\u201d]}"
    result = clean_grammar_output(raw)
    assert '\u201c' not in result and '\u201d' not in result
    expected = '{"kolors": ["A sepia tone, soft grainy texture"]}'
    assert result == expected


def test_clean_grammar_output_normalizes_smart_single_quotes():
    """Left/right single curly quotes (U+2018, U+2019) must be replaced with straight ASCII apostrophes."""
    raw = '{"prompt": "a cat\u2019s eye"}'  # \u2019 is right single curly quote
    result = clean_grammar_output(raw)
    assert '\u2019' not in result and '\u2018' not in result
    expected = '{"prompt": "a cat\'s eye"}'
    assert result == expected


def test_clean_grammar_output_strips_thinking_blocks():
    """Thinking blocks (``...``) must be removed before JSON extraction."""
    raw = '``thinking block\n\n```json\n{"origin": ["a", "b"]}\n```\n'
    result = clean_grammar_output(raw)
    assert 'thinking' not in result
    expected = '{"origin": ["a", "b"]}'
    assert result == expected


def test_clean_grammar_output_no_json_returns_stripped_input():
    """When no JSON object/array is found, clean returns the stripped input unchanged."""
    raw = "Here's a description of what I want to see."
    result = clean_grammar_output(raw)
    assert result == raw


def test_clean_grammar_output_strips_thinking_blocks_with_smart_quotes():
    """Thinking block removal and smart quote normalization must compose correctly in one pass."""
    raw = '``thinking\n\n```json\n{"kolors": [\u201cA sepia tone\u201d]}\n```\n'
    result = clean_grammar_output(raw)
    assert '\u201c' not in result and '\u201d' not in result
    expected = '{"kolors": ["A sepia tone"]}'
    assert result == expected


# ---------------------------------------------------------------------------
# _api_root — pure URL transformation (no side effects, no mocks needed)
# ---------------------------------------------------------------------------

def test_api_root_strips_v1_suffix():
    """LM Studio base URLs conventionally end in /v1; _api_root must strip it."""
    assert _api_root("http://localhost:1234/v1") == "http://localhost:1234"


def test_api_root_handles_trailing_slash_with_v1():
    """Trailing slash before /v1 should still resolve to the server root."""
    assert _api_root("http://localhost:1234/v1/") == "http://localhost:1234"


def test_api_root_leaves_non_v1_urls_unchanged():
    """Non-LM-Studio URLs (no /v1 segment) should round-trip untouched."""
    url = "http://example.com/api"
    assert _api_root(url) == url


def test_api_root_does_not_strip_mid_path_v1():
    """/v1 appearing in the middle of a path must NOT be stripped — only trailing /v1 is removed."""
    url = "http://localhost:1234/some/v1/path/extra"
    assert _api_root(url) == url


def test_api_root_handles_v1_with_query():
    """A URL with query params after /v1 should round-trip unchanged (not endswith /v1)."""
    url = "http://localhost:1234/v1?param=value"
    assert _api_root(url) == url


def test_api_root_case_sensitive_v1():
    """/V1 or /V1/ variants must not be stripped — only lowercase /v1 is the LM Studio convention."""
    url_upper = "http://localhost:1234/V1"
    assert _api_root(url_upper) == url_upper

    url_mixed = "http://localhost:1234/v1/"
    assert _api_root(url_mixed) == "http://localhost:1234"  # lowercase stripped per contract
