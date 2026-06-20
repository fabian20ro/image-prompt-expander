import pytest
import json
from pathlib import Path
from src.grammar_generator import cache_grammar, get_cached_grammar

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
