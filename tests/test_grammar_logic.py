"""Tests for grammar_generator logic."""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grammar_generator import get_system_prompt, hash_prompt

def test_get_system_prompt_default(tmp_path):
    # Given
    generic_content = "generic content"
    (tmp_path / "system_prompt.txt").write_text(generic_content)
    
    # When
    result = get_system_prompt(model=None, templates_dir=tmp_path)
    
    # Then
    assert result == generic_content

def test_get_system_prompt_model_specific(tmp_path):
    # Given
    model_name = "flux2-klein-4b"
    model_prefix = "flux2-klein"
    specific_content = "specific content"
    (tmp_path / f"system_prompt_{model_prefix}.txt").write_text(specific_content)
    
    # When
    result = get_system_prompt(model=model_name, templates_dir=tmp_path)
    
    # Then
    assert result == specific_content

def test_get_system_prompt_fallback(tmp_path):
    # Given
    generic_content = "generic content"
    (tmp_path / "system_prompt.txt").write_text(generic_content)
    
    # When
    result = get_system_prompt(model="unknown-model", templates_dir=tmp_path)
    
    # Then
    assert result == generic_content

def test_hash_prompt():
    # Given
    prompt = "a cat"
    model = "model-a"
    
    # When
    h1 = hash_prompt(prompt, model)
    h2 = hash_prompt(prompt, model)
    h3 = hash_prompt(prompt)
    
    # Then
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 12
