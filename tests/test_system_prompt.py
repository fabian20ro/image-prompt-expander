import pytest
from pathlib import Path
from src.grammar_generator import get_system_prompt

def test_get_system_prompt_default(tmp_path):
    # Setup: create a dummy template file
    template_file = tmp_path / "system_prompt.txt"
    template_file.write_text("generic prompt")
    
    # When
    prompt = get_system_prompt(templates_dir=tmp_path)
    
    # Then
    assert prompt == "generic prompt"

def test_get_system_prompt_model_specific(tmp_path):
    # Setup: create a model-specific template file
    model_template_file = tmp_path / "system_prompt_flux2-klein.txt"
    model_template_file.write_text("model prompt")
    
    # When
    prompt = get_system_prompt(model="flux2-klein-4b", templates_dir=tmp_path)
    
    # Then
    assert prompt == "model prompt"

def test_get_system_prompt_fallback(tmp_path):
    # Setup: create a dummy template file
    template_file = tmp_path / "system_prompt.txt"
    template_file.write_text("generic prompt")
    
    # When
    prompt = get_system_prompt(model="some-unknown-model", templates_dir=tmp_path)
    
    # Then
    assert prompt == "generic prompt"
