from pathlib import Path
from src.grammar_generator import get_system_prompt, hash_prompt, get_cached_grammar, cache_grammar, get_cached_raw_response
import json

def test_get_system_prompt_default(tmp_path):
    # Setup: create a dummy template file
    template_file = tmp_path / "system_prompt.txt"
    template_file.write_text("generic prompt")
    
    # When
    prompt = get_system_prompt(templates_dir=tmp_path)
    
    # Then
    assert prompt == "generic prompt"

def test_project_system_prompt_is_ernie_specific():
    prompt = get_system_prompt()
    assert "ERNIE-Image-Turbo" in prompt
    assert "Tracery" in prompt
    assert "Subject: exact identity" in prompt
    assert "Description and details" in prompt
    assert "Style and medium" in prompt
    assert "Technical and capture finish" in prompt
    assert "exactly 7 distinct" in prompt
    assert "Never use 2–4 alternatives" in prompt
