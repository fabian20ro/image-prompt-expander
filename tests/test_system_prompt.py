import pytest
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

def test_hash_prompt():
    """Test that hash_prompt generates consistent 12-char hex hashes."""
    prompt = "a dragon"
    model = "flux2-klein"
    h1 = hash_prompt(prompt, model)
    h2 = hash_prompt(prompt, model)
    assert h1 == h2
    assert len(h1) == 12
    
    h3 = hash_prompt(prompt)
    assert len(h3) == 12
    assert h1 != h3

def test_cache_workflow(tmp_path, monkeypatch):
    """Test the full cache/get workflow for grammars and raw responses."""
    import src.grammar_generator as gg
    # Use monkeypatch to safely swap CACHE_DIR for this test
    monkeypatch.setattr(gg, "CACHE_DIR", tmp_path)
    
    prompt = "a dragon"
    model = "flux2-klein"
    p_hash = hash_prompt(prompt, model)
    grammar = '{"origin": ["dragon"]}'
    raw = '{"thinking": "yes", "content": "grammar"}'
    
    # Test cache_grammar
    g, cached, raw_res = cache_grammar(p_hash, grammar, raw, prompt, model)
    assert g == grammar
    assert cached is False
    assert raw_res == raw
    
    # Test get_cached_grammar
    assert get_cached_grammar(p_hash) == grammar
    
    # Test get_cached_raw_response
    assert get_cached_raw_response(p_hash) == raw
    
    # Test miss
    assert get_cached_grammar("non-existent-hash") is None
    assert get_cached_raw_response("non-existent-hash") is None
