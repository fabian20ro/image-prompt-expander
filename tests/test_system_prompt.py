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


def test_system_prompt_contains_all_content_type_patterns():
    """Verify the template includes all five documented content-type patterns."""
    prompt = get_system_prompt()
    for pattern in ["Portrait", "Product", "Poster/infographic", "Comic", "Landscape/concept art"]:
        assert f"- {pattern}:" in prompt, f"Missing content-type pattern: {pattern}"


def test_system_prompt_enforces_single_quote_visible_text():
    """Verify the template requires wrapping visible text in single quotes."""
    prompt = get_system_prompt()
    assert "wrap each visible string in single quotes" in prompt


def test_system_prompt_forbids_vague_placeholders():
    """The template explicitly lists forbidden vague placeholders to prevent low-quality prompts.

    These phrases are listed as examples of what the generator must never emit.
    Their presence in the system prompt confirms the quality guardrail is active.
    """
    prompt = get_system_prompt()
    for placeholder in [
        '"some text"',
        '"a button"',
        '"relevant facts"',
        '"and so on"',
    ]:
        assert (
            placeholder in prompt
        ), f"Forbidden vague placeholder guardrail missing: {placeholder}"


def test_system_prompt_forbids_markdown_and_commentary():
    """Verify the template explicitly forbids Markdown formatting, commentary, and reasoning output.

    The generator must return pure JSON — never prose, explanations, or markdown-wrapped blocks.
    These prohibitions are listed in the system prompt to prevent low-quality responses.
    """
    prompt = get_system_prompt()
    for prohibition in [
        "No Markdown",
        "commentary",
        "reasoning",
    ]:
        assert (
            prohibition in prompt
        ), f"Output guardrail missing from system prompt: {prohibition}"


def test_system_prompt_forbids_duplicate_alternatives():
    """Verify the template explicitly forbids duplicating or near-duplicating Tracery alternatives.

    The grammar generator must produce truly distinct options per rule — not inflated counts of
    nearly identical entries. This guardrail prevents wasted variation and low-quality grammars.
    """
    prompt = get_system_prompt()
    for prohibition in [
        "duplicate",
        "rephrase",
    ]:
        assert (
            prohibition in prompt
        ), f"Anti-duplicate guardrail missing from system prompt: {prohibition}"


def test_system_prompt_enforces_output_quality_constraints():
    """Verify the template enforces word count, contradiction avoidance, style consistency, and translation rules.

    These constraints prevent low-quality or inconsistent outputs across series generation.
    """
    prompt = get_system_prompt()
    for constraint in [
        "50–150 words",
        "contradictory",
        "consistent across a series",
        "Do not translate it",
    ]:
        assert (
            constraint in prompt
        ), f"Output quality constraint missing from system prompt: {constraint}"


def test_system_prompt_forbids_inference_metadata():
    """Verify the template forbids embedding inference metadata into prompts.

    The generator must never write guidance scale, quantization info, or numeric resolution
    into the expanded prompt — the application controls those parameters separately.
    """
    prompt = get_system_prompt()
    for forbidden in [
        "guidance scale",
        "quantization",
        "numeric output resolution",
    ]:
        assert (
            forbidden in prompt
        ), f"Inference metadata prohibition missing from system prompt: {forbidden}"
