"""Tests for grammar_generator module."""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import patch
from grammar_generator import clean_grammar_output, get_system_prompt, hash_prompt

class TestCleanGrammarOutput(unittest.TestCase):
    """Tests for the clean_grammar_output function."""

    # Tests for removing <think> tags
    def test_removes_think_tags_single_line(self):
        # Verifies single-line think tags are stripped from output
        # Given
        input_text = '<think>some thinking</think>{"origin": "#prompt#"}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#"}'

    def test_removes_think_tags_multiline(self):
        # Verifies multi-line think tags are stripped from output
        # Given
        input_text = '''<think>
This is some
multi-line thinking
content here
</think>{"origin": "#prompt#"}'''

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#"}'

    def test_removes_think_tags_with_nested_json(self):
        # Verifies think tags containing JSON-like content are removed
        # Given
        input_text = '<think>Let me think about {"key": "value"}</think>{"origin": "#test#"}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#test#"}'

    # Tests for removing markdown code blocks
    def test_removes_json_code_block_with_newline(self):
        # Verifies ```json blocks with newline after marker are handled
        # Given
        input_text = '''```json
{"origin": "#prompt#", "prompt": ["test"]}
```'''

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#", "prompt": ["test"]}'

    def test_removes_json_code_block_without_newline(self):
        # Verifies ```json blocks without newline after marker are handled
        # Given
        input_text = '```json{"origin": "#prompt#"}```'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#"}'

    def test_removes_multiple_think_blocks(self):
        # Verifies multiple think blocks are all removed
        # Given
        input_text = '<think>first</think>text<think>second</think>{"origin": "#test#"}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#test#"}'

    def test_handles_no_tags(self):
        # Verifies input with no special tags remains unchanged (except strip)
        # Given
        input_text = '{"origin": "#test#"}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#test#"}'

    def test_removes_tracery_code_block(self):
        # Verifies ```tracery blocks are handled
        # Given
        input_text = '''```tracery
{"origin": "#prompt#"}
```'''

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#"}'

    def test_removes_plain_code_block(self):
        # Verifies plain ``` blocks without language marker are handled
        # Given
        input_text = '''```
{"origin": "#prompt#"}
```'''

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#"}'

    def test_removes_code_block_with_extra_whitespace(self):
        # Verifies code blocks with extra whitespace are handled
        # Given
        input_text = '''```json
{"origin": "#prompt#", "prompt": ["test"]}
```'''

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#", "prompt": ["test"]}'

    # Tests for combined scenarios
    def test_removes_think_and_code_block(self):
        # Verifies both think tags and code blocks are removed together
        # Given
        input_text = '''<think>
Let me create a grammar...
</think>

```json
{"origin": "#prompt#", "prompt": ["a #subject#"]}
```'''

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#", "prompt": ["a #subject#"]}'

    def test_handles_plain_json(self):
        # Verifies plain JSON without any markers passes through unchanged
        # Given
        input_text = '{"origin": "#prompt#"}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#"}'

    def test_extracts_json_from_surrounding_text(self):
        # Verifies JSON is extracted when surrounded by non-JSON text
        # Given
        input_text = 'Here is the grammar: {"origin": "#prompt#"} Hope this helps!'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#"}'

    def test_handles_complex_nested_json(self):
        # Verifies complex nested JSON structures are preserved
        # Given
        input_text = '''```json
{
    "origin": "#prompt#",
    "prompt": ["#subject# in #setting#"],
    "subject": ["dragon", "phoenix"],
    "setting": ["mountains", "forest"]
}
```'''

        # When
        result = clean_grammar_output(input_text)

        # Then
        import json
        parsed = json.loads(result)
        assert parsed["origin"] == "#prompt#"
        assert "dragon" in parsed["subject"]
        assert "mountains" in parsed["setting"]

    # Tests for smart quote normalization
    def test_replaces_smart_double_quotes(self):
        # Verifies curly double quotes are replaced with straight quotes
        # Given
        input_text = '{\u201corigin\u201d: \u201c#prompt#\u201d}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert '\u201c' not in result
        assert '\u201d' not in result
        import json
        parsed = json.loads(result)
        assert parsed["origin"] == "#prompt#"

    def test_replaces_smart_single_quotes(self):
        # Verifies curly single quotes are replaced with straight quotes
        # Given
        input_text = '{"origin": "#prompt#", "test": "it\u2019s working"}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert '\u2018' not in result
        assert '\u2019' not in result
        assert "it's working" in result

    def test_replaces_mixed_smart_quotes(self):
        # Verifies both left and right curly quotes are normalized
        # Given
        input_text = '{\u201cprompt\u201d: [\u201ca dragon in the sky\u201d], \u201ctest\u201d: \u201cit\u2019s great\u201d}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        import json
        parsed = json.loads(result)
        assert parsed["test"] == "it's great"


    def test_handles_malformed_json_after_start_brace(self):
        # Verifies that if JSON parsing fails after finding a brace, 
        # it returns the original string instead of crashing.
        # Given
        input_text = 'Prefix {' + 'invalid json'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == 'Prefix {invalid json'

    def test_replaces_smart_quotes_in_code_block(self):
        # Verifies smart quotes inside code blocks are normalized
        # Given
        input_text = '''```json
{
    "origin": "#prompt#",
    "payload": {
        "key": "value",
        "nested": [1, 2, 3]
    }
}
```'''

        # When
        result = clean_grammar_output(input_text)

        # Then
        import json
        parsed = json.loads(result)
        assert parsed["origin"] == "#prompt#"
        assert parsed["payload"]["key"] == "value"
        assert parsed["payload"]["nested"] == [1, 2, 3]


class TestGetSystemPrompt:
    """Tests for get_system_prompt."""

    def test_get_system_prompt_flux2_normalization(self, tmp_path):
        # Given
        model = "flux2-klein-4b"
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        system_prompt_file = templates_dir / "system_prompt_flux2-klein.txt"
        system_prompt_file.write_text("flux2-klein prompt")

        # When
        result = get_system_prompt(model, templates_dir=templates_dir)
        # Then
        assert result == "flux2-klein prompt"

    def test_get_system_prompt_fallback(self, tmp_path):
        # Given
        model = "some-other-model"
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        system_prompt_file = templates_dir / "system_prompt.txt"
        system_prompt_file.write_text("generic prompt")

        # When
        result = get_system_prompt(model, templates_dir=templates_dir)
        # Then
        assert result == "generic prompt"

    def test_get_system_prompt_none(self, tmp_path):
        # Given
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        system_prompt_file = templates_dir / "system_prompt.txt"
        system_prompt_file.write_text("generic prompt")

        # When
        result = get_system_prompt(None, templates_dir=templates_dir)
        # Then
        assert result == "generic prompt"

class TestHashPrompt:
    """Tests for hash_prompt."""
    def test_hash_prompt_consistency(self):
        prompt = "a cute cat"
        model = "flux2-klein"
        h1 = hash_prompt(prompt, model)
        h2 = hash_prompt(prompt, model)
        assert h1 == h2
        assert len(h1) == 12

    def test_hash_prompt_with_model_change(self):
        prompt = "a cute cat"
        h1 = hash_prompt(prompt, "model-a")
        h2 = hash_prompt(prompt, "model-b")
        assert h1 != h2

    def test_hash_prompt_no_model(self):
        prompt = "a cute cat"
        h1 = hash_prompt(prompt)
        h2 = hash_prompt(prompt)
        assert h1 == h2

    def test_handles_json_array(self):
        # Verifies that if the LLM returns a JSON array, it is preserved
        # Given
        input_text = ''
        # When
        result = clean_grammar_output(input_text)
        # Then
        assert result == '[1, 2, 3]'

    def test_handles_json_object_with_extra_content(self):
        # Verifies that content after the JSON object is discarded
        # Given
        input_text = '{"a": 1} extra'
        # When
        result = clean_grammar_output(input_text)
        # Then
        assert result == '{"a": 1}'

    def test_handles_json_in_text_array(self):
        # Verifies extraction of JSON array from within text
        # Given
        input_text = 'Here is an array: [1, 2, 3] and more'
        # When
        result = clean_grammar_output(input_text)
        # Then
        assert result == '[1, 2, 3]'
    def test_handles_json_array(self):
        # Verifies that if the LLM returns a JSON array, it is preserved
        # Given
        input_text = ''
        # When
        result = clean_grammar_output(input_text)
        # Then
        assert result == '[1, 2, 3]'

    def test_handles_json_object_with_extra_content(self):
        # Verifies that content after the JSON object is discarded
        # Given
        input_text = '{"a": 1} extra'
        # When
        result = clean_grammar_output(input_text)
        # Then
        assert result == '{"a": 1}'

    def test_handles_json_in_text_array(self):
        # Verifies extraction of JSON array from within text
        # Given
        input_text = 'Here is an array: [1, 2, 3] and more'
        # When
        result = clean_grammar_output(input_text)
        # Then
        assert result == '[1, 2, 3]'

    def test_handles_json_array(self):
        # Verifies that if the LLM returns a JSON array, it is preserved
        # Given
        input_text = '```json\n[1, 2, 3]\n```'
        # When
        result = clean_grammar_output(input_text)
        # Then
        assert result == '[1, 2, 3]'

    def test_handles_json_object_with_extra_content(self):
        # Verifies that content after the JSON object is discarded
        # Given
        input_text = '{"a": 1} extra'
        # When
        result = clean_grammar_output(input_text)
        # Then
        assert result == '{"a": 1}'

    def test_handles_json_in_text_array(self):
        # Verifies extraction of JSON array from within text
        # Given
        input_text = 'Here is an array: [1, 2, 3] and more'
        # When
        result = clean_grammar_output(input_text)
        # Then
        assert result == '[1, 2, 3]'
