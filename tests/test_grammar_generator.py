"""Tests for grammar_generator module."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grammar_generator import clean_grammar_output


class TestCleanGrammarOutput:
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
</think>
{"origin": "#prompt#"}'''

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
{"origin": "#prompt#"}
```'''

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert result == '{"origin": "#prompt#"}'

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
