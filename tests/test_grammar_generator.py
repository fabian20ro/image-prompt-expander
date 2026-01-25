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

    # Tests for smart quote normalization

    def test_replaces_smart_double_quotes(self):
        # Verifies curly double quotes are replaced with straight quotes
        # Given - use Unicode escapes to avoid Python parsing issues
        # U+201C (left double quote) and U+201D (right double quote)
        input_text = '{\u201corigin\u201d: \u201c#prompt#\u201d}'

        # When
        result = clean_grammar_output(input_text)

        # Then - smart quotes should be converted to straight quotes
        assert '\u201c' not in result
        assert '\u201d' not in result
        import json
        parsed = json.loads(result)
        assert parsed["origin"] == "#prompt#"

    def test_replaces_smart_single_quotes(self):
        # Verifies curly single quotes are replaced with straight quotes
        # Given - U+2018 (left single) and U+2019 (right single)
        input_text = '{"origin": "#prompt#", "test": "it\u2019s working"}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        assert '\u2018' not in result
        assert '\u2019' not in result
        assert "it's working" in result

    def test_replaces_mixed_smart_quotes(self):
        # Verifies both left and right curly quotes are normalized
        # Given - smart quotes in JSON syntax (keys and string delimiters)
        # This matches real LLM output where smart quotes replace JSON syntax quotes
        input_text = '{\u201cprompt\u201d: [\u201ca dragon in the sky\u201d], \u201ctest\u201d: \u201cit\u2019s great\u201d}'

        # When
        result = clean_grammar_output(input_text)

        # Then
        import json
        parsed = json.loads(result)
        assert parsed["prompt"] == ["a dragon in the sky"]
        assert parsed["test"] == "it's great"

    def test_smart_quotes_in_code_block(self):
        # Verifies smart quotes inside code blocks are normalized
        # Given
        input_text = '```json\n{\u201corigin\u201d: \u201c#prompt#\u201d}\n```'

        # When
        result = clean_grammar_output(input_text)

        # Then
        import json
        parsed = json.loads(result)
        assert parsed["origin"] == "#prompt#"

    def test_smart_quotes_with_think_tags(self):
        # Verifies full pipeline: think removal + code block + quote normalization
        # Given
        input_text = '<think>\nLet me create a grammar with \u201csmart quotes\u201d...\n</think>\n\n```json\n{\u201corigin\u201d: \u201c#prompt#\u201d, \u201cdesc\u201d: \u201ca \u2018fancy\u2019 thing\u201d}\n```'

        # When
        result = clean_grammar_output(input_text)

        # Then
        import json
        parsed = json.loads(result)
        assert parsed["origin"] == "#prompt#"
        assert parsed["desc"] == "a 'fancy' thing"
