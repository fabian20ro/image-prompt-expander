"""Tests for grammar_generator module."""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import patch, MagicMock
from grammar_generator import clean_grammar_output, get_system_prompt, hash_prompt, get_cached_grammar

class TestCleanGrammarOutput(unittest.TestCase):
    """Tests for the clean_grammar_output function."""

    def test_removes_think_tags_single_line(self):
        input_text = '<think>some thinking</think>{"origin": "#prompt#"}'
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#"}'

    def test_removes_think_tags_multiline(self):
        input_text = '''<think>
This is some
multi-line thinking
content here
</think>{"origin": "#prompt#"}'''
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#"}'

    def test_removes_think_tags_with_nested_json(self):
        input_text = '<think>Let me think about {"key": "value"}</think>{"origin": "#test#"}'
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#test#"}'

    def test_removes_json_code_block_with_newline(self):
        input_text = '''```json
{"origin": "#prompt#", "prompt": ["test"]}
```'''
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#", "prompt": ["test"]}'

    def test_removes_json_code_block_without_newline(self):
        input_text = '```json{"origin": "#prompt#"}```'
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#"}'

    def test_removes_multiple_think_blocks(self):
        input_text = '<think>first</think>text<think>second</think>{"origin": "#test#"}'
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#test#"}'

    def test_handles_no_tags(self):
        input_text = '{"origin": "#test#"}'
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#test#"}'

    def test_removes_tracery_code_block(self):
        input_text = '''```tracery
{"origin": "#prompt#"}
```'''
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#"}'

    def test_removes_plain_code_block(self):
        input_text = '''```
{"origin": "#prompt#"}
```'''
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#"}'

    def test_removes_code_block_with_extra_whitespace(self: unittest.TestCase):
        input_text = '''```json
{"origin": "#prompt#", "prompt": ["test"]}
```'''
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#", "prompt": ["test"]}'

    def test_removes_think_and_code_block(self):
        input_text = '''<think>
Let me create a grammar...
</think>

```json
{"origin": "#prompt#", "prompt": ["a #subject#"]}
```'''
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#", "prompt": ["a #subject#"]}'

    def test_handles_plain_json(self):
        input_text = '{"origin": "#prompt#"}'
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#"}'

    def test_extracts_json_from_surrounding_text(self):
        input_text = 'Here is the grammar: {"origin": "#prompt#"} Hope this helps!'
        result = clean_grammar_output(input_text)
        assert result == '{"origin": "#prompt#"}'

    def test_handles_complex_nested_json(self):
        input_text = '''```json
{
    "origin": "#prompt#",
    "prompt": ["#subject# in #setting#"],
    "subject": ["dragon", "phoenix"],
    "setting": ["mountains", "forest"]
}
```'''
        result = clean_grammar_output(input_text)
        import json
        parsed = json.loads(result)
        assert parsed["origin"] == "#prompt#"
        assert "dragon" in parsed["subject"]
        assert "mountains" in parsed["setting"]


class TestGetSystemPrompt(unittest.TestCase):
    """Tests for the get_system_prompt function."""

    @patch("grammar_generator.Path")
    def test_get_system_prompt_with_model_normalization(self, mock_path_class):
        # Given
        mock_templates_dir = MagicMock(spec=Path)
        mock_path_class.return_value = mock_templates_dir
        mock_path_class.return_value.__truediv__.return_value = mock_templates_dir
        
        mock_specific_file = MagicMock(spec=Path)
        mock_specific_file.exists.return_value = True
        mock_specific_file.read_text.return_value = "model specific prompt"
        mock_templates_dir.__truediv__.return_value = mock_specific_file
        
        model = "flux2-klein-4b"

        # When
        result = get_system_prompt(model=model, templates_dir=mock_templates_dir)

        # Then
        assert result == "model specific prompt"

    @patch("grammar_generator.Path")
    def test_get_system_prompt_fallback(self, mock_path_class):
        # Given
        mock_templates_dir = MagicMock(spec=Path)
        mock_path_class.return_value = mock_templates_dir
        mock_path_class.return_value.__truediv__.return_value = mock_templates_dir
        
        mock_generic_file = MagicMock(spec=Path)
        mock_generic_file.exists.return_value = False
        mock_generic_file.read_text.return_value = "generic prompt"
        mock_templates_dir.__truediv__.return_value = mock_generic_file
        
        model = None

        # When
        result = get_system_prompt(model=model, templates_dir=mock_templates_dir)

        # Then
        assert result == "generic prompt"

    @patch("grammar_generator.Path")
    def test_get_system_prompt_custom_templates_dir(self, mock_path_class):
        # Given
        custom_dir = MagicMock(spec=Path)
        mock_path_class.return_value = custom_dir
        mock_path_class.return_value.__truediv__.return_value = custom_dir
        
        mock_generic_file = MagicMock(spec=Path)
        mock_generic_file.exists.return_value = False
        mock_generic_file.read_text.return_value = "custom prompt"
        custom_dir.__truediv__.return_value = mock_generic_file
        
        model = None

        # When
        result = get_system_prompt(model=model, templates_dir=custom_dir)

        # Then
        assert result == "custom prompt"


class TestHashPrompt(unittest.TestCase):
    """Tests for the hash_prompt function."""

    def test_hash_prompt_no_model(self):
        prompt = "a sunset in the mountains"
        result = hash_prompt(prompt)
        assert len(result) == 12
        assert isinstance(result, str)

    def test_hash_prompt_with_model(self):
        prompt = "a sunset in the mountains"
        model = "flux2-klein"
        result_no_model = hash_prompt(prompt)
        result_with_model = hash_prompt(prompt, model=model)
        assert result_no_model != result_with_model
        assert len(result_with_model) == 12


class TestCache(unittest.TestCase):
    """Tests for the caching mechanisms."""

    @patch("grammar_generator.CACHE_DIR.mkdir")
    @patch("grammar_generator.Path.exists")
    @patch("grammar_generator.Path.read_text")
    def test_get_cached_grammar_exists(self, mock_read, mock_exists, mock_mkdir):
        mock_exists.return_value = True
        mock_read.return_value = '{"origin": "#test#"}'
        prompt_hash = "abcdef123456"
        result = get_cached_grammar(prompt_hash)
        assert result == '{"origin": "#test#"}'

    @patch("grammar_generator.CACHE_DIR.mkdir")
    @patch("grammar_generator.Path.exists")
    def test_get_cached_grammar_not_exists(self, mock_exists, mock_mkdir):
        mock_exists.return_value = False
        prompt_hash = "abcdef123456"
        result = get_cached_grammar(prompt_hash)
        assert result is None

    @patch("grammar_generator.CACHE_DIR.mkdir")
    @patch("grammar_generator.Path.write_text")
    @patch("grammar_generator.datetime")
    def test_cache_grammar(self, mock_datetime, mock_write, mock_mkdir):
        prompt_hash = "abcdef123456"
        grammar = '{"origin": "#prompt#"}'
        raw_response = '<think>...</think>{"origin": "#prompt#"}'
        user_prompt = "a sunset"
        model = "flux2-klein"
        mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"

        result = cache_grammar(prompt_hash, grammar, raw_response, user_prompt, model)

        assert isinstance(result, Path)
        assert mock_write.call_count == 3 
