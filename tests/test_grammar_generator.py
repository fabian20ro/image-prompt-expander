import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import patch, MagicMock
from grammar_generator import (
    clean_grammar_output,
    get_system_prompt,
    hash_prompt,
    get_cached_grammar,
    cache_grammar,
    get_cached_raw_response,
    generate_grammar
)

class TestCache(unittest.TestCase):
    """Tests for the caching mechanisms."""

    @patch("grammar_generator.CACHE_DIR")
    def test_get_cached_grammar_exists(self, mock_cache_dir):
        # Setup mock for CACHE_DIR / filename
        mock_file = MagicMock(spec=Path)
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = '{"origin": "#test#"}'
        mock_cache_dir.__truediv__.return_value = mock_file

        prompt_hash = "abcdef123456"
        result = get_cached_grammar(prompt_hash)
        assert result == '{"origin": "#test#"}'

    @patch("grammar_generator.CACHE_DIR")
    def test_get_cached_grammar_not_exists(self, mock_cache_dir):
        # Setup mock for CACHE_DIR / filename
        mock_file = MagicMock(spec=Path)
        mock_file.exists.return_value = False
        mock_cache_dir.__truediv__.return_value = mock_file

        prompt_hash = "abcdef123456"
        result = get_cached_grammar(prompt_hash)
        assert result is None

    @patch("grammar_generator.CACHE_DIR")
    @patch("grammar_generator.Path.write_text")
    @patch("grammar_generator.datetime")
    def test_cache_grammar(self, mock_datetime, mock_write, mock_cache_dir):
        prompt_hash = "abcdef123456"
        grammar = '{"origin": "#prompt#"}'
        raw_response = '<think>...</think>{"origin": "#prompt#"}'
        user_prompt = "a sunset"
        model = "flux2-klein"
        mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"

        result = cache_grammar(prompt_hash, grammar, raw_response, user_prompt, model)

        assert isinstance(result, tuple)
        assert result[0] == grammar
        assert result[1] is False
        assert result[2] == raw_response

class TestGrammarTools(unittest.TestCase):
    """Tests for grammar utility functions."""

    def test_clean_grammar_output_empty(self):
        self.assertEqual(clean_grammar_output(""), "")
        self.assertEqual(clean_grammar_output("   "), "")

    def test_clean_grammar_output_with_thinking_and_code_blocks(self):
        input_str = '<think>some thought</think>```json\n{"key": "value"}\n```'
        expected = '{"key": "value"}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_with_smart_quotes(self):
        # Test with smart double quotes to ensure valid JSON output
        input_str = '{\u201ckey\u201d: \u201cvalue\u201d}'
        expected = '{"key": "value"}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_extracts_json(self):
        input_str = 'Some noise here {"a": 1} and some noise there'
        expected = '{"a": 1}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_hash_prompt(self):
        prompt = "a beautiful sunset"
        model = "flux2-klein"
        h1 = hash_prompt(prompt, model)
        h2 = hash_prompt(prompt, model)
        h3 = hash_prompt(prompt, "other-model")

        self.assertEqual(len(h1), 12)
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    @patch("grammar_generator.Path.exists")
    @patch("grammar_generator.Path.read_text")
    @patch("grammar_generator.paths")
    def test_get_system_prompt(self, mock_paths, mock_read_text, mock_exists):
        mock_paths.templates_dir = Path("/tmp/templates")
        mock_exists.return_value = True
        mock_read_text.return_value = "system prompt content"

        result = get_system_prompt(model="flux2-klein")
        self.assertEqual(result, "system prompt content")

    def test_get_system_prompt_normalization(self):
        from grammar_generator import get_system_prompt
        from pathlib import Path
        from unittest.mock import patch
        with patch("grammar_generator.Path.exists") as mock_exists, \
             patch("grammar_generator.Path.read_text") as mock_read_text, \
             patch("grammar_generator.paths") as mock_paths:
            mock_paths.templates_dir = Path("/tmp/templates")
            mock_exists.side_effect = lambda *args, **kwargs: str(args[0] if args else None) == str(Path("/tmp/templates/system_prompt_flux2-klein.txt"))
            mock_read_text.return_value = "model_specific_content"
            self.assertEqual(get_system_prompt(model="flux2-klein-4b"), "model_specific_content")

class TestGenerateGrammar(unittest.TestCase):
    @patch("grammar_generator.OpenAI")
    @patch("grammar_generator.get_system_prompt")
    @patch("grammar_generator.cache_grammar")
    @patch("grammar_generator.get_cached_grammar")
    def test_generate_grammar_invalid_json(self, mock_get_cached, mock_cache, mock_get_prompt, mock_openai):
        # Setup
        mock_get_cached.return_value = None
        mock_get_prompt.return_value = '{"test": "prompt"}'

        # Mock OpenAI response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Not JSON"
        mock_client.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            generate_grammar("test prompt")

        self.assertIn("LLM returned invalid JSON grammar after cleaning", str(context.exception))

if __name__ == "__main__":
    unittest.main()
