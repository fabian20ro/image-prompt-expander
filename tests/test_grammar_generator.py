import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import patch, MagicMock
from grammar_generator import clean_grammar_output, get_system_prompt, hash_prompt, get_cached_grammar, cache_grammar, get_cached_raw_response

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
