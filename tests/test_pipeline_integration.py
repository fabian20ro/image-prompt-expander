
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from pipeline import PipelineExecutor, PipelineResult

@pytest.fixture
def mock_grammar_file(tmp_path):
    grammar_file = tmp_path / "test.tracery.json"
    grammar_file.write_text('{"root": {"_repeat": ["A"]}}')
    meta_file = tmp_path / "test.metaprompt.json"
    meta_file.write_text('{"user_prompt": "a dragon"}')
    return grammar_file

def test_run_from_grammar_success(mock_grammar_file, tmp_path):
    executor = PipelineExecutor()
    
    with patch("pipeline.generate_grammar", return_value=("{\"root\": {\"_repeat\": [\"A\"]}}", False, None)):
        with patch("pipeline.PipelineExecutor.run_from_grammar_text", return_value=PipelineResult(success=True, run_id="test_id")):
            result = executor.run_from_grammar(mock_grammar_file)
            assert result.success is True
            assert result.run_id == "test_id"

def test_run_from_grammar_no_meta(mock_grammar_file):
    executor = PipelineExecutor()
    
    with patch("pipeline.generate_grammar", return_value=("{\"root\": {\"_repeat\": [\"A\"]}}", False, None)):
        with patch("pipeline.PipelineExecutor.run_from_grammar_text", return_value=PipelineResult(success=True, run_id="test_id")):
            result = executor.run_from_grammar(mock_grammar_file)
            assert result.success is True
            assert result.run_id == "test_id"
