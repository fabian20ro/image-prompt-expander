import sys
from pathlib import Path

# Add src and project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.grammar_generator import clean_grammar_output
print(f"Result: {clean_grammar_output('{\"a\": 1} extra')}")
