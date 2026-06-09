import json
import re

def clean_grammar_output_new(grammar: str) -> str:
    # Remove <think>...</think> blocks (including multiline)
    grammar = re.sub(r'<think>.*?</think>', '', grammar, flags=re.DOTALL)

    # Remove markdown code block markers and extract content
    code_block_match = re.search(r'```(?:json|tracery)?\s*(.*?)```', grammar, flags=re.DOTALL)
    if code_block_match:
        grammar = code_block_match.group(1)

    grammar = grammar.strip()

    # Normalize smart/curly quotes
    quote_replacements = {
        '\u201c': '"',
        '\u201d': '"',
        '\u2018': "'",
        '\u2019': "'",
    }
    for smart, straight in quote_replacements.items():
        grammar = grammar.replace(smart, straight)

    # Try to find the first valid JSON object or array in the text
    match = re.search(r'[\{\[]', grammar)
    if match:
        start_idx = match.start()
        try:
            decoder = json.JSONDecoder()
            _, end_pos = decoder.raw_decode(grammar[start_idx:])
            grammar = grammar[start_idx:start_idx + end_pos]
        except (json.JSONDecodeError, ValueError):
            pass

    return grammar

# Tests
print(f"Object test: {clean_grammar_output_new('{\"a\": 1} extra')}")
print(f"Array test: {clean_grammar_output_new('[1, 2, 3] extra')}")
print(f"Code block array: {clean_grammar_output_new('```json\n[1, 2, 3]\n```')}")
print(f"JSON in text array: {clean_grammar_output_new('Here is an array: [1, 2, 3] and more')}")
