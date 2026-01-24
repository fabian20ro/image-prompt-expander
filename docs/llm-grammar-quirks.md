# LLM Grammar Generation Quirks

This document tracks model-specific quirks when generating Tracery grammars via LM Studio.

## GLM-4 (glm-4.6v)

### Thinking Blocks
GLM-4 outputs `<think>...</think>` blocks containing reasoning before the actual response. These must be stripped from output.

**Solution**: `grammar_generator.py` uses regex to remove `<think>.*?</think>` blocks.

### Markdown Code Blocks
Often wraps JSON output in markdown code blocks (` ```json ... ``` `).

**Solution**: Extract content between code block markers.

### Invalid Tracery Syntax
May generate dictionary-based conditional lookups that Tracery doesn't support:
```json
{
  "time_lighting": {
    "dawn": ["soft light"],
    "night": ["moonlight"]
  }
}
```
This syntax is NOT valid Tracery - rule values must be arrays of strings.

**Solution**: Updated system prompt to emphasize that ALL values must be arrays, and to use "scene packages" for coherent variations instead of conditional mappings.

## General Issues (All Models)

### Missing Rule Definitions
Models sometimes reference rules that don't exist (e.g., `#weather_details#` without defining it).

**Symptoms**: Tracery throws `IndexError: Cannot choose from an empty sequence`

**Solution**: System prompt explicitly states every `#reference#` must have a corresponding rule.

### Nested Arrays
Some models output nested arrays instead of flat string arrays:
```json
"options": [["option1"], ["option2"]]  // WRONG
"options": ["option1", "option2"]       // CORRECT
```

**Solution**: Could add post-processing to flatten, or emphasize in prompt.

### JSON Syntax Errors
Common issues:
- Trailing commas after last array element
- Single quotes instead of double quotes
- Unescaped special characters

**Solution**: `clean_grammar_output()` attempts to extract valid JSON from malformed output.

## Model Recommendations

| Model | Quality | Speed | Notes |
|-------|---------|-------|-------|
| GLM-4.6v | Good | Fast | Needs thinking block removal |
| Qwen 2.5 | Good | Medium | Follows instructions well |
| Llama 3.x | Variable | Fast | May need more examples |
| Mistral | Good | Fast | Generally reliable |

## Testing New Models

When testing a new model:

1. Run with `--dry-run` first to inspect raw grammar output
2. Check for:
   - Thinking/reasoning blocks
   - Markdown formatting
   - Valid JSON structure
   - All values are string arrays
   - All referenced rules are defined
3. Update `clean_grammar_output()` if new patterns emerge
4. Document quirks in this file
