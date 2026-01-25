"""LM Studio integration for generating Tracery grammars."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI


# Default LM Studio endpoint
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"

# Cache directory for generated grammars
CACHE_DIR = Path(__file__).parent.parent / "generated" / "grammars"


def get_system_prompt(model: str | None = None) -> str:
    """
    Load the system prompt from the templates directory.

    Args:
        model: Model name to select appropriate system prompt.
               Supports z-image-turbo (camera-first) and flux2-klein (prose-based).
               Falls back to generic system_prompt.txt if model-specific not found.

    Returns:
        The system prompt content
    """
    templates_dir = Path(__file__).parent.parent / "templates"

    # Determine the model-specific prompt filename
    if model:
        # Normalize model name for file lookup (e.g., flux2-klein-4b -> flux2-klein)
        if model.startswith("flux2-klein"):
            prompt_name = "flux2-klein"
        else:
            prompt_name = model

        model_specific_path = templates_dir / f"system_prompt_{prompt_name}.txt"
        if model_specific_path.exists():
            return model_specific_path.read_text()

    # Fallback to generic system prompt
    generic_path = templates_dir / "system_prompt.txt"
    return generic_path.read_text()


def hash_prompt(user_prompt: str) -> str:
    """Generate a short hash for caching grammars by prompt."""
    return hashlib.sha256(user_prompt.encode()).hexdigest()[:12]


def get_cached_grammar(prompt_hash: str) -> str | None:
    """
    Check if a grammar exists in cache for the given prompt hash.

    Args:
        prompt_hash: The hash of the user prompt

    Returns:
        The cached grammar content, or None if not cached
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{prompt_hash}.tracery.json"

    if cache_file.exists():
        return cache_file.read_text()
    return None


def cache_grammar(prompt_hash: str, grammar: str, raw_response: str, user_prompt: str) -> Path:
    """
    Save a grammar to the cache.

    Args:
        prompt_hash: The hash of the user prompt
        grammar: The cleaned grammar content
        raw_response: The raw LLM response (including thinking blocks)
        user_prompt: The original user prompt (for metadata)

    Returns:
        Path to the cached grammar file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Save the cleaned grammar
    cache_file = CACHE_DIR / f"{prompt_hash}.tracery.json"
    cache_file.write_text(grammar)

    # Save the raw LLM response (with thinking blocks etc.)
    raw_file = CACHE_DIR / f"{prompt_hash}.raw.txt"
    raw_file.write_text(raw_response)

    # Save metadata
    metadata_file = CACHE_DIR / f"{prompt_hash}.metaprompt.json"
    metadata = {
        "user_prompt": user_prompt,
        "created_at": datetime.now().isoformat(),
        "hash": prompt_hash
    }
    metadata_file.write_text(json.dumps(metadata, indent=2))

    return cache_file


def generate_grammar(
    user_prompt: str,
    base_url: str = LM_STUDIO_BASE_URL,
    api_key: str = LM_STUDIO_API_KEY,
    use_cache: bool = True,
    temperature: float = 0.7,
    model: str | None = None,
) -> tuple[str, bool]:
    """
    Generate a Dada Engine grammar for the given prompt using LM Studio.

    Args:
        user_prompt: The user's image description
        base_url: LM Studio API base URL
        api_key: API key (LM Studio doesn't require a real key)
        use_cache: Whether to check/use cached grammars
        temperature: LLM temperature for generation
        model: Image model name to select appropriate system prompt

    Returns:
        Tuple of (grammar content, was_cached)
    """
    prompt_hash = hash_prompt(user_prompt)

    # Check cache first
    if use_cache:
        cached = get_cached_grammar(prompt_hash)
        if cached:
            return cached, True

    # Generate new grammar via LM Studio
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    system_prompt = get_system_prompt(model)

    response = client.chat.completions.create(
        model="local-model",  # LM Studio uses whatever model is loaded
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )

    raw_response = response.choices[0].message.content

    # Clean up the grammar (remove any markdown code blocks if present)
    grammar = clean_grammar_output(raw_response)

    # Cache the result
    if use_cache:
        cache_grammar(prompt_hash, grammar, raw_response, user_prompt)

    return grammar, False


def clean_grammar_output(grammar: str) -> str:
    """
    Clean LLM output by removing thinking blocks, markdown code blocks, and extra whitespace.

    Args:
        grammar: Raw LLM output

    Returns:
        Cleaned JSON grammar content
    """
    # Remove <think>...</think> blocks (including multiline)
    grammar = re.sub(r'<think>.*?</think>', '', grammar, flags=re.DOTALL)

    # Remove markdown code block markers and extract content
    # Handle ```json, ```tracery, or plain ``` markers
    code_block_match = re.search(r'```(?:json|tracery)?\s*\n(.*?)```', grammar, flags=re.DOTALL)
    if code_block_match:
        grammar = code_block_match.group(1)

    grammar = grammar.strip()

    # Validate it's valid JSON
    try:
        json.loads(grammar)
    except json.JSONDecodeError:
        # Try to extract JSON object from the text
        json_match = re.search(r'\{[\s\S]*\}', grammar)
        if json_match:
            grammar = json_match.group(0)

    return grammar
