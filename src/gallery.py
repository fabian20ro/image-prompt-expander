"""HTML gallery generation for image prompt outputs."""

import html
import re
from pathlib import Path


def create_gallery(
    output_dir: Path,
    prefix: str,
    prompts: list[str],
    images_per_prompt: int,
    grammar: str | None = None,
    raw_response_file: str | None = None,
) -> Path:
    """Create initial gallery with placeholders for all expected images.

    Args:
        output_dir: Directory where gallery.html will be created
        prefix: Prefix used for image filenames
        prompts: List of prompt texts
        images_per_prompt: Number of images generated per prompt
        grammar: Optional Tracery grammar JSON to display
        raw_response_file: Optional filename for raw LLM response link

    Returns:
        Path to the created gallery.html file
    """
    gallery_path = output_dir / f"{prefix}_gallery.html"
    total_images = len(prompts) * images_per_prompt

    # Build cards for all expected images
    cards_html = []
    for prompt_idx, prompt_text in enumerate(prompts):
        for image_idx in range(images_per_prompt):
            image_filename = f"{prefix}_{prompt_idx}_{image_idx}.png"
            escaped_prompt = html.escape(prompt_text)

            # Check if image already exists (for resume scenarios)
            image_path = output_dir / image_filename
            if image_path.exists():
                card = f'''    <div class="card" data-image="{image_filename}">
      <a href="{image_filename}" target="_blank">
        <img src="{image_filename}" loading="lazy">
      </a>
      <div class="prompt">{escaped_prompt}</div>
    </div>'''
            else:
                card = f'''    <div class="card" data-image="{image_filename}">
      <div class="placeholder">Pending...</div>
      <div class="prompt">{escaped_prompt}</div>
    </div>'''
            cards_html.append(card)

    # Count existing images
    completed = sum(1 for p in prompts for i in range(images_per_prompt)
                   if (output_dir / f"{prefix}_{prompts.index(p)}_{i}.png").exists())

    gallery_html = _build_gallery_html(prefix, cards_html, completed, total_images, grammar, raw_response_file)
    gallery_path.write_text(gallery_html)

    return gallery_path


def update_gallery(
    gallery_path: Path,
    image_path: Path,
    prompt: str,
    completed: int,
    total: int,
) -> None:
    """Update gallery to show newly generated image.

    Args:
        gallery_path: Path to the gallery.html file
        image_path: Path to the newly generated image
        prompt: The prompt text for this image
        completed: Number of images completed so far
        total: Total number of images to generate
    """
    if not gallery_path.exists():
        return

    html_content = gallery_path.read_text()
    image_filename = image_path.name

    # Find the card for this image and replace placeholder with actual image
    # Pattern matches the placeholder div for this specific image
    placeholder_pattern = (
        rf'(<div class="card" data-image="{re.escape(image_filename)}">)\s*'
        rf'<div class="placeholder">Pending\.\.\.</div>'
    )

    replacement = (
        rf'\1\n      <a href="{image_filename}" target="_blank">\n'
        rf'        <img src="{image_filename}" loading="lazy">\n'
        rf'      </a>'
    )

    html_content = re.sub(placeholder_pattern, replacement, html_content)

    # Update the status count
    status_pattern = r'<p class="status">Generated: \d+ / \d+ images</p>'
    status_replacement = f'<p class="status">Generated: {completed} / {total} images</p>'
    html_content = re.sub(status_pattern, status_replacement, html_content)

    gallery_path.write_text(html_content)


def generate_gallery_for_directory(prompts_dir: Path) -> Path:
    """Generate a gallery for an existing prompts directory.

    Args:
        prompts_dir: Directory containing prompt files and images

    Returns:
        Path to the created gallery.html file

    Raises:
        ValueError: If no metadata file found or no prompts found
    """
    import json

    # Find metadata file
    meta_files = list(prompts_dir.glob("*_metadata.json"))
    if not meta_files:
        raise ValueError(f"No metadata file found in {prompts_dir}")

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")
    images_per_prompt = metadata.get("image_generation", {}).get("images_per_prompt", 1)

    # Load prompts
    prompt_files = sorted(prompts_dir.glob(f"{prefix}_*.txt"))
    prompt_files = [f for f in prompt_files if f.stem.count('_') == 1]

    if not prompt_files:
        raise ValueError(f"No prompt files found in {prompts_dir}")

    prompts = [f.read_text() for f in prompt_files]

    # Load grammar if available
    grammar = None
    grammar_file = prompts_dir / f"{prefix}_grammar.json"
    if grammar_file.exists():
        grammar = grammar_file.read_text()

    # Check for raw response file
    raw_response_file = None
    raw_file = prompts_dir / f"{prefix}_raw_response.txt"
    if raw_file.exists():
        raw_response_file = f"{prefix}_raw_response.txt"

    # Create gallery
    gallery_path = create_gallery(prompts_dir, prefix, prompts, images_per_prompt, grammar, raw_response_file)

    return gallery_path


def _build_gallery_html(
    prefix: str,
    cards_html: list[str],
    completed: int,
    total: int,
    grammar: str | None = None,
    raw_response_file: str | None = None,
) -> str:
    """Build the complete gallery HTML document."""
    cards_joined = "\n".join(cards_html)

    # Build header section with optional grammar display and raw response link
    header_section = ""
    if grammar or raw_response_file:
        header_parts = []
        if raw_response_file:
            header_parts.append(f'<a href="{raw_response_file}" class="raw-link">View Raw LLM Response</a>')
        header_section = f'''
  <div class="header-links">
    {" ".join(header_parts)}
  </div>'''

    # Build collapsible grammar section
    grammar_section = ""
    if grammar:
        escaped_grammar = html.escape(grammar)
        grammar_section = f'''
  <details class="grammar-section">
    <summary>Tracery Grammar</summary>
    <pre>{escaped_grammar}</pre>
  </details>'''

    return f'''<!DOCTYPE html>
<html>
<head>
  <title>Gallery: {prefix}</title>
  <meta charset="utf-8">
  <style>
    body {{ font-family: system-ui; padding: 20px; background: #1a1a1a; color: #fff; }}
    h1 {{ margin-bottom: 10px; }}
    .header-links {{ margin-bottom: 15px; }}
    .header-links a {{ color: #6af; text-decoration: none; margin-right: 20px; }}
    .header-links a:hover {{ text-decoration: underline; }}
    .grammar-section {{ margin-bottom: 20px; background: #2a2a2a; border-radius: 8px; }}
    .grammar-section summary {{ padding: 12px 16px; cursor: pointer; color: #888; font-size: 14px; }}
    .grammar-section summary:hover {{ color: #aaa; }}
    .grammar-section pre {{ margin: 0; padding: 16px; background: #222; font-size: 12px; color: #8f8; overflow-x: auto; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }}
    .status {{ color: #888; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: #2a2a2a; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; aspect-ratio: 3/4; object-fit: cover; cursor: pointer; }}
    .card .placeholder {{ width: 100%; aspect-ratio: 3/4; background: #333; display: flex; align-items: center; justify-content: center; color: #666; }}
    .card .prompt {{ padding: 12px; font-size: 13px; color: #aaa; max-height: 150px; overflow-y: auto; }}
  </style>
</head>
<body>
  <h1>Gallery: {prefix}</h1>{header_section}{grammar_section}
  <p class="status">Generated: {completed} / {total} images</p>
  <div class="grid">
{cards_joined}
  </div>
</body>
</html>
'''
