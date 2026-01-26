"""HTML gallery generation for image prompt outputs."""

import html
import re
from pathlib import Path

from filelock import FileLock

from html_components import (
    LogPanel,
    ProgressBar,
    Buttons,
    NavHeader,
    GalleryStyles,
    SSEClient,
)


def create_gallery(
    output_dir: Path,
    prefix: str,
    prompts: list[str],
    images_per_prompt: int,
    grammar: str | None = None,
    raw_response_file: str | None = None,
    interactive: bool = False,
    run_id: str | None = None,
) -> Path:
    """Create initial gallery with placeholders for all expected images.

    Args:
        output_dir: Directory where gallery.html will be created
        prefix: Prefix used for image filenames
        prompts: List of prompt texts
        images_per_prompt: Number of images generated per prompt
        grammar: Optional Tracery grammar JSON to display
        raw_response_file: Optional filename for raw LLM response link
        interactive: If True, include editable grammar and action buttons
        run_id: Run ID for API calls (required if interactive)

    Returns:
        Path to the created gallery.html file
    """
    gallery_path = output_dir / f"{prefix}_gallery.html"
    total_images = len(prompts) * max(images_per_prompt, 1)

    # Build cards for all prompts
    cards_html = []
    for prompt_idx, prompt_text in enumerate(prompts):
        escaped_prompt = html.escape(prompt_text)

        if images_per_prompt == 0:
            # No images expected - just show prompt with generate button
            card = _build_card_html(
                f"{prefix}_{prompt_idx}_0.png", escaped_prompt, prompt_idx, 0,
                exists=False, interactive=interactive, no_image_expected=True
            )
            cards_html.append(card)
        else:
            for image_idx in range(images_per_prompt):
                image_filename = f"{prefix}_{prompt_idx}_{image_idx}.png"

                # Check if image already exists (for resume scenarios)
                image_path = output_dir / image_filename
                if image_path.exists():
                    card = _build_card_html(
                        image_filename, escaped_prompt, prompt_idx, image_idx,
                        exists=True, interactive=interactive
                    )
                else:
                    card = _build_card_html(
                        image_filename, escaped_prompt, prompt_idx, image_idx,
                        exists=False, interactive=interactive
                    )
                cards_html.append(card)

    # Count existing images
    completed = sum(1 for p_idx, p in enumerate(prompts) for i in range(max(images_per_prompt, 1))
                   if (output_dir / f"{prefix}_{p_idx}_{i}.png").exists())

    gallery_html = _build_gallery_html(
        prefix, cards_html, completed, total_images, grammar, raw_response_file,
        interactive=interactive, run_id=run_id
    )
    gallery_path.write_text(gallery_html)

    return gallery_path


def _build_card_html(
    image_filename: str,
    escaped_prompt: str,
    prompt_idx: int,
    image_idx: int,
    exists: bool,
    interactive: bool = False,
    no_image_expected: bool = False,
) -> str:
    """Build HTML for a single image card."""
    action_buttons = ""
    if interactive:
        action_buttons = f'''
      <div class="card-actions">
        <button class="btn-small btn-primary" onclick="generateImage({prompt_idx}, {image_idx})">Generate</button>
        <button class="btn-small btn-secondary" onclick="enhanceImage({prompt_idx}, {image_idx})">Enhance</button>
      </div>'''

    if exists:
        return f'''    <div class="card" data-image="{image_filename}" data-prompt-idx="{prompt_idx}" data-image-idx="{image_idx}">
      <a href="{image_filename}" target="_blank">
        <img src="{image_filename}" loading="lazy">
      </a>
      <div class="prompt">{escaped_prompt}</div>{action_buttons}
    </div>'''
    elif no_image_expected:
        # Show prompt-only card with generate button
        return f'''    <div class="card prompt-only" data-image="{image_filename}" data-prompt-idx="{prompt_idx}" data-image-idx="{image_idx}">
      <div class="placeholder no-image">#{prompt_idx}</div>
      <div class="prompt">{escaped_prompt}</div>{action_buttons}
    </div>'''
    else:
        return f'''    <div class="card" data-image="{image_filename}" data-prompt-idx="{prompt_idx}" data-image-idx="{image_idx}">
      <div class="placeholder">Pending...</div>
      <div class="prompt">{escaped_prompt}</div>{action_buttons}
    </div>'''


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

    # Use file locking to prevent concurrent update corruption
    lock_path = gallery_path.with_suffix('.html.lock')
    with FileLock(lock_path, timeout=10):
        html_content = gallery_path.read_text()
        image_filename = image_path.name

        # Find the card for this image and replace placeholder with actual image
        # Pattern matches the placeholder div for this specific image
        placeholder_pattern = (
            rf'(<div class="card" data-image="{re.escape(image_filename)}"[^>]*>)\s*'
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


def generate_gallery_for_directory(prompts_dir: Path, interactive: bool = False) -> Path:
    """Generate a gallery for an existing prompts directory.

    Args:
        prompts_dir: Directory containing prompt files and images
        interactive: If True, include editable grammar and action buttons

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

    # Get run_id from directory name
    run_id = prompts_dir.name if interactive else None

    # Create gallery
    gallery_path = create_gallery(
        prompts_dir, prefix, prompts, images_per_prompt, grammar, raw_response_file,
        interactive=interactive, run_id=run_id
    )

    return gallery_path


def _build_nav_header() -> str:
    """Build navigation header with back link."""
    return NavHeader.html()


def _build_interactive_grammar_section(grammar: str, run_id: str) -> str:
    """Build the interactive grammar section with edit capabilities."""
    escaped_grammar = html.escape(grammar)
    return f'''
  <div class="grammar-section-interactive">
    <div class="grammar-header">
      <span class="grammar-title">Tracery Grammar</span>
      <div class="grammar-actions">
        <button id="btn-save-grammar" class="btn-small btn-primary">Save</button>
        <button id="btn-regenerate" class="btn-small btn-secondary">Regenerate Prompts</button>
      </div>
    </div>
    <textarea id="grammar-editor" class="grammar-editor">{escaped_grammar}</textarea>
  </div>
'''


def _build_interactive_action_bar(run_id: str) -> str:
    """Build the action bar with generate/enhance all buttons."""
    return f'''
  <div class="action-bar">
    <button id="btn-generate-all" class="btn-primary">Generate All Images</button>
    <button id="btn-enhance-all" class="btn-secondary">Enhance All</button>
    <button id="btn-archive" class="btn-secondary">Save to Archive</button>
    <div class="action-spacer"></div>
    <button id="btn-clear-queue" class="btn-secondary">Clear Queue</button>
    <button id="btn-kill" class="btn-danger">Kill Current</button>
  </div>
'''


def _build_interactive_progress_bar() -> str:
    """Build the fixed progress bar at bottom."""
    return ProgressBar.html()


def _build_log_panel() -> str:
    """Build the collapsible log panel HTML."""
    return LogPanel.html()


def _build_interactive_js(run_id: str) -> str:
    """Build JavaScript for interactive gallery features."""
    log_js = LogPanel.js()
    sse_js = SSEClient.js()

    return f'''
<script>
(function() {{
  const RUN_ID = "{run_id}";
  const grammarEditor = document.getElementById('grammar-editor');
  const btnSaveGrammar = document.getElementById('btn-save-grammar');
  const btnRegenerate = document.getElementById('btn-regenerate');
  const btnGenerateAll = document.getElementById('btn-generate-all');
  const btnEnhanceAll = document.getElementById('btn-enhance-all');
  const btnClearQueue = document.getElementById('btn-clear-queue');
  const btnKill = document.getElementById('btn-kill');
  const progressBar = document.getElementById('progress-bar');
  const progressMessage = document.getElementById('progress-message');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const logPanel = document.getElementById('log-panel');

  // Shared log panel functions
{log_js}

  // Shared SSE connection logic
{sse_js}

  function initSSE() {{
    const es = connectSSE();
    if (!es) return;

    es.addEventListener('status', (e) => {{
      const data = JSON.parse(e.data);
      if (data.current) {{
        progressBar.classList.remove('hidden');
        progressMessage.textContent = `Running: ${{data.current.type}}`;
        if (data.current.progress) {{
          const p = data.current.progress;
          const pct = p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
          progressFill.style.width = pct + '%';
          progressText.textContent = `${{p.current}}/${{p.total}}`;
          if (p.message) progressMessage.textContent = p.message;
        }}
      }}
    }});

    es.addEventListener('task_started', (e) => {{
      progressBar.classList.remove('hidden');
      const task = JSON.parse(e.data);
      progressMessage.textContent = `Running: ${{task.type}}`;
    }});

    es.addEventListener('task_progress', (e) => {{
      const data = JSON.parse(e.data);
      const pct = data.total > 0 ? Math.round((data.current / data.total) * 100) : 0;
      progressFill.style.width = pct + '%';
      progressText.textContent = `${{data.current}}/${{data.total}}`;
      if (data.message) progressMessage.textContent = data.message;
    }});

    es.addEventListener('task_completed', (e) => {{
      const data = JSON.parse(e.data);
      progressMessage.textContent = 'Completed';
      // Reload page if this was a regenerate_prompts task for this gallery
      if (data.result && data.result.task_type === 'regenerate_prompts' && data.result.run_id === RUN_ID) {{
        setTimeout(() => window.location.reload(), 500);
      }}
    }});

    es.addEventListener('task_failed', (e) => {{
      const data = JSON.parse(e.data);
      progressMessage.textContent = 'Failed: ' + data.error;
    }});

    es.addEventListener('task_cancelled', (e) => {{
      progressMessage.textContent = 'Cancelled';
    }});

    es.addEventListener('queue_updated', (e) => {{
      const data = JSON.parse(e.data);
      if (data.current) {{
        progressBar.classList.remove('hidden');
        progressMessage.textContent = `Running: ${{data.current.type}}`;
        if (data.current.progress) {{
          const p = data.current.progress;
          const pct = p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
          progressFill.style.width = pct + '%';
          progressText.textContent = `${{p.current}}/${{p.total}}`;
          if (p.message) progressMessage.textContent = p.message;
        }}
      }} else if (data.pending_count > 0) {{
        progressBar.classList.remove('hidden');
        progressMessage.textContent = `${{data.pending_count}} task(s) pending...`;
        progressFill.style.width = '0%';
        progressText.textContent = '';
      }} else {{
        // No current task and no pending - hide after brief delay
        setTimeout(() => {{
          progressBar.classList.add('hidden');
          progressFill.style.width = '0%';
        }}, 1500);
      }}
    }});

    es.addEventListener('image_ready', (e) => {{
      const data = JSON.parse(e.data);
      if (data.run_id === RUN_ID) {{
        updateImage(data.path);
      }}
    }});

    es.addEventListener('task_log', (e) => {{
      const data = JSON.parse(e.data);
      appendLog(data.timestamp, data.message);
      // Auto-open log panel when logs arrive
      if (logPanel && !logPanel.open) {{
        logPanel.open = true;
      }}
    }});
  }}

  function updateImage(filename) {{
    const card = document.querySelector(`[data-image="${{filename}}"]`);
    if (!card) return;

    const placeholder = card.querySelector('.placeholder');
    if (placeholder) {{
      const link = document.createElement('a');
      link.href = filename;
      link.target = '_blank';
      const img = document.createElement('img');
      img.src = filename + '?t=' + Date.now();
      img.loading = 'lazy';
      link.appendChild(img);
      placeholder.replaceWith(link);
    }} else {{
      const img = card.querySelector('img');
      if (img) img.src = filename + '?t=' + Date.now();
    }}
  }}

  // API helpers
  async function apiPost(url, body = {{}}) {{
    try {{
      const resp = await fetch(url, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body),
      }});
      if (!resp.ok) {{
        const err = await resp.json();
        throw new Error(err.detail || 'Request failed');
      }}
      return await resp.json();
    }} catch (err) {{
      alert('Error: ' + err.message);
      throw err;
    }}
  }}

  async function apiPut(url, body = {{}}) {{
    try {{
      const resp = await fetch(url, {{
        method: 'PUT',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body),
      }});
      if (!resp.ok) {{
        const err = await resp.json();
        throw new Error(err.detail || 'Request failed');
      }}
      return await resp.json();
    }} catch (err) {{
      alert('Error: ' + err.message);
      throw err;
    }}
  }}

  // Button handlers
  if (btnSaveGrammar) {{
    btnSaveGrammar.addEventListener('click', async () => {{
      await apiPut(`/api/gallery/${{RUN_ID}}/grammar`, {{
        grammar: grammarEditor.value
      }});
      alert('Grammar saved');
    }});
  }}

  if (btnRegenerate) {{
    btnRegenerate.addEventListener('click', async () => {{
      await apiPost(`/api/gallery/${{RUN_ID}}/regenerate`);
      progressBar.classList.remove('hidden');
      progressMessage.textContent = 'Regenerating prompts...';
    }});
  }}

  if (btnGenerateAll) {{
    btnGenerateAll.addEventListener('click', async () => {{
      await apiPost(`/api/gallery/${{RUN_ID}}/generate-all`, {{
        images_per_prompt: 1,
        resume: true
      }});
      progressBar.classList.remove('hidden');
      progressMessage.textContent = 'Queued image generation...';
    }});
  }}

  if (btnEnhanceAll) {{
    btnEnhanceAll.addEventListener('click', async () => {{
      await apiPost(`/api/gallery/${{RUN_ID}}/enhance-all`, {{
        softness: 0.5
      }});
      progressBar.classList.remove('hidden');
      progressMessage.textContent = 'Queued enhancement...';
    }});
  }}

  if (btnClearQueue) {{
    btnClearQueue.addEventListener('click', async () => {{
      await apiPost('/api/queue/clear');
    }});
  }}

  if (btnKill) {{
    btnKill.addEventListener('click', async () => {{
      await apiPost('/api/worker/kill');
    }});
  }}

  const btnArchive = document.getElementById('btn-archive');
  if (btnArchive) {{
    btnArchive.addEventListener('click', async () => {{
      if (!confirm('Save this gallery to archive?')) return;
      try {{
        const resp = await apiPost(`/api/gallery/${{RUN_ID}}/archive`);
        alert(resp.message || 'Archived successfully');
      }} catch (err) {{ /* error shown by apiPost */ }}
    }});
  }}

  // Global functions for per-image buttons
  window.generateImage = async function(promptIdx, imageIdx) {{
    await apiPost(`/api/gallery/${{RUN_ID}}/image/${{promptIdx}}/generate`, {{
      image_idx: imageIdx
    }});
    progressBar.classList.remove('hidden');
    progressMessage.textContent = `Generating image ${{promptIdx}}_${{imageIdx}}...`;
  }};

  window.enhanceImage = async function(promptIdx, imageIdx) {{
    await apiPost(`/api/gallery/${{RUN_ID}}/image/${{promptIdx}}/enhance`, {{
      image_idx: imageIdx,
      softness: 0.5
    }});
    progressBar.classList.remove('hidden');
    progressMessage.textContent = `Enhancing image ${{promptIdx}}_${{imageIdx}}...`;
  }};

  // Start SSE
  initSSE();
}})();
</script>
'''


def _build_interactive_styles() -> str:
    """Build additional CSS for interactive gallery."""
    return (
        NavHeader.css() +
        GalleryStyles.css() +
        Buttons.css() +
        ProgressBar.css() +
        '''
    .progress-container { display: flex; align-items: center; gap: 12px; }
    .progress-fill { height: 100%; background: #4a9eff; transition: width 0.3s; }

    /* Adjust body padding */
    body { padding-bottom: 80px; }
''' +
        LogPanel.css()
    )


def _build_gallery_html(
    prefix: str,
    cards_html: list[str],
    completed: int,
    total: int,
    grammar: str | None = None,
    raw_response_file: str | None = None,
    interactive: bool = False,
    run_id: str | None = None,
) -> str:
    """Build the complete gallery HTML document."""
    cards_joined = "\n".join(cards_html)

    # Base tag for interactive galleries to resolve relative URLs correctly
    base_tag = f'<base href="/gallery/{run_id}/">' if interactive and run_id else ""

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

    # Build grammar section
    grammar_section = ""
    if grammar:
        if interactive and run_id:
            grammar_section = _build_interactive_grammar_section(grammar, run_id)
        else:
            escaped_grammar = html.escape(grammar)
            grammar_section = f'''
  <details class="grammar-section">
    <summary>Tracery Grammar</summary>
    <pre>{escaped_grammar}</pre>
  </details>'''

    # Build interactive sections
    nav_header = _build_nav_header() if interactive else ""
    action_bar = _build_interactive_action_bar(run_id) if interactive and run_id else ""
    log_panel = _build_log_panel() if interactive else ""
    progress_bar = _build_interactive_progress_bar() if interactive else ""
    interactive_js = _build_interactive_js(run_id) if interactive and run_id else ""
    extra_styles = _build_interactive_styles() if interactive else ""

    return f'''<!DOCTYPE html>
<html>
<head>
  <title>Gallery: {prefix}</title>
  <meta charset="utf-8">
  {base_tag}
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
    .card .placeholder.no-image {{ aspect-ratio: 1/1; background: #252525; color: #555; font-size: 24px; font-weight: bold; }}
    .card.prompt-only {{ border: 1px dashed #444; }}
    .card .prompt {{ padding: 12px; font-size: 13px; color: #aaa; max-height: 150px; overflow-y: auto; }}
{extra_styles}
  </style>
</head>
<body>{nav_header}
  <h1>Gallery: {prefix}</h1>{header_section}{grammar_section}{action_bar}{log_panel}
  <p class="status">Generated: {completed} / {total} images</p>
  <div class="grid">
{cards_joined}
  </div>
{progress_bar}
{interactive_js}
</body>
</html>
'''
