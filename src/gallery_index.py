"""Master gallery index generation for all prompt runs."""

import html
import json
import re
from pathlib import Path


def generate_master_index(generated_dir: Path) -> Path:
    """
    Scan generated/prompts/ for all run directories and create
    a master index.html linking to all gallery.html files.

    Args:
        generated_dir: Path to the generated/ directory

    Returns:
        Path to the created index.html file
    """
    prompts_dir = generated_dir / "prompts"
    index_path = generated_dir / "index.html"

    # Scan for directories matching YYYYMMDD_HHMMSS_* pattern
    runs = []
    if prompts_dir.exists():
        for run_dir in prompts_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if directory name matches timestamp pattern
            if not re.match(r'^\d{8}_\d{6}_', run_dir.name):
                continue

            run_info = _extract_run_info(run_dir)
            if run_info:
                runs.append(run_info)

    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x["timestamp"], reverse=True)

    # Generate index HTML
    index_html = _build_index_html(runs)
    index_path.write_text(index_html)

    return index_path


def _extract_run_info(run_dir: Path) -> dict | None:
    """
    Extract information about a run directory.

    Args:
        run_dir: Path to a run directory

    Returns:
        Dictionary with run info, or None if not a valid run
    """
    # Find metadata file
    meta_files = list(run_dir.glob("*_metadata.json"))
    if not meta_files:
        return None

    try:
        metadata = json.loads(meta_files[0].read_text())
    except (json.JSONDecodeError, IOError):
        return None

    prefix = metadata.get("prefix", "image")

    # Find gallery file
    gallery_file = run_dir / f"{prefix}_gallery.html"
    if not gallery_file.exists():
        return None

    # Find first image for thumbnail
    thumbnail = None
    for i in range(100):  # Check first 100 potential images
        for j in range(10):  # Check first 10 images per prompt
            img_path = run_dir / f"{prefix}_{i}_{j}.png"
            if img_path.exists():
                thumbnail = img_path.relative_to(run_dir.parent.parent)
                break
        if thumbnail:
            break

    # Extract timestamp from directory name
    dir_parts = run_dir.name.split("_")
    if len(dir_parts) >= 2:
        timestamp = f"{dir_parts[0]}_{dir_parts[1]}"
    else:
        timestamp = run_dir.name

    # Format timestamp for display (YYYYMMDD_HHMMSS -> YYYY-MM-DD HH:MM:SS)
    display_time = timestamp
    if len(timestamp) == 15 and timestamp[8] == "_":
        display_time = (
            f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} "
            f"{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
        )

    # Get image count
    image_count = len(list(run_dir.glob(f"{prefix}_*_*.png")))

    # Get prompt count
    prompt_count = metadata.get("count", 0)

    return {
        "dir_name": run_dir.name,
        "timestamp": timestamp,
        "display_time": display_time,
        "user_prompt": metadata.get("user_prompt", "Unknown prompt"),
        "prefix": prefix,
        "gallery_path": f"prompts/{run_dir.name}/{prefix}_gallery.html",
        "thumbnail": str(thumbnail) if thumbnail else None,
        "image_count": image_count,
        "prompt_count": prompt_count,
        "model": metadata.get("image_generation", {}).get("model", "N/A"),
    }


def _build_index_html(runs: list[dict]) -> str:
    """Build the master index HTML document."""
    cards_html = []

    for run in runs:
        escaped_prompt = html.escape(run["user_prompt"])
        truncated_prompt = (escaped_prompt[:100] + "...") if len(escaped_prompt) > 100 else escaped_prompt

        if run["thumbnail"]:
            thumbnail_html = f'<img src="{run["thumbnail"]}" loading="lazy">'
        else:
            thumbnail_html = '<div class="no-thumbnail">No images</div>'

        card = f'''    <a href="{run["gallery_path"]}" class="card">
      <div class="thumbnail">
        {thumbnail_html}
      </div>
      <div class="info">
        <div class="prompt" title="{escaped_prompt}">{truncated_prompt}</div>
        <div class="meta">
          <span class="time">{run["display_time"]}</span>
          <span class="stats">{run["image_count"]} images | {run["prompt_count"]} prompts</span>
          <span class="model">{run["model"]}</span>
        </div>
      </div>
    </a>'''
        cards_html.append(card)

    cards_joined = "\n".join(cards_html) if cards_html else '<p class="empty">No galleries found. Generate some images first!</p>'
    run_count = len(runs)

    return f'''<!DOCTYPE html>
<html>
<head>
  <title>Image Prompt Generator - Gallery Index</title>
  <meta charset="utf-8">
  <style>
    body {{ font-family: system-ui; padding: 20px; background: #1a1a1a; color: #fff; margin: 0; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ margin-bottom: 8px; }}
    .subtitle {{ color: #888; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }}
    .card {{ background: #2a2a2a; border-radius: 12px; overflow: hidden; text-decoration: none; color: inherit; transition: transform 0.2s, box-shadow 0.2s; display: block; }}
    .card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }}
    .thumbnail {{ aspect-ratio: 4/3; background: #333; overflow: hidden; }}
    .thumbnail img {{ width: 100%; height: 100%; object-fit: cover; }}
    .no-thumbnail {{ width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666; }}
    .info {{ padding: 16px; }}
    .prompt {{ font-size: 14px; color: #ddd; margin-bottom: 12px; line-height: 1.4; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .meta {{ display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #888; }}
    .meta .time {{ color: #6af; }}
    .meta .stats {{ color: #aaa; }}
    .meta .model {{ color: #8f8; }}
    .empty {{ color: #666; text-align: center; padding: 60px 20px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Image Prompt Generator</h1>
    <p class="subtitle">{run_count} generation runs</p>
    <div class="grid">
{cards_joined}
    </div>
  </div>
</body>
</html>
'''
