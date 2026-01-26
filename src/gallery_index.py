"""Master gallery index generation for all prompt runs."""

import html
import json
import re
from pathlib import Path


def generate_master_index(generated_dir: Path, interactive: bool = False) -> Path:
    """
    Scan generated/prompts/ for all run directories and create
    a master index.html linking to all gallery.html files.

    Args:
        generated_dir: Path to the generated/ directory
        interactive: If True, include generation form and SSE support

    Returns:
        Path to the created index.html file
    """
    prompts_dir = generated_dir / "prompts"
    saved_dir = generated_dir / "saved"
    index_path = generated_dir / "index.html"

    # Scan active runs
    active_runs = []
    if prompts_dir.exists():
        for run_dir in prompts_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if directory name matches timestamp pattern
            if not re.match(r'^\d{8}_\d{6}_', run_dir.name):
                continue

            run_info = _extract_run_info(run_dir, is_archive=False)
            if run_info:
                active_runs.append(run_info)

    # Scan archived runs
    archived_runs = []
    if saved_dir.exists():
        for archive_dir in saved_dir.iterdir():
            if not archive_dir.is_dir():
                continue

            run_info = _extract_run_info(archive_dir, is_archive=True)
            if run_info:
                archived_runs.append(run_info)

    # Sort by timestamp (newest first)
    active_runs.sort(key=lambda x: x["timestamp"], reverse=True)
    archived_runs.sort(key=lambda x: x["timestamp"], reverse=True)

    # Generate index HTML
    index_html = _build_index_html(active_runs, archived_runs, interactive=interactive)
    index_path.write_text(index_html)

    return index_path


def _extract_run_info(run_dir: Path, is_archive: bool = False) -> dict | None:
    """
    Extract information about a run directory.

    Args:
        run_dir: Path to a run directory
        is_archive: Whether this is an archived run

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

    # Find first image for thumbnail using glob (more efficient)
    thumbnail = None
    thumbnail_file = None
    images = sorted(run_dir.glob(f"{prefix}_*_*.png"))
    if images:
        thumbnail = images[0].relative_to(run_dir.parent.parent)
        thumbnail_file = images[0].name

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

    # Build paths based on whether this is an archive
    if is_archive:
        gallery_path = f"saved/{run_dir.name}/{prefix}_gallery.html"
    else:
        gallery_path = f"prompts/{run_dir.name}/{prefix}_gallery.html"

    # Get backup info if available
    backup_info = metadata.get("backup_info", {})
    backup_reason = backup_info.get("backup_reason", "")

    return {
        "dir_name": run_dir.name,
        "timestamp": timestamp,
        "display_time": display_time,
        "user_prompt": metadata.get("user_prompt", "Unknown prompt"),
        "prefix": prefix,
        "gallery_path": gallery_path,
        "thumbnail": str(thumbnail) if thumbnail else None,
        "thumbnail_file": thumbnail_file,
        "image_count": image_count,
        "prompt_count": prompt_count,
        "model": metadata.get("image_generation", {}).get("model", "N/A"),
        "is_archive": is_archive,
        "backup_reason": backup_reason,
    }


def _build_generation_form() -> str:
    """Build the new generation form HTML."""
    return '''
  <div class="form-section">
    <h2>New Generation</h2>
    <form id="generate-form">
      <div class="form-row">
        <div class="form-group flex-grow">
          <label for="prompt">Prompt</label>
          <input type="text" id="prompt" name="prompt" required placeholder="a dragon flying over mountains">
        </div>
        <div class="form-group">
          <label for="prefix">Prefix</label>
          <input type="text" id="prefix" name="prefix" value="image" placeholder="image">
        </div>
      </div>

      <details class="settings-section" open>
        <summary>Generation Settings</summary>
        <div class="form-row">
          <div class="form-group">
            <label for="count">Prompt Count</label>
            <input type="number" id="count" name="count" value="50" min="1">
          </div>
          <div class="form-group">
            <label for="model">Model</label>
            <select id="model" name="model">
              <option value="z-image-turbo" selected>z-image-turbo</option>
              <option value="flux2-klein-4b">flux2-klein-4b</option>
              <option value="flux2-klein-9b">flux2-klein-9b</option>
            </select>
          </div>
          <div class="form-group">
            <label for="temperature">Temperature</label>
            <input type="number" id="temperature" name="temperature" value="0.7" step="0.1" min="0" max="2">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group checkbox-group">
            <label>
              <input type="checkbox" id="no_cache" name="no_cache">
              Force regenerate grammar (ignore cache)
            </label>
          </div>
        </div>
      </details>

      <details class="settings-section">
        <summary>Image Settings</summary>
        <div class="form-row">
          <div class="form-group checkbox-group">
            <label>
              <input type="checkbox" id="generate_images" name="generate_images">
              Generate Images
            </label>
          </div>
          <div class="form-group">
            <label for="images_per_prompt">Images/Prompt</label>
            <input type="number" id="images_per_prompt" name="images_per_prompt" value="1" min="1">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label for="width">Width</label>
            <input type="number" id="width" name="width" value="864" step="8">
          </div>
          <div class="form-group">
            <label for="height">Height</label>
            <input type="number" id="height" name="height" value="1152" step="8">
          </div>
          <div class="form-group">
            <label for="steps">Steps</label>
            <input type="number" id="steps" name="steps" placeholder="auto" min="1">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label for="quantize">Quantize</label>
            <select id="quantize" name="quantize">
              <option value="8" selected>8-bit</option>
              <option value="6">6-bit</option>
              <option value="5">5-bit</option>
              <option value="4">4-bit</option>
              <option value="3">3-bit</option>
            </select>
          </div>
          <div class="form-group">
            <label for="seed">Seed</label>
            <input type="number" id="seed" name="seed" placeholder="random">
          </div>
          <div class="form-group">
            <label for="max_prompts">Max Prompts</label>
            <input type="number" id="max_prompts" name="max_prompts" placeholder="all" min="1">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group checkbox-group">
            <label>
              <input type="checkbox" id="tiled_vae" name="tiled_vae" checked>
              Tiled VAE (memory efficient)
            </label>
          </div>
        </div>
      </details>

      <details class="settings-section">
        <summary>Enhancement Settings</summary>
        <div class="form-row">
          <div class="form-group checkbox-group">
            <label>
              <input type="checkbox" id="enhance" name="enhance">
              Enhance with SeedVR2
            </label>
          </div>
          <div class="form-group">
            <label for="enhance_softness">Softness</label>
            <input type="number" id="enhance_softness" name="enhance_softness" value="0.5" step="0.1" min="0" max="1">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group checkbox-group">
            <label>
              <input type="checkbox" id="enhance_after" name="enhance_after">
              Enhance after all images (saves memory)
            </label>
          </div>
        </div>
      </details>

      <div class="form-actions">
        <button type="submit" class="btn-primary">Start Grammar and Prompt Generation</button>
      </div>
    </form>
  </div>
'''


def _build_queue_status_bar() -> str:
    """Build the queue status bar HTML."""
    return '''
  <div id="queue-status" class="queue-status hidden">
    <div class="queue-info">
      <span id="queue-text">Idle</span>
    </div>
    <div id="progress-container" class="progress-container hidden">
      <div class="progress-bar">
        <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
      </div>
      <span id="progress-text">0%</span>
    </div>
    <div class="queue-actions">
      <button id="btn-kill" class="btn-danger btn-small hidden">Kill</button>
      <button id="btn-clear" class="btn-secondary btn-small hidden">Clear Queue</button>
    </div>
  </div>
'''


def _build_log_panel() -> str:
    """Build the collapsible log panel HTML."""
    return '''
  <details id="log-panel" class="log-panel">
    <summary>
      <span class="log-title">Generation Logs</span>
      <span id="log-count" class="log-count">0</span>
      <button id="btn-clear-logs" class="btn-small btn-secondary" onclick="event.preventDefault(); clearLogs();">Clear</button>
    </summary>
    <div id="log-content" class="log-content"></div>
  </details>
'''


def _build_interactive_js() -> str:
    """Build the JavaScript for form submission and SSE."""
    return '''
<script>
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, initializing...');

  const form = document.getElementById('generate-form');
  const queueStatus = document.getElementById('queue-status');
  const queueText = document.getElementById('queue-text');
  const progressContainer = document.getElementById('progress-container');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const btnKill = document.getElementById('btn-kill');
  const btnClear = document.getElementById('btn-clear');
  const logPanel = document.getElementById('log-panel');
  const logContent = document.getElementById('log-content');
  const logCount = document.getElementById('log-count');

  if (!form) {
    console.error('Form not found!');
    return;
  }
  console.log('Form found:', form);

  let eventSource = null;
  let logLineCount = 0;
  const MAX_LOG_LINES = 500;

  // Log panel functions
  function appendLog(timestamp, message) {
    if (!logContent) return;

    const line = document.createElement('div');
    line.className = 'log-line';
    if (message.toLowerCase().includes('error')) line.className += ' error';
    else if (message.toLowerCase().includes('warning')) line.className += ' warning';

    const time = timestamp ? new Date(timestamp).toLocaleTimeString() : '';
    line.innerHTML = `<span class="timestamp">${time}</span>${escapeHtml(message)}`;

    logContent.appendChild(line);
    logLineCount++;
    logCount.textContent = logLineCount;

    // Auto-scroll to bottom
    logContent.scrollTop = logContent.scrollHeight;

    // Trim old lines if too many
    while (logContent.children.length > MAX_LOG_LINES) {
      logContent.removeChild(logContent.firstChild);
      logLineCount--;
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  window.clearLogs = function() {
    if (logContent) {
      logContent.innerHTML = '';
      logLineCount = 0;
      logCount.textContent = '0';
    }
  };

  // Connect to SSE with backoff
  let sseRetryCount = 0;
  const MAX_SSE_RETRIES = 10;

  function connectSSE() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }

    if (sseRetryCount >= MAX_SSE_RETRIES) {
      console.warn('SSE: Max retries reached, giving up');
      return;
    }

    try {
      eventSource = new EventSource('/api/events');
    } catch (e) {
      console.error('SSE: Failed to create EventSource', e);
      return;
    }

    eventSource.onopen = () => {
      console.log('SSE connected');
      sseRetryCount = 0; // Reset on successful connection
    };

    eventSource.onerror = (e) => {
      console.error('SSE error', e);
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      sseRetryCount++;
      const delay = Math.min(3000 * Math.pow(2, sseRetryCount - 1), 30000);
      console.log(`SSE: Retry ${sseRetryCount}/${MAX_SSE_RETRIES} in ${delay}ms`);
      setTimeout(connectSSE, delay);
    };

    eventSource.addEventListener('status', (e) => {
      const data = JSON.parse(e.data);
      updateStatus(data);
    });

    eventSource.addEventListener('queue_updated', (e) => {
      const data = JSON.parse(e.data);
      updateStatus(data);
    });

    eventSource.addEventListener('task_started', (e) => {
      const task = JSON.parse(e.data);
      queueText.textContent = `Running: ${task.type}`;
      btnKill.classList.remove('hidden');
      progressContainer.classList.remove('hidden');
      queueStatus.classList.remove('hidden');
    });

    eventSource.addEventListener('task_progress', (e) => {
      const data = JSON.parse(e.data);
      const pct = data.total > 0 ? Math.round((data.current / data.total) * 100) : 0;
      progressFill.style.width = pct + '%';
      progressText.textContent = `${data.current}/${data.total}`;
      if (data.message) {
        queueText.textContent = data.message;
      }
    });

    eventSource.addEventListener('task_completed', (e) => {
      const data = JSON.parse(e.data);
      queueText.textContent = 'Completed';
      btnKill.classList.add('hidden');
      progressContainer.classList.add('hidden');
      // Reload to show new gallery
      if (data.result && data.result.run_id) {
        setTimeout(() => location.reload(), 1000);
      }
    });

    eventSource.addEventListener('task_failed', (e) => {
      const data = JSON.parse(e.data);
      queueText.textContent = 'Failed: ' + (data.error || 'Unknown error');
      btnKill.classList.add('hidden');
      progressContainer.classList.add('hidden');
    });

    eventSource.addEventListener('task_cancelled', (e) => {
      queueText.textContent = 'Cancelled';
      btnKill.classList.add('hidden');
      progressContainer.classList.add('hidden');
    });

    eventSource.addEventListener('task_log', (e) => {
      const data = JSON.parse(e.data);
      appendLog(data.timestamp, data.message);
      // Auto-open log panel when logs arrive
      if (logPanel && !logPanel.open) {
        logPanel.open = true;
      }
    });

    eventSource.addEventListener('ping', () => {});
  }

  function updateStatus(data) {
    const pending = data.pending_count || 0;
    const current = data.current;

    if (current) {
      queueStatus.classList.remove('hidden');
      btnKill.classList.remove('hidden');
      progressContainer.classList.remove('hidden');
      queueText.textContent = current.progress?.message || `Running: ${current.type}`;

      if (current.progress && current.progress.total > 0) {
        const pct = Math.round((current.progress.current / current.progress.total) * 100);
        progressFill.style.width = pct + '%';
        progressText.textContent = `${current.progress.current}/${current.progress.total}`;
      }
    } else if (pending > 0) {
      queueStatus.classList.remove('hidden');
      queueText.textContent = `${pending} task(s) pending`;
      btnClear.classList.remove('hidden');
      btnKill.classList.add('hidden');
      progressContainer.classList.add('hidden');
    } else {
      queueStatus.classList.add('hidden');
    }
  }

  // Form submission
  form.addEventListener('submit', async function(e) {
    e.preventDefault();
    console.log('Form submitted');

    const formData = new FormData(form);
    const promptValue = formData.get('prompt');

    if (!promptValue || !promptValue.trim()) {
      alert('Please enter a prompt');
      return;
    }

    const data = {
      prompt: promptValue.trim(),
      prefix: formData.get('prefix') || 'image',
      count: parseInt(formData.get('count')) || 50,
      model: formData.get('model') || 'z-image-turbo',
      temperature: parseFloat(formData.get('temperature')) || 0.7,
      no_cache: form.querySelector('#no_cache')?.checked || false,
      generate_images: form.querySelector('#generate_images')?.checked || false,
      images_per_prompt: parseInt(formData.get('images_per_prompt')) || 1,
      width: parseInt(formData.get('width')) || 864,
      height: parseInt(formData.get('height')) || 1152,
      steps: formData.get('steps') ? parseInt(formData.get('steps')) : null,
      quantize: parseInt(formData.get('quantize')) || 8,
      seed: formData.get('seed') ? parseInt(formData.get('seed')) : null,
      max_prompts: formData.get('max_prompts') ? parseInt(formData.get('max_prompts')) : null,
      tiled_vae: form.querySelector('#tiled_vae')?.checked || false,
      enhance: form.querySelector('#enhance')?.checked || false,
      enhance_softness: parseFloat(formData.get('enhance_softness')) || 0.5,
      enhance_after: form.querySelector('#enhance_after')?.checked || false,
    };

    console.log('Sending data:', data);

    // Show status immediately
    queueStatus.classList.remove('hidden');
    queueText.textContent = 'Submitting...';

    try {
      const resp = await fetch('/api/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data),
      });

      console.log('Response status:', resp.status);

      if (!resp.ok) {
        const err = await resp.json();
        console.error('Error response:', err);
        alert('Error: ' + (err.detail || 'Unknown error'));
        queueStatus.classList.add('hidden');
        return;
      }

      const result = await resp.json();
      console.log('Success:', result);
      queueText.textContent = result.message;
    } catch (err) {
      console.error('Fetch error:', err);
      alert('Error: ' + err.message);
      queueStatus.classList.add('hidden');
    }
  });

  // Kill button
  btnKill.addEventListener('click', async () => {
    try {
      await fetch('/api/worker/kill', {method: 'POST'});
    } catch (err) {
      console.error('Kill failed', err);
    }
  });

  // Clear queue button
  btnClear.addEventListener('click', async () => {
    try {
      await fetch('/api/queue/clear', {method: 'POST'});
      btnClear.classList.add('hidden');
      queueStatus.classList.add('hidden');
    } catch (err) {
      console.error('Clear failed', err);
    }
  });

  // Start SSE connection
  connectSSE();
});
</script>
'''


def _build_interactive_styles() -> str:
    """Build additional CSS for interactive mode."""
    return '''
    /* Form styles */
    .form-section { background: #2a2a2a; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
    .form-section h2 { margin: 0 0 20px 0; font-size: 18px; }
    .form-row { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
    .form-group { display: flex; flex-direction: column; gap: 6px; min-width: 120px; }
    .form-group.flex-grow { flex: 1; min-width: 200px; }
    .form-group label { font-size: 12px; color: #888; }
    .form-group input, .form-group select { background: #1a1a1a; border: 1px solid #444; border-radius: 6px; padding: 8px 12px; color: #fff; font-size: 14px; }
    .form-group input:focus, .form-group select:focus { outline: none; border-color: #6af; }
    .form-group input::placeholder { color: #666; }
    .checkbox-group { flex-direction: row; align-items: center; }
    .checkbox-group label { display: flex; align-items: center; gap: 8px; font-size: 14px; color: #ddd; cursor: pointer; }
    .checkbox-group input[type="checkbox"] { width: 16px; height: 16px; }
    .settings-section { background: #222; border-radius: 8px; margin-bottom: 16px; }
    .settings-section summary { padding: 12px 16px; cursor: pointer; color: #888; font-size: 14px; list-style: none; }
    .settings-section summary::-webkit-details-marker { display: none; }
    .settings-section summary::before { content: "\\25B6"; margin-right: 8px; font-size: 10px; display: inline-block; transition: transform 0.2s; }
    .settings-section[open] summary::before { transform: rotate(90deg); }
    .settings-section[open] { padding-bottom: 16px; }
    .settings-section > div { padding: 0 16px; }
    .form-actions { margin-top: 20px; }
    .btn-primary { background: #4a9eff; color: #fff; border: none; border-radius: 8px; padding: 12px 24px; font-size: 16px; cursor: pointer; font-weight: 500; }
    .btn-primary:hover { background: #3d8be0; }
    .btn-secondary { background: #444; color: #fff; border: none; border-radius: 6px; padding: 8px 16px; font-size: 14px; cursor: pointer; }
    .btn-secondary:hover { background: #555; }
    .btn-danger { background: #d44; color: #fff; border: none; border-radius: 6px; padding: 8px 16px; font-size: 14px; cursor: pointer; }
    .btn-danger:hover { background: #c33; }
    .btn-small { padding: 6px 12px; font-size: 12px; }

    /* Queue status bar */
    .queue-status { position: fixed; bottom: 0; left: 0; right: 0; background: #2a2a2a; border-top: 1px solid #444; padding: 12px 20px; display: flex; align-items: center; gap: 16px; z-index: 1000; }
    .queue-status.hidden { display: none; }
    .queue-info { flex: 1; font-size: 14px; color: #ddd; }
    .progress-container { display: flex; align-items: center; gap: 12px; }
    .progress-container.hidden { display: none; }
    .progress-bar { width: 200px; height: 8px; background: #444; border-radius: 4px; overflow: hidden; }
    .progress-fill { height: 100%; background: #4a9eff; transition: width 0.3s; }
    .queue-actions { display: flex; gap: 8px; }
    .queue-actions .hidden { display: none; }

    /* Adjust container padding for status bar */
    body { padding-bottom: 80px; }

    /* Log panel */
    .log-panel { background: #2a2a2a; border-radius: 8px; margin-bottom: 24px; }
    .log-panel summary { padding: 12px 16px; cursor: pointer; display: flex; align-items: center; gap: 12px; list-style: none; }
    .log-panel summary::-webkit-details-marker { display: none; }
    .log-title { color: #888; font-size: 14px; }
    .log-count { background: #444; color: #ddd; font-size: 11px; padding: 2px 8px; border-radius: 10px; }
    .log-content { max-height: 300px; overflow-y: auto; padding: 0 16px 16px; font-family: monospace; font-size: 12px; line-height: 1.6; }
    .log-line { color: #aaa; white-space: pre-wrap; word-break: break-all; }
    .log-line .timestamp { color: #6af; margin-right: 8px; }
    .log-line.error { color: #f88; }
    .log-line.warning { color: #fa0; }
'''


def _build_card_html(run: dict, interactive: bool, is_archive: bool = False) -> str:
    """Build HTML for a single gallery card."""
    escaped_prompt = html.escape(run["user_prompt"])
    truncated_prompt = (escaped_prompt[:100] + "...") if len(escaped_prompt) > 100 else escaped_prompt

    if run["thumbnail_file"] and interactive:
        # In interactive mode, use the gallery route for images
        if is_archive:
            thumbnail_src = f'/archive/{run["dir_name"]}/{run["thumbnail_file"]}'
        else:
            thumbnail_src = f'/gallery/{run["dir_name"]}/{run["thumbnail_file"]}'
        thumbnail_html = f'<img src="{thumbnail_src}" loading="lazy">'
    elif run["thumbnail"]:
        # In static mode, use relative path
        thumbnail_html = f'<img src="{run["thumbnail"]}" loading="lazy">'
    else:
        thumbnail_html = '<div class="no-thumbnail">No images</div>'

    # Use gallery route in interactive mode, relative path in static mode
    if interactive:
        gallery_href = f'/archive/{run["dir_name"]}' if is_archive else f'/gallery/{run["dir_name"]}'
    else:
        gallery_href = run["gallery_path"]

    card_class = "card archive" if is_archive else "card"

    # Add archive badge if this is an archive
    badge_html = ""
    if is_archive:
        reason = run.get("backup_reason", "archived")
        reason_label = {
            "pre_regenerate": "Pre-Regen",
            "pre_enhance": "Pre-Enhance",
            "manual_archive": "Saved",
        }.get(reason, "Backup")
        badge_html = f'<span class="archive-badge">{reason_label}</span>'

    return f'''    <a href="{gallery_href}" class="{card_class}">
      <div class="thumbnail">
        {thumbnail_html}{badge_html}
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


def _build_index_html(active_runs: list[dict], archived_runs: list[dict] = None, interactive: bool = False) -> str:
    """Build the master index HTML document."""
    if archived_runs is None:
        archived_runs = []

    # Build active run cards
    active_cards_html = []
    for run in active_runs:
        card = _build_card_html(run, interactive, is_archive=False)
        active_cards_html.append(card)

    active_cards_joined = "\n".join(active_cards_html) if active_cards_html else '<p class="empty">No galleries found. Generate some images first!</p>'
    run_count = len(active_runs)

    # Build archived run cards
    archived_section = ""
    if archived_runs:
        archived_cards_html = []
        for run in archived_runs:
            card = _build_card_html(run, interactive, is_archive=True)
            archived_cards_html.append(card)
        archived_cards_joined = "\n".join(archived_cards_html)
        archived_section = f'''
    <h2 class="section-title">Saved Archives ({len(archived_runs)})</h2>
    <div class="grid">
{archived_cards_joined}
    </div>'''

    # Build interactive sections
    form_html = _build_generation_form() if interactive else ""
    log_panel_html = _build_log_panel() if interactive else ""
    queue_html = _build_queue_status_bar() if interactive else ""
    js_html = _build_interactive_js() if interactive else ""
    extra_styles = _build_interactive_styles() if interactive else ""

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
    .section-title {{ margin: 32px 0 16px; font-size: 16px; color: #888; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }}
    .card {{ background: #2a2a2a; border-radius: 12px; overflow: hidden; text-decoration: none; color: inherit; transition: transform 0.2s, box-shadow 0.2s; display: block; }}
    .card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }}
    .card.archive {{ border: 2px solid #665500; }}
    .card.archive:hover {{ border-color: #997700; }}
    .thumbnail {{ aspect-ratio: 4/3; background: #333; overflow: hidden; position: relative; }}
    .thumbnail img {{ width: 100%; height: 100%; object-fit: cover; }}
    .no-thumbnail {{ width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666; }}
    .archive-badge {{ position: absolute; top: 8px; right: 8px; background: #665500; color: #fff; font-size: 10px; padding: 2px 8px; border-radius: 4px; }}
    .info {{ padding: 16px; }}
    .prompt {{ font-size: 14px; color: #ddd; margin-bottom: 12px; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }}
    .meta {{ display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #888; }}
    .meta .time {{ color: #6af; }}
    .meta .stats {{ color: #aaa; }}
    .meta .model {{ color: #8f8; }}
    .empty {{ color: #666; text-align: center; padding: 60px 20px; }}
{extra_styles}
  </style>
</head>
<body>
  <div class="container">
    <h1>Image Prompt Generator</h1>
    <p class="subtitle">{run_count} generation runs</p>
{form_html}{log_panel_html}
    <div class="grid">
{active_cards_joined}
    </div>{archived_section}
  </div>
{queue_html}
{js_html}
</body>
</html>
'''
