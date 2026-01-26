---
name: serve
description: Start the web UI server
allowed-tools:
  - Bash(python src/cli.py --serve:*)
  - Bash(source venv/bin/activate:*)
---

# Web Server

Start the image-prompt-expander web server.

## Usage

- `/serve` - Start the web UI

## Steps

1. Activate venv: `source venv/bin/activate`
2. Start server: `python src/cli.py --serve`
3. Report server URL: http://localhost:8000

## Notes

- Server opens browser automatically at http://localhost:8000
- Features: generation form, gallery browser, queue management
- Real-time progress updates via SSE
- Use Ctrl+C to stop the server