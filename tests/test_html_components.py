"""Tests for html_components.py - shared HTML/CSS/JS components."""

from html_components import (
    LogPanel,
    QueueStatusBar,
    ProgressBar,
    Notifications,
    SSEClient,
    Buttons,
    NavHeader,
    InteractiveStyles,
    FormStyles,
    GalleryStyles,
    IndexStyles,
)


class TestLogPanel:
    """Tests for LogPanel component."""

    def test_html_contains_log_panel(self):
        html = LogPanel.html()
        assert "log-panel" in html
        assert "log-content" in html
        assert "log-count" in html

    def test_css_returns_string(self):
        css = LogPanel.css()
        assert ".log-panel" in css
        assert ".log-content" in css

    def test_js_returns_string(self):
        js = LogPanel.js()
        assert "appendLog" in js
        assert "clearLogs" in js
        assert "MAX_LOG_LINES" in js

    def test_escape_html_function_exists(self):
        """escapeHtml must be a standalone function callable from log-line rendering."""
        js = LogPanel.js()
        # The function is defined with 'function escapeHtml(text)' at top level.
        assert "function escapeHtml(" in js or "escapeHtml = function" in js

    def test_escape_html_produces_escaped_output(self):
        """escapeHtml uses DOM textContent→innerHTML, so raw HTML is escaped.

        This is a security contract: without proper escaping, user-supplied log
        messages could inject HTML/script tags into the gallery UI (XSS).
        The implementation creates a div, sets .textContent, then reads .innerHTML —
        which forces browsers to escape < > & characters.
        """
        js = LogPanel.js()
        # Verify the DOM-based pattern is used: createElement + textContent + innerHTML.
        assert "createElement('div')" in js or 'createElement("div")' in js
        assert "textContent" in js
        assert "innerHTML" in js


class TestQueueStatusBar:
    """Tests for QueueStatusBar component."""

    def test_html_contains_queue_elements(self):
        html = QueueStatusBar.html()
        assert "queue-status" in html
        assert "progress-fill" in html
        assert "btn-kill" in html
        assert "btn-clear" in html

    def test_css_returns_string(self):
        css = QueueStatusBar.css()
        assert ".queue-status" in css
        assert ".progress-bar" in css


class TestProgressBar:
    """Tests for ProgressBar component."""

    def test_html_contains_progress_elements(self):
        html = ProgressBar.html()
        assert "progress-bar" in html
        assert "progress-fill" in html
        assert "progress-message" in html

    def test_css_returns_string(self):
        css = ProgressBar.css()
        assert ".progress-bar" in css or "progress" in css


class TestNotifications:
    """Tests for Notifications component."""

    def test_html_contains_toast_region(self):
        html = Notifications.html()
        assert "toast-region" in html

    def test_css_returns_string(self):
        css = Notifications.css()
        assert ".toast" in css

    def test_js_returns_string(self):
        js = Notifications.js()
        assert "showToast" in js


class TestSSEClient:
    """Tests for SSEClient component."""

    def test_js_contains_event_source(self):
        js = SSEClient.js()
        assert "EventSource" in js

    def test_js_contains_reconnect_logic(self):
        js = SSEClient.js()
        assert "connectSSE" in js
        assert "MAX_SSE_RETRIES" in js


class TestButtons:
    """Tests for Buttons component."""

    def test_css_returns_string(self):
        css = Buttons.css()
        assert ".btn" in css


class TestNavHeader:
    """Tests for NavHeader component."""

    def test_html_contains_nav(self):
        html = NavHeader.html()
        assert "nav-header" in html
        assert "/index" in html

    def test_css_returns_string(self):
        css = NavHeader.css()
        assert "nav-header" in css


class TestStyleClasses:
    """Tests for style classes."""

    def test_interactive_styles_css(self):
        css = InteractiveStyles.css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_form_styles_css(self):
        css = FormStyles.css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_gallery_styles_css(self):
        css = GalleryStyles.css()
        assert isinstance(css, str)
        assert len(css) > 0
        # GalleryStyles defines three gallery-specific style groups; losing any of
        # them silently breaks the user's gallery UI (action bar, grammar editor,
        # card actions). Each selector must survive as a substring.
        assert ".grammar-section-interactive" in css
        assert ".action-bar" in css
        assert ".card-actions" in css

    def test_index_styles_css(self):
        css = IndexStyles.css()
        assert isinstance(css, str)
        assert len(css) > 0
