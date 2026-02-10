# Markdown Viewer Chrome Extension

A simple, local-first Chrome extension that automatically renders Markdown files (`.md`, `.markdown`) as styled HTML using GitHub-flavored styles.

## 🚀 Installation

1.  **Open Extensions Management**:
    -   Navigate to `chrome://extensions/` in your Chrome browser.
2.  **Enable Developer Mode**:
    -   Toggle the switch in the top-right corner to **ON**.
3.  **Load Unpacked Extension**:
    -   Click the **Load unpacked** button (top-left).
    -   Select the `markdown-viewer-extension` folder from this project.

## ⚠️ Critical Step: Allow File Access
By default, Chrome blocks extensions from running on local files (`file://`). You **MUST** enable this manually:

1.  Find "Markdown Viewer" in your list of extensions.
2.  Click the **Details** button.
3.  Scroll down to the "Allow access to file URLs" toggle.
4.  **Turn it ON**.

## Usage
Simply drag and drop any `.md` file into a Chrome tab, or open a local file directly. It should now render as styled HTML!

## Development
-   **Core Logic**: `content.js` detects markdown mime-types or file extensions and uses `marked.js` to render them.
-   **Styling**: `styles.css` provides the visual theme.
