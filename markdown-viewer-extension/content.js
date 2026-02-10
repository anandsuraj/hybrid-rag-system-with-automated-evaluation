// Check if the content is likely raw markdown.
// Chrome wraps text files in a <pre> tag with style="word-wrap: break-word; white-space: pre-wrap;"
function isMarkdown() {
    const contentType = document.contentType;
    if (contentType && (contentType === 'text/markdown' || contentType === 'text/x-markdown')) {
        return true;
    }
    // Fallback: check URL extension
    return /\.(md|markdown)$/i.test(window.location.pathname);
}

function createDownloadButton() {
    const btn = document.createElement('button');
    btn.className = 'markdown-viewer-download-btn';
    btn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Save as PDF
    `;
    btn.title = "Save as PDF";
    btn.onclick = () => window.print();
    document.body.appendChild(btn);
}

function renderMarkdown() {
    // Get the raw text
    // If Chrome wraps it in a <pre>, we need that content. Otherwise, body text.
    let rawText = document.body.innerText;
    
    // Clear the existing body
    document.body.innerHTML = '<div class="markdown-body"></div>';
    const container = document.querySelector('.markdown-body');

    // Use marked library (injected via manifest)
    try {
        container.innerHTML = marked.parse(rawText);
        createDownloadButton(); // Add button after rendering
    } catch (e) {
        console.error("Markdown rendering failed:", e);
        container.innerHTML = "<p>Error rendering Markdown.</p><pre>" + rawText + "</pre>";
    }
}

if (isMarkdown()) {
    renderMarkdown();
}
