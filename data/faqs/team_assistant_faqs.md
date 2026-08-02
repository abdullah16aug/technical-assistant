# TeamAssistant FAQs

## What formats can TeamAssistant ingest?
TeamAssistant currently supports PDF, Markdown (.md), Plain Text (.txt), and Word Documents (.docx).

## What is the maximum file size for uploads?
The maximum upload size is 20 MB per file.

## How does TeamAssistant handle security and authentication?
Authentication uses Microsoft Entra ID. Access is controlled via IAM roles (Administrator, Data Engineer, Viewer). No credentials are stored in source code, and all data transmission is encrypted via HTTPS.

## Why is my query returning "No answer found"?
This usually happens if the relevant document has not been uploaded, or if the question is too vague. Try rephrasing your question, uploading the latest documentation, or ensuring that the Web Search Fallback is enabled.

## How can I improve the response time?
If responses are slow, administrators can optimize the chunk size, reduce the Top-K retrieval limit, or enable embedding caching. Average response time should be between 2 to 4 seconds.

## Can TeamAssistant read text inside images?
No, currently TeamAssistant does not index images. Scanned PDFs must go through OCR (Optical Character Recognition) before being uploaded.

## Does TeamAssistant support languages other than English?
No, the system currently only supports English documentation.

## What is Web Search Fallback?
If the internal knowledge base does not contain the answer to your question, the system will automatically search the web (via DuckDuckGo or Tavily) and generate an answer using external sources.
