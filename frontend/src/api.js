const BASE = "http://localhost:5000/api";

/**
 * Upload a PDF file to the backend for indexing.
 * Returns the server response JSON on success, throws on error.
 */
export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Upload failed");
  return data;
}

/**
 * Send a question and stream the answer back via SSE.
 *
 * WHY not use EventSource here?
 * The browser's EventSource API only supports GET requests.
 * Our /api/chat needs POST (to send a JSON body with the question).
 * So we use fetch() with a ReadableStream reader instead — same protocol,
 * more control.
 *
 * @param {string} question
 * @param {function} onSources  - called once with sources array
 * @param {function} onToken    - called for each text token streamed
 * @param {function} onDone     - called with final usage stats
 * @param {function} onError    - called if something goes wrong
 */
export async function streamChat(question, document_id, { onSources, onToken, onDone, onError }) {
  try {
    const res = await fetch(`${BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, document_id, stream: true }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Chat request failed");
    }

    // Read the SSE stream line by line
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by "\n\n"
      // We may receive partial events — buffer until we have a full one
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // last element may be incomplete — keep in buffer

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;

        const jsonStr = line.slice(6); // strip "data: " prefix
        try {
          const event = JSON.parse(jsonStr);

          if (event.type === "sources") onSources?.(event.data);
          else if (event.type === "token")  onToken?.(event.data);
          else if (event.type === "done")   onDone?.(event.data);
          else if (event.type === "error")  onError?.(new Error(event.data));
        } catch {
          // Malformed JSON — skip
        }
      }
    }
  } catch (err) {
    onError?.(err);
  }
}

/**
 * Check if the backend is running and has a PDF indexed.
 */
export async function checkHealth() {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}