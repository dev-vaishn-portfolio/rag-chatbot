import { useState, useRef, useEffect } from "react";
import { streamChat } from "../api";
import SourceCard from "./Sourcecard";
import styles from "./ChatWindow.module.css";

const SUGGESTED = [
  "Summarize this document",
  "What are the key points?",
  "What topics does this cover?",
  "What conclusions are drawn?",
];

export default function ChatWindow({ documentInfo }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendQuestion(question) {
    if (!question.trim() || isStreaming) return;

    const userId = Date.now();
    const assistantId = Date.now() + 1;

    setMessages(prev => [
      ...prev,
      { role: "user", text: question, id: userId },
      { role: "assistant", text: "", sources: null, usage: null, id: assistantId, streaming: true },
    ]);
    setInput("");
    setIsStreaming(true);

    await streamChat(question, documentInfo.document_id, {
      onSources(sources) {
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, sources } : m));
      },
      onToken(token) {
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, text: m.text + token } : m));
      },
      onDone(usage) {
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, streaming: false, usage } : m));
        setIsStreaming(false);
      },
      onError(err) {
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, text: `Error: ${err.message}`, streaming: false, error: true } : m));
        setIsStreaming(false);
      },
    });
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(input); }
  }

  return (
    <div className={styles.window}>
      <div className={styles.topbar}>
        {documentInfo ? (
          <div className={styles.docInfo}>
            <span className={styles.docDot} />
            <span className={styles.docName}>{documentInfo.filename}</span>
            <span className={styles.docStats}>{documentInfo.total_chunks} chunks · {documentInfo.total_words?.toLocaleString()} words</span>
          </div>
        ) : (
          <span className={styles.docEmpty}>No document loaded</span>
        )}
      </div>

      <div className={styles.messages}>
        {!documentInfo && (
          <div className={styles.placeholder}>
            <div className={styles.placeholderIcon}>◈</div>
            <p className={styles.placeholderTitle}>Upload a PDF to start</p>
            <p className={styles.placeholderSub}>Your questions will be answered using only the content of your document.</p>
          </div>
        )}

        {documentInfo && messages.length === 0 && (
          <div className={styles.suggestions}>
            <p className={styles.suggestLabel}>Try asking…</p>
            <div className={styles.suggestGrid}>
              {SUGGESTED.map(q => (
                <button key={q} className={styles.suggestBtn} onClick={() => sendQuestion(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={`${styles.msgRow} ${styles[msg.role]} animate-in`}>
            <div className={styles.msgMeta}>
              <span className={styles.roleLabel}>{msg.role === "user" ? "you" : "rag"}</span>
            </div>
            <div className={styles.msgContent}>
              {msg.sources && msg.sources.length > 0 && (
                <div className={styles.sourcesSection}>
                  <p className={styles.sourcesLabel}>↗ retrieved {msg.sources.length} chunks</p>
                  <div className={styles.sourcesList}>
                    {msg.sources.map((s, i) => <SourceCard key={s.chunk_index} source={s} index={i} />)}
                  </div>
                </div>
              )}
              <div className={`${styles.bubble} ${msg.error ? styles.errorBubble : ""}`}>
                <p className={styles.text}>
                  {msg.text || (msg.streaming ? "" : "…")}
                  {msg.streaming && <span className={styles.cursor}>▋</span>}
                </p>
              </div>
              {msg.usage && (
                <p className={styles.usage}>{msg.usage.input_tokens} in · {msg.usage.output_tokens} out</p>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className={styles.inputArea}>
        <div className={`${styles.inputWrapper} ${!documentInfo ? styles.disabled : ""}`}>
          <textarea
            className={styles.textarea}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={documentInfo ? "Ask anything about your document…" : "Upload a document first"}
            disabled={!documentInfo || isStreaming}
            rows={1}
          />
          <button
            className={styles.sendBtn}
            onClick={() => sendQuestion(input)}
            disabled={!documentInfo || isStreaming || !input.trim()}
          >
            {isStreaming ? <span className={styles.stopDot} /> : "↑"}
          </button>
        </div>
        <p className={styles.hint}>Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  );
}