import { useState, useRef } from "react";
import { uploadPDF } from "../api";
import styles from "./UploadPanel.module.css";

export default function UploadPanel({ onUploaded }) {
  const [status, setStatus] = useState("idle");
  const [info, setInfo] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  async function handleFile(file) {
    if (!file || !file.name.endsWith(".pdf")) {
      setErrorMsg("Please upload a PDF file.");
      setStatus("error");
      return;
    }
    setStatus("uploading");
    setErrorMsg("");
    try {
      const data = await uploadPDF(file);
      setInfo(data);
      setStatus("success");
      onUploaded(data);
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }

  function onInputChange(e) {
    handleFile(e.target.files[0]);
  }
  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.logo}>◈ RAG</span>
        <span className={styles.tagline}>Ask questions from any document</span>
      </div>

      <div
        className={`${styles.dropzone} ${dragging ? styles.dragging : ""} ${
          status === "success" ? styles.done : ""
        }`}
        onDrop={onDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onClick={() => status !== "success" && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          onChange={onInputChange}
          style={{ display: "none" }}
        />

        {status === "idle" && (
          <div className={styles.dropContent}>
            <div className={styles.dropIcon}>⬆</div>
            <p className={styles.dropTitle}>Drop your PDF here</p>
            <p className={styles.dropSub}>or click to browse</p>
          </div>
        )}

        {status === "uploading" && (
          <div className={styles.dropContent}>
            <div className={styles.spinner} />
            <p className={styles.dropTitle}>Indexing document…</p>
            <p className={styles.dropSub}>
              Chunking → Embedding → Building vector index
            </p>
          </div>
        )}

        {status === "success" && info && (
          <div className={styles.dropContent}>
            <div className={styles.successIcon}>✓</div>
            <p className={styles.dropTitle}>{info.filename}</p>
            <div className={styles.statRow}>
              <Stat label="chunks" value={info.total_chunks} />
              <Stat label="words" value={info.total_words?.toLocaleString()} />
              <Stat label="dims" value={info.embedding_dim} />
            </div>
          </div>
        )}

        {status === "error" && (
          <div className={styles.dropContent}>
            <div className={styles.errorIcon}>✕</div>
            <p className={styles.dropTitle}>Upload failed</p>
            <p className={styles.dropSub}>{errorMsg}</p>
            <button
              className={styles.retryBtn}
              onClick={(e) => {
                e.stopPropagation();
                setStatus("idle");
              }}
            >
              Try again
            </button>
          </div>
        )}
      </div>

      {status === "idle" && (
        <div className={styles.steps}>
          {[
            ["01", "PDF is parsed", "PyPDF extracts raw text"],
            ["02", "Text is chunked", "500-word sliding window"],
            ["03", "Chunks embedded", "all-MiniLM-L6-v2 → 384-dim vectors"],
            ["04", "FAISS indexed", "Similarity search ready"],
          ].map(([n, title, sub]) => (
            <div key={n} className={styles.step}>
              <span className={styles.stepNum}>{n}</span>
              <div>
                <p className={styles.stepTitle}>{title}</p>
                <p className={styles.stepSub}>{sub}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {status === "success" && (
        <button
          className={styles.changeBtn}
          onClick={() => {
            setStatus("idle");
            setInfo(null);
          }}
        >
          ↩ Upload different document
        </button>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statValue}>{value}</span>
      <span className={styles.statLabel}>{label}</span>
    </div>
  );
}
