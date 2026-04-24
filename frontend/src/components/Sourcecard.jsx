import { useState } from "react";
import styles from "./SourceCard.module.css";

export default function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);

  const scoreColor =
    source.relevance_pct >= 70 ? "var(--green)" :
    source.relevance_pct >= 40 ? "var(--amber)" :
    "var(--text-muted)";

  return (
    <div className={styles.card}>
      <div className={styles.header} onClick={() => setExpanded(e => !e)}>
        <div className={styles.left}>
          <span className={styles.index}>#{index + 1}</span>
          <span className={styles.chunkLabel}>chunk {source.chunk_index}</span>
        </div>
        <div className={styles.right}>
          <div className={styles.barTrack}>
            <div className={styles.barFill} style={{ width: `${source.relevance_pct}%`, background: scoreColor }} />
          </div>
          <span className={styles.score} style={{ color: scoreColor }}>{source.relevance_pct}%</span>
          <span className={styles.toggle}>{expanded ? "▲" : "▼"}</span>
        </div>
      </div>
      {expanded && (
        <div className={styles.body}>
          <p className={styles.preview}>{source.text_preview}</p>
        </div>
      )}
    </div>
  );
}