import { useState } from "react";
import UploadPanel from "./components/UploadPanel";
import ChatWindow from "./components/ChatWindow";
import "./index.css";
import styles from "./App.module.css";

export default function App() {
  const [documentInfo, setDocumentInfo] = useState(null);

  return (
    <div className={styles.layout}>
      <aside className={styles.sidebar}>
        <UploadPanel onUploaded={setDocumentInfo} />
      </aside>
      <main className={styles.main}>
        <ChatWindow documentInfo={documentInfo} />
      </main>
    </div>
  );
}