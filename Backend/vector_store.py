import faiss
import numpy as np
import pickle
import os


class VectorStore:
    """
    Wraps FAISS to store, save, load, and search embedding vectors.
    
    WHY a class? Because we need to keep TWO things in sync:
    1. The FAISS index — knows about vectors, but NOT original text
    2. The chunks list — has the original text, but NOT vectors
    
    FAISS only stores numbers. When it says "chunk #7 is most similar",
    we need to look up chunks[7] to get the actual text back.
    This class keeps them together.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []  # Parallel list: chunks[i] matches index vector i
    
    def build(self, chunks: list[dict], embeddings: np.ndarray):
        """
        Build the FAISS index from scratch.
        
        WHY IndexFlatIP?
        - "Flat" = exact search (checks every vector, no approximation)
        - "IP" = Inner Product = dot product similarity
        - Since we normalized our vectors, dot product = cosine similarity
        - For small PDFs (< 10,000 chunks), exact search is fast enough
        - For very large corpora, you'd switch to IndexIVFFlat (approximate,
          but much faster). That's an interview talking point!
        
        The index takes a float32 array — we cast explicitly to be safe.
        """
        embeddings_f32 = embeddings.astype(np.float32)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_f32)
        self.chunks = chunks
        
        print(f"[VectorStore] Built index with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Find the top_k most similar chunks to a query vector.
        
        Returns list of chunks with an added 'score' field (similarity score).
        Higher score = more similar (range: 0 to 1 for normalized vectors).
        
        WHY top_k=3?
        - Too few: might miss relevant context
        - Too many: fills the LLM prompt with irrelevant text (worse answers)
        - 3-5 is the industry standard starting point
        """
        if self.index is None:
            raise ValueError("Index not built yet. Upload a PDF first.")
        
        # FAISS expects shape (num_queries, embedding_dim)
        # We're searching one query at a time, so shape is (1, 384)
        query_f32 = query_embedding.astype(np.float32).reshape(1, -1)
        
        scores, indices = self.index.search(query_f32, top_k)
        
        # scores shape: (1, top_k) — squeeze to 1D
        # indices shape: (1, top_k) — these are positions in self.chunks
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # FAISS returns -1 when fewer results than top_k
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)
        
        return results
    
    def save(self, save_dir: str):
        """
        Persist the index and chunks to disk.
        
        WHY save?
        Without this, every server restart loses the index and the user
        must re-upload. For production you'd use a persistent vector DB
        (Pinecone, Weaviate, ChromaDB). For this project, disk save is fine.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        
        with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        
        print(f"[VectorStore] Saved to {save_dir}/")
    
    def load(self, save_dir: str):
        """Load a previously saved index from disk."""
        index_path = os.path.join(save_dir, "index.faiss")
        chunks_path = os.path.join(save_dir, "chunks.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No saved index found at {save_dir}")
        
        self.index = faiss.read_index(index_path)
        
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        
        print(f"[VectorStore] Loaded index with {self.index.ntotal} vectors")
    
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

