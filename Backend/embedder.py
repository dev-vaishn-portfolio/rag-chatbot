from sentence_transformers import SentenceTransformer
import numpy as np

# We load the model ONCE when this module is imported.
# WHY: Loading an ML model takes ~2-3 seconds. If we loaded it
# on every request, the app would be unbearably slow.
# By loading at module level, it's loaded once when Flask starts.
#
# WHY this model (all-MiniLM-L6-v2)?
# - Small (80MB) — fast to download and run
# - Produces 384-dimensional vectors — good balance of quality vs speed
# - Free, runs locally — no API cost for embeddings
# - Specifically designed for semantic similarity tasks (exactly what we need)
print("[Embedder] Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Embedder] Model ready.")


def embed_text(text: str) -> np.ndarray:
    """
    Convert a single string into a vector (1D numpy array of 384 floats).
    
    We use this to embed the USER'S QUESTION at query time.
    It must use the SAME model as the chunks — otherwise the vectors
    live in different "spaces" and similarity search becomes meaningless.
    Think of it like: you can't compare temperatures in Celsius vs Fahrenheit
    without converting first.
    """
    embedding = model.encode(text, convert_to_numpy=True)
    # Normalize to unit length — makes cosine similarity = dot product
    # which is what FAISS IndexFlatIP uses (faster than L2 for this use case)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def embed_chunks(chunks: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    Embed all text chunks and return:
    1. The original chunks list (unchanged, for reference)
    2. A 2D numpy array of shape (num_chunks, 384)
       where each row is one chunk's embedding vector
    
    WHY batch encoding?
    model.encode() with a list of texts processes them in batches
    internally — much faster than calling encode() in a loop.
    
    show_progress_bar=True lets you see progress for large PDFs.
    """
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"[Embedder] Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32  # Process 32 chunks at a time
    )
    
    # Normalize all vectors at once (faster than one by one)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    print(f"[Embedder] Done. Embedding shape: {embeddings.shape}")
    # e.g. "Embedding shape: (24, 384)" → 24 chunks, each a 384-dim vector
    
    return chunks, embeddings