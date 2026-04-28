import os
import time
import requests
import numpy as np

# We use the Hugging Face Inference API instead of a local model to save RAM!
# This requires NO local model weights (saves ~1GB of RAM).
# For best results in production, set HF_TOKEN in your .env file.
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

print("[Embedder] Initialized API Embedder using HuggingFace Inference API.")

def get_hf_embeddings(texts: list[str], max_retries: int = 4) -> np.ndarray:
    """Helper function to call HF API with retries for cold-start (503)."""
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
    for attempt in range(max_retries):
        response = requests.post(API_URL, headers=headers, json={"inputs": texts})
        
        if response.status_code == 200:
            return np.array(response.json())
        elif response.status_code == 503:
            # The model is currently loading into HF's servers
            print(f"[Embedder] HF Model is loading... Retrying in 10s (Attempt {attempt+1}/{max_retries})")
            time.sleep(10)
        else:
            raise Exception(f"HuggingFace API Error ({response.status_code}): {response.text}")
            
    raise Exception("HuggingFace API failed: Model took too long to load.")


def embed_text(text: str) -> np.ndarray:
    """
    Convert a single string into a vector (1D numpy array of 384 floats).
    """
    embedding = get_hf_embeddings([text])[0]
    # Normalize to unit length for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def embed_chunks(chunks: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    Embed all text chunks via API and return:
    1. The original chunks list
    2. A 2D numpy array of shape (num_chunks, 384)
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"[Embedder] Embedding {len(texts)} chunks via HF API...")
    
    # Process in batches to avoid payload size limits and timeouts
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_emb = get_hf_embeddings(batch_texts)
            all_embeddings.extend(batch_emb)
        except Exception as e:
            print(f"[Embedder] Error during batch embedding: {e}")
            raise e
            
    embeddings = np.array(all_embeddings)
    
    # Normalize all vectors at once
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    print(f"[Embedder] Done. Embedding shape: {embeddings.shape}")
    
    return chunks, embeddings