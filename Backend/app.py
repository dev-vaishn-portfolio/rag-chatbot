import os
import json
import tempfile
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()  # Loads GROQ_API_KEY from .env file BEFORE importing submodules

from pdf_processor import process_pdf
from embedder import embed_chunks, embed_text
from vector_store import VectorStore
from rag_engine import query_rag, query_rag_stream
import uuid

# Dictionary to hold active vector stores by document_id
active_indices = {}

app = Flask(__name__)
# Limit upload size to 16MB for security
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# CORS lets our React frontend (localhost:5173) talk to this Flask server
# (localhost:5000). Without it, browsers block cross-origin requests.
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:3000"]}})

# Where to save the FAISS index between restarts
INDEX_DIR = "./saved_index"


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check — useful to confirm the server is running."""
    return jsonify({
        "status": "ok",
        "active_sessions": len(active_indices)
    })


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """
    Accepts a PDF file, runs the full indexing pipeline, saves the index.
    
    Flow:
    1. Save uploaded file to a temp location
    2. Extract text & chunk it (pdf_processor)
    3. Embed all chunks (embedder)
    4. Build FAISS index (vector_store)
    5. Save index to disk
    6. Return stats to frontend
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    # Save to a temporary file — we need a real file path for PdfReader
    # WHY tempfile? It auto-cleans up and avoids naming collisions
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Step 1: PDF → chunks
        chunks = process_pdf(tmp_path)
        
        # Step 2: chunks → embeddings (numpy array)
        chunks, embeddings = embed_chunks(chunks)
        
        # Step 3: Build FAISS index for this specific document
        doc_id = str(uuid.uuid4())
        vs = VectorStore()
        vs.build(chunks, embeddings)
        
        # Step 4: Save to disk for persistence across restarts
        save_path = os.path.join(INDEX_DIR, doc_id)
        vs.save(save_path)
        
        active_indices[doc_id] = vs
        
        return jsonify({
            "message": "PDF indexed successfully!",
            "document_id": doc_id,
            "filename": file.filename,
            "total_chunks": len(chunks),
            "total_words": sum(c["word_count"] for c in chunks),
            "embedding_dim": embeddings.shape[1]
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    
    except Exception as e:
        print(f"[Error] Upload failed: {e}")
        return jsonify({"error": "Failed to process PDF. See server logs."}), 500
    
    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    doc_id = data.get("document_id")
    if not doc_id or doc_id not in active_indices:
        return jsonify({"error": "Invalid or missing document_id. Please upload a PDF first."}), 400
        
    vs = active_indices[doc_id]
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    # Embed the query and search
    query_embedding = embed_text(query)
    results = vs.search(query_embedding, top_k=3)
    
    return jsonify({
        "query": query,
        "results": [
            {
                "chunk_index": r["chunk_index"],
                "text": r["text"][:300] + "...",  # Truncate for readability
                "score": round(r["score"], 4),
                "word_count": r["word_count"]
            }
            for r in results
        ]
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Request body must include a 'question' field"}), 400
        
    doc_id = data.get("document_id")
    if not doc_id or doc_id not in active_indices:
        return jsonify({"error": "Invalid or missing document_id. Please upload a PDF first."}), 400
        
    vs = active_indices[doc_id]

    question = data.get("question", "").strip()
    use_stream = data.get("stream", True)  # Default to streaming

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    if not vs.is_ready():
        return jsonify({"error": "No PDF indexed yet. Upload a PDF first."}), 400

    # ── Non-streaming mode ────────────────────────────────────────────────────
    if not use_stream:
        try:
            result = query_rag(question, vs)
            return jsonify(result)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            print(f"[Error] Chat failed: {e}")
            return jsonify({"error": "Failed to generate answer. See server logs."}), 500

    # ── Streaming mode (SSE) ──────────────────────────────────────────────────
    def generate():
        try:
            for event in query_rag_stream(question, vs):
                # Serialize the event dict to a JSON string and wrap in SSE format
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "data": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            # Prevent nginx/proxies from buffering the stream
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        }
    )


if __name__ == "__main__":
    # Load previously saved indices on startup
    if os.path.exists(INDEX_DIR):
        for doc_id in os.listdir(INDEX_DIR):
            doc_path = os.path.join(INDEX_DIR, doc_id)
            if os.path.isdir(doc_path):
                try:
                    vs = VectorStore()
                    vs.load(doc_path)
                    active_indices[doc_id] = vs
                    print(f"[App] Loaded index {doc_id} from disk.")
                except Exception as e:
                    print(f"[App] Could not load saved index {doc_id}: {e}")
    
    print("[App] Starting Flask server...")
    # Port is usually provided by the hosting environment via the PORT env var
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)