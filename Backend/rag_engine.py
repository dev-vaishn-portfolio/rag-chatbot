import os
from groq import Groq
from embedder import embed_text


# Initialize the Groq client once at module load
# WHY: Same reason as the embedding model — avoid recreating on every request
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the RAG prompt — this is the most important function in the project.

    WHY this prompt structure?
    1. We give the LLM ONLY the retrieved context, not the whole PDF.
       This keeps the prompt short, focused, and cheap.
    2. We explicitly instruct it to NOT answer from outside knowledge.
       Without this, Claude might blend PDF content with its training data
       — great for general chat, bad for document Q&A (you lose traceability).
    3. We include chunk numbers so the LLM can cite sources.
       This is the "grounding" that GenAI roles care deeply about.

    The format:
    ─────────────────────────────────────────
    You are a document assistant...

    CONTEXT:
    [Chunk 1] ...text...
    [Chunk 2] ...text...
    [Chunk 3] ...text...

    QUESTION: ...

    ANSWER:
    ─────────────────────────────────────────
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        score_pct = round(chunk["score"] * 100, 1)
        context_blocks.append(
            f"[Chunk {i} | relevance: {score_pct}%]\n{chunk['text']}"
        )

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a precise document assistant. Your job is to answer questions based ONLY on the provided context chunks extracted from the user's uploaded document.

Rules:
- Answer using ONLY information from the context below
- If the answer is not in the context, say: "I couldn't find that information in the document."
- Be concise and direct
- When relevant, mention which chunk(s) your answer comes from

CONTEXT:
{context_str}

QUESTION: {question}

ANSWER:"""

    return prompt


def query_rag(question: str, vs, top_k: int = 3) -> dict:
    """
    Full RAG query pipeline:
    1. Embed the question
    2. Retrieve top_k most relevant chunks from FAISS
    3. Build a grounded prompt
    4. Call Claude and get the answer
    5. Return answer + the source chunks (for frontend to display)

    WHY return source chunks?
    Showing users WHERE the answer came from builds trust and lets them
    verify. This is called "citation" or "attribution" — a must-have
    feature for any production RAG system.
    """
    if not vs.is_ready():
        raise ValueError("No document indexed. Please upload a PDF first.")

    # Step 1: Embed the question (same model as chunks — critical!)
    print(f"[RAG] Embedding question: '{question}'")
    query_embedding = embed_text(question)

    # Step 2: Retrieve relevant chunks
    retrieved_chunks = vs.search(query_embedding, top_k=top_k)
    print(f"[RAG] Retrieved {len(retrieved_chunks)} chunks. "
          f"Top score: {retrieved_chunks[0]['score']:.3f}")

    # Step 3: Build the grounded prompt
    prompt = build_prompt(question, retrieved_chunks)

    # Step 4: Call Groq
    print("[RAG] Calling Groq API...")
    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = message.choices[0].message.content

    # Step 5: Return answer + metadata
    return {
        "answer": answer,
        "sources": [
            {
                "chunk_index": c["chunk_index"],
                "text_preview": c["text"][:200] + "...",
                "score": round(c["score"], 4),
                "relevance_pct": round(c["score"] * 100, 1)
            }
            for c in retrieved_chunks
        ],
        "model": message.model,
        "input_tokens": message.usage.prompt_tokens if message.usage else 0,
        "output_tokens": message.usage.completion_tokens if message.usage else 0
    }


def query_rag_stream(question: str, vs, top_k: int = 3):
    """
    Streaming version of query_rag using Groq.
    """
    if not vs.is_ready():
        yield {"type": "error", "data": "No document indexed. Please upload a PDF first."}
        return

    # Retrieve chunks first (not streamed — this is fast)
    query_embedding = embed_text(question)
    retrieved_chunks = vs.search(query_embedding, top_k=top_k)

    # Send sources immediately so the frontend can show them
    # before the answer text starts streaming
    sources = [
        {
            "chunk_index": c["chunk_index"],
            "text_preview": c["text"][:200] + "...",
            "score": round(c["score"], 4),
            "relevance_pct": round(c["score"] * 100, 1)
        }
        for c in retrieved_chunks
    ]
    yield {"type": "sources", "data": sources}

    # Build prompt and stream Groq's response
    prompt = build_prompt(question, retrieved_chunks)

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            yield {"type": "token", "data": chunk.choices[0].delta.content}

    # Final message
    yield {
        "type": "done",
        "data": {
            "input_tokens": 0,
            "output_tokens": 0
        }
    }