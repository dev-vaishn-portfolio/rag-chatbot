from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.
    
    WHY: PdfReader reads each page and pulls the raw text.
    Some PDFs are scanned images — those return empty strings.
    For this project, we assume text-based PDFs (resumes, docs, notes).
    """
    reader = PdfReader(file_path)
    full_text = ""
    
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:  # Some pages may be empty or image-only
            full_text += f"\n[Page {page_num + 1}]\n{page_text}"
    
    return full_text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping chunks.
    
    WHY chunk_size=500?
    - Too large → expensive to embed, less precise retrieval
    - Too small → chunks lose context (a sentence without its paragraph)
    - 500 words is a sweet spot for most documents
    
    WHY overlap=50?
    - Prevents information loss at boundaries
    - If a key sentence falls at the end of chunk 1, it also appears
      at the start of chunk 2, so retrieval still finds it
    
    Returns a list of dicts — each chunk carries its text AND metadata
    (page position, chunk index). This metadata becomes valuable later
    when showing users "where" an answer came from.
    """
    # Split into words first — word-level chunking is more natural than character-level
    words = text.split()
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": end
        })
        
        # Move forward by (chunk_size - overlap)
        # This creates the sliding window with overlap
        start += chunk_size - overlap
        chunk_index += 1
    
    return chunks


def process_pdf(file_path: str) -> list[dict]:
    """
    Full pipeline: PDF file → list of text chunks with metadata.
    This is the single function app.py will call.
    """
    print(f"[PDF Processor] Extracting text from: {file_path}")
    text = extract_text_from_pdf(file_path)
    
    if not text:
        raise ValueError("No text could be extracted from this PDF. It may be a scanned image.")
    
    print(f"[PDF Processor] Extracted {len(text.split())} words total")
    
    chunks = chunk_text(text)
    print(f"[PDF Processor] Created {len(chunks)} chunks")
    
    return chunks