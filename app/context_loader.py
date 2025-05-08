import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from app.db import SessionLocal, ContextChunk
import os

CHUNK_SIZE = 500  # words
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
model = SentenceTransformer(EMBEDDING_MODEL)

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

def load_and_store_context(file_path):
    source_name = os.path.basename(file_path)
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    embeddings = model.encode(chunks).tolist()
    db = SessionLocal()

    for content, embedding in zip(chunks, embeddings):
        db.add(ContextChunk(source=source_name, content=content, embedding=embedding))

    db.commit()
    db.close()
    print(f"Loaded {len(chunks)} chunks from {file_path}")
