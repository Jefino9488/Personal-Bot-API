import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from app.db import get_db, ContextChunk
import os
import logging
import hashlib
from cachetools import LRUCache, cached
import time
from sqlalchemy import func

# Get configuration from environment variables
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))  # words
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # words of overlap between chunks
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))

logger = logging.getLogger(__name__)

# Initialize the embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Create an LRU cache for embeddings
embedding_cache = LRUCache(maxsize=CACHE_SIZE)

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file with error handling."""
    try:
        start_time = time.time()
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        logger.info(f"PDF extraction completed in {time.time() - start_time:.2f} seconds")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        raise

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks for better context preservation.
    Using overlapping chunks improves context retrieval quality.
    """
    words = text.split()
    total_words = len(words)

    if total_words <= chunk_size:
        return [text]  # Return the entire text as one chunk if it's small enough

    chunks = []
    for i in range(0, total_words, chunk_size - overlap):
        # Ensure we don't go beyond the text length
        end_idx = min(i + chunk_size, total_words)
        # Create chunk from words[i:end_idx]
        chunk = " ".join(words[i:end_idx])
        chunks.append(chunk)

        # Break if we've reached the end of the text
        if end_idx == total_words:
            break

    logger.info(f"Created {len(chunks)} chunks with {overlap} words overlap")
    return chunks

@cached(cache=embedding_cache)
def get_embedding(text_hash, text):
    """Generate embedding for text with caching based on hash."""
    start_time = time.time()
    embedding = model.encode([text])[0].tolist()
    logger.debug(f"Embedding generated in {time.time() - start_time:.2f} seconds")
    return embedding

def load_and_store_context(file_path):
    """
    Load a document, chunk it, generate embeddings, and store in database.
    Uses improved chunking with overlap and caching for better performance.
    """
    start_time = time.time()
    source_name = os.path.basename(file_path)

    # Check if document already exists in database
    with get_db() as db:
        existing_count = db.query(func.count(ContextChunk.id)).filter(
            ContextChunk.source == source_name
        ).scalar()

        if existing_count > 0:
            logger.info(f"Document {source_name} already exists with {existing_count} chunks. Skipping.")
            return

    # Extract and chunk text
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    # Process chunks in batches to avoid memory issues
    batch_size = 50
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]

        with get_db() as db:
            for content in batch_chunks:
                # Create a hash of the content for caching
                content_hash = hashlib.md5(content.encode()).hexdigest()

                # Get embedding with caching
                embedding = get_embedding(content_hash, content)

                # Add to database
                db.add(ContextChunk(
                    source=source_name, 
                    content=content, 
                    embedding=embedding
                ))

        logger.info(f"Processed batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")

    total_time = time.time() - start_time
    logger.info(f"Loaded {total_chunks} chunks from {file_path} in {total_time:.2f} seconds")
