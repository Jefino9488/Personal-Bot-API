import hashlib
import logging
import os
import time
from functools import lru_cache

from cachetools import TTLCache
from sentence_transformers import SentenceTransformer
from sqlalchemy.sql import text

from app.db import get_db
from app.gemini import ask_gemini

# Get configuration from environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Cache TTL in seconds
CACHE_SIZE = int(os.getenv("RESPONSE_CACHE_SIZE", "100"))

logger = logging.getLogger(__name__)

# Initialize the embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Create a TTL cache for question responses
response_cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)

# Create a cache for embeddings
@lru_cache(maxsize=100)
def get_question_embedding(question_hash):
    """Generate and cache embedding for a question."""
    return _generate_embedding(question_hash)

def _generate_embedding(question_hash):
    """Internal function to generate embedding from cache key."""
    # Extract the original question from the hash (in a real implementation, 
    # you might need a different approach to recover the question)
    # For this example, we'll use a global dict to store question -> hash mapping
    if question_hash in question_to_embedding:
        return question_to_embedding[question_hash]
    return None

# Dictionary to store question -> embedding mapping
question_to_embedding = {}

def get_relevant_context(question, top_k=TOP_K_RESULTS):
    """
    Retrieve the most relevant context chunks for a question.
    Uses vector similarity search with caching for performance.
    """
    start_time = time.time()

    # Create a hash of the question for caching
    question_hash = hashlib.md5(question.encode()).hexdigest()

    # Generate or retrieve cached embedding
    if question_hash not in question_to_embedding:
        q_embedding = model.encode([question])[0].tolist()
        question_to_embedding[question_hash] = q_embedding
    else:
        q_embedding = question_to_embedding[question_hash]

    with get_db() as db:
        try:
            # Use the pgvector extension for efficient vector similarity search
            sql = text("""
                SELECT content FROM context_chunks
                ORDER BY embedding <-> CAST(:embedding AS vector)
                LIMIT :top_k
            """)

            results = db.execute(sql, {"embedding": q_embedding, "top_k": top_k}).fetchall()
            context = "\n\n".join([row[0] for row in results])

            logger.info(f"Retrieved {len(results)} context chunks in {time.time() - start_time:.2f} seconds")
            return context
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            # Return empty context in case of error to allow graceful degradation
            return ""

async def handle_question_async(question):
    """Asynchronous version of handle_question."""
    # Check cache first
    question_hash = hashlib.md5(question.encode()).hexdigest()
    if question_hash in response_cache:
        logger.info("Cache hit for question")
        return response_cache[question_hash]

    # Get context and ask Gemini in parallel
    context = get_relevant_context(question)
    response = await ask_gemini(question, context)

    # Cache the response
    response_cache[question_hash] = response
    return response

def handle_question(question):
    """
    Handle a question by retrieving relevant context and querying Gemini.
    Uses caching for improved performance.
    """
    start_time = time.time()

    # Check cache first
    question_hash = hashlib.md5(question.encode()).hexdigest()
    if question_hash in response_cache:
        logger.info("Cache hit for question")
        return response_cache[question_hash]

    try:
        # Get relevant context
        context = get_relevant_context(question)

        # Query Gemini
        response = ask_gemini(question, context)

        # Cache the response
        response_cache[question_hash] = response

        logger.info(f"Question handled in {time.time() - start_time:.2f} seconds")
        return response
    except Exception as e:
        logger.error(f"Error handling question: {str(e)}")
        return "I'm sorry, I encountered an error while processing your question. Please try again later."
