import hashlib
import logging
import os
import time
from functools import lru_cache

from cachetools import TTLCache
from sentence_transformers import SentenceTransformer
from sqlalchemy.sql import text

from app.db import get_db
from app.gemini import ask_gemini, ask_gemini_async

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

    Args:
        question (str): The question to find context for
        top_k (int): Number of top results to retrieve

    Returns:
        str: Concatenated context chunks, or empty string on error

    Raises:
        No exceptions are raised; errors are logged and empty string is returned
    """
    start_time = time.time()
    logger.info(f"Getting relevant context for question: '{question[:50]}...' (top_k={top_k})")

    try:
        # Create a hash of the question for caching
        question_hash = hashlib.md5(question.encode()).hexdigest()

        # Generate or retrieve cached embedding
        embedding_start = time.time()
        if question_hash not in question_to_embedding:
            logger.debug(f"Generating new embedding for question (hash: {question_hash[:8]})")
            q_embedding = model.encode([question])[0].tolist()
            question_to_embedding[question_hash] = q_embedding
        else:
            logger.debug(f"Using cached embedding for question (hash: {question_hash[:8]})")
            q_embedding = question_to_embedding[question_hash]
        logger.debug(f"Embedding processing took {time.time() - embedding_start:.3f} seconds")

        with get_db() as db:
            try:
                # Use the pgvector extension for efficient vector similarity search
                db_start = time.time()
                sql = text("""
                           SELECT content
                           FROM context_chunks
                           ORDER BY embedding <-> CAST(:embedding AS vector)
                           LIMIT :top_k
                           """)

                results = db.execute(sql, {"embedding": q_embedding, "top_k": top_k}).fetchall()

                if not results:
                    logger.warning("No context chunks found in database. Check if context was loaded properly.")
                    return ""

                context = "\n\n".join([row[0] for row in results])
                logger.debug(f"Database query took {time.time() - db_start:.3f} seconds")
                logger.info(f"Retrieved {len(results)} context chunks in {time.time() - start_time:.3f} seconds")

                # Log a sample of the context for debugging
                context_sample = context[:100] + "..." if len(context) > 100 else context
                logger.debug(f"Context sample: {context_sample}")

                return context
            except Exception as e:
                logger.error(f"Database error retrieving context: {str(e)}", exc_info=True)
                # Return empty context in case of error to allow graceful degradation
                return ""
    except Exception as e:
        logger.error(f"Unexpected error in get_relevant_context: {str(e)}", exc_info=True)
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
    response = await ask_gemini_async(question, context)

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
