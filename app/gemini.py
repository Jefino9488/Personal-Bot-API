import os
import requests
import httpx
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio

# Get configuration from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "10"))  # Timeout in seconds
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

logger = logging.getLogger(__name__)

# Define which exceptions should trigger a retry
RETRY_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.RequestException,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RequestError
)

def create_prompt(question, context):
    """Create a well-structured prompt for the Gemini model."""
    return f"""Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question: {question}
If the answer cannot be found in the context, say "I don't have enough information to answer this question."
"""

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    reraise=True
)
def ask_gemini(question, context):
    """
    Query the Gemini API with retry logic for resilience.
    Uses exponential backoff for retries.
    """
    start_time = time.time()
    prompt = create_prompt(question, context)

    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,  # Lower temperature for more focused responses
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 1024,
        }
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json=body,
            timeout=GEMINI_TIMEOUT
        )

        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        data = response.json()
        candidates = data.get("candidates", [])

        if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
            result = candidates[0]["content"]["parts"][0]["text"]
            logger.info(f"Gemini API call completed in {time.time() - start_time:.2f} seconds")
            return result
        else:
            logger.warning("Unexpected response structure from Gemini API")
            return "I'm sorry, I couldn't process your question properly. Please try again."

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from Gemini API: {str(e)}")
        if response.status_code == 429:
            # Rate limit error - implement exponential backoff
            logger.warning("Rate limit hit, implementing backoff")
            time.sleep(2)  # Simple backoff
            raise e  # Will be caught by retry decorator
        return "I'm sorry, there was an error processing your request. Please try again later."

    except RETRY_EXCEPTIONS as e:
        logger.error(f"Network error with Gemini API: {str(e)}")
        raise  # Will be caught by retry decorator

    except Exception as e:
        logger.error(f"Unexpected error with Gemini API: {str(e)}")
        return "I'm sorry, an unexpected error occurred. Please try again later."

async def ask_gemini_async(question, context):
    """
    Asynchronous version of ask_gemini using httpx.
    """
    start_time = time.time()
    prompt = create_prompt(question, context)

    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 1024,
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                json=body,
                timeout=GEMINI_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()
            candidates = data.get("candidates", [])

            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                result = candidates[0]["content"]["parts"][0]["text"]
                logger.info(f"Async Gemini API call completed in {time.time() - start_time:.2f} seconds")
                return result
            else:
                logger.warning("Unexpected response structure from Gemini API")
                return "I'm sorry, I couldn't process your question properly. Please try again."

    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        logger.error(f"Error with async Gemini API call: {str(e)}")
        return "I'm sorry, there was an error processing your request. Please try again later."
