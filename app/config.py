"""
Configuration module for the Personal Bot API.

This module centralizes all configuration settings loaded from environment variables,
providing default values and type conversion where appropriate.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "10"))  # Timeout in seconds
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

# Database Settings
DATABASE_URL = os.getenv("DATABASE_URL")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))

# Caching Settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Cache TTL in seconds
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))
RESPONSE_CACHE_SIZE = int(os.getenv("RESPONSE_CACHE_SIZE", "100"))

# Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))  # words
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # words of overlap between chunks
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))

# Rate Limiting
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "100"))
RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))

# Monitoring
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Application Settings
RESUME_PATH = os.getenv("RESUME_PATH", "resume.pdf")
