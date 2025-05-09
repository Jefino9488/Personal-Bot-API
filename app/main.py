from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field, field_validator
from app.db import init_db
from app.context_loader import load_and_store_context
from app.ask import handle_question, handle_question_async
import logging
import time
import os
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, start_http_server
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "100"))
RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))

# Lazy-init Prometheus metrics
REQUEST_COUNT = None
REQUEST_LATENCY = None

# Initialize FastAPI app
app = FastAPI(
    title="Personal Bot API",
    description="API for answering questions based on personal context",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)

    if ENABLE_METRICS and REQUEST_COUNT and REQUEST_LATENCY:
        REQUEST_COUNT.labels(
            'personal_bot_api',
            request.url.path,
            request.method,
            response.status_code
        ).inc()

        REQUEST_LATENCY.labels(
            'personal_bot_api',
            request.url.path
        ).observe(time.time() - start_time)

    return response

# Pydantic model with updated validator
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

    @field_validator('question')
    @classmethod
    def question_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or just whitespace')
        return v.strip()

class HealthResponse(BaseModel):
    status: str
    version: str

# Application startup hook
@app.on_event("startup")
async def startup():
    global REQUEST_COUNT, REQUEST_LATENCY
    try:
        # Initialize Redis-based rate limiter
        redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        await FastAPILimiter.init(redis_client)
        logger.info("Rate limiter initialized successfully")

        # Initialize database and create tables
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully.")

        # Load and embed PDF context
        logger.info("Loading PDF context...")
        load_and_store_context("resume.pdf")
        logger.info("PDF context loaded successfully.")

        # Initialize Prometheus metrics
        if ENABLE_METRICS and REQUEST_COUNT is None:
            REQUEST_COUNT = Counter(
                'request_count', 'App Request Count',
                ['app_name', 'endpoint', 'method', 'status_code']
            )
            REQUEST_LATENCY = Histogram(
                'request_latency_seconds', 'Request latency in seconds',
                ['app_name', 'endpoint']
            )
            start_http_server(METRICS_PORT)
            logger.info(f"Metrics server started on port {METRICS_PORT}")

    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/ask")
async def ask(
    req: AskRequest,
    rate_limiter: Optional[RateLimiter] = Depends(RateLimiter(times=RATE_LIMIT, seconds=RATE_LIMIT_PERIOD))
):
    try:
        response = handle_question(req.question)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

@app.post("/ask/async")
async def ask_async(
    req: AskRequest,
    rate_limiter: Optional[RateLimiter] = Depends(RateLimiter(times=RATE_LIMIT, seconds=RATE_LIMIT_PERIOD))
):
    try:
        response = await handle_question_async(req.question)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing question asynchronously: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )
