# Personal Bot API

A FastAPI application that answers questions based on personal context loaded from a resume PDF. The application uses vector embeddings and the Gemini API to provide accurate responses to questions about the resume content.

## Features

- **Context-Aware Responses**: Answers questions based on the content of a resume PDF
- **Vector Search**: Uses pgvector for efficient semantic search
- **Rate Limiting**: Protects the API from abuse
- **Caching**: Redis-based caching for improved performance
- **Metrics**: Prometheus metrics for monitoring
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Async Support**: Both synchronous and asynchronous endpoints

## Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- Redis
- Google Gemini API key

## Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd PersonalBotApi
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. Copy the example environment file and update it with your settings:
   ```bash
   copy .env.example .env
   ```

4. Update the `.env` file with your Gemini API key and other settings.

5. Place your resume PDF file in the root directory as `resume.pdf`.

6. Start the application:
   ```bash
   uvicorn app.main:app --reload --port 8001
   ```

### Docker Deployment

1. Make sure Docker and Docker Compose are installed on your system.

2. Create a `.env` file with your Gemini API key:
   ```bash
   GEMINI_API_KEY=your-api-key-here
   ```

3. Start the application using Docker Compose:
   ```bash
   docker-compose up -d
   ```

## Configuration

The application can be configured using environment variables:

### Database Settings
- `DATABASE_URL`: PostgreSQL connection string
- `DB_POOL_SIZE`: Database connection pool size
- `DB_MAX_OVERFLOW`: Maximum number of connections to overflow
- `DB_POOL_TIMEOUT`: Connection timeout in seconds

### Caching Settings
- `REDIS_URL`: Redis connection string
- `CACHE_TTL`: Cache time-to-live in seconds
- `EMBEDDING_CACHE_SIZE`: Maximum number of embeddings to cache
- `RESPONSE_CACHE_SIZE`: Maximum number of responses to cache

### API Settings
- `GEMINI_API_KEY`: Google Gemini API key
- `GEMINI_TIMEOUT`: Timeout for Gemini API requests in seconds
- `GEMINI_MAX_RETRIES`: Maximum number of retries for Gemini API requests

### Chunking Settings
- `CHUNK_SIZE`: Size of text chunks for processing
- `CHUNK_OVERLAP`: Overlap between chunks
- `TOP_K_RESULTS`: Number of top results to consider

### Rate Limiting
- `RATE_LIMIT`: Maximum number of requests per period
- `RATE_LIMIT_PERIOD`: Rate limit period in seconds

### Monitoring
- `ENABLE_METRICS`: Enable Prometheus metrics
- `METRICS_PORT`: Port for Prometheus metrics

### Embedding Model
- `EMBEDDING_MODEL`: Model to use for text embeddings

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the API.

### Ask a Question
```
POST /ask
```
Body:
```json
{
  "question": "What skills are mentioned in the resume?"
}
```

### Ask a Question (Async)
```
POST /ask/async
```
Body:
```json
{
  "question": "What skills are mentioned in the resume?"
}
```

## Monitoring

Prometheus metrics are available at port 9090 when `ENABLE_METRICS` is set to `true`.

## Development

### Project Structure

- `app/main.py`: Main FastAPI application
- `app/db.py`: Database connection and models
- `app/context_loader.py`: PDF loading and context embedding
- `app/ask.py`: Question handling logic
- `app/gemini.py`: Integration with Google Gemini API
- `app/config.py`: Centralized configuration management
- `tests/`: Unit tests for the application

### Testing

Run the tests with pytest:

```bash
pytest tests/
```

For test coverage report:

```bash
pytest tests/ --cov=app --cov-report=term
```

### CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. Runs linting with flake8
2. Performs security checks with safety
3. Runs tests with coverage reporting
4. Builds and publishes the Docker image to GitHub Container Registry

## License

[MIT License](LICENSE)
