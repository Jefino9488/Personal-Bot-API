from sqlalchemy import create_engine, Column, Integer, Text, String, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv
import logging
from contextlib import contextmanager
from sqlalchemy.engine.url import make_url

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
is_sqlite = make_url(DATABASE_URL).get_backend_name() == "sqlite"

# Configure engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT,
    pool_pre_ping=True,  # Verify connections before using them
    echo=False,  # Set to True for SQL query logging
)

# Create scoped session factory
SessionFactory = sessionmaker(bind=engine)
SessionLocal = scoped_session(SessionFactory)

Base = declarative_base()

# Model for context chunks
class ContextChunk(Base):
    __tablename__ = "context_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False)  # e.g., "resume.pdf"
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384) if not is_sqlite else String)  # fallback for SQLite

    # Create an index on the source column for faster filtering
    __table_args__ = (
        Index('idx_source', 'source'),
    )

@contextmanager
def get_db():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        db.close()

def init_db():
    """Initialize database tables and indexes."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise
