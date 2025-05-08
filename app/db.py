from sqlalchemy import create_engine, Column, Integer, Text, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

# Model for context chunks
class ContextChunk(Base):
    __tablename__ = "context_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False)  # e.g., "resume.pdf"
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # 384 = size of MiniLM embeddings

def init_db():
    Base.metadata.create_all(bind=engine)
