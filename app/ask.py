from sqlalchemy.sql import text
from app.db import SessionLocal, ContextChunk
from app.gemini import ask_gemini
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

def get_relevant_context(question, top_k=5):
    db = SessionLocal()
    q_embedding = model.encode([question]).tolist()[0]

    # Cast the embedding parameter to vector type
    sql = text("""
        SELECT content FROM context_chunks
        ORDER BY embedding <-> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    results = db.execute(sql, {"embedding": q_embedding, "top_k": top_k}).fetchall()
    db.close()
    return "\n\n".join([row[0] for row in results])

def handle_question(question):
    context = get_relevant_context(question)
    return ask_gemini(question, context)