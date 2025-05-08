from fastapi import FastAPI
from pydantic import BaseModel
from app.db import init_db
from app.context_loader import load_and_store_context
from app.ask import handle_question
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    return {"response": handle_question(req.question)}

if __name__ == "__main__":
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully.")
        logger.info("Loading PDF context...")
        load_and_store_context(r'D:\Downloads\resume.pdf')
        logger.info("PDF context loaded successfully.")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise