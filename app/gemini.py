import os
import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def ask_gemini(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        json=body
    )

    if response.status_code == 200:
        candidates = response.json().get("candidates", [])
        return candidates[0]["content"]["parts"][0]["text"] if candidates else "No response from Gemini."
    else:
        return f"Error: {response.text}"
