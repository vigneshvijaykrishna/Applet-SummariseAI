from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import logging
import os

# current model is 'llama-3.1-8b-instant'
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
allow_methods=["GET", "POST", "OPTIONS"], # Explicitly list these
    allow_headers=["*"],
)

# --- API KEY FROM ENV ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment variables")

class SummaryRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize_text(request: SummaryRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Summarize this text into 3 clear bullet points."},
            {"role": "user", "content": request.text}
        ]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10.0
            )

            if response.status_code != 200:
                logging.error(response.text)
                raise HTTPException(status_code=500, detail="AI Service Error")

            result = response.json()
            return {"summary": result["choices"][0]["message"]["content"]}

        except Exception as e:
            logging.error(str(e))
            raise HTTPException(status_code=500, detail="Internal Server Error")
