import os
import asyncio
import uuid
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
import edge_tts
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Gemini API Key
KEY_FILE = "../api_key.txt"
GEMINI_API_KEY = ""
if os.path.exists(KEY_FILE):
    try:
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            GEMINI_API_KEY = f.read().strip()
    except Exception:
        pass

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

_gemini_client = None
if GEMINI_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[INFO] Gemini API connected.")
    except Exception as e:
        print(f"[WARN] Could not init Gemini: {e}")

class SpeakRequest(BaseModel):
    words: List[str]

def _build_sentence_with_gemini(raw_words: str) -> str:
    if not _gemini_client:
        return raw_words

    prompt = (
        "You are a helpful assistant for a gesture-to-speech app used by "
        "people who communicate through hand gestures. The user has signed "
        "the following words in order:\n\n"
        f"  {raw_words}\n\n"
        "Turn them into ONE short, natural English sentence (max 15 words). "
        "Make the sentence as simple as possible without losing meaning, and ensure it is grammatically correct. "
        "Do NOT add extra meaning the words don't imply. "
        "Reply with ONLY the sentence, no quotes, no explanation."
    )

    try:
        response = _gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        sentence = response.text.strip()
        if sentence:
            print(f"[GEMINI] \"{raw_words}\" → \"{sentence}\"")
            return sentence
    except Exception as e:
        print(f"[GEMINI ERROR] {e}")

    return raw_words

@app.post("/api/speak")
async def speak(request: SpeakRequest):
    if not request.words:
        raise HTTPException(status_code=400, detail="No words provided")
    
    raw_words = " ".join(request.words)
    print(f"Received request for words: {raw_words}")
    
    # 1. Ask Gemini for the natural sentence
    sentence = _build_sentence_with_gemini(raw_words)
    
    # 2. Generate Audio via edge-tts
    try:
        communicate = edge_tts.Communicate(sentence, "en-US-GuyNeural")
        
        # Save to a temporary file
        temp_dir = tempfile.gettempdir()
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(temp_dir, filename)
        
        await communicate.save(filepath)
        
        # Return the file as an attachment
        return FileResponse(filepath, media_type="audio/mpeg", filename=filename)
        
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
