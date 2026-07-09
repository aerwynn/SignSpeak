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
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ──────────────────────────────────────────────────────────
LLM_TIMEOUT_SECONDS = 5  # Hard timeout for LLM race

# ── Load Gemini API Key ─────────────────────────────────────────────
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

# ── Load Groq API Key ──────────────────────────────────────────────
GROQ_KEY_FILE = "../api_key_groq.txt"
GROQ_API_KEY = ""
if os.path.exists(GROQ_KEY_FILE):
    try:
        with open(GROQ_KEY_FILE, "r", encoding="utf-8") as f:
            GROQ_API_KEY = f.read().strip()
    except Exception:
        pass

if not GROQ_API_KEY:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

if GROQ_API_KEY:
    print("[INFO] Groq API key loaded.")
else:
    print("[WARN] No Groq API key found.")

# ── Shared Prompt Builder ───────────────────────────────────────────
def _build_prompt(raw_words: str) -> str:
    return (
        "You are an AI assistant powering 'SignSpeak', a real-time web application "
        "that translates hand gestures into spoken English to help non-verbal individuals communicate. "
        "The user has signed the following raw keywords in order using hand gestures:\n\n"
        f"  {raw_words}\n\n"
        "Your task is to convert these raw keywords into a single, grammatically correct English sentence. "
        "Follow these rules strictly:\n"
        "1. Make the sentence simple, natural, and logical to speak out loud.\n"
        "2. If the words include a list of items/needs (e.g., Water, Food, Medicine), connect them appropriately (e.g., 'I need water, food, and medicine.').\n"
        "3. Words like 'Emergency', 'Help', 'Attention', or 'Hello' should be treated as independent exclamations or alerts, usually punctuated separately. (e.g., 'Its an emergency! I need medicine.' or 'Hello, attention please! I need water.').\n"
        "4. Do NOT merge an alert word into a noun phrase. For example, 'Emergency Medicine' should become 'Emergency! I need medicine.', NOT 'I need emergency medicine.'\n"
        "5. Do not hallucinate bizarre contexts or merge unrelated concepts (e.g., do NOT say 'medicine for the water').\n"
        "6. Output ONLY the final spoken sentence. Do not include quotes, greetings, or explanations."
    )

# ── Gemini Call (async) ─────────────────────────────────────────────
async def _call_gemini(raw_words: str) -> str:
    """Call Gemini API in a thread pool (its SDK is synchronous)."""
    if not _gemini_client:
        raise RuntimeError("Gemini client not available")

    prompt = _build_prompt(raw_words)

    def _sync_call():
        response = _gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        return response.text.strip()

    result = await asyncio.get_event_loop().run_in_executor(None, _sync_call)
    if not result:
        raise RuntimeError("Gemini returned empty response")
    print(f"[GEMINI] \"{raw_words}\" → \"{result}\"")
    return result

# ── Groq Call (async) ───────────────────────────────────────────────
async def _call_groq(raw_words: str) -> str:
    """Call Groq API via httpx (fully async, no SDK needed)."""
    if not GROQ_API_KEY:
        raise RuntimeError("Groq API key not available")

    prompt = _build_prompt(raw_words)

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 60,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        data = response.json()
        result = data["choices"][0]["message"]["content"].strip()

    if not result:
        raise RuntimeError("Groq returned empty response")
    print(f"[GROQ] \"{raw_words}\" → \"{result}\"")
    return result

# ── Race Both APIs ──────────────────────────────────────────────────
async def _race_llm_apis(raw_words: str) -> str:
    """
    Fire Gemini and Groq concurrently. Return the first successful response.
    If both fail or neither responds within LLM_TIMEOUT_SECONDS, return raw_words.
    """
    tasks = []
    if _gemini_client:
        tasks.append(asyncio.create_task(_call_gemini(raw_words)))
    if GROQ_API_KEY:
        tasks.append(asyncio.create_task(_call_groq(raw_words)))

    if not tasks:
        print("[FALLBACK] No LLM APIs available, using raw words.")
        return raw_words

    pending = set(tasks)
    loop = asyncio.get_event_loop()
    start_time = loop.time()
    
    while pending:
        elapsed = loop.time() - start_time
        timeout = max(0, LLM_TIMEOUT_SECONDS - elapsed)
        if timeout <= 0:
            print("[TIMEOUT] LLM APIs timed out.")
            break
            
        done, pending = await asyncio.wait(
            pending,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in done:
            try:
                result = task.result()
                if result:
                    # Success! Cancel remaining and return
                    for p in pending:
                        p.cancel()
                    return result
            except Exception as e:
                print(f"[LLM ERROR] A task failed: {e}")
                
    # If we get here, all tasks either failed or timed out
    for p in pending:
        p.cancel()
        
    print(f"[FALLBACK] Using raw words: \"{raw_words}\"")
    return raw_words


class SpeakRequest(BaseModel):
    words: List[str]

@app.post("/api/speak")
async def speak(request: SpeakRequest):
    if not request.words:
        raise HTTPException(status_code=400, detail="No words provided")
    
    raw_words = " ".join(request.words)
    print(f"Received request for words: {raw_words}")
    
    # 1. Race LLM APIs for the natural sentence (5s timeout)
    sentence = await _race_llm_apis(raw_words)
    
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
