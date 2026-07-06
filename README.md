# SignSpeak

SignSpeak is a real-time web application that translates hand gestures into spoken English. I built this to help non-verbal individuals or those with speech impairments communicate more efficiently using computer vision and LLMs.

It uses a device's webcam to track hand landmarks locally in the browser, accumulates recognized gestures into a buffer, and passes them to Google's Gemini AI. Gemini processes the raw gesture sequence into a contextually accurate, grammatically correct sentence, which is then synthesized into speech.

## System Architecture

The project is split into a client-side React SPA for real-time inference and a Python backend for heavy LLM and TTS processing.

### Frontend (Client-side Inference)
- **Framework:** React 19, TypeScript, and Vite.
- **Computer Vision:** We use `@mediapipe/tasks-vision` for on-device hand tracking. Running inference directly in the browser via WebGL allows for real-time gesture recognition without any network latency.
- **State Management:** The React app maintains a buffer of recognized words. Once the user hits the spacebar, the buffer is dispatched to the backend.

### Backend (LLM & TTS Pipeline)
- **API Layer:** Built with FastAPI (Python 3.9+) to provide a lightweight, asynchronous REST interface.
- **Natural Language Processing:** Uses the `google-genai` SDK (`gemini-3.1-flash-lite` model). When the backend receives an array of raw words (e.g., `["I", "hungry", "food"]`), it prompts Gemini to construct a natural English sentence (`"I am hungry and would like some food."`).
- **Voice Synthesis:** We use `edge-tts` to convert the generated sentence into a high-quality audio file (mp3) which is streamed back to the frontend for playback.

## Use Cases
- Giving a voice to people with conditions like ALS, cerebral palsy, or vocal cord damage.
- Communicating in places where speaking isn't possible (like super loud environments).
- Helping patients in intensive care easily tell their caregivers what they need (like "Food", "Water", or "Pain").

## Running the project locally

You'll need Python (3.9+) and Node.js installed on your computer.

### 1. Start the Backend API
Open a terminal, go into the `backend` folder, and install the Python dependencies:
```bash
pip install -r requirements.txt
```
Make sure you have your Gemini API key ready. You can either export it as an environment variable (`GEMINI_API_KEY`) or the app will prompt you for it.

Start the FastAPI server (it runs on port 8000 by default):
```bash
python -m uvicorn app:app --reload
```
*(Leave this terminal running in the background!)*

### 2. Start the Frontend
Open a new terminal, go into the `frontend` folder, and install the dependencies:
```bash
npm install
```
Start the Vite development server:
```bash
npm run dev
```

### 3. Usage
1. Go to `http://localhost:5173` in your browser.
2. Allow webcam access when prompted.
3. Show your hand to the camera to select gestures from the right side of the screen. The app will string the words together at the bottom.
4. **Hit the SPACEBAR** to send your gesture buffer to the backend for AI translation and audio playback!
5. **Press 'C'** if you want to clear your current words and start over.
