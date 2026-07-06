# SignSpeak

SignSpeak is a web app that translates hand gestures into spoken English in real-time. We built this to help people who are non-verbal or have speech impairments communicate more easily.

It uses your webcam to track hand movements, and then passes those gestures to Google's Gemini AI, which puts together a natural-sounding sentence and speaks it out loud.

## How we built it
- **Frontend:** React and Vite. We use Google MediaPipe for fast, on-device hand tracking right in the browser.
- **Backend:** Python and FastAPI. This handles the Gemini API integration to build the sentences, and uses Microsoft Edge TTS to generate the voice.

## Use Cases
- Giving a voice to people with conditions like ALS, cerebral palsy, or vocal cord damage.
- Communicating in places where speaking isn't possible (like super loud environments).
- Helping patients in intensive care easily tell their caregivers what they need (like "Food", "Water", or "Pain").

## How to run it locally

You'll need Python (3.9+) and Node.js installed on your computer.

### 1. Start the Backend
Open a terminal, go into the `backend` folder, and install the Python dependencies:
```bash
pip install -r requirements.txt
```
Then start the server:
```bash
python -m uvicorn app:app --reload
```
*(Leave this terminal running in the background!)*

### 2. Start the Frontend
Open a new terminal, go into the `frontend` folder, and install the node packages:
```bash
npm install
```
Then start the React app:
```bash
npm run dev
```

### 3. Using the App
1. Go to `http://localhost:5173` in your browser.
2. Allow webcam access when prompted.
3. Show your hand to the camera to select gestures from the right side of the screen. The app will string the words together at the bottom.
4. **Hit the SPACEBAR** to have the AI translate your gestures into a complete sentence and speak it!
5. **Press 'C'** if you want to clear your current words and start over.

*Note: Since the backend uses Gemini, you might be asked to paste in your Gemini API key in the terminal the first time you run it.*
