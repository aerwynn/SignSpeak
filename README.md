# GestureVoice AI (SignSpeak)

**GestureVoice AI** is a real-time, browser-based accessibility application designed to bridge the communication gap for individuals who are non-verbal or have speech impairments. By leveraging advanced computer vision and Large Language Models, the application translates physical hand gestures into natural, contextually appropriate spoken English.

## 🌟 Key Features
- **Real-Time Hand Tracking:** Utilizes MediaPipe for 60 FPS on-device hand tracking via WebGL.
- **AI Sentence Construction:** Uses Google Gemini AI to take raw gesture inputs and construct grammatically correct, natural-sounding sentences.
- **Text-to-Speech:** Features Microsoft Edge TTS for high-quality, human-like voice synthesis.
- **Modern UI:** Built with React and Vite featuring a clean, accessible glassmorphism interface.

## 💡 Primary Use Cases
- **Assistive Technology:** Provides a voice for individuals with conditions such as ALS, cerebral palsy, or temporary vocal cord damage.
- **Non-Verbal Communication:** Allows seamless communication in environments where speaking is not possible or appropriate.
- **Medical & Caregiver Settings:** Enables patients in intensive care to communicate fundamental needs (e.g., "Food", "Water", "Pain") instantly.

## 🛠️ Tech Stack
- **Frontend:** React, Vite, Google MediaPipe
- **Backend:** Python, FastAPI, Google Gemini API, Microsoft Edge TTS

## 🚀 Setup Instructions

### Prerequisites
- Python (version 3.9 or higher)
- Node.js

### 1. Set up the Python Backend
1. Open a terminal and navigate to the `backend` folder.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Python server:
   ```bash
   python -m uvicorn app:app --reload
   ```
*(Keep this terminal open, it needs to stay running!)*

### 2. Set up the React Frontend
1. Open a second, new terminal window and navigate to the `frontend` folder.
2. Install the necessary Node packages:
   ```bash
   npm install
   ```
3. Start the website:
   ```bash
   npm run dev
   ```

## 🎮 How to Use
1. Open your web browser and go to `http://localhost:5173`.
2. Allow the browser to access your webcam.
3. Show your hand to the camera to build a sentence using the gestures shown on the right side of the screen. The application will track your hand landmarks and accumulate the translated words into a sentence buffer.
4. **Press `SPACEBAR`** to have the AI translate your gestures into a natural English sentence and speak it out loud!
5. **Press `C`** to clear your current gesture buffer.

*Note: Since the backend uses Gemini AI, you will need a valid Gemini API key. The application will prompt you to enter one in the terminal the first time you run it.*
