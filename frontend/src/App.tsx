import { useEffect, useRef, useState } from 'react';
import { FilesetResolver, GestureRecognizer, DrawingUtils } from '@mediapipe/tasks-vision';
import { gestureLabelToWord, detectTwoHandCombo, detectCustomGesture } from './lib/gestureUtils';
import { Mic, Loader2 } from 'lucide-react';
import './index.css';

const HOLD_THRESHOLD = 20;

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [activeWord, setActiveWord] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [sentenceBuffer, setSentenceBuffer] = useState<string[]>([]);
  const [holdProgress, setHoldProgress] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const recognizerRef = useRef<GestureRecognizer | null>(null);
  const holdCounterRef = useRef(0);
  const lastGestureRef = useRef<string | null>(null);
  const wordAddedRef = useRef(false);
  const sentenceRef = useRef<string[]>([]);
  const smoothedLandmarksRef = useRef<any[]>([]);
  
  useEffect(() => {
    sentenceRef.current = sentenceBuffer;
  }, [sentenceBuffer]);

  useEffect(() => {
    async function initMediaPipe() {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      
      const recognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.7,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.6
      });
      
      recognizerRef.current = recognizer;
      startCamera();
    }
    initMediaPipe();
  }, []);

  const startCamera = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: { ideal: 1920 }, 
            height: { ideal: 1080 },
            frameRate: { ideal: 60 }
          }
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      } catch (err) {
        console.error("Camera access denied or unavailable", err);
      }
    }
  };

  const speakSentence = async () => {
    if (sentenceRef.current.length === 0 || isSpeaking || isProcessing) return;
    
    setIsProcessing(true);
    const wordsToSpeak = [...sentenceRef.current];
    setSentenceBuffer([]);
    holdCounterRef.current = 0;
    wordAddedRef.current = true;

    try {
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const response = await fetch(`${API_URL}/api/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ words: wordsToSpeak })
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        setIsProcessing(false);
        setIsSpeaking(true);
        audio.onended = () => {
          setIsSpeaking(false);
          URL.revokeObjectURL(url);
        };
        audio.play();
      } else {
        setIsProcessing(false);
        setIsSpeaking(false);
      }
    } catch (e) {
      console.error(e);
      setIsProcessing(false);
      setIsSpeaking(false);
    }
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.code === 'Space') {
      speakSentence();
    } else if (e.code === 'KeyC') {
      setSentenceBuffer([]);
      holdCounterRef.current = 0;
      wordAddedRef.current = true;
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  const drawLandmarks = (ctx: CanvasRenderingContext2D, originalLandmarks: any) => {
    // Mirror the X coordinates for drawing on the mirrored canvas
    const landmarks = originalLandmarks.map((lm: any) => ({
      ...lm,
      x: 1.0 - lm.x
    }));
    
    const connections = GestureRecognizer.HAND_CONNECTIONS;
    const drawingUtils = new DrawingUtils(ctx);
    
    ctx.lineWidth = 3;
    drawingUtils.drawConnectors(landmarks, connections, {
      color: "#8A8D98",
      lineWidth: 3
    });
    
    drawingUtils.drawLandmarks(landmarks, {
      color: "#2EC4B6",
      lineWidth: 2,
      fillColor: "#121216",
      radius: (data) => { return data.from ? 4 : 5; }
    });
  };

  const drawBoundingBox = (ctx: CanvasRenderingContext2D, originalLandmarks: any, w: number, h: number, word: string, score: number) => {
    let minX = w, minY = h, maxX = 0, maxY = 0;
    for (const lm of originalLandmarks) {
      // Mirror the X coordinate for the bounding box
      const x = (1.0 - lm.x) * w;
      const y = lm.y * h;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    const pad = 30;
    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(w, maxX + pad);
    maxY = Math.min(h, maxY + pad);

    ctx.strokeStyle = "#2EC4B6";
    ctx.lineWidth = 2;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

    if (word) {
      const text = `${word} : ${Math.round(score * 100)}%`;
      ctx.font = "bold 16px Inter";
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = "#2EC4B6";
      ctx.fillRect(minX, minY - 30, tw + 20, 30);
      ctx.fillStyle = "#121216";
      ctx.fillText(text, minX + 10, minY - 10);
    }
  };

  useEffect(() => {
    let animationId: number;
    
    const renderLoop = async () => {
      if (videoRef.current && canvasRef.current && recognizerRef.current && videoRef.current.readyState >= 2) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        
        if (ctx) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const w = canvas.width;
          const h = canvas.height;

          // Draw video mirrored
          ctx.save();
          ctx.scale(-1, 1);
          ctx.drawImage(video, -w, 0, w, h);
          ctx.restore();

          const nowInMs = Date.now();
          const results = recognizerRef.current.recognizeForVideo(video, nowInMs);

          let currentWord = null;
          let currentConf = 0;
          let currentLandmarks = results.landmarks;

          if (results.landmarks && results.landmarks.length > 0) {
            // Smooth landmarks using EMA to prevent visual shaking
            if (smoothedLandmarksRef.current.length !== results.landmarks.length) {
              smoothedLandmarksRef.current = JSON.parse(JSON.stringify(results.landmarks));
            } else {
              const alpha = 0.35; // Smoothing factor
              for (let i = 0; i < results.landmarks.length; i++) {
                for (let j = 0; j < results.landmarks[i].length; j++) {
                  smoothedLandmarksRef.current[i][j].x = smoothedLandmarksRef.current[i][j].x * (1 - alpha) + results.landmarks[i][j].x * alpha;
                  smoothedLandmarksRef.current[i][j].y = smoothedLandmarksRef.current[i][j].y * (1 - alpha) + results.landmarks[i][j].y * alpha;
                }
              }
            }
            currentLandmarks = smoothedLandmarksRef.current;

            for (let i=0; i<currentLandmarks.length; i++) {
              drawLandmarks(ctx, currentLandmarks[i]);
            }

            if (results.gestures.length >= 2) {
              const l1 = results.gestures[0][0].categoryName;
              const l2 = results.gestures[1][0].categoryName;
              const combo = detectTwoHandCombo(l1, l2);
              if (combo) {
                currentWord = combo;
                currentConf = Math.min(results.gestures[0][0].score, results.gestures[1][0].score);
              }
            }

            if (!currentWord && results.gestures.length > 0) {
              const gestureObj = results.gestures[0][0];
              const mapped = gestureLabelToWord(gestureObj.categoryName);
              if (mapped) {
                currentWord = mapped;
                currentConf = gestureObj.score;
              }
            }

            if (!currentWord) {
              // Pass the smoothed, un-mirrored landmarks to the custom logic
              const custom = detectCustomGesture(currentLandmarks[0]);
              if (custom) {
                currentWord = custom;
                currentConf = 0.85;
              }
            }

            for (let i=0; i<currentLandmarks.length; i++) {
                drawBoundingBox(ctx, currentLandmarks[i], w, h, currentWord || "", currentConf);
            }
          } else {
            smoothedLandmarksRef.current = [];
          }

          if (currentWord && currentWord === lastGestureRef.current) {
            if (!wordAddedRef.current) holdCounterRef.current++;
          } else {
            holdCounterRef.current = 0;
            wordAddedRef.current = false;
          }

          lastGestureRef.current = currentWord;

          if (holdCounterRef.current >= HOLD_THRESHOLD && !wordAddedRef.current) {
            setSentenceBuffer(prev => [...prev, currentWord as string]);
            wordAddedRef.current = true;
          }

          setActiveWord(currentWord);
          setConfidence(currentConf);
          setHoldProgress(holdCounterRef.current);
        }
      }
      animationId = requestAnimationFrame(renderLoop);
    };
    
    renderLoop();
    return () => cancelAnimationFrame(animationId);
  }, []);

  const progressRatio = Math.min(holdProgress / HOLD_THRESHOLD, 1.0);

  return (
    <div className="app-container">
      {/* LEFT: Camera Viewport */}
      <div className="main-viewport">
        {/* IMPORTANT: autoPlay and muted are required for the camera stream to play without user interaction */}
        <video ref={videoRef} className="video-hidden" autoPlay muted playsInline />
        <canvas ref={canvasRef} className="canvas-layer" />
        
        {/* Top Overlay */}
        <div className="top-overlay glass-overlay">
          <div className="badge-container">
            {activeWord ? (
              <>
                <div 
                  className="active-gesture-badge" 
                  style={{ background: progressRatio < 1 ? 'var(--badge-bg)' : 'var(--badge-hold)' }}
                >
                  <h1>{activeWord}</h1>
                </div>
                <div className="confidence-bar-container">
                   <div 
                     className="confidence-fill"
                     style={{
                        width: `${confidence * 100}%`,
                        background: confidence < 0.4 ? 'var(--red)' : confidence < 0.7 ? 'var(--accent)' : 'var(--green)'
                     }}
                   />
                   <span className="confidence-text">{activeWord} : {Math.round(confidence * 100)}%</span>
                </div>
              </>
            ) : (
              <span className="empty-text">No gesture detected</span>
            )}
          </div>
          
          {/* Circular Progress */}
          <div className="circular-progress">
             <svg viewBox="0 0 50 50">
                <circle cx="25" cy="25" r="22" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="4" />
                <circle 
                  className="progress-ring-circle"
                  cx="25" cy="25" r="22" 
                  fill="none" 
                  stroke={progressRatio < 0.5 ? 'var(--cyan)' : 'var(--green)'} 
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeDasharray="138.23"
                  strokeDashoffset={138.23 - (progressRatio * 138.23)}
                />
             </svg>
             <span>{Math.round(progressRatio * 100)}%</span>
          </div>
        </div>

        {/* Bottom Overlay */}
        <div className="bottom-overlay glass-overlay">
          <div className="sentence-buffer">
             {sentenceBuffer.length > 0 ? (
               sentenceBuffer.map((w, i) => (
                 <span key={i} className="chip">{w}</span>
               ))
             ) : (
                !isSpeaking && !isProcessing && <span className="empty-text">Show a gesture to start building a sentence...</span>
              )}
           </div>
           
           {isProcessing && (
             <div className="processing-badge processing-pulse">
               <Loader2 size={20} className="spinner" />
               PROCESSING...
             </div>
           )}

           {isSpeaking && (
             <div className="speaking-badge speaking-pulse">
               <Mic size={20} />
               SPEAKING
             </div>
           )}
        </div>
      </div>

      {/* RIGHT: Gesture Guide Panel */}
      <div className="side-panel glass-panel">
        <div className="side-panel-content">
          <h2 className="side-panel-title">GESTURE GUIDE</h2>
          <div className="side-panel-divider" />
          
          <div className="gesture-list">
            {[
              ["Thumb Up", "Yes"], ["Thumb Down", "No"], ["Open Palm", "Hello"],
              ["Closed Fist", "Help"], ["Point Up", "Attention"], ["Victory Sign", "Water"],
              ["Spread Hand", "Please"], ["Pinch", "Pain"], ["Three Fingers", "Food"],
              ["Pinky Only", "Bathroom"], ["Shaka", "Medicine"], ["Horns", "Me"],
              ["Finger Gun", "Go"], ["Both Fists", "Emergency"], ["Palm + Thumb", "Hello"]
            ].map(([g, w]) => {
              const active = activeWord === w;
              return (
                <div key={g} className={`gesture-row ${active ? 'active-row' : ''}`}>
                  <span className="gesture-row-name">{g}</span>
                  <span className="gesture-row-word">{w}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="side-panel-footer">
           <div className="shortcut-row">
             <div className="shortcut-key">SPACE</div>
             <span>Speak sentence</span>
           </div>
           <div className="shortcut-row">
             <div className="shortcut-key">C</div>
             <span>Clear buffer</span>
           </div>
        </div>
      </div>
    </div>
  );
}
