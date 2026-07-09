import time
import threading
import asyncio
import uuid
import os
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
import edge_tts
import pygame

# Initialize pygame mixer for audio playback silently
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
pygame.mixer.init()

GESTURE_WORD_MAP = {
    "Closed_Fist":  "Help",
    "Open_Palm":    "Hello",
    "Pointing_Up":  "Attention",
    "Thumb_Down":   "No",
    "Thumb_Up":     "Yes",
    "Victory":      "Water",
    "ILoveYou":     "Please",
}

def gesture_label_to_word(label):
    return GESTURE_WORD_MAP.get(label, None)


# ── Two-Hand Combo Map ─────────────────────────────────────────────
TWO_HAND_COMBOS = [
    ({"Closed_Fist"},              "Emergency"),
    ({"Open_Palm", "Thumb_Up"},    "I'm Okay"),
]


def detect_two_hand_combo(label1, label2):
    """Check if two MediaPipe gesture labels form a known two-hand combo."""
    pair = {label1, label2}
    for combo_set, word in TWO_HAND_COMBOS:
        if pair == combo_set:
            return word
    return None


# ── Finger-State Helpers ───────────────────────────────────────────
def get_finger_states(landmarks):
    """
    Determine which fingers are extended based on landmark positions.
    Returns [thumb, index, middle, ring, pinky] as booleans.
    """
    fingers = []

    # Thumb: extended if tip (4) is farther from index_mcp (5) than ip (3)
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    index_mcp = landmarks[5]
    dist_tip = math.hypot(thumb_tip.x - index_mcp.x,
                          thumb_tip.y - index_mcp.y)
    dist_ip  = math.hypot(thumb_ip.x  - index_mcp.x,
                          thumb_ip.y  - index_mcp.y)
    fingers.append(dist_tip > dist_ip)

    # Index (8/6), Middle (12/10), Ring (16/14), Pinky (20/18)
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip_id, pip_id in zip(tip_ids, pip_ids):
        fingers.append(landmarks[tip_id].y < landmarks[pip_id].y)

    return fingers


def _is_pinching(landmarks):
    """Check if thumb and index tips are close relative to hand size."""
    thumb_tip  = landmarks[4]
    index_tip  = landmarks[8]
    wrist      = landmarks[0]
    middle_mcp = landmarks[9]

    pinch_dist = math.hypot(thumb_tip.x - index_tip.x,
                            thumb_tip.y - index_tip.y)
    hand_size  = math.hypot(wrist.x - middle_mcp.x,
                            wrist.y - middle_mcp.y)

    if hand_size < 0.01:
        return False
    return (pinch_dist / hand_size) < 0.25


def detect_custom_gesture(landmarks):
    """
    Detect custom gestures via finger-state analysis.
    Called only when MediaPipe's built-in recogniser returns no match.
    """
    if _is_pinching(landmarks):
        return "Pain"

    thumb, index, middle, ring, pinky = get_finger_states(landmarks)

    if not thumb and index and middle and ring and not pinky:
        return "Food"
    if not thumb and not index and not middle and not ring and pinky:
        return "Bathroom"
    if thumb and not index and not middle and not ring and pinky:
        return "Medicine"
    if not thumb and index and not middle and not ring and pinky:
        return "Me"
    if thumb and index and not middle and not ring and not pinky:
        return "Go"

    return None


is_speaking = False

_gemini_client = None
KEY_FILE = "api_key.txt"

def load_or_prompt_api_key():
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r", encoding="utf-8") as f:
                key = f.read().strip()
                if key:
                    return key
        except Exception:
            pass

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        try:
            with open(KEY_FILE, "w", encoding="utf-8") as f:
                f.write(env_key)
        except Exception:
            pass
        return env_key

    print("\n" + "="*70)
    print(" GEMINI API KEY SETUP")
    print("="*70)
    print("To enable AI-enhanced sentences, please enter your Gemini API key.")
    print("This will be saved to 'api_key.txt' so you only have to do this once.")
    print("If you want to skip and run offline (raw words only), just press Enter.")
    print("="*70)
    
    try:
        user_key = input("Enter Gemini API Key: ").strip()
        if user_key:
            with open(KEY_FILE, "w", encoding="utf-8") as f:
                f.write(user_key)
            print(f"[INFO] API key saved to '{KEY_FILE}' successfully!\n")
            return user_key
    except Exception as e:
        print(f"[WARN] Could not save API key: {e}")
        
    print("[INFO] Running in offline mode (speaking raw words).\n")
    return ""

GEMINI_API_KEY = load_or_prompt_api_key()

if GEMINI_API_KEY:
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[INFO] Gemini API connected — sentences will be AI-enhanced.")
    except Exception as e:
        print(f"[WARN] Could not init Gemini: {e}. Falling back to raw words.")
else:
    print("[INFO] No Gemini API key provided. Sentences will be spoken as raw words.")


def _build_sentence_with_gemini(raw_words):
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
        print(f"[GEMINI ERROR] {e} — falling back to raw words.")

    return raw_words


async def _generate_and_play_tts(text):
    global is_speaking
    try:
        # High quality Azure/Edge male voice
        communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
        
        # Use a unique filename to avoid locking issues if spoken rapidly
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        await communicate.save(filename)
        
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
            
        pygame.mixer.music.unload()
        try:
            os.remove(filename)
        except Exception:
            pass
            
    except Exception as e:
        print(f"[TTS ERROR] {e}")
    finally:
        is_speaking = False


def _tts_worker(raw_words):
    text = _build_sentence_with_gemini(raw_words)
    asyncio.run(_generate_and_play_tts(text))


def speak_sentence(sentence_buffer):
    global is_speaking
    if not sentence_buffer or is_speaking:
        return sentence_buffer

    raw_words = " ".join(sentence_buffer)
    print(f"[TTS] Words: \"{raw_words}\"")

    is_speaking = True
    t = threading.Thread(target=_tts_worker, args=(raw_words,), daemon=True)
    t.start()

    return []


# ═══════════════════════════════════════════════════════════════════
#   DISPLAY CONSTANTS
# ═══════════════════════════════════════════════════════════════════

DISPLAY_H = 1080
DISPLAY_W = 1920
PANEL_W   = 420
CAM_DISPLAY_W = DISPLAY_W - PANEL_W   # 1500
CAM_DISPLAY_H = DISPLAY_H             # 1080

# Colour palette (BGR)
COL_BG_DARK    = (25, 25, 30)
COL_BG_PANEL   = (30, 30, 38)
COL_ACCENT     = (50, 160, 250)
COL_GREEN      = (100, 220, 80)
COL_CYAN       = (220, 200, 60)
COL_WHITE      = (240, 240, 245)
COL_GRAY       = (130, 130, 140)
COL_PROGRESS   = (80, 220, 120)
COL_PROG_BG    = (60, 60, 70)
COL_SENTENCE   = (80, 180, 255)
COL_SPEAKING   = (80, 200, 255)
COL_BADGE_BG   = (55, 120, 60)
COL_BADGE_HOLD = (40, 90, 180)
COL_RED        = (80, 80, 230)
COL_JOINT      = (80, 255, 160)
COL_BOX        = (100, 255, 100)

FINGER_COLORS = [
    (100, 100, 255),   # thumb
    (100, 220, 255),   # index
    (100, 255, 100),   # middle
    (255, 220, 100),   # ring
    (255, 100, 200),   # pinky
]

CV_FONT = cv2.FONT_HERSHEY_DUPLEX

# ── PIL Sans-Serif Fonts for PRE-RENDERED STATIC PANEL ────────────
def _load_font(size, bold=False):
    candidates = (
        [("segoeuib.ttf", "segoeui.ttf"),
         ("arialbd.ttf",  "arial.ttf"),
         ("calibrib.ttf", "calibri.ttf")]
    )
    for bold_name, regular_name in candidates:
        name = bold_name if bold else regular_name
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()

FONT_SM      = _load_font(18)
FONT_MD      = _load_font(24)
FONT_MD_BOLD = _load_font(24, bold=True)
FONT_LG_BOLD = _load_font(32, bold=True)

GESTURE_GUIDE = [
    ("Thumb Up",      "Yes"),
    ("Thumb Down",    "No"),
    ("Open Palm",     "Hello"),
    ("Closed Fist",   "Help"),
    ("Point Up",      "Attention"),
    ("Victory Sign",  "Water"),
    ("Spread Hand",   "Please"),
    ("Pinch",         "Pain"),
    ("Three Fingers", "Food"),
    ("Pinky Only",    "Bathroom"),
    ("Shaka",         "Medicine"),
    ("Horns",         "Me"),
    ("Finger Gun",    "Go"),
    ("Both Fists",    "Emergency"),
    ("Palm + Thumb",  "I'm Okay"),
]

def _bgr_to_rgb(bgr):
    return (bgr[2], bgr[1], bgr[0])

def _pil_text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

# ═══════════════════════════════════════════════════════════════════
#   DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _pre_render_side_panel():
    """Renders the static text/background for the gesture guide ONCE to fix lag."""
    h, w = DISPLAY_H, PANEL_W
    # Create empty solid background
    panel_np = np.zeros((h, w, 3), dtype=np.uint8)
    panel_np[:] = COL_BG_PANEL
    
    # Convert to PIL to draw pretty text
    pil_img = Image.fromarray(cv2.cvtColor(panel_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Title
    draw.text((20, 18), "GESTURE GUIDE", font=FONT_LG_BOLD, fill=_bgr_to_rgb(COL_ACCENT))
    draw.line([(20, 58), (w - 20, 58)], fill=_bgr_to_rgb(COL_ACCENT), width=2)

    row_h = 38
    start_y = 75
    for i, (gesture, word) in enumerate(GESTURE_GUIDE):
        y = start_y + i * row_h
        draw.text((20, y), gesture, font=FONT_MD, fill=_bgr_to_rgb(COL_GRAY))
        tw, _ = _pil_text_size(draw, word, FONT_MD_BOLD)
        draw.text((w - tw - 20, y), word, font=FONT_MD_BOLD, fill=_bgr_to_rgb(COL_CYAN))

    # Keyboard shortcuts at the bottom
    shortcut_y = h - 80
    draw.line([(20, shortcut_y - 10), (w - 20, shortcut_y - 10)], fill=_bgr_to_rgb(COL_GRAY), width=1)
    sc_col = _bgr_to_rgb(COL_WHITE)
    draw.text((20, shortcut_y + 2),  "SPACE   Speak sentence", font=FONT_SM, fill=sc_col)
    draw.text((20, shortcut_y + 26), "C          Clear buffer", font=FONT_SM, fill=sc_col)
    draw.text((20, shortcut_y + 50), "Q          Quit", font=FONT_SM, fill=sc_col)

    # Convert back to BGR numpy array
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _overlay_rect(frame, pt1, pt2, color, alpha):
    """Draw a semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_landmarks_cv2(frame, landmarks):
    h, w, _ = frame.shape
    finger_groups = [
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        [(0, 5), (5, 6), (6, 7), (7, 8)],
        [(5, 9), (9, 10), (10, 11), (11, 12)],
        [(9, 13), (13, 14), (14, 15), (15, 16)],
        [(13, 17), (17, 18), (18, 19), (19, 20)],
    ]
    palm_connections = [(0, 17), (0, 5), (5, 9), (9, 13), (13, 17)]

    points = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))

    for s, e in palm_connections:
        cv2.line(frame, points[s], points[e], COL_GRAY, 2, cv2.LINE_AA)

    for fi, conns in enumerate(finger_groups):
        col = FINGER_COLORS[fi]
        for s, e in conns:
            cv2.line(frame, points[s], points[e], col, 2, cv2.LINE_AA)

    for i, pt in enumerate(points):
        radius = 6 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, pt, radius, COL_JOINT, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, radius, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hand_box(frame, landmarks, gesture_text, confidence):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

    pad = 28
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)

    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_BOX, 2, cv2.LINE_AA)

    if gesture_text:
        pct = int(confidence * 100)
        label = f"{gesture_text} : {pct}%"
        font_scale = 0.7
        (tw, th), baseline = cv2.getTextSize(label, CV_FONT, font_scale, 2)
        label_y1 = max(0, y1 - th - 14)
        label_y2 = y1
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 12, label_y2), COL_BOX, -1)
        cv2.putText(frame, label, (x1 + 6, label_y2 - 5),
                    CV_FONT, font_scale, COL_BG_DARK, 2, cv2.LINE_AA)


def _draw_confidence_bar(frame, x, y, bar_w, bar_h, word, confidence):
    pct = int(confidence * 100)
    fill_w = int(bar_w * min(confidence, 1.0))

    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), COL_PROG_BG, -1)
    if confidence < 0.4:
        col = COL_RED
    elif confidence < 0.7:
        col = COL_ACCENT
    else:
        col = COL_GREEN
    cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), col, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), COL_WHITE, 1)

    text = f"{word} : {pct}%"
    cv2.putText(frame, text, (x + 8, y + bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_WHITE, 1, cv2.LINE_AA)


def _draw_progress_arc(frame, cx, cy, radius, progress, target):
    cv2.circle(frame, (cx, cy), radius, COL_PROG_BG, 3, cv2.LINE_AA)

    if target <= 0:
        return
    ratio = min(progress / target, 1.0)
    angle = int(360 * ratio)

    if ratio < 0.5:
        r = ratio * 2
        col = (int(220 * (1 - r) + 80 * r),
               int(200 * (1 - r) + 220 * r),
               int(60  * (1 - r) + 120 * r))
    else:
        r = (ratio - 0.5) * 2
        col = (int(80  * (1 - r) + 50  * r),
               int(220 * (1 - r) + 220 * r),
               int(120 * (1 - r) + 250 * r))

    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius),
                    -90, 0, angle, col, 3, cv2.LINE_AA)

    pct = f"{int(ratio * 100)}%"
    (tw, th), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, pct, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)


def draw_ui(frame, panel_bg, active_word, confidence, sentence_buffer, hold_progress, hold_target):
    """Build the UI using fast OpenCV operations entirely."""
    cam_h, cam_w = CAM_DISPLAY_H, CAM_DISPLAY_W
    canvas = cv2.resize(frame, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)

    # Top Bar
    top_h = 90
    _overlay_rect(canvas, (0, 0), (cam_w, top_h), COL_BG_DARK, 0.65)

    if active_word:
        badge_text = active_word.upper()
        (tw, th), _ = cv2.getTextSize(badge_text, CV_FONT, 1.2, 2)
        pad = 14
        bx1, by1 = 18, 12
        bx2 = bx1 + tw + pad * 2 + 6
        by2 = by1 + th + pad * 2
        bg_col = COL_BADGE_BG if hold_progress < hold_target else COL_BADGE_HOLD
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg_col, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), COL_GREEN, 2)
        cv2.putText(canvas, badge_text, (bx1 + pad + 3, by2 - pad + 4),
                    CV_FONT, 1.2, COL_WHITE, 2, cv2.LINE_AA)
        
        if confidence > 0:
            _draw_confidence_bar(canvas, 18, top_h - 28, 320, 22, active_word, confidence)
    else:
        cv2.putText(canvas, "No gesture detected", (22, 45), CV_FONT, 0.8, COL_GRAY, 1, cv2.LINE_AA)

    arc_cx = cam_w - 60
    arc_cy = 46
    _draw_progress_arc(canvas, arc_cx, arc_cy, 30, hold_progress, hold_target)

    # Bottom Sentence Bar
    bot_h = 64
    _overlay_rect(canvas, (0, cam_h - bot_h), (cam_w, cam_h), COL_BG_DARK, 0.7)

    if sentence_buffer:
        chip_x = 18
        chip_y = cam_h - bot_h + 16
        for word in sentence_buffer:
            (tw, th), _ = cv2.getTextSize(word, CV_FONT, 0.7, 1)
            chip_w = tw + 24
            chip_h = th + 18
            cv2.rectangle(canvas, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), COL_SENTENCE, -1)
            cv2.rectangle(canvas, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), (60, 140, 220), 1)
            cv2.putText(canvas, word, (chip_x + 12, chip_y + th + 9), CV_FONT, 0.7, COL_WHITE, 1, cv2.LINE_AA)
            chip_x += chip_w + 12
            if chip_x > cam_w - 180:
                break
    else:
        if not is_speaking:
            cv2.putText(canvas, "Show a gesture to start building a sentence...", (22, cam_h - 22), CV_FONT, 0.6, COL_GRAY, 1, cv2.LINE_AA)

    if is_speaking:
        badge = "SPEAKING"
        (bw, bh), _ = cv2.getTextSize(badge, CV_FONT, 0.8, 2)
        bx = cam_w - bw - 30
        by = cam_h - bot_h + 16
        cv2.rectangle(canvas, (bx - 12, by), (bx + bw + 12, by + bh + 16), COL_SPEAKING, -1)
        cv2.putText(canvas, badge, (bx, by + bh + 8), CV_FONT, 0.8, COL_BG_DARK, 2, cv2.LINE_AA)

    # Fast Horizontal Concat to attach side panel
    full = cv2.hconcat([canvas, panel_bg.copy()])

    # Apply active row highlight directly onto the full canvas (right side)
    if active_word:
        row_h = 38
        start_y = 75
        for i, (gesture, word) in enumerate(GESTURE_GUIDE):
            if word == active_word:
                y = start_y + i * row_h
                _overlay_rect(full, (cam_w + 6, y - 4), (DISPLAY_W - 6, y + row_h - 10), COL_BADGE_BG, 0.55)
                # Redraw white active text over highlight using OpenCV font so it pops out
                cv2.putText(full, gesture, (cam_w + 20, y + 24), cv2.FONT_HERSHEY_DUPLEX, 0.75, COL_WHITE, 1, cv2.LINE_AA)
                
                (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
                cv2.putText(full, word, (DISPLAY_W - tw - 20, y + 24), cv2.FONT_HERSHEY_DUPLEX, 0.8, COL_GREEN, 2, cv2.LINE_AA)
                break

    return full


def ensure_model_exists():
    model_path = "gesture_recognizer.task"
    if not os.path.exists(model_path):
        print("[INFO] Model not found locally. Downloading gesture_recognizer.task ...")
        url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        urllib.request.urlretrieve(url, model_path)
        print("[INFO] Download complete!")
    return model_path


def main():
    model_path = ensure_model_exists()
    
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6)
    
    detector = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("GestureVoice AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GestureVoice AI", DISPLAY_W, DISPLAY_H)

    # PRE-RENDER UI ONCE (NO LAG)
    print("[INFO] Pre-rendering UI...")
    STATIC_PANEL = _pre_render_side_panel()

    sentence_buffer = []

    HOLD_THRESHOLD  = 20
    hold_counter    = 0
    last_gesture    = None
    word_added      = False

    KEY_COOLDOWN    = 0.5
    last_key_time   = 0.0

    frame_timestamp_ms = 0

    print("╔══════════════════════════════════════════════════╗")
    print("║          GestureVoice AI  —  Running!           ║")
    print("║  Show your hand to the camera to get started.   ║")
    print("║  SPACE = Speak  |  C = Clear  |  Q = Quit       ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        frame_timestamp_ms += 33

        results = detector.recognize_for_video(mp_image, frame_timestamp_ms)

        active_word = None
        confidence  = 0.0

        # Scale frame up BEFORE drawing landmarks (so they render at 1080p)
        frame = cv2.resize(frame, (CAM_DISPLAY_W, CAM_DISPLAY_H),
                           interpolation=cv2.INTER_LINEAR)

        if results.hand_landmarks:
            for hand_lm in results.hand_landmarks:
                draw_landmarks_cv2(frame, hand_lm)

            # ── Two-hand combo detection (highest priority) ──
            if (len(results.gestures) >= 2
                    and results.gestures[0] and results.gestures[1]):
                label1 = results.gestures[0][0].category_name
                label2 = results.gestures[1][0].category_name
                combo = detect_two_hand_combo(label1, label2)
                if combo:
                    active_word = combo
                    confidence  = min(results.gestures[0][0].score,
                                      results.gestures[1][0].score)

            # ── Single-hand detection ──
            if active_word is None:
                if results.gestures and results.gestures[0]:
                    gesture_obj = results.gestures[0][0]
                    gesture_label = gesture_obj.category_name
                    mapped = gesture_label_to_word(gesture_label)
                    if mapped:
                        active_word = mapped
                        confidence  = gesture_obj.score

                # Fallback: custom landmark-based gesture detection
                if active_word is None:
                    custom = detect_custom_gesture(results.hand_landmarks[0])
                    if custom:
                        active_word = custom
                        confidence  = 0.85   # fixed confidence for custom

            # Draw bounding box around every detected hand
            for hand_lm in results.hand_landmarks:
                draw_hand_box(frame, hand_lm, active_word, confidence)

        if active_word is not None and active_word == last_gesture:
            if not word_added:
                hold_counter += 1
        else:
            hold_counter = 0
            word_added   = False

        last_gesture = active_word

        if hold_counter >= HOLD_THRESHOLD and not word_added:
            sentence_buffer.append(active_word)
            print(f"[+] Added \"{active_word}\"  →  "
                  f"Buffer: {' '.join(sentence_buffer)}")
            word_added = True

        display = draw_ui(frame, STATIC_PANEL, active_word, confidence,
                          sentence_buffer, hold_counter, HOLD_THRESHOLD)
        cv2.imshow("GestureVoice AI", display)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        if key == ord('q'):
            print("[INFO] Quitting GestureVoice AI. Goodbye!")
            break

        elif key == ord('c') and (now - last_key_time) > KEY_COOLDOWN:
            last_key_time = now
            sentence_buffer.clear()
            hold_counter = 0
            word_added   = True
            print("[INFO] Sentence buffer cleared.")

        elif key == ord(' ') and (now - last_key_time) > KEY_COOLDOWN:
            last_key_time = now
            sentence_buffer = speak_sentence(sentence_buffer)
            hold_counter = 0
            word_added   = True

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("[INFO] Resources released. Application closed.")


if __name__ == "__main__":
    main()
