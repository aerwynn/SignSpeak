export const GESTURE_WORD_MAP: Record<string, string> = {
    "Closed_Fist":  "Help",
    "Open_Palm":    "Hello",
    "Pointing_Up":  "Attention",
    "Thumb_Down":   "No",
    "Thumb_Up":     "Yes",
    "Victory":      "Water",
    "ILoveYou":     "Please", // Mapped to Spread Hand in UI
};

export function gestureLabelToWord(label: string): string | null {
    return GESTURE_WORD_MAP[label] || null;
}

const TWO_HAND_COMBOS = [
    { combo: ["Closed_Fist", "Closed_Fist"], word: "Emergency" },
    { combo: ["Open_Palm", "Thumb_Up"], word: "Hello" },
    { combo: ["Thumb_Up", "Open_Palm"], word: "Hello" }, // order doesn't matter
];

export function detectTwoHandCombo(label1: string, label2: string): string | null {
    for (const item of TWO_HAND_COMBOS) {
        if (item.combo.includes(label1) && item.combo.includes(label2)) {
            return item.word;
        }
    }
    return null;
}

export function detectCustomGesture(landmarks: any[]): string | null {
    if (!landmarks || landmarks.length < 21) return null;

    // Helper to calculate distance
    const dist = (p1: any, p2: any) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

    // _is_pinching check
    const thumb_tip = landmarks[4];
    const index_tip = landmarks[8];
    const wrist = landmarks[0];
    const middle_mcp = landmarks[9];

    const pinch_dist = dist(thumb_tip, index_tip);
    const hand_size = dist(wrist, middle_mcp);
    
    if (hand_size > 0.01 && (pinch_dist / hand_size) < 0.25) {
        return "Pain";
    }

    const pinky_mcp = landmarks[17];
    
    // In Shaka (Medicine), the thumb is stretched far away from the pinky base.
    // In Bathroom (Pinky Only), the thumb is tucked in over the palm, close to the pinky base.
    const thumb = dist(thumb_tip, pinky_mcp) > (hand_size * 1.1);

    const index = dist(landmarks[8], wrist) > dist(landmarks[6], wrist);
    const middle = dist(landmarks[12], wrist) > dist(landmarks[10], wrist);
    const ring = dist(landmarks[16], wrist) > dist(landmarks[14], wrist);
    const pinky = dist(landmarks[20], wrist) > dist(landmarks[18], wrist);

    if (!thumb && index && middle && ring && !pinky) return "Food";
    if (!thumb && !index && !middle && !ring && pinky) return "Bathroom";
    if (thumb && !index && !middle && !ring && pinky) return "Medicine";
    if (!thumb && index && !middle && !ring && pinky) return "Me";
    if (thumb && index && !middle && !ring && !pinky) return "Go";

    return null;
}
