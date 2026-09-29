import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import GESTURE_MODEL_PATH

class GestureEngine:
    def __init__(self):
        self.recognizer = None
        self.is_ready = False
        self.init_recognizer()

    def init_recognizer(self):
        if not os.path.exists(GESTURE_MODEL_PATH):
            print(f"Warning: Gesture model not found at {GESTURE_MODEL_PATH}")
            return

        try:
            base_options = python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            self.is_ready = True
            print("Gesture Engine initialized successfully!")
        except Exception as e:
            print(f"Error initializing GestureRecognizer: {e}")

    def detect_gesture(self, frame_bgr, threshold=0.50, face_bbox=None):
        """
        Detects hand gestures in a BGR frame with optional face bbox anatomical verification.
        Returns:
            is_raised (bool): True if an open palm / raised hand belonging to the person is detected
            gesture_info: list of dicts with gesture details and landmark coordinates
        """
        if not self.is_ready or self.recognizer is None or frame_bgr is None:
            return False, []

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            results = self.recognizer.recognize(mp_image)
        except Exception as e:
            # Fallback if frame is invalid
            return False, []

        is_raised_hand = False
        detected_hands = []

        if results.gestures and len(results.gestures) > 0:
            for i, gesture_list in enumerate(results.gestures):
                if not gesture_list:
                    continue
                top_gesture = gesture_list[0]
                category_name = top_gesture.category_name
                score = top_gesture.score

                # Extract bounding box & key landmarks
                bbox = None
                wrist_y = 1.0
                wrist_x = 0.5
                fingertip_y = 0.0

                if results.hand_landmarks and len(results.hand_landmarks) > i:
                    landmarks = results.hand_landmarks[i]
                    xs = [lm.x * w for lm in landmarks]
                    ys = [lm.y * h for lm in landmarks]
                    min_x, max_x = max(0, int(min(xs))), min(w, int(max(xs)))
                    min_y, max_y = max(0, int(min(ys))), min(h, int(max(ys)))
                    bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

                    wrist_y = landmarks[0].y
                    wrist_x = landmarks[0].x
                    fingertip_y = landmarks[12].y

                # Strict Hand Rise condition:
                # 1. Category in ['Open_Palm', 'Pointing_Up', 'Thumb_Up', 'Victory', 'Raised_Hand'] with score >= threshold
                # 2. Biomechanical check: Fingertips elevated above wrist & extended
                # 3. Body Alignment check: Hand must originate from person's body relative to face bbox
                is_this_hand_raised = False
                
                # Check gesture category score
                if category_name in ['Open_Palm', 'Pointing_Up', 'Thumb_Up', 'Victory', 'Raised_Hand'] and score >= threshold:
                    is_this_hand_raised = True
                
                # Biomechanical check: Fingertips above wrist & fingers extended upwards
                if not is_this_hand_raised and results.hand_landmarks and len(results.hand_landmarks) > i:
                    landmarks = results.hand_landmarks[i]
                    wrist_y = landmarks[0].y
                    mcp_y = landmarks[9].y # Middle finger MCP (knuckle)
                    index_tip_y = landmarks[8].y
                    middle_tip_y = landmarks[12].y
                    ring_tip_y = landmarks[16].y
                    pinky_tip_y = landmarks[20].y

                    # At least 2 fingertips significantly higher than wrist (smaller Y in normalized coordinates)
                    tips_above_wrist = sum([
                        (wrist_y - tip_y) > 0.08 for tip_y in [index_tip_y, middle_tip_y, ring_tip_y, pinky_tip_y]
                    ])
                    fingers_extended = (mcp_y - middle_tip_y) > 0.04 or (mcp_y - index_tip_y) > 0.04

                    if tips_above_wrist >= 2 and fingers_extended:
                        is_this_hand_raised = True

                # Body & Face Spatial Anatomical Alignment Verification
                if is_this_hand_raised and face_bbox is not None:
                    fx, fy, fw, fh = face_bbox
                    # Convert face center and dimensions to normalized 0.0 - 1.0 coordinates
                    norm_face_cx = (fx + fw / 2.0) / w
                    norm_face_cy = (fy + fh / 2.0) / h
                    norm_face_w = fw / float(w)
                    norm_face_h = fh / float(h)

                    # 1. Hand must not be far off to the side (horizontal distance <= 3.0x face width)
                    horiz_dist = abs(wrist_x - norm_face_cx)
                    # 2. Hand/Wrist must be above waist level relative to face (vertical distance <= 3.5x face height below face top)
                    vert_dist = wrist_y - (fy / float(h))

                    is_body_aligned = (horiz_dist <= (norm_face_w * 3.0 + 0.15)) and (vert_dist <= (norm_face_h * 3.5 + 0.20)) and (fingertip_y <= (fy / float(h) + norm_face_h * 1.5))
                    if not is_body_aligned:
                        is_this_hand_raised = False

                if is_this_hand_raised:
                    is_raised_hand = True

                detected_hands.append({
                    'gesture': category_name,
                    'score': float(score),
                    'is_raised': is_this_hand_raised,
                    'bbox': bbox
                })

        return is_raised_hand, detected_hands

def analyze_top_apparel_color(img_bgr, face_bbox=None):
    """
    Analyzes upper torso region below detected face to classify top apparel/dress color.
    """
    if img_bgr is None:
        return "Unknown"
    
    h, w = img_bgr.shape[:2]
    
    if face_bbox is not None:
        fx, fy, fw, fh = face_bbox
        # Define torso region below face
        torso_y1 = min(h - 1, int(fy + fh * 1.1))
        torso_y2 = min(h, int(fy + fh * 3.5))
        torso_x1 = max(0, int(fx - fw * 0.3))
        torso_x2 = min(w, int(fx + fw * 1.3))
        
        if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
            torso_crop = img_bgr[int(h*0.4):int(h*0.8), int(w*0.3):int(w*0.7)]
        else:
            torso_crop = img_bgr[torso_y1:torso_y2, torso_x1:torso_x2]
    else:
        torso_crop = img_bgr[int(h*0.4):int(h*0.8), int(w*0.3):int(w*0.7)]

    if torso_crop is None or torso_crop.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    mean_s = np.mean(s_channel)
    mean_v = np.mean(v_channel)
    mean_h = np.mean(h_channel)

    if mean_v < 50:
        return "Black"
    elif mean_v > 200 and mean_s < 40:
        return "White"
    elif mean_s < 40:
        return "Gray"

    # Color ranges based on Hue (0 - 180 in OpenCV)
    if mean_h < 10 or mean_h > 170:
        return "Red"
    elif 10 <= mean_h < 25:
        return "Orange"
    elif 25 <= mean_h < 35:
        return "Yellow"
    elif 35 <= mean_h < 85:
        return "Green"
    elif 85 <= mean_h < 130:
        return "Blue"
    elif 130 <= mean_h < 170:
        return "Purple"

    return "Blue"

# Global Gesture Engine Singleton
gesture_engine = GestureEngine()
