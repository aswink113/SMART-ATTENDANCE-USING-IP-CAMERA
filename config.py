import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
CAPTURES_DIR = os.path.join(BASE_DIR, 'captures')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH = os.path.join(BASE_DIR, 'attendance.db')

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Face and Gesture Models
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, 'face_recognition_sface_2021dec.onnx')
GESTURE_MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_recognizer.task')
FACE_CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'face_classifier.pkl')

SECRET_KEY = 'smart-attendance-system-secret-key-2026'

# Default Camera and Punch Rule Defaults
DEFAULT_COOLDOWN_MINUTES = 5
DEFAULT_FACE_THRESHOLD = 0.40  # Cosine distance / confidence threshold
DEFAULT_GESTURE_THRESHOLD = 0.50
