import os
import cv2
import numpy as np
import pickle
import threading
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, FACE_CLASSIFIER_PATH, DATASET_DIR
from database import get_all_employees

class FaceEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.detector = None
        self.recognizer = None
        self.model = None
        self.encoder = None
        self.known_embeddings = []
        self.known_labels = []
        self.is_trained = False
        self.init_models()
        self.load_or_train()

    def init_models(self):
        if os.path.exists(YUNET_MODEL_PATH) and os.path.exists(SFACE_MODEL_PATH):
            self.detector = cv2.FaceDetectorYN.create(
                model=YUNET_MODEL_PATH,
                config='',
                input_size=(320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=SFACE_MODEL_PATH,
                config=''
            )
        else:
            print("Warning: Face model files not found in models/ directory.")

    def extract_face_and_embedding(self, img_bgr, apply_augmentation=False):
        """Detects the largest face in an image and extracts 128-d embedding(s).
        Returns best_face and list of embeddings (1 or more if augmented)."""
        if self.detector is None or self.recognizer is None or img_bgr is None:
            return None, [] if apply_augmentation else (None, None)

        h, w = img_bgr.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
            h, w = img_bgr.shape[:2]

        with self.lock:
            try:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(img_bgr)

                if faces is None or len(faces) == 0:
                    return (None, []) if apply_augmentation else (None, None)

                # Select face with largest area & highest confidence
                best_face = max(faces, key=lambda f: float(f[2] * f[3]) * (float(f[14]) if len(f) >= 15 else 1.0))
                
                # Check minimum face size
                if best_face[2] < 20 or best_face[3] < 20:
                    return (None, []) if apply_augmentation else (None, None)

                aligned_face = self.recognizer.alignCrop(img_bgr, best_face)
                if aligned_face is None or aligned_face.shape[:2] != (112, 112):
                    return (None, []) if apply_augmentation else (None, None)
                
                # Base normalized feature
                feature = self.recognizer.feature(aligned_face)[0]
                feature = feature / (np.linalg.norm(feature) + 1e-7)

                if not apply_augmentation:
                    # Apply subtle CLAHE equalization test to enhance live camera feed match score
                    lab = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                    cl = clahe.apply(l)
                    equ_img = cv2.merge((cl, a, b))
                    equ_bgr = cv2.cvtColor(equ_img, cv2.COLOR_LAB2BGR)
                    feat_equ = self.recognizer.feature(equ_bgr)[0]
                    feat_equ = feat_equ / (np.linalg.norm(feat_equ) + 1e-7)
                    
                    # Return mean vector between raw feature and CLAHE feature for high illumination invariance
                    blended_feat = (feature * 0.75 + feat_equ * 0.25)
                    blended_feat = blended_feat / (np.linalg.norm(blended_feat) + 1e-7)
                    return best_face, blended_feat

                # Augmentation variants for training robust high-precision models
                embeddings = [feature]

                # Variant 1: CLAHE Illumination Normalized
                lab = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                cl = clahe.apply(l)
                equ_img = cv2.merge((cl, a, b))
                equ_bgr = cv2.cvtColor(equ_img, cv2.COLOR_LAB2BGR)
                feat_equ = self.recognizer.feature(equ_bgr)[0]
                embeddings.append(feat_equ / (np.linalg.norm(feat_equ) + 1e-7))

                # Variant 2: Horizontal Flip
                flipped_img = cv2.flip(img_bgr, 1)
                self.detector.setInputSize((w, h))
                _, flipped_faces = self.detector.detect(flipped_img)
                if flipped_faces is not None and len(flipped_faces) > 0:
                    best_flipped = max(flipped_faces, key=lambda f: float(f[2] * f[3]))
                    aligned_flipped = self.recognizer.alignCrop(flipped_img, best_flipped)
                    if aligned_flipped is not None and aligned_flipped.shape[:2] == (112, 112):
                        feat_flip = self.recognizer.feature(aligned_flipped)[0]
                        embeddings.append(feat_flip / (np.linalg.norm(feat_flip) + 1e-7))

                # Variant 3: Slight Brightness adjustment (+15)
                bright_img = cv2.convertScaleAbs(img_bgr, alpha=1.08, beta=15)
                aligned_bright = self.recognizer.alignCrop(bright_img, best_face)
                if aligned_bright is not None and aligned_bright.shape[:2] == (112, 112):
                    feat_bright = self.recognizer.feature(aligned_bright)[0]
                    embeddings.append(feat_bright / (np.linalg.norm(feat_bright) + 1e-7))

                # Variant 4: Slight Darkening adjustment (-15) for low-light robustness
                dark_img = cv2.convertScaleAbs(img_bgr, alpha=0.92, beta=-15)
                aligned_dark = self.recognizer.alignCrop(dark_img, best_face)
                if aligned_dark is not None and aligned_dark.shape[:2] == (112, 112):
                    feat_dark = self.recognizer.feature(aligned_dark)[0]
                    embeddings.append(feat_dark / (np.linalg.norm(feat_dark) + 1e-7))

                return best_face, embeddings
            except Exception as e:
                print(f"Feature extraction error: {e}")
                return (None, []) if apply_augmentation else (None, None)

    def train_model(self):
        """Extracts embeddings for all employee photos (with augmentation) and trains KNN / Cosine classifier."""
        employees = get_all_employees()
        embeddings = []
        labels = []

        print(f"Starting Face Recognition training for {len(employees)} employees...")

        for emp in employees:
            emp_id = emp['emp_id']
            for photo_path in emp.get('photos', []):
                if not os.path.exists(photo_path):
                    continue
                img = cv2.imread(photo_path)
                if img is None:
                    continue
                face, embs = self.extract_face_and_embedding(img, apply_augmentation=True)
                for emb in embs:
                    if emb is not None:
                        embeddings.append(emb)
                        labels.append(emp_id)

        if len(embeddings) == 0:
            print("No face images available to train.")
            with self.lock:
                self.is_trained = False
                self.known_embeddings = []
                self.known_labels = []
            if os.path.exists(FACE_CLASSIFIER_PATH):
                try:
                    os.remove(FACE_CLASSIFIER_PATH)
                except Exception:
                    pass
            return False, "No face photos found. Please add employee photos."

        known_embeddings = np.array(embeddings)
        known_labels = np.array(labels)

        # Train a KNN classifier with cosine metric
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)
        knn = KNeighborsClassifier(n_neighbors=min(3, len(embeddings)), metric='cosine')
        knn.fit(known_embeddings, labels_encoded)

        # Save model
        with open(FACE_CLASSIFIER_PATH, 'wb') as f:
            pickle.dump({
                'knn': knn,
                'encoder': encoder,
                'embeddings': known_embeddings,
                'labels': known_labels
            }, f)

        with self.lock:
            self.model = knn
            self.encoder = encoder
            self.known_embeddings = known_embeddings
            self.known_labels = known_labels
            self.is_trained = True

        print(f"Face model trained successfully with {len(embeddings)} augmented samples across {len(set(labels))} employees!")
        return True, f"Trained successfully on {len(embeddings)} face samples across {len(set(labels))} employees."

    def load_or_train(self):
        """Loads existing model from disk or triggers training."""
        if os.path.exists(FACE_CLASSIFIER_PATH):
            try:
                with open(FACE_CLASSIFIER_PATH, 'rb') as f:
                    data = pickle.load(f)
                    with self.lock:
                        self.model = data.get('knn')
                        self.encoder = data.get('encoder')
                        self.known_embeddings = data.get('embeddings', [])
                        self.known_labels = data.get('labels', [])
                        self.is_trained = True
                print("Face recognition model loaded from cache.")
                return
            except Exception as e:
                print(f"Error loading face model: {e}")

        # Fallback to train
        self.train_model()

    def detect_faces(self, frame_bgr):
        """Detects all faces in frame and returns bounding boxes and landmarks."""
        if self.detector is None or frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        with self.lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame_bgr)

        if faces is None or len(faces) == 0:
            return []

        return faces

    def recognize_face(self, frame_bgr, face_data, threshold=0.45):
        """
        Recognizes a detected face against known embeddings using top-k cosine similarity aggregation per employee.
        Returns (emp_id, confidence, bbox) where emp_id is 'Unknown' if below threshold.
        """
        x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
        if not self.is_trained or self.recognizer is None or len(self.known_embeddings) == 0:
            return "Unknown", 0.0, (x, y, w, h)

        with self.lock:
            try:
                aligned_face = self.recognizer.alignCrop(frame_bgr, face_data)
                if aligned_face is None or aligned_face.shape[:2] != (112, 112):
                    return "Unknown", 0.0, (x, y, w, h)
                feature = self.recognizer.feature(aligned_face)[0]
                known_embs = self.known_embeddings.copy()
                known_lbls = self.known_labels.copy()
            except Exception as e:
                return "Unknown", 0.0, (x, y, w, h)

        try:
            # Normalize vectors
            norm_feature = feature / (np.linalg.norm(feature) + 1e-7)
            norm_known = known_embs / (np.linalg.norm(known_embs, axis=1, keepdims=True) + 1e-7)
            similarities = np.dot(norm_known, norm_feature)

            # Group similarities by employee identity label to calculate top-k score per employee
            unique_labels = np.unique(known_lbls)
            best_emp_id = "Unknown"
            best_emp_score = 0.0

            for emp_id in unique_labels:
                emp_indices = np.where(known_lbls == emp_id)[0]
                emp_sims = similarities[emp_indices]
                
                # Take top-2 similarity max and mean
                top_k = min(2, len(emp_sims))
                top_k_sims = np.partition(emp_sims, -top_k)[-top_k:]
                max_sim = float(np.max(emp_sims))
                top_avg = float(np.mean(top_k_sims))
                
                # Calibrated similarity score mapping
                # SFace Cosine similarity typically ranges 0.35 - 0.75 for same identity
                raw_score = 0.70 * max_sim + 0.30 * top_avg
                # Calibrate score to standard 0-100% human confidence scale
                calibrated_score = float(np.clip((raw_score - 0.20) / (0.65 - 0.20), 0.0, 1.0))

                if calibrated_score > best_emp_score:
                    best_emp_score = calibrated_score
                    best_emp_id = emp_id

            if best_emp_score >= threshold:
                return best_emp_id, float(best_emp_score), (x, y, w, h)
            else:
                return "Unknown", float(best_emp_score), (x, y, w, h)
        except Exception:
            return "Unknown", 0.0, (x, y, w, h)

# Global Face Engine Singleton
face_engine = FaceEngine()
