import os
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# 1. Configuration
DATASET_PATH = "Dataset" 
MODEL_FILENAME = 'face_classification_model.pkl'
ENCODER_FILENAME = 'label_encoder.pkl'

print(f"Current Working Directory: {os.getcwd()}")

# Check if Dataset folder exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: Could not find a folder named '{DATASET_PATH}' in the current directory.")
    print("Please make sure your Dataset folder is in the same place as this script.")
    exit()

print("✅ Found Dataset folder. Initializing MTCNN and FaceNet...")
detector = MTCNN()
embedder = FaceNet()

def extract_face(img):
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    return face

# 2. Extract Embeddings
print("Scanning Dataset folder...")
embeddings = []
labels = []

folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
print(f"Found {len(folders)} student folders: {folders}")

if len(folders) == 0:
    print("❌ ERROR: The Dataset folder is empty or doesn't contain subfolders.")
    exit()

for person in folders:
    person_path = os.path.join(DATASET_PATH, person)
    images = os.listdir(person_path)
    print(f"  -> Processing {len(images)} images for: {person}")
    
    for img_name in images:
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"     [Warning] Could not read image: {img_name}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = extract_face(img)
        
        if face is None:
            print(f"     [Warning] No face detected in: {img_name}")
            continue
            
        embedding = embedder.embeddings([face])[0]
        embeddings.append(embedding)
        labels.append(person)

print(f"✅ Total valid faces successfully processed: {len(embeddings)}")

if len(embeddings) == 0:
    print("❌ ERROR: No faces were extracted. Cannot train the model.")
    exit()

# 3. Train the Model
print("Training Classification Model...")
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

model = SVC(kernel='linear', probability=True)
model.fit(embeddings, labels_encoded)

# 4. Save the Models Locally
with open(MODEL_FILENAME, 'wb') as f:
    pickle.dump(model, f)
    
with open(ENCODER_FILENAME, 'wb') as f:
    pickle.dump(encoder, f)

print(f"🎉 Success! Models saved as {MODEL_FILENAME} and {ENCODER_FILENAME}")