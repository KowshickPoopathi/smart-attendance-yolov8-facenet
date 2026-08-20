"""
Smart Attendance Monitoring System using YOLOv8
-------------------------------------------------
Pipeline: Camera Feed -> YOLOv8 Face Detection -> Face Embeddings (facenet-pytorch)
          -> Match against enrolled student database -> Mark Attendance (CSV)

NOTE: This version uses facenet-pytorch for recognition instead of
face_recognition/dlib. dlib requires compiling from source with CMake +
a C++ compiler, and has no prebuilt wheels for newer Python versions
(e.g. Python 3.14), which is why the dlib install was failing.
facenet-pytorch ships prebuilt wheels and needs no compilation.

Requirements:
    pip install ultralytics opencv-python numpy torch torchvision facenet-pytorch

Model:
    Download a face-trained YOLOv8 checkpoint (e.g. "yolov8n-face.pt")
    and place it in the same folder as this script, OR pass its path
    via the FACE_MODEL_PATH variable below.

Folder structure expected for enrollment:
    students/
        Kowshick/
            photo1.jpg
            photo2.jpg
        Pranav/
            photo1.jpg
        ...

Usage:
    1. Put student photos in the "students/" folder as shown above.
    2. Run:  python smart_attendance_yolov8.py --enroll
       (builds the face embedding database, run this once / whenever you add students)
    3. Run:  python smart_attendance_yolov8.py
       (starts live webcam attendance marking)

IMPORTANT: Run these commands from the SAME folder where this .py file
is saved. If you see "No such file or directory", cd into that folder
first, e.g.:  cd Downloads
"""

import os
import cv2
import csv
import pickle
import argparse
import datetime
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
FACE_MODEL_PATH = "yolov8n-face.pt"     # path to YOLOv8 face-detection weights
STUDENTS_DIR = "students"               # folder containing student photo subfolders
ENCODINGS_FILE = "encodings.pkl"        # stores known face embeddings
ATTENDANCE_FILE = "attendance.csv"      # attendance log output
MATCH_THRESHOLD = 0.9                   # lower = stricter match (embedding distance)
CAMERA_SOURCE = 0                       # 0 = default webcam, or RTSP/video path


# ---------------------------------------------------------------------------
# SHARED: lazy-load the embedding model (facenet-pytorch), once per process
# ---------------------------------------------------------------------------
_embedder = None
_device = None

def get_embedder():
    global _embedder, _device
    if _embedder is None:
        import torch
        from facenet_pytorch import InceptionResnetV1
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _embedder = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
        print(f"[INFO] Face embedding model loaded on {_device}")
    return _embedder, _device


def get_embedding(face_crop_bgr):
    """Given a BGR face crop (numpy array), return a 512-d embedding vector."""
    import torch
    from facenet_pytorch import fixed_image_standardization

    embedder, device = get_embedder()

    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))

    face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
    face_tensor = fixed_image_standardization(face_tensor)
    face_tensor = face_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = embedder(face_tensor)

    return embedding.cpu().numpy()[0]


# ---------------------------------------------------------------------------
# STEP 1: ENROLLMENT — build known-face embeddings from the students/ folder
# ---------------------------------------------------------------------------
def enroll_students():
    known_embeddings = []
    known_names = []

    if not os.path.isdir(STUDENTS_DIR):
        print(f"[ERROR] '{STUDENTS_DIR}' folder not found. Create it and add student photos first.")
        return

    for student_name in os.listdir(STUDENTS_DIR):
        student_folder = os.path.join(STUDENTS_DIR, student_name)
        if not os.path.isdir(student_folder):
            continue

        for img_file in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_file)
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARN] Could not read {img_path}, skipped.")
                    continue

                embedding = get_embedding(image)
                known_embeddings.append(embedding)
                known_names.append(student_name)
                print(f"[OK] Enrolled {student_name} -> {img_file}")
            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_embeddings, "names": known_names}, f)

    print(f"\nEnrollment complete. {len(known_names)} face(s) saved to {ENCODINGS_FILE}")


# ---------------------------------------------------------------------------
# STEP 2: LOAD RESOURCES — YOLOv8 model + known embeddings
# ---------------------------------------------------------------------------
def load_face_model():
    from ultralytics import YOLO
    if not os.path.exists(FACE_MODEL_PATH):
        print(f"[ERROR] Model file '{FACE_MODEL_PATH}' not found. "
              f"Download a YOLOv8 face-detection checkpoint and place it here.")
        exit(1)
    return YOLO(FACE_MODEL_PATH)


def load_known_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        print(f"[ERROR] '{ENCODINGS_FILE}' not found. Run with --enroll first.")
        exit(1)
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


# ---------------------------------------------------------------------------
# STEP 3: ATTENDANCE LOGGING
# ---------------------------------------------------------------------------
marked_today = set()

def init_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def mark_attendance(name):
    if name in marked_today:
        return
    marked_today.add(name)
    now = datetime.datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    print(f"[ATTENDANCE] {name} marked present at {now.strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------
# STEP 4: DETECTION + RECOGNITION LOOP
# ---------------------------------------------------------------------------
def detect_faces(model, frame):
    """Run YOLOv8 face detection, return list of (x1, y1, x2, y2) boxes."""
    results = model(frame, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def find_best_match(embedding, known_embeddings, known_names):
    """Return (name, distance) of the closest known embedding."""
    if not known_embeddings:
        return "Unknown", None

    distances = [np.linalg.norm(embedding - known) for known in known_embeddings]
    best_index = int(np.argmin(distances))
    best_distance = distances[best_index]

    if best_distance < MATCH_THRESHOLD:
        return known_names[best_index], best_distance
    return "Unknown", best_distance


def process_frame(frame, model, known_encodings, known_names):
    face_boxes = detect_faces(model, frame)
    h, w = frame.shape[:2]

    for (x1, y1, x2, y2) in face_boxes:
        # clamp box to frame bounds to avoid crop errors
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        embedding = get_embedding(face_crop)
        name, distance = find_best_match(embedding, known_encodings, known_names)

        if name != "Unknown":
            mark_attendance(name)

        color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


# ---------------------------------------------------------------------------
# STEP 5: MAIN — live camera attendance loop
# ---------------------------------------------------------------------------
def run_attendance_system():
    print("Loading YOLOv8 face detection model...")
    model = load_face_model()

    print("Loading known face embeddings...")
    known_encodings, known_names = load_known_encodings()

    # warm up the embedding model once so the first frame isn't slow
    get_embedder()

    init_attendance_file()

    print("Starting camera feed. Press 'q' to quit.")
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        print("[ERROR] Could not open camera/video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream or camera error.")
            break

        frame = process_frame(frame, model, known_encodings, known_names)

        cv2.imshow("Smart Attendance - YOLOv8", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Attendance saved to '{ATTENDANCE_FILE}'.")
    print(f"Marked present: {sorted(marked_today)}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Attendance System using YOLOv8")
    parser.add_argument("--enroll", action="store_true",
                         help="Build face embeddings database from students/ folder")
    args = parser.parse_args()

    if args.enroll:
        enroll_students()
    else:
        run_attendance_system()
