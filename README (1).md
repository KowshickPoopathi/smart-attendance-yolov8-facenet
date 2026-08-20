# Smart Attendance Monitoring System (YOLOv8 + FaceNet)

An AI-powered attendance system that uses **YOLOv8** for real-time face detection and **FaceNet (facenet-pytorch)** for face recognition, deployed on a **Raspberry Pi 4** for standalone operation. Recognized attendance is automatically logged to an Excel spreadsheet.

Built as a capstone/academic project (B.E. ECE, Sathyabama Institute of Science and Technology), based on:
- *Smart Attendance Monitoring System using Face Analytics*
- *Smart AI Based Attendance Monitoring System Using YOLOv8*

---

## Features

- **Real-time face detection** using YOLOv8 (`yolov8n-face.pt`)
- **Face recognition** via FaceNet embeddings (InceptionResnetV1, pretrained on VGGFace2)
- **Automatic attendance logging** to `.xlsx` (name, date, time) using `openpyxl`
- **Standalone edge deployment** on Raspberry Pi 4 — no laptop/PC required at runtime
- **Headless mode support** — auto-detects if no display is connected (runs over SSH without a video window)
- Optional **Arduino integration** for visual/audio feedback (green LED = recognized, red LED = unknown, buzzer alert)

---

## Tech Stack

| Component | Technology |
|---|---|
| Face Detection | YOLOv8 (Ultralytics) |
| Face Recognition | facenet-pytorch (InceptionResnetV1) |
| Attendance Logging | openpyxl (Excel) |
| Hardware | Raspberry Pi 4 (2GB), Pi Camera / USB webcam |
| Optional Feedback | Arduino Uno (LED + buzzer) |
| Language | Python 3 |

---

## Project Structure

```
smart-attendance-yolov8-facenet/
├── smart_attendance_pi.py     # Main script (detection + recognition + logging)
├── yolov8n-face.pt            # YOLOv8 face detection weights (download separately)
├── encodings.pkl              # Saved face embeddings (generated after enrollment)
├── students/                  # Enrollment photos, organized per student
│   ├── kowshick/
│   ├── pranesh/
│   └── nithish/
├── attendance.xlsx            # Auto-generated attendance log
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/smart-attendance-yolov8-facenet.git
cd smart-attendance-yolov8-facenet
```

### 2. Create a virtual environment
```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install ultralytics --no-deps
pip install facenet-pytorch --no-deps
pip install opencv-python-headless openpyxl numpy pyyaml requests tqdm matplotlib pandas seaborn
```

> On Raspberry Pi, install `torch` and `torchvision` via `apt` (system packages) rather than `pip`, then create the venv with `--system-site-packages` so it inherits them.

### 4. Download YOLOv8 face detection weights
Download `yolov8n-face.pt` from [lindevs/yolov8-face releases](https://github.com/lindevs/yolov8-face/releases) and place it in the project root.

---

## Usage

### Enroll students
Add photos of each student to `students/<name>/`, then run:
```bash
python3 smart_attendance_pi.py --enroll
```
This generates `encodings.pkl` containing face embeddings for all enrolled students.

### Run live attendance
```bash
python3 smart_attendance_pi.py
```
The system will detect and recognize faces via the camera feed and log attendance automatically to `attendance.xlsx`. Runs headless (no display window) when connected over SSH.

---

## Sample Output

`attendance.xlsx` records:

| Name | Date | Time |
|---|---|---|
| Kowshick | 2026-08-14 | 09:02:15 |
| Pranesh | 2026-08-14 | 09:03:41 |

---

## Future Improvements

- Web dashboard for viewing attendance records
- Edge-AI acceleration via NPU (e.g., Luckfox Aura RV1126B, RKNN conversion)
- Cloud sync of attendance logs
- Multi-camera support for multiple entry points

---

## Authors

Kowshick — B.E. Electronics and Communication Engineering, Sathyabama Institute of Science and Technology
