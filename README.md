# Facial-Attendance-marking-System-using-MongoDb

A smart attendance system using **OpenCV**, **face_recognition**, **dlib**, and **MongoDB**, enhanced with **blink-based liveness detection** to prevent spoofing with photos.

---

## 🔧 Features

✅ Face recognition using `face_recognition`  
✅ Blink detection using `dlib` (EAR)  
✅ Liveness detection (prevents photo-based spoofing)  
✅ Attendance recorded in:
- 🗃️ MongoDB
- 🧾 `attendance.csv`  
✅ Realtime camera preview with recognition name & match %  
✅ Displays “Attendance marked” on screen  
✅ Duplicate MongoDB user prevention  
✅ Command-line based (no GUI dependency)

---
## 📁 Folder Structure
├── mongodb_face_attendance.py # Main attendance system <br>
├── insert_to_mongodb.py # Insert face images to MongoDB <br>
├── images/ # Folder with user images<br>
├── attendance.csv # Logged attendance file<br>
├── shape_predictor_68_face_landmarks.dat # Dlib landmark model<br>

🧠 Usage Instructions
1. Add Student Images
Place student face images in the images/ folder

File name format = "Full Name.jpg" (e.g., Atishay Jain.jpg)

Images must show only one clear face

2. Download Landmark Predictor File
Download the facial landmarks model from Kaggle:

📥 shape_predictor_68_face_landmarks.dat

Unzip and place shape_predictor_68_face_landmarks.dat in the project root.

pip install opencv-python face_recognition dlib pymongo scipy numpy

3. Insert Faces into MongoDB
   python insert_to_mongodb.py
4. Run Attendance System
  python mongodb_face_attendance.py

Live webcam stream will open<br>
Blink to confirm liveness<br>
Attendance marked in MongoDB and CSV<br>
Name, match %, and “Attendance marked” message shown on screen

🧾 Output
attendance.csv logs all recognized attendance

MongoDB document structure:
json
{<br>
  "name": "Atishay Jain",<br>
  "encoding": [...],<br>
  "attendance": [<br>
    { "timestamp": "2025-06-27 12:45:00", "status": "Present" }<br>
  ]<br>
}
