# mongodb_face_attendance.py with dlib-based blink detection + CSV logging

import cv2
import numpy as np
import face_recognition
from pymongo import MongoClient
from datetime import datetime
from scipy.spatial import distance as dist
import dlib
import csv
import os

# MongoDB local connection
client = MongoClient("mongodb://localhost:27017")
db = client["face_attendance"]
collection = db["users"]

# Dlib facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

CSV_FILE = "attendance.csv"

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Ensure CSV file exists with header
def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Timestamp"])

# Load encodings and names from MongoDB
def load_known_faces_from_db():
    known_encodings = []
    known_names = []
    for user in collection.find():
        if 'encoding' in user:
            known_encodings.append(np.array(user['encoding']))
            known_names.append(user['name'])
    return known_encodings, known_names

# Mark attendance in MongoDB and CSV
def mark_attendance_db(name):
    now = datetime.now()
    ts = now.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[📝] Marking attendance for {name} at {ts}")

    user = collection.find_one({"name": name})
    if user and not isinstance(user.get("attendance"), list):
        collection.update_one({"name": name}, {"$set": {"attendance": []}})
        print(f"[⚠️] Fixed 'attendance' field structure for {name}")

    result = collection.update_one(
        {"name": name},
        {"$push": {"attendance": {"timestamp": ts, "status": "Present"}}}
    )
    print(f"[🔍] MongoDB Update Result: matched={result.matched_count}, modified={result.modified_count}")

    # Direct CSV write
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, ts])

# Face recognition loop with blink detection
def run_attendance_system():
    ensure_csv()
    known_encodings, known_names = load_known_faces_from_db()
    print(f"Loaded encodings for: {known_names}")
    attendance_set = set()
    blinked = set()

    cap = cv2.VideoCapture(0)
    recent_marked = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        try:
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        except Exception as e:
            print(f"Encoding error: {e}")
            face_encodings = []

        dets = face_detector(gray)

        for encoding, loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            face_dist = face_recognition.face_distance(known_encodings, encoding)

            name = 'Unknown'
            match_pct = 0
            status_text = ''

            if matches:
                best_match_index = np.argmin(face_dist)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    match_pct = round((1 - face_dist[best_match_index]) * 100, 2)

                    if name not in attendance_set:
                        for d in dets:
                            shape = predictor(gray, d)
                            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                            leftEye = shape_np[LEFT_EYE]
                            rightEye = shape_np[RIGHT_EYE]
                            leftEAR = eye_aspect_ratio(leftEye)
                            rightEAR = eye_aspect_ratio(rightEye)
                            ear = (leftEAR + rightEAR) / 2.0

                            if ear < 0.20:
                                blinked.add(name)
                            elif name in blinked:
                                attendance_set.add(name)
                                mark_attendance_db(name)
                                blinked.remove(name)
                                recent_marked[name] = datetime.now()

            top, right, bottom, left = [v * 2 for v in loc]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            text = f'{name} ({match_pct}%)'
            cv2.putText(frame, text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Show 'Attendance marked' message if recently marked
            if name in recent_marked:
                if (datetime.now() - recent_marked[name]).seconds < 3:
                    cv2.putText(frame, 'Attendance marked', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('Liveness Face Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[📊] Attendance Summary")
    for name in attendance_set:
        print(f" - {name} marked present")
    print(f"[✅] Total Present: {len(attendance_set)}")

if __name__ == '__main__':
    run_attendance_system()