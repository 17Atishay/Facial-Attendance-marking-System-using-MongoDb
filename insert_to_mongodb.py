# insert_to_mongodb.py

import base64
import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
import os

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["face_attendance"]
collection = db["users"]

# Path to image folder
IMAGE_PATH = "images"

def encode_image_to_binary(path):
    with open(path, "rb") as img_file:
        return Binary(img_file.read())

def insert_user_to_db(name, image_path):
    # Check if user already exists
    if collection.find_one({"name": name}):
        print(f"[⚠️] User '{name}' already exists. Skipping insert.")
        return

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        print(f"[❌] No face found in {name}'s image. Skipping.")
        return

    binary_image = encode_image_to_binary(image_path)
    face_encoding = encodings[0].tolist()

    user = {
        "name": name,
        "image": binary_image,
        "encoding": face_encoding,
        "attendance": []
    }
    collection.insert_one(user)
    print(f"[✅] Inserted {name} into MongoDB.")

def load_and_insert_all():
    for filename in os.listdir(IMAGE_PATH):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            name = os.path.splitext(filename)[0]
            path = os.path.join(IMAGE_PATH, filename)
            insert_user_to_db(name, path)

    print(f"[📦] Finished inserting. Databases: {client.list_database_names()}")

if __name__ == "__main__":
    load_and_insert_all()
