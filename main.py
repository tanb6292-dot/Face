from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import face_recognition
import shutil
import os

app = FastAPI()

known_face_encodings = []
known_face_names = []

try:
    suspect_image = face_recognition.load_image_file("suspect.jpg")
    encoding = face_recognition.face_encodings(suspect_image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append("တရားခံ (Suspect)")
    print("တရားခံပုံကို အောင်မြင်စွာ Load လုပ်ပြီးပါပြီ။")
except Exception as e:
    print(f"Error loading suspect image: {e}")

@app.post("/check-video/")
async def check_video(file: UploadFile = File(...)):
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    cap = cv2.VideoCapture(temp_video_path)
    found_suspect = False
    matched_name = ""
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 5 != 0:
            continue
            
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] <= 0.6:
                    found_suspect = True
                    matched_name = known_face_names[best_match_index]
                    break
        if found_suspect:
            break
            
    cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        
    if found_suspect:
        return {"status": "Suspect Found", "message": f"🚨 တရားခံ ({matched_name}) ကို တွေ့ရှိရပါသည်!"}
    else:
        return {"status": "Not Found", "message": "တရားခံ ရှာမတွေ့ပါ။"}
