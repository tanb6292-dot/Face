import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
import shutil
import os

app = FastAPI()

@app.post("/check-video/")
async def check_video(file: UploadFile = File(...)):
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    cap = cv2.VideoCapture(temp_video_path)
    
    # suspect.jpg ကို ဖတ်ရန်
    suspect_img = cv2.imread("suspect.jpg", cv2.IMREAD_GRAYSCALE)
    if suspect_img is None:
        cap.release()
        os.remove(temp_video_path)
        return "Suspect image not found on server"

    match_found = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # ဖရိမ်တိုင်းကို သေချာစစ်ဆေးရန် (1 ဖရိမ်ခြားတစ်ခါ)
        if frame_count % 2 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                
                # ပုံအရွယ်အစား မတူနိုင်သည်ကို ဖြေရှင်းရန် Suspect ပုံကို မျက်နှာအရွယ်အစားအလိုက် ညှိခြင်း
                try:
                    suspect_resized = cv2.resize(suspect_img, (w, h))
                    res = cv2.matchTemplate(face_roi, suspect_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    
                    print(f"Frame {frame_count} - Match Score: {max_val}") # Render Logs တွင် Score ကို ကြည့်ရန်

                    # ⚠️ Threshold ကို 0.25 သို့ အလွန်လွယ်ကူအောင် လျှော့ချထားပါသည်
                    if max_val > 0.25:
                        match_found = True
                        break
                except Exception as e:
                    continue

        if match_found:
            break

    cap.release()
    os.remove(temp_video_path)

    if match_found:
        return "Found Match!"
    else:
        return "Not Found"
        
