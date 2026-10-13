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
    
    suspect_img = cv2.imread("suspect.jpg", cv2.IMREAD_GRAYSCALE)
    if suspect_img is None:
        cap.release()
        os.remove(temp_video_path)
        return {"result": "Suspect image not found", "percentage": "0%"}

    suspect_gray = cv2.resize(suspect_img, (100, 100))
    highest_score = 0.0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(gray_frame, (100, 100))
            
            res = cv2.matchTemplate(frame_resized, suspect_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > highest_score:
                highest_score = max_val

    cap.release()
    os.remove(temp_video_path)

    # ရလာတဲ့ ရမှတ်ကို ရာခိုင်နှုန်း (%) အဖြစ် ပြောင်းလဲခြင်း (0 ထက်ငယ်ပါက 0 ဟု သတ်မှတ်မည်)
    match_percentage = max(0.0, highest_score) * 100

    if match_percentage > 15.0:  # လိုအပ်သော Threshold တန်ဖိုး
        return {
            "status": "Found Match",
            "percentage": f"{match_percentage:.2f}%"
        }
    else:
        return {
            "status": "Not Found",
            "percentage": f"{match_percentage:.2f}%"
        }

