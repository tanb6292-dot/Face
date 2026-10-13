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
        return "Suspect image not found on server"

    suspect_gray = cv2.resize(suspect_img, (100, 100))
    match_found = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # အချိန်မလွန်စေရန် ၁၅ ဖရိမ်မှ ၁ ဖရိမ်နှုန်းဖြင့် အမြန်စစ်ဆေးမည်
        if frame_count % 15 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(gray_frame, (100, 100))
            
            res = cv2.matchTemplate(frame_resized, suspect_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > 0.25:
                match_found = True
                break

    cap.release()
    os.remove(temp_video_path)

    if match_found:
        return "Found Match!"
    else:
        return "Not Found"
