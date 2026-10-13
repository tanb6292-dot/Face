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
        return {"status": "Error", "percentage": "0%"}

    # Suspect ပုံကို အရွယ်အစားညှိခြင်းနှင့် အလင်းအမှောင်ပြင်ဆင်ခြင်း
    suspect_gray = cv2.resize(suspect_img, (100, 100))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    suspect_processed = clahe.apply(suspect_gray)

    highest_score = 0.0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # ဖရိမ် ၅ ဖရိမ်တစ်ခါ စစ်ဆေးခြင်းဖြင့် ပိုမိုစိပ်ငှက်တိကျစေရန်
        if frame_count % 5 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(gray_frame, (100, 100))
            
            # အလင်းအမှောင် အားနည်းချက်များကို ကုစားရန် CLAHE ကို အသုံးပြုခြင်း
            frame_processed = clahe.apply(frame_resized)
            
            res = cv2.matchTemplate(frame_processed, suspect_processed, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > highest_score:
                highest_score = max_val

    cap.release()
    os.remove(temp_video_path)

    match_percentage = max(0.0, highest_score) * 100

    # လိုအပ်သော Threshold တန်ဖိုး (၁၅% နှင့်အထက်ဆိုလျှင် တွေ့ရှိသည်ဟု သတ်မှတ်မည်)
    if match_percentage > 10.0:
        return {
            "status": "Found Match",
            "percentage": f"{match_percentage:.2f}%"
        }
    else:
        return {
            "status": "Not Found",
            "percentage": f"{match_percentage:.2f}%"
    }
    
