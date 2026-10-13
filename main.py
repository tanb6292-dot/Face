import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os

app = FastAPI()

# တရားခံပုံကို စတင်ဖွင့်ဖတ်ခြင်း
SUSPECT_PATH = "suspect.jpg"

@app.get("/")
def home():
    return {"message": "CCTV AI Face Recognition API is running with OpenCV!"}

@app.post("/check-video/")
async def check_video(file: UploadFile = File(...)):
    # တရားခံပုံ ရှိမရှိ စစ်ဆေးခြင်း
    if not os.path.exists(SUSPECT_PATH):
        raise HTTPException(status_code=400, detail="Suspect image not found on server.")

    # အသုံးပြုသူ ပို့လိုက်သော ဗီဒီယိုဖိုင်ကို ယာယီသိမ်းဆည်းခြင်း
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # OpenCV ဖြင့် ဗီဒီယိုကို ဖတ်ရှုခြင်း
    cap = cv2.VideoCapture(video_path)
    suspect_img = cv2.imread(SUSPECT_PATH, cv2.IMREAD_GRAYSCALE)
    
    if suspect_img is None:
        cap.release()
        os.remove(video_path)
        raise HTTPException(status_code=500, detail="Could not read suspect image.")

    suspect_gray = cv2.resize(suspect_img, (100, 100))
    match_found = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # ဖရိမ်တိုင်းကို Efficiency ကောင်းစေရန် Frame ကျော်ပြီး စစ်ဆေးခြင်း (ဥပမာ - ၁၀ ဖရိမ်တစ်ခါ)
        if frame_count % 10 == 0:
                suspect_gray = cv2.resize(suspect_img, (100, 100))
    match_found = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # ဖရိမ်ကျော်မှုကို လျှော့ချလိုက်သည် (ဥပမာ - ၃ ဖရိမ်တစ်ခါ စစ်မည်)
        if frame_count % 3 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (100, 100))
                
                res = cv2.matchTemplate(face_resized, suspect_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                # ⚠️ တူညီမှုနှုန်း Threshold ကို 0.4 သို့ လျှော့ချပေးလိုက်သည် (ပိုမိုလွယ်ကူစွာ ဖမ်းမိစေရန်)
                if max_val > 0.4:
                    match_found = True
                    break

        if match_found:
            break

    cap.release()
    os.makedirs("temp", exist_ok=True)
    if os.path.exists(video_path):
        os.remove(video_path)

    if match_found:
        return {"status": "matched", "message": "တရားခံကို ဗီဒီယိုထဲတွင် တွေ့ရှိရပါသည်!"}
    else:
        return {"status": "not_matched", "message": "တရားခံအား မတွေ့ရှိရပါ။"}
    
