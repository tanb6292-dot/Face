FROM python:3.9-slim

# လိုအပ်သော System dependencies များ ထည့်သွင်းခြင်း (dlib အတွက် cmake လိုအပ်သည်)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# လိုအပ်သော Python libraries များ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ကုဒ်များနှင့် တရားခံပုံကို ကူးယူခြင်း
COPY . .

# ဆာဗာ စတင်ရန် Command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
