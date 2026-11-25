FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# OpenCV 실행에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU 전용 PyTorch를 먼저 설치 → 이미지 용량이 크게 줄어듦 (GPU가 필요하면 이 줄 삭제)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# YOLO Pose 가중치를 빌드할 때 미리 받아 둠 → 실행 중 인터넷이 없어도 동작
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
