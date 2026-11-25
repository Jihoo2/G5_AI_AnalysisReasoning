FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU 전용 PyTorch → 이미지 용량 절약 (GPU가 필요하면 이 줄 삭제)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 모델 파일(.joblib, .pt)은 Git에 없으므로 실행할 때 models/ 폴더를 연결하세요:
#   docker run -p 8001:8001 -v $(pwd)/models:/app/models g5-ai-server
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
