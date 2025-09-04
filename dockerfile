# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 기본 패키지
RUN apt-get update && apt-get install -y \
      python3 python3-pip python3-venv python3-dev git wget \
      && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 및 필수 라이브러리 설치
RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install transformers sentencepiece mecab-python3 pandas tqdm

# 코드 복사
WORKDIR /workspace
COPY . /workspace

# 기본 커맨드: bash로 진입
CMD ["bash"]


# docker run -it --gpus all -v C:\Users\Server7\LLM:/workspace -p 8888:8888 --name gec-training gec-env:latest
# --rm

# docker exec -i -t gec-training bash

# PYTHONPATH=/workspace python3 ./transformer/luna/train_s2s.py
