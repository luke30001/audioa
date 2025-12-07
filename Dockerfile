FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HF_HUB_CACHE=/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY runpod/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runpod/handler.py runpod/download_model.py ./

RUN python download_model.py

ENV RUNPOD_DEBUG=0
CMD ["python", "-u", "handler.py"]
