"""
Pre-fetches Whisper weights so the container is ready when the pod starts.
"""

import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-large-v3")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def main():
    AutoProcessor.from_pretrained(MODEL_ID)
    AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )


if __name__ == "__main__":
    main()
