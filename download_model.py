"""
Pre-fetches Whisper weights so the container is ready when the pod starts.
"""

import os

from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-large-v3")


def main():
    os.makedirs("/models", exist_ok=True)
    # Download weights and processor files without loading the model into memory
    # Uses standard HuggingFace cache structure so from_pretrained() can find it
    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir="/models",
    )


if __name__ == "__main__":
    main()
