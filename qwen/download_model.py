"""One-off: download model weights into the `qwen-weights` Modal Volume.

    modal run qwen/download_model.py                                  # 9B test model (L4 profile)
    modal run qwen/download_model.py --repo-id Qwen/Qwen3.8-27B-FP8   # 27B model

Weights land at /root/weights/<name part of the repo id>, the path `serve.py` reads,
so a cold start loads from disk instead of re-downloading from Hugging Face.
"""

import modal

app = modal.App("qwen-downloader")
volume = modal.Volume.from_name("qwen-weights", create_if_missing=True)

image = modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]")


@app.function(volumes={"/root/weights": volume}, image=image, timeout=1800)
def download(repo_id: str = "Qwen/Qwen3.5-9B"):
    import os

    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    local_dir = f"/root/weights/{repo_id.split('/')[-1]}"
    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.pt", "*.bin"],  # safetensors only
    )
    volume.commit()
    print("Download complete and volume committed!")
