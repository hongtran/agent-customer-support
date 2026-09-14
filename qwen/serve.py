"""Serve a Qwen model on Modal behind vLLM's OpenAI-compatible API.

Pick the setup with `PROFILE` below, download its weights once
(`modal run qwen/download_model.py --repo-id <repo>`), then `make modal-deploy`.
The app reaches it as `modal/<served_name>` — see `_modal_client` in
`agent_customer_support/llm/__init__.py`.

`PROFILE` is a module constant, not an env var, on purpose: Modal re-imports this
file inside the container, so a value read from the deploy shell would not be there
and the container would start a different model than the one it was sized for.

This is a `@modal.web_server` Function, not an `@app.server`, on purpose: a Server
rejects every request with 503 while no container is ready, so with scale-to-zero
each cold start failed the chat turn. A web Function holds the request through the
cold start instead (answering 303 every 150 s, which the OpenAI SDK follows).

Auth is vLLM's `--api-key`, not Modal's proxy auth, so the app can use a plain
OpenAI client with a Bearer token. Without the key, anyone with the URL could spend
the GPU.
"""

import os
import subprocess

import modal

PROFILES: dict[str, dict] = {
    # Cheap test of the app plumbing (routing, JSON answers, citations).
    # T4 does NOT work for Qwen3.5: on Turing, vLLM hung compiling the Triton
    # linear-attention kernels until the 20 min startup timeout killed it. L4 (Ada)
    # gets the fused kernels and is in the vLLM recipe's supported list.
    "l4-9b": {
        "repo_id": "Qwen/Qwen3.5-9B",
        "served_name": "qwen3.5-9b",
        "gpu": "L4",
        "concurrency": 4,
        "args": [
            # Compose prompts run ~12.4K tokens plus the app's 4K output reserve (dev), so
            # 16K is too tight. This does not cost memory: the KV pool is sized by
            # --gpu-memory-utilization (~82K tokens here), and Qwen3.5 keeps KV only
            # for its 8 full-attention layers.
            "--max-model-len", "32768",
            "--max-num-seqs", "4",
            "--gpu-memory-utilization", "0.92",
            # Text only: skips loading the vision tower, which the app never uses.
            "--language-model-only",
            # Qwen3.5 has no reasoning-effort levels, only thinking on/off; the app's
            # `reasoning_effort` template kwarg is ignored by this template.
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
            # This repo ships no generation_config.json, and the app sends no
            # temperature, so without this vLLM samples at a raw temperature 1.0.
            # Qwen's recommended non-thinking settings. (vLLM only takes temperature,
            # top_p, top_k, min_p and repetition_penalty from here.)
            "--override-generation-config",
            '{"temperature": 0.7, "top_p": 0.8, "top_k": 20}',
        ],
    },
    # The real target: Qwen3.8-27B FP8 on one H100.
    "h100-27b": {
        "repo_id": "Qwen/Qwen3.8-27B-FP8",
        "served_name": "qwen3.8-27b",
        "gpu": "H100",
        "concurrency": 32,
        "args": [
            # Enough for compose prompts; shorter than the native 262K leaves more KV
            # cache for parallel requests.
            "--max-model-len", "32768",
            "--kv-cache-dtype", "fp8",
            # The model's built-in multi-token-prediction head: faster output.
            # Drop this flag first if the server fails to start.
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
        ],
    },
}  # fmt: skip

PROFILE = "l4-9b"
CFG = PROFILES[PROFILE]
# Same layout download_model.py writes: /root/weights/<name part of the repo id>.
MODEL_DIR = f"/root/weights/{CFG['repo_id'].split('/')[-1]}"
MINUTES = 60

app = modal.App("qwen-vllm")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.29.0")
    .env({"VLLM_LOG_STATS_INTERVAL": "10"})
)

weights_vol = modal.Volume.from_name("qwen-weights")
# torch.compile / CUDA graph artifacts — reusing them makes later cold starts shorter.
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=CFG["gpu"],
    volumes={"/root/weights": weights_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[modal.Secret.from_name("qwen-vllm-api-key")],  # provides VLLM_API_KEY
    # Scale to zero: stay up 5 min after the last request, then stop billing.
    scaledown_window=5 * MINUTES,
    # Upper bound on one request; a knowledge compose call is well under this.
    timeout=10 * MINUTES,
)
@modal.concurrent(max_inputs=CFG["concurrency"])
# Model load + compile on a cold container takes minutes, not seconds.
@modal.web_server(port=8000, startup_timeout=20 * MINUTES)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL_DIR,
        "--served-model-name",
        CFG["served_name"],
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        # KeyError here (not an open server) if the secret is missing.
        "--api-key",
        os.environ["VLLM_API_KEY"],
        # REQUIRED for every Qwen3.x profile: moves thinking into
        # `reasoning_content`, so `content` holds only the answer and
        # ComposedAnswer's JSON parses. It also makes vLLM apply the json_schema
        # constraint after thinking ends, not to the thinking itself.
        "--reasoning-parser",
        "qwen3",
        *CFG["args"],
    ]
    subprocess.Popen(cmd)
