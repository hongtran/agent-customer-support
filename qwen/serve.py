"""Serve a Qwen model on Modal behind vLLM's OpenAI-compatible API.

Pick the setup with `QWEN_PROFILE` (see `PROFILES`), download its weights once
(`modal run qwen/download_model.py --repo-id <repo>`), then
`make modal-deploy PROFILE=<profile>`. Each profile is its own Modal app with its own
URL, so several can be deployed at the same time.
The app reaches it as `modal/<served_name>` — see `_modal_client` in
`agent_customer_support/llm/__init__.py`.

`QWEN_PROFILE` is read from the deploy shell AND written into the image env: Modal
re-imports this file inside the container, and a value that lived only in the deploy
shell would not be there, so the container would start a different model than the
one it was sized for.

Cold starts restore a memory snapshot (CPU + GPU) taken after vLLM has loaded,
compiled and warmed up, then put to sleep — so they skip the model load and compile
that used to take 5-7 minutes. The first container after a deploy still pays the
full start once, to build the snapshot.

This is a web endpoint, not an `@app.server`, on purpose: a Server rejects every
request with 503 while no container is ready, so with scale-to-zero each cold start
failed the chat turn. A web endpoint holds the request through the cold start instead
(answering 303 every 150 s, which the OpenAI SDK follows).

Auth is vLLM's `--api-key`, not Modal's proxy auth, so the app can use a plain
OpenAI client with a Bearer token. Without the key, anyone with the URL could spend
the GPU. The key only guards /v1/*, which is why vLLM listens on localhost and the
public URL is a proxy that forwards /v1/* alone (see `_proxy_app`).
"""

import json
import os
import subprocess
import time
import urllib.request

import modal

PROFILES: dict[str, dict] = {
    # Cheap test of the app plumbing (routing, JSON answers, citations).
    # T4 does NOT work for Qwen3.5: on Turing, vLLM hung compiling the Triton
    # linear-attention kernels until the 20 min startup timeout killed it. L4 (Ada)
    # gets the fused kernels and is in the vLLM recipe's supported list.
    "l4-9b": {
        "repo_id": "Qwen/Qwen3.5-9B",
        "served_name": "qwen3.5-9b",
        # The first app keeps its original names, so its URL
        # (<workspace>--qwen-vllm-serve.modal.run) does not change.
        "app_name": "qwen-vllm",
        "label": "qwen-vllm-serve",
        "gpu": "L4",
        "concurrency": 8,
        "args": [
            # Compose prompts run ~12.4K tokens plus the app's 4K output reserve (dev), so
            # 16K is too tight. This does not cost memory: the KV pool is sized by
            # --gpu-memory-utilization (~82K tokens here), and Qwen3.5 keeps KV only
            # for its 8 full-attention layers.
            "--max-model-len", "32768",
            "--max-num-seqs", "8",
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
    # Same 9B model, FP8 weights (W8A8, dynamic per-token activation scales) from
    # RedHatAI. About half the weight memory of BF16, so more of the L4 is left for
    # KV cache. L4 (Ada, sm_89) runs FP8 natively. Its own app and URL, so it can run
    # next to `l4-9b` for a side-by-side comparison.
    "l4-9b-fp8": {
        "repo_id": "RedHatAI/Qwen3.5-9B-FP8-dynamic",
        "served_name": "qwen3.5-9b-fp8",
        "app_name": "qwen-vllm-9b-fp8",
        "label": "qwen-vllm-9b-fp8-serve",
        "gpu": "L4",
        "concurrency": 8,
        # Same flags as `l4-9b`, so any difference comes from the weights alone.
        "args": [
            "--max-model-len", "32768",
            "--max-num-seqs", "8",
            "--gpu-memory-utilization", "0.92",
            "--language-model-only",
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
            "--override-generation-config",
            '{"temperature": 0.7, "top_p": 0.8, "top_k": 20}',
        ],
    },
    # The real target: Qwen3.8-27B FP8 on one H100.
    "h100-27b": {
        "repo_id": "Qwen/Qwen3.8-27B-FP8",
        "served_name": "qwen3.8-27b",
        "app_name": "qwen-vllm-27b",
        "label": "qwen-vllm-27b-serve",
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

# Chosen in the deploy shell: `QWEN_PROFILE=l4-9b-fp8 modal deploy qwen/serve.py`
# (or `make modal-deploy PROFILE=l4-9b-fp8`). The value is also baked into the image
# env below, so when Modal re-imports this file inside the container it reads the
# same profile the app was deployed with, not the default.
PROFILE = os.environ.get("QWEN_PROFILE", "l4-9b")
if PROFILE not in PROFILES:
    raise ValueError(f"Unknown QWEN_PROFILE {PROFILE!r}; pick one of {sorted(PROFILES)}")
CFG = PROFILES[PROFILE]
# Same layout download_model.py writes: /root/weights/<name part of the repo id>.
MODEL_DIR = f"/root/weights/{CFG['repo_id'].split('/')[-1]}"
MINUTES = 60

# vLLM listens here, on localhost only. The public port belongs to `_proxy_app`.
VLLM_PORT = 8001
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"

# One Modal app per profile: deploying one profile never replaces another.
app = modal.App(CFG["app_name"])

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.29.0")
    .env(
        {
            # Pins the profile inside the container (see PROFILE above).
            "QWEN_PROFILE": PROFILE,
            "VLLM_LOG_STATS_INTERVAL": "10",
            # Turns on vLLM's /sleep and /wake_up endpoints, which the snapshot needs.
            # It also turns on /collective_rpc and other admin routes that --api-key
            # does NOT guard (it only checks /v1/*) — hence the proxy below.
            "VLLM_SERVER_DEV_MODE": "1",
            # torch.compile's worker subprocesses break snapshot creation.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

weights_vol = modal.Volume.from_name("qwen-weights")
# torch.compile / CUDA graph artifacts — reusing them makes the snapshot build shorter.
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


def _post(path: str, body: bytes | None = None, headers: dict | None = None) -> None:
    req = urllib.request.Request(
        VLLM_URL + path, data=body or b"", headers=headers or {}, method="POST"
    )
    urllib.request.urlopen(req, timeout=5 * MINUTES).close()


def _wait_ready(proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{VLLM_URL}/health", timeout=2).close()
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError("vLLM did not become healthy in time")


@app.cls(
    image=image,
    gpu=CFG["gpu"],
    volumes={"/root/weights": weights_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[modal.Secret.from_name("qwen-vllm-api-key")],  # provides VLLM_API_KEY
    # Scale to zero: stay up 2 min after the last request, then stop billing.
    scaledown_window=2 * MINUTES,
    # Upper bound on one request; a knowledge compose call is well under this.
    timeout=10 * MINUTES,
    # Covers the slow path: model load + compile when a snapshot is being built.
    startup_timeout=20 * MINUTES,
    # Snapshot CPU *and* GPU memory after `start`, so a cold start restores a
    # loaded, compiled, warmed-up vLLM instead of repeating all of it. Snapshots are
    # built only for deployed apps (`modal deploy`, not `modal serve`), and every
    # redeploy that changes this file builds new ones.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=CFG["concurrency"])
class Qwen:
    @modal.enter(snap=True)
    def start(self):
        cmd = [
            "vllm",
            "serve",
            MODEL_DIR,
            "--served-model-name",
            CFG["served_name"],
            "--host",
            "127.0.0.1",
            "--port",
            str(VLLM_PORT),
            # Read here, during the snapshot: rotating the secret needs a redeploy.
            # KeyError here (not an open server) if the secret is missing.
            "--api-key",
            os.environ["VLLM_API_KEY"],
            # REQUIRED for every Qwen3.x profile: moves thinking into
            # `reasoning_content`, so `content` holds only the answer and
            # ComposedAnswer's JSON parses. It also makes vLLM apply the json_schema
            # constraint after thinking ends, not to the thinking itself.
            "--reasoning-parser",
            "qwen3",
            # Lets `start` move the weights to CPU before the snapshot is taken.
            "--enable-sleep-mode",
            *CFG["args"],
        ]
        self.proc = subprocess.Popen(cmd)
        _wait_ready(self.proc, timeout=18 * MINUTES)
        # A few real requests first, so lazy init (compile, CUDA graphs, tokenizer)
        # is already done inside the snapshot, not on the first user turn.
        warmup = json.dumps(
            {
                "model": CFG["served_name"],
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Xin chào"}],
            }
        ).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['VLLM_API_KEY']}",
        }
        for _ in range(3):
            _post("/v1/chat/completions", warmup, headers)
        # Level 1: weights go to CPU memory, the KV cache is dropped. The snapshot
        # is then mostly CPU memory, which restores faster than a full GPU image.
        _post("/sleep?level=1")

    @modal.enter(snap=False)
    def wake_up(self):
        # Runs after every restore: weights back to the GPU, KV cache re-allocated.
        _post("/wake_up")
        _wait_ready(self.proc, timeout=2 * MINUTES)

    @modal.exit()
    def stop(self):
        self.proc.terminate()

    # An ASGI app, like the web_server before it, holds a request through a cold
    # start instead of answering 503. The label comes from the profile, so each
    # app has its own URL: <workspace>--<label>.modal.run.
    @modal.asgi_app(label=CFG["label"])
    def serve(self):
        return _proxy_app()


def _proxy_app():
    """Forward only /v1/* to vLLM.

    Everything else vLLM serves in dev mode (/sleep, /wake_up, /collective_rpc,
    /invocations, ...) is not behind --api-key, so it must not be reachable from the
    public URL: /sleep alone would let anyone knock the server out.
    """
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from starlette.background import BackgroundTask

    client = httpx.AsyncClient(base_url=VLLM_URL, timeout=None)
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    # Hop-by-hop or length headers: the proxy re-frames the body itself.
    drop = {"host", "content-length", "transfer-encoding", "connection"}

    @api.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request):
        upstream = client.build_request(
            request.method,
            f"/v1/{path}",
            params=request.query_params,
            headers={k: v for k, v in request.headers.items() if k.lower() not in drop},
            content=await request.body(),
        )
        resp = await client.send(upstream, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers={k: v for k, v in resp.headers.items() if k.lower() not in drop},
            background=BackgroundTask(resp.aclose),
        )

    return api
