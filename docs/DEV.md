# Developer Setup

## LLM layer (enterprise_llm_service)

`enterprise_llm_service` is installed from a local wheel rather than a registry package,
because it has a private GitLab dependency in its declared dependency tree.
Runtime dependencies are declared in `pyproject.toml`; the wheel itself is installed
separately without its deps.

### Steps to reproduce

1. Install the wheel (no deps):
   ```
   poetry run pip install --no-deps "/Users/hongtran/Projects/enterprise-llm-service/dist/enterprise_llm_service-1.0.3-py3-none-any.whl"
   ```

2. Runtime deps (`anthropic`, `tiktoken`, `google-genai`, `openai`, `environs`, `tenacity`)
   are already declared in `pyproject.toml` and installed by `poetry install`.

3. `enterprise_llm_service.config.global_config` reads several env vars at module
   import time with **no defaults** — they must be set before importing. Copy `.env-example`
   to `.env` and fill in real values (or keep the stubs for local dev without those services):
   ```
   cp .env-example .env
   ```
   Required stubs (already in `.env-example`):
   - `QDRANT_ENDPOINT`, `QDRANT_API_KEY`
   - `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
   - `TOGETHERAI_API_KEY` (OpenAI client is instantiated at import time with this key)

4. Verify:
   ```
   poetry run python -c "from enterprise_llm_service.llm_inference import ai_completion_with_tools; from enterprise_llm_service.llm_inference.llm_inference_base import ai_completion; print('ok')"
   ```

### Note on `ai_completion` import path

In wheel version 1.0.3, `ai_completion` is defined in
`enterprise_llm_service.llm_inference.llm_inference_base` but is **not re-exported** from
the package `__init__.py`. Use the submodule import:

```python
# Correct:
from enterprise_llm_service.llm_inference import ai_completion_with_tools
from enterprise_llm_service.llm_inference.llm_inference_base import ai_completion

# This will fail (ai_completion not in __init__.py):
# from enterprise_llm_service.llm_inference import ai_completion, ai_completion_with_tools
```

### Reproducibility concern

The `pip install --no-deps <wheel>` step is a **manual step** not captured by Poetry.
Anyone setting up from scratch must run step 1 above before `poetry install`.
The wheel path is local; if the wheel is rebuilt (version bump), update the path accordingly.
