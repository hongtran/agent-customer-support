# tests/conftest.py
import os

os.environ.setdefault("PRODUCT_COLLECTION", "cenlab")
os.environ.setdefault("ENVIRONMENT", "dev")
os.environ.setdefault("AGENT_MODEL", "gpt-5.4-mini")
os.environ.setdefault("DYNAMODB_ENDPOINT_URL", "http://localhost:8000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "local")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "local")
os.environ.setdefault("AWS_REGION", "ap-southeast-1")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:4566")  # LocalStack
os.environ.setdefault("S3_BUCKET_ATTACHMENTS", "agent-customer-support-attachments")
os.environ.setdefault("QDRANT_ENDPOINT", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "local")
os.environ.setdefault("GOOGLE_API_KEY", "fake-google-for-tests")
os.environ.setdefault("QA_COLLECTION", "cenlab_qa")
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-for-tests")
os.environ.setdefault("TOGETHERAI_API_KEY", "fake-togetherai-for-tests")

# respx 0.21 + httpx 0.28 / httpcore 1.x compatibility fix:
# httpcore 1.x passes method as bytes; respx's HTTPCoreMocker.to_httpx_request
# forwards the raw bytes into httpx.Request, which keeps method as bytes.
# respx then compares bytes b'POST' against string 'POST' — no match.
# Fix: decode bytes method before constructing the httpx.Request.
import httpx as _httpx
from respx.mocks import HTTPCoreMocker as _HTTPCoreMocker


@classmethod  # type: ignore[misc]
def _patched_to_httpx_request(cls, **kwargs):
    request = kwargs["request"]
    method = request.method
    if isinstance(method, bytes):
        method = method.decode()
    raw_url = (
        request.url.scheme,
        request.url.host,
        request.url.port,
        request.url.target,
    )
    from respx.patterns import parse_url

    return _httpx.Request(
        method,
        parse_url(raw_url),
        headers=request.headers,
        stream=request.stream,
        extensions=request.extensions,
    )


_HTTPCoreMocker.to_httpx_request = _patched_to_httpx_request
