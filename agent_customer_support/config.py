from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "prod"]
ReasoningEffort = Literal["minimal", "low", "medium", "high"]

# Enforced per-environment reasoning profile. Deliberately not overridable by env
# vars: prod always reasons hard, dev always stays cheap and fast.
_REASONING_EFFORT_BY_ENV: dict[str, ReasoningEffort] = {"dev": "low", "prod": "high"}

# On reasoning models the cap covers reasoning tokens *plus* visible output, so it
# has to sit well above the old 5000 — otherwise a high-effort turn can burn the
# whole budget on reasoning and return empty text.
_MAX_OUTPUT_TOKENS_BY_ENV: dict[str, int] = {"dev": 4000, "prod": 8000}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Deployment environment — drives the enforced reasoning profile below.
    environment: Environment = "dev"

    # RAG (Qdrant read path)
    qdrant_endpoint: str = "http://localhost:6333"
    qdrant_api_key: str = "dummy"
    google_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 3072
    product_collection: str = "cenlab"
    qa_collection: str = "cenlab_qa_dev"
    qa_lead_threshold: float = 0.85
    agent_model: str = "gpt-5.4-mini"

    # Per-agent model overrides — default to agent_model if unset
    triage_model: str | None = "gpt-5.4-mini"
    knowledge_model: str | None = "gpt-5.6-luna"
    knowledge_contextualize_model: str | None = "gpt-5.4-mini"
    verification_model: str | None = "gpt-5.4-mini"
    flow_model: str | None = "gpt-5.4-mini"
    guardrail_model: str | None = "gpt-5.4-mini"

    dynamodb_endpoint_url: str | None = None
    dynamodb_auto_create_tables: bool = True
    aws_region: str = "ap-southeast-1"

    # Attachment storage (S3). Image bytes never go into DynamoDB — the conversation
    # is one item and DynamoDB caps items at 400 KB. Set s3_endpoint_url to
    # http://localhost:4566 for LocalStack; leave unset to hit real AWS.
    s3_endpoint_url: str | None = None
    s3_bucket_attachments: str = "agent-customer-support-attachments"
    s3_auto_create_bucket: bool = True
    s3_presign_expiry_seconds: int = 3600
    # Rejected at the API boundary before any LLM spend. 5 MB sits near Anthropic's
    # own per-image ceiling, so anything larger would fail downstream anyway.
    max_attachment_bytes: int = 5 * 1024 * 1024

    # Document images (S3, read-only). Screenshots and button glyphs extracted from the
    # source user guides, keyed by application slug: <prefix>/<slug>/imageNN.png. A
    # separate bucket from s3_bucket_attachments on purpose — user uploads are private
    # per-conversation data, these are shared assets with a different lifecycle.
    s3_bucket_doc_images: str = "elsa-enterprise-llm-service-local"
    doc_images_prefix: str = "doc_images"
    # Which names exist under a slug is read from S3 and cached in-process. The bucket
    # only changes when a document is re-ingested, so a coarse TTL is fine and keeps
    # this to one listing per application per process.
    doc_images_cache_seconds: int = 600
    # Ceiling on how many images one reply may carry, applied after composition.
    max_reply_images: int = 5

    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 3600

    zalo_cs_webhook_url: str | None = None

    # Auth. jwt_secret has no usable default on purpose — a shipped signing secret is
    # the same class of bug as no auth at all, so server startup refuses to run without
    # it (see server.py lifespan) rather than quietly signing tokens anyone can forge.
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24h

    # comma-separated list of allowed CORS origins
    cors_allowed_origins: str = "http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    # table names
    table_customers: str = "acs_customers"
    table_flows: str = "acs_flows"
    table_conversations: str = "acs_conversations"
    table_requests: str = "acs_requests"
    table_qa: str = "acs_qa"

    # observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    def model_for(self, agent: str) -> str:
        return getattr(self, f"{agent}_model", None) or self.agent_model

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        """Reasoning effort for OpenAI reasoning models — fixed by environment."""
        return _REASONING_EFFORT_BY_ENV[self.environment]

    @property
    def max_output_tokens(self) -> int:
        """Output token ceiling (reasoning + visible text) — fixed by environment."""
        return _MAX_OUTPUT_TOKENS_BY_ENV[self.environment]


@lru_cache
def get_settings() -> Settings:
    return Settings()
