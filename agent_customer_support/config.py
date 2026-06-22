from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # RAG (Qdrant read path)
    qdrant_endpoint: str = "http://localhost:6333"
    qdrant_api_key: str = "dummy"
    google_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 3072
    product_collection: str = "cenlab"
    qa_collection: str = "cenlab_qa"
    agent_model: str = "gpt-4o-mini"

    # Per-agent model overrides — default to agent_model if unset
    triage_model: str | None = "gpt-4o-mini"
    knowledge_model: str | None = "gpt-4o"
    knowledge_contextualize_model: str | None = "gpt-4o"
    verification_model: str | None = "gpt-4o-mini"
    flow_model: str | None = "gpt-4o-mini"

    def model_for(self, agent: str) -> str:
        return getattr(self, f"{agent}_model", None) or self.agent_model

    dynamodb_endpoint_url: str | None = None
    aws_region: str = "ap-southeast-1"

    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 3600

    zalo_cs_webhook_url: str | None = None
    admin_token: str = ""

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
