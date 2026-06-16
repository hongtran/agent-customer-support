from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    rag_base_url: str = "http://localhost:7799"
    product_collection: str = "cenlab"
    agent_model: str = "gpt-4o-mini"

    # Per-agent model overrides — default to agent_model if unset
    triage_model: str | None = "gpt-4o-mini"
    knowledge_model: str | None = "gpt-4o"
    knowledge_grader_model: str | None = "gpt-4o-mini"
    knowledge_contextualize_model: str | None = "gpt-4o-mini"
    verification_model: str | None = "gpt-4o-mini"
    flow_model: str | None = "gpt-4o-mini"
    diagnostic_model: str | None = "gpt-4o-mini"

    def model_for(self, agent: str) -> str:
        return getattr(self, f"{agent}_model", None) or self.agent_model

    dynamodb_endpoint_url: str | None = None
    aws_region: str = "ap-southeast-1"

    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 3600

    zalo_cs_webhook_url: str | None = None

    # table names
    table_customers: str = "acs_customers"
    table_flows: str = "acs_flows"
    table_conversations: str = "acs_conversations"
    table_requests: str = "acs_requests"

    # observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
