"""Environment-based configuration. Secrets are only ever read from the environment."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./voiceflow.db"
    # sha256(api_key) -> tenant_id. Keys themselves are never stored.
    api_key_hashes: dict[str, str] = field(default_factory=dict)
    max_body_bytes: int = 1_000_000
    run_rate_limit_per_min: int = 10
    auth_fail_limit_per_min: int = 10
    llm_provider: str = "mock"
    scheduler_provider: str = "mock"
    env: str = "development"


def load_settings() -> Settings:
    """VOICEFLOW_API_KEYS format: ``tenantA:keyA,tenantB:keyB``."""
    hashes: dict[str, str] = {}
    for pair in filter(None, os.environ.get("VOICEFLOW_API_KEYS", "").split(",")):
        tenant, _, key = pair.partition(":")
        if tenant and key:
            hashes[hash_key(key)] = tenant
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "sqlite:///./voiceflow.db"),
        api_key_hashes=hashes,
        max_body_bytes=int(os.environ.get("MAX_BODY_BYTES", "1000000")),
        run_rate_limit_per_min=int(os.environ.get("RUN_RATE_LIMIT_PER_MIN", "10")),
        auth_fail_limit_per_min=int(os.environ.get("AUTH_FAIL_LIMIT_PER_MIN", "10")),
        llm_provider=os.environ.get("LLM_PROVIDER", "mock"),
        scheduler_provider=os.environ.get("SCHEDULER_PROVIDER", "mock"),
        env=os.environ.get("APP_ENV", "development"),
    )
