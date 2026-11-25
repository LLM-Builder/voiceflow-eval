import pytest
from fastapi.testclient import TestClient

from app.config import Settings, hash_key
from app.main import create_app

A = {"Authorization": "Bearer key-a"}
B = {"Authorization": "Bearer key-b"}


@pytest.fixture()
def client(tmp_path):
    cfg = Settings(database_url=f"sqlite:///{tmp_path}/t.db",
                   api_key_hashes={hash_key("key-a"): "tenant-a", hash_key("key-b"): "tenant-b"},
                   run_rate_limit_per_min=100, auth_fail_limit_per_min=5)
    return TestClient(create_app(cfg))


PROMPT_V1 = {"name": "sales", "strategy": "zero_shot", "system_prompt": "Qualify leads.",
             "params": {"mock_profile": "weak"}}
PROMPT_V2 = {"name": "sales", "strategy": "structured_instruction", "system_prompt": "Qualify leads carefully.",
             "params": {"mock_profile": "full"}}
