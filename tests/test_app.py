from fastapi.testclient import TestClient
from backend import main


def test_local_chat_page_uses_v1_fields() -> None:
    """테스트용 HTML 페이지가 v1에 맞게 나오는지 확인한다."""
    response = TestClient(main.app).get("/app/")

    assert response.status_code == 200
    assert 'const CHAT_URL = "/v1/chat/messages"' in response.text
    assert "thread_id" in response.text
    assert "request_id" in response.text
    assert "response.body.getReader()" in response.text
    assert "addTrackCards" in response.text
    assert "store_url" in response.text


def test_cold_app_import_does_not_require_feature_keys_or_db():
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    for key in ("DATABASE_URL", "OPENROUTER_LLM_API_KEY", "OPENROUTER_EMBEDDING_API_KEY",
                "OPENROUTER_STT_API_KEY", "LLM_MODEL"):
        env.pop(key, None)
    env["PYTHON_DOTENV_DISABLED"] = "1"
    result = subprocess.run([sys.executable, "-c", """
import sys
from fastapi.testclient import TestClient
from backend.main import app
assert 'db.models' not in sys.modules
assert TestClient(app).get('/health').json() == {'status': 'ok'}
"""], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
