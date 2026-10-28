"""머문음 Agent와 네 환경을 멱등 생성한다. 모델 호출·운영 자동 채점은 하지 않는다."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    client = Client()
    try:
        agent = client.request_with_retries("GET", "/api/v1/agents/meomuneum").json()
    except LangSmithNotFoundError:
        agent = client.request_with_retries("POST", "/api/v1/agents", json={
            "id": "meomuneum", "name": "머문음", "source_type": "EXTERNAL",
        }).json()
    for environment in agent["environments"]:
        expected = "meomuneum-" + environment["environment"].lower()
        if environment["tracer_session_name"] != expected:
            raise ValueError(f"Unexpected tracing destination: {environment['environment']}")
    print(json.dumps({"id": agent["id"], "name": agent["name"], "environments": [
        {"environment": e["environment"], "project": e["tracer_session_name"], "id": e["tracer_session_id"]}
        for e in agent["environments"]
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
