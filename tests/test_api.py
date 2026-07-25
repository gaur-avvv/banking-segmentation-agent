from fastapi.testclient import TestClient

from banking_agent.api import create_app


def test_web_ui_and_agent_card_are_available():
    client = TestClient(create_app("data"))
    assert client.get("/health").status_code == 200
    assert client.get("/ui").status_code == 200
    card = client.get("/.well-known/agent.json").json()
    assert card["name"] == "banking-segmentation-orchestrator"
    assert {skill["id"] for skill in card["skills"]} >= {"segment_customers", "compare_segments"}
