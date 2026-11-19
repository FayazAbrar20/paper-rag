import pytest
from fastapi.testclient import TestClient

import api
import ui


@pytest.fixture
def client(monkeypatch):
    def fake_answer(question):
        return {"answer": f"Echo: {question} [S1]", "sources": [{"paper": "Sarcasm", "page": 3, "score": 0.8}]}

    monkeypatch.setattr(api, "answer", fake_answer)
    return TestClient(api.app)


def test_ask_returns_answer_and_sources(client):
    response = client.post("/ask", json={"question": "  What was the best F1?  "})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Echo: What was the best F1? [S1]",
        "sources": [{"paper": "Sarcasm", "page": 3, "score": 0.8}],
    }


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_ask_rejects_empty_or_too_long_questions(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_ask_reports_unavailable_language_model(monkeypatch):
    def broken_answer(question):
        raise api.requests.ConnectionError("Ollama is not running")

    monkeypatch.setattr(api, "answer", broken_answer)
    response = TestClient(api.app).post("/ask", json={"question": "Hello?"})
    assert response.status_code == 503
    assert "Language model unavailable" in response.json()["detail"]


def test_chat_reply_lists_sources():
    reply = ui.format_reply({"answer": "F1 was 0.72 [S1].", "sources": [{"paper": "Sarcasm", "page": 3}]})
    assert reply == "F1 was 0.72 [S1].\n\n**Sources**\n- Sarcasm, page 3"


def test_chat_reply_for_refusal_has_no_sources_section():
    assert ui.format_reply({"answer": "I don't know based on the papers.", "sources": []}) == "I don't know based on the papers."
