"""The graph retains one conversation across compiled invocations."""

from uuid import uuid4

from backend.chat.graph import nodes, workflow


def test_thread_messages_survive_graph_rebuild(monkeypatch):
    seen = []

    def classify(_client, messages):
        seen.append(messages)
        return {"intent": "guide"}

    monkeypatch.setattr(nodes, "classify", classify)
    config = {"configurable": {"thread_id": str(uuid4()), "client": object()}}
    workflow.build_graph().invoke({"message": "first", "messages": ["first"]}, config=config)
    workflow.build_graph().invoke({"message": "second", "messages": ["second"]}, config=config)
    assert seen == [["first"], ["first", "second"]]
