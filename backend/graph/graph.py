"""추천 파이프라인 LangGraph 조립.

노드 구현이 끝나면 여기서 그래프 형태(분기·재시도 등)를 조정한다.
"""

from langgraph.graph import END, START, StateGraph

from backend.graph.nodes import embed_node, reason_node, search_node
from backend.graph.state import RecommendationState


def build_graph():
    """RecommendationState 기반 StateGraph를 구성해 컴파일한다.

    최종 답변 생성(SSE 스트리밍, stream_answer)은 그래프 밖에서 그대로 처리한다.
    """

    graph = StateGraph(RecommendationState)

    graph.add_node("embed", embed_node)
    graph.add_node("search", search_node)
    graph.add_node("reason", reason_node)

    graph.add_edge(START, "embed")
    graph.add_edge("embed", "search")
    graph.add_edge("search", "reason")
    graph.add_edge("reason", END)

    # TODO: tracks가 비어있으면 reason을 건너뛰고 바로 END로 가는
    # 조건부 분기(add_conditional_edges)를 추가할지 검토.

    return graph.compile()
