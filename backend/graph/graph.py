"""추천 파이프라인 LangGraph 조립.

노드 구현이 끝나면 여기서 그래프 형태(분기·재시도 등)를 조정한다.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from backend.graph.nodes import classify_node, embed_node, reason_node, search_node
from backend.graph.state import RecommendationState

# NOTE: 채팅방을 나가면 대화 기억도 사라져도 되는 요구사항(장기 저장 불필요)이라
# DB 기반 checkpointer 대신 프로세스 메모리에만 저장하는 InMemorySaver를 쓴다.
# 이 인스턴스와 컴파일된 그래프는 요청마다 새로 만들면 안 되고, 서버가 떠
# 있는 동안 한 번만 만들어져 재사용돼야 thread_id별 기억이 실제로 유지된다.
# NOTE: InMemorySaver는 프로세스 메모리라서, 서버가 여러 워커/인스턴스로 뜨면
# 같은 thread_id 요청이 다른 프로세스로 갈 수 있어 기억이 안 보일 수 있다.
# 지금 규모에선 괜찮지만 배포 구조가 바뀌면(워커 여러 개, 여러 대) 재검토할 것.
_checkpointer = InMemorySaver()


def route_by_intent(state: RecommendationState) -> str:
    """classify_node가 채운 intent에 따라 다음 노드를 정한다."""

    # TODO: guide/clarify/lookup 분기가 생기면 각자의 노드로 연결한다.
    # 지금은 recommend가 아니면 바로 끝낸다.

    if state["intent"] == "recommend":
        return "embed"
    return END


def build_graph():
    """RecommendationState 기반 StateGraph를 구성해 컴파일한다.

    최종 답변 생성(SSE 스트리밍, stream_answer)은 그래프 밖에서 그대로 처리한다.
    """

    graph = StateGraph(RecommendationState)

    graph.add_node("classify", classify_node)
    graph.add_node("embed", embed_node)
    graph.add_node("search", search_node)
    graph.add_node("reason", reason_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_by_intent, {"embed": "embed", END: END})
    graph.add_edge("embed", "search")
    graph.add_edge("search", "reason")
    graph.add_edge("reason", END)

    # TODO: tracks가 비어있으면 reason을 건너뛰고 바로 END로 가는
    # 조건부 분기(add_conditional_edges)를 추가할지 검토.
    # TODO: classify에서 recommend로 판단했더라도, embed 이후(예: search 결과가
    # 없거나 사용자가 지원 불가능한 조건을 요구한 경우) guide/clarify 같은 다른
    # 분기로 다시 튀어야 하는 경우가 있을 수 있다. 지금은 embed→search→reason이
    # 한 번 들어가면 끝까지 직진하는 구조라 이런 재분기가 없다.

    return graph.compile(checkpointer=_checkpointer)
