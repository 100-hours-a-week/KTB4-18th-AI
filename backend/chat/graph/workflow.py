"""추천 파이프라인 LangGraph 조립.

노드 구현이 끝나면 여기서 그래프 형태(분기·재시도 등)를 조정한다.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from backend.chat.graph.nodes import (
    classify_node,
    embed_node,
    jev_node,
    lookup_node,
    search_node,
)
from backend.chat.graph.state import RecommendationState

# NOTE: 채팅방을 나가면 대화 기억도 사라져도 되는 요구사항(장기 저장 불필요)이라
# DB 기반 checkpointer 대신 프로세스 메모리에만 저장하는 InMemorySaver를 쓴다.
# 그래프를 다시 구성해도 thread_id별 기억이 이어지도록 이 저장소 인스턴스는 공유한다.
# NOTE: InMemorySaver는 프로세스 메모리라서, 서버가 여러 워커/인스턴스로 뜨면
# 같은 thread_id 요청이 다른 프로세스로 갈 수 있어 기억이 안 보일 수 있다.
# 지금 규모에선 괜찮지만 배포 구조가 바뀌면(워커 여러 개, 여러 대) 재검토할 것.
_checkpointer = InMemorySaver()


def route_entry(state: RecommendationState) -> str:
    """고속 모드가 켜진 대화면 classify(LLM) 대신 jev로 들어간다."""

    return "jev" if state.get("fast_mode") else "classify"


def route_by_jev(state: RecommendationState) -> str:
    """추천은 jev_node가 Jev와 동시에 임베딩까지 해뒀으니 바로 검색으로 보낸다. 조회는 jev_node가
    곡을 찾았으면 끝내고, 못 찾았으면 제목·아티스트 추출을 위해 classify로 보낸다.
    guide/out_of_scope는 고정 안내라 끝낸다. Jev가 실패했거나 미지원 조건을 감지했으면 classify로 보낸다."""

    if state["jev_failed"] or state["jev_unsupported"]:
        return "classify"
    intent = state["intent"]
    if intent == "recommend":
        return "search"
    if intent == "lookup":
        return END if state["lookup_status"] == "found" else "classify"
    return END


def route_by_intent(state: RecommendationState) -> str:
    """classify_node가 채운 intent·조건 필드에 따라 다음 노드를 정한다.

    guide/clarify/out_of_scope와, recommend인데 지원 불가능한 조건·정보 부족인
    경우는 더 할 DB 작업이 없으므로 그래프를 바로 끝낸다. prepare_recommendation이
    intent와 각 필드를 보고 어떤 고정/생성 안내를 내려줄지 결정한다.
    """

    intent = state["intent"]
    count = state.get("requested_count")
    if intent in ("recommend", "lookup") and count is not None and count < 1:
        return END
    if intent == "recommend":
        if state["recommend_unsupported_condition"] or not state["recommend_has_enough_info"]:
            return END
        return "embed"
    if intent == "lookup":
        return "lookup"
    return END


def build_graph(*, photo: bool = False):
    """RecommendationState 기반 StateGraph를 구성해 컴파일한다.
    최종 답변 생성(SSE 스트리밍, stream_answer/stream_lookup_answer)은 그래프
    밖에서 그대로 처리한다.
    """

    graph = StateGraph(RecommendationState)

    graph.add_node("classify", classify_node)
    graph.add_node("embed", embed_node)
    graph.add_node("search", search_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("jev", jev_node)

    # NOTE: 사진은 VLM이 만든 검색 조건으로 진입하므로 텍스트 의도 분류를 생략한다.
    if photo:
        graph.add_edge(START, "embed")
    else:
        graph.add_conditional_edges(START, route_entry, {"classify": "classify", "jev": "jev"})
    graph.add_conditional_edges(
        "classify", route_by_intent, {"embed": "embed", "lookup": "lookup", END: END},
    )
    graph.add_conditional_edges(
        "jev", route_by_jev, {"search": "search", "classify": "classify", END: END},
    )
    graph.add_edge("embed", "search")
    graph.add_edge("search", END)
    graph.add_edge("lookup", END)

    return graph.compile(checkpointer=_checkpointer)


def record_shown_tracks(thread_id: str, tracks: list[dict]) -> None:
    """검색 성공이 아니라 SSE 카드 전송 시점에 최근 목록과 중복 제외 ID를 저장한다."""
    if not tracks:
        return
    build_graph().update_state(
        {"configurable": {"thread_id": thread_id}},
        {"shown_track_ids": [track["track_id"] for track in tracks],
         "recent_tracks": [
             {"rank": rank, "track_id": track["track_id"],
              "title": track["title"], "artist": track["artist"]}
             for rank, track in enumerate(tracks, start=1)
         ]},
        # 검색·조회 모두 종료된 상태를 갱신하며 후속 노드를 예약하지 않는다.
        as_node="search",
    )
