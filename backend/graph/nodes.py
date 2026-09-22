"""RecommendationState를 받아 부분 상태(dict)를 반환하는 LangGraph 노드.

각 노드는 backend/recommendation.py, backend/music_search.py에 있는 기존 로직을
이 자리로 옮겨오면서(필요하면 기능 단위로 잘게 쪼개면서) 채운다.
"""
from state import RecommendationState
from ..recommendation import embed_query

from backend.graph.state import RecommendationState


def embed_node(state: RecommendationState) -> dict:
    """사용자 메시지를 검색용 벡터로 변환한다."""
    query_vector = embed_query(state["genai_client"], state["message"])
    return {"query_vector": query_vector}

def search_node(state: RecommendationState) -> dict:
    """query_vector로 pgvector 검색을 수행해 후보 곡과 무드 태그를 가져온다.

    TODO: recommendation.prepare_recommendation() 안의 db.search.search() 호출부 이관.
    """

    raise NotImplementedError


def reason_node(state: RecommendationState) -> dict:
    """검색된 곡마다 추천 이유를 채운다.

    TODO: recommendation.assign_reasons() 로직 이관.
    """

    raise NotImplementedError


def answer_node(state: RecommendationState) -> dict:
    """최종 추천 답변을 생성한다.

    TODO: recommendation.answer_input()/answer_instructions() 로직 이관.
    지금은 main.py의 stream_answer()가 SSE 스트리밍을 그래프 밖에서 처리하는데,
    이 노드 안으로 들여올지 계속 밖에 둘지 결정 필요.
    """

    raise NotImplementedError
