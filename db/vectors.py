import numpy as np


def decode_vector(raw) -> np.ndarray:
    """DB 저장 형식 → numpy"""
    if raw is None:
        return None
    return np.asarray(raw, dtype=np.float32)


def l2norm(v: np.ndarray) -> np.ndarray:
    """바로 코사인 유사도 계산되게 미리 정규화"""
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-10)
