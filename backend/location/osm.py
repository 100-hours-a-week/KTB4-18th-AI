"""OSM 지형 측정: 좌표 주변 지형까지의 거리, 들어 있는 용도지역·국립공원, 5km 안 봉우리.

입력 파일은 scripts/location/extract_osm.py가 만든 features.jsonl이다(LOCATION_OSM_FEATURES로 경로 변경).
shapely와 파일은 선택 사항이다. 둘 중 하나라도 없으면 available()이 False이고, 서비스는 카카오만으로 판정한다.
서버에 OSM을 어떻게 둘지(메모리·PostGIS)는 아직 정하지 않았다. 지금은 첫 요청 때 파일을 메모리에 올린다.
"""

import json
import math
import os
import threading
from functools import cache
from pathlib import Path

from backend.core.config import PROJECT_ROOT

DEFAULT_FEATURES = PROJECT_ROOT / "out" / "osm" / "features.jsonl"
SEARCH_M = 2000
PEAK_SEARCH_M = 5000
TOP_N = 3

_load_lock = threading.Lock()


def features_path() -> Path:
    return Path(os.getenv("LOCATION_OSM_FEATURES", "") or DEFAULT_FEATURES)


def available() -> bool:
    try:
        import shapely  # noqa: F401
    except ImportError:
        return False
    return features_path().exists()


@cache
def _index():
    """(행 목록, 도형 배열, 공간 인덱스). 약 20만 개·1~2초. 프로세스당 한 번만 읽는다"""
    import shapely

    rows = [json.loads(line) for line in features_path().open()]
    geoms = shapely.from_wkb([bytes.fromhex(r["wkb"]) for r in rows])
    return rows, geoms, shapely.STRtree(geoms)


def _query_around(lat: float, lng: float, radius_m: float):
    """반경 사각형 안 지형의 (행, 거리 m, 면적 ㎡)를 가까운 순으로"""
    import shapely
    from shapely.geometry import Point, box

    # 동시에 들어온 첫 요청들이 파일을 여러 번 읽지 않게 한다
    with _load_lock:
        rows, geoms, tree = _index()

    kx = 111_320 * math.cos(math.radians(lat))
    ky = 110_540
    idx = tree.query(box(lng - radius_m / kx, lat - radius_m / ky, lng + radius_m / kx, lat + radius_m / ky))
    # 좌표를 (lat, lng) 중심의 미터 평면으로 옮긴다. 수 km 범위라 등장방형 근사로 충분하다
    local = shapely.transform(geoms[idx], lambda c: (c - [lng, lat]) * [kx, ky])
    distances = shapely.distance(local, Point(0, 0))
    areas = shapely.area(local)  # 선·점은 0
    return sorted(((rows[i], float(d), float(a)) for i, d, a in zip(idx, distances, areas)), key=lambda t: t[1])


def measure(lat: float, lng: float) -> dict:
    """summarize_osm_nature·combine_kakao_osm에 넣을 측정값. relief는 서비스가 고도 API로 채운다"""
    nearest, landuse, protected = {}, [], []
    for row, dist, area in _query_around(lat, lng, SEARCH_M):
        # 용도지역·공원 구역은 거리 대신 "그 안에 있나"만 본다
        if row["kind"] == "landuse":
            if dist == 0:
                landuse.append(row["sub"])
            continue
        if row["kind"] == "protected":
            if dist == 0:
                protected.append(row["name"])
            continue
        hits = nearest.setdefault(row["kind"], [])
        if len(hits) < TOP_N:
            hits.append({
                "distance_m": round(dist), "area_m2": round(area),
                "name": row["name"], "sub": row["sub"], "ele": row["ele"],
            })

    peaks = []
    for row, dist, _ in _query_around(lat, lng, PEAK_SEARCH_M):
        if row["kind"] != "mountain" or not row["name"] or dist > PEAK_SEARCH_M:
            continue
        try:
            ele = float(row["ele"])
        except (TypeError, ValueError):  # 높이가 없거나 "632 m" 같은 형식
            continue
        peaks.append({"distance_m": round(dist), "name": row["name"], "ele": ele})
    return {"nearest": nearest, "landuse": landuse, "protected": protected, "peaks": peaks}
