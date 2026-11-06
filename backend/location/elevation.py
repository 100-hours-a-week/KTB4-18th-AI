"""주변 ±1km 고도 기복. OSM 산 판정(context.summarize_osm_nature)과 산 이름 고르기에 쓴다.

assets/korea_9s.npy는 scripts/location/build_dem.py가 Copernicus DEM GLO-90으로 만든 한국 고도 격자다(int16, m).
Open-Meteo 고도 API도 같은 자료다. 요청마다 좌표 81개를 물으면 무료 한도(일 1만)가 하루 120번 남짓 추천에 끝나서
파일로 들고 있는다. 9초 격자(남북 약 280m)로 줄였지만 35곳의 산 판정은 Open-Meteo 값과 같았다.
memmap으로 열어 요청마다 81칸만 디스크에서 읽으므로 메모리를 거의 쓰지 않는다.
"""

import math
from functools import cache
from pathlib import Path

import numpy as np

ASSET = Path(__file__).parent / "assets" / "korea_9s.npy"
# 격자 [0, 0]이 북서쪽 끝(위도 39, 경도 124)이고 1도를 400칸(9초)으로 나눈다. build_dem.py와 같아야 한다
TOP_LAT, LEFT_LNG, CELLS_PER_DEG = 39, 124, 400
# 9×9 지점, 250m 간격 → ±1km. 산 판정 기준(context.MOUNTAIN_MIN_RELIEF_M)을 이 간격으로 맞췄다
GRID_STEP_M = 250
OFFSETS = range(-4, 5)


@cache
def _grid() -> np.ndarray:
    return np.load(ASSET, mmap_mode="r")


def relief(lat: float, lng: float) -> dict | None:
    """주변 9×9 지점의 최저·최고 고도와 그 차이(기복). 격자 범위(한국) 밖이면 None"""
    grid = _grid()
    d_lat = GRID_STEP_M / 110_540
    d_lng = GRID_STEP_M / (111_320 * math.cos(math.radians(lat)))
    rows = [math.floor((TOP_LAT - (lat + i * d_lat)) * CELLS_PER_DEG) for i in OFFSETS]
    cols = [math.floor((lng + j * d_lng - LEFT_LNG) * CELLS_PER_DEG) for j in OFFSETS]
    # 음수 칸 번호는 numpy가 반대쪽 끝으로 읽어 버리므로 범위를 먼저 확인한다
    if min(rows) < 0 or min(cols) < 0 or max(rows) >= grid.shape[0] or max(cols) >= grid.shape[1]:
        return None
    heights = grid[np.ix_(rows, cols)]
    low, high = int(heights.min()), int(heights.max())
    return {"min_m": low, "max_m": high, "relief_m": high - low}
