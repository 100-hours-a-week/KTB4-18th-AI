"""
임베딩 — CLAP 512 벡터와 mood 태그 56개

모델 로딩이 1~3분 걸리므로 프로세스당 1회만 하고 재사용한다.
지연 로딩(lazy)이라 실제로 임베딩을 요청할 때 처음 로드된다.

  오디오 → [CLAP]           → 512 벡터  ← 추천 순위 산출
  오디오 → [EffNet 백본]    → 1280
             └ [mood head]  → 태그 56개 ← 필터 및 추천 이유

EffNet 1280 벡터 자체는 저장하지 않는다. 서비스 흐름에 곡→곡 유사도 경로가
없기 때문이다. mood head가 백본 출력을 입력으로 받으므로 백본은 거쳐야 한다.
"""
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

ESSENTIA_FILES = {
    "discogs-effnet-bs64-1.pb":
        "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
        "discogs-effnet-bs64-1.pb",
    "mtg_jamendo_moodtheme-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/classification-heads/"
        "mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "mtg_jamendo_moodtheme-discogs-effnet-1.json":
        "https://essentia.upf.edu/models/classification-heads/"
        "mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json",
}

TOP_TAGS = 10
MIN_TAG_PROB = 0.05


def _fetch_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in ESSENTIA_FILES.items():
        dst = MODEL_DIR / name
        if not dst.exists():
            print(f"  모델 다운로드: {name}")
            urllib.request.urlretrieve(url, dst)


def _find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class Embedder:
    """모델을 한 번만 로드하고 재사용한다."""

    def __init__(self):
        self._clap = None
        self._backbone = None
        self._head = None
        self._tag_names = None
        self._ffmpeg = None

    # ── 지연 로딩 ───────────────────────────────────────
    def _load_clap(self):
        if self._clap is None:
            import laion_clap
            # amodel을 명시하면 기본 체크포인트와 차원이 어긋나 size mismatch가
            # 발생한다(768 vs 1024). 기본값끼리 짝을 맞춘다.
            m = laion_clap.CLAP_Module(enable_fusion=False)
            m.load_ckpt()
            self._clap = m
        return self._clap

    def _load_essentia(self):
        if self._backbone is None:
            import essentia
            essentia.log.warningActive = False
            essentia.log.infoActive = False
            from essentia.standard import (
                TensorflowPredict2D, TensorflowPredictEffnetDiscogs)
            _fetch_models()
            self._backbone = TensorflowPredictEffnetDiscogs(
                graphFilename=str(MODEL_DIR / "discogs-effnet-bs64-1.pb"),
                output="PartitionedCall:1",   # :1 임베딩, :0 은 400 style 확률
            )
            self._head = TensorflowPredict2D(
                graphFilename=str(
                    MODEL_DIR / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"))
            self._tag_names = json.loads(
                (MODEL_DIR / "mtg_jamendo_moodtheme-discogs-effnet-1.json")
                .read_text(encoding="utf-8"))["classes"]
        return self._backbone, self._head, self._tag_names

    # ── 텍스트 ─────────────────────────────────────────
    def embed_text(self, text: str) -> np.ndarray:
        """질의 문장 → 512 벡터.

        CLAP 텍스트 인코더는 roberta-base 기반 영어 전용이다. 한국어를 넣으면
        내용과 무관하게 거의 동일한 좌표가 나오므로, 반드시 영어 소리 서술을
        입력해야 한다.
        """
        m = self._load_clap()
        v = np.asarray(m.get_text_embedding([text, ""], use_tensor=False))[0]
        return v / np.maximum(np.linalg.norm(v), 1e-10)

    # ── 오디오 ─────────────────────────────────────────
    def embed_preview(self, url: str) -> tuple[np.ndarray, dict]:
        """미리듣기 URL → (512 벡터, mood 태그 dict)

        오디오 파일은 임시 디렉터리에 받았다가 함수가 끝나면 삭제된다.
        곡 단위 스트리밍 처리라 디스크에는 처리 중인 곡 하나만 존재한다.
        """
        from essentia.standard import MonoLoader

        backbone, head, tag_names = self._load_essentia()
        clap = self._load_clap()
        if self._ffmpeg is None:
            self._ffmpeg = _find_ffmpeg()

        with tempfile.TemporaryDirectory() as tmp:
            m4a = Path(tmp) / "a.m4a"
            wav = Path(tmp) / "a.wav"

            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r, open(m4a, "wb") as f:
                f.write(r.read())

            # EffNet은 16kHz. MonoLoader가 m4a를 직접 읽고 리샘플한다.
            audio = MonoLoader(filename=str(m4a), sampleRate=16000,
                               resampleQuality=4)()
            emb = backbone(audio)                      # (패치 수, 1280)
            tag_probs = np.mean(head(emb), axis=0)     # (56,)

            # CLAP은 48kHz wav를 기대한다
            subprocess.run(
                [self._ffmpeg, "-y", "-loglevel", "error", "-i", str(m4a),
                 "-ac", "1", "-ar", "48000", str(wav)], check=True)
            v = np.asarray(clap.get_audio_embedding_from_filelist(
                x=[str(wav)], use_tensor=False))[0]

        v = v / np.maximum(np.linalg.norm(v), 1e-10)
        return v.astype(np.float32), self.top_tags(tag_probs, tag_names)

    @staticmethod
    def top_tags(probs: np.ndarray, names: list[str]) -> dict:
        out = {}
        for j in np.argsort(-probs)[:TOP_TAGS]:
            p = float(probs[j])
            if p < MIN_TAG_PROB:
                break
            out[names[j]] = round(p, 3)
        return out


# 프로세스당 하나만 쓰면 되므로 모듈 수준에 둔다
embedder = Embedder()
