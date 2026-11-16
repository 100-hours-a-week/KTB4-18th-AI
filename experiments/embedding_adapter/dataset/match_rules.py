"""
곡 매칭 규칙 (2_match_melon, overlap, 3_tail_songs가 함께 쓴다)

  norm         소문자, 악센트·특수문자 제거
  title_key    버전 표기(Remix, Inst., Live 등)가 있으면 None. 소리가 다른 곡이므로 매칭하지 않는다
  artist_keys  "에픽하이 (EPIK HIGH)" → [에픽하이, epikhigh]
  title_aliases  "으르렁 (Growl)" → [Growl]  (괄호 속 다른 제목, 매칭 v3·iTunes 검색용)
"""
import re
import unicodedata

PARENS = r"\(.*?\)|\[.*?\]"
# NOTE: 단어 단위로만 본다. 부분 문자열이면 forever(ver), Delivery(live)에도 걸린다
VERSION_RE = re.compile(r"\b(remix|mix|inst|instrumental|live|acoustic|edit|mr|remaster\w*|ver|version|remake"
                        r"|acapella|accapella|a cappella|piano|sessions?|unplugged|demo)\b")
# NOTE: 이름에 ver, mix가 들어가도 소리는 같은 곡. Original Mix는 EDM에서 원곡 표기
SAME_SOUND_WORDS = ("explicit", "clean", "taylor", "original mix")
HANGUL = re.compile(r"[가-힣]")


def norm(s: str) -> str:
    """악센트 제거(Ü→U) → 소문자화 → 글자/숫자만 남김"""
    # NOTE: NFKD로 쪼개면 한글도 자모로 분리되므로 악센트만 버리고 NFC로 다시 합친다
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[^0-9a-z가-힣ㄱ-ㅎぁ-んァ-ン一-龥]", "", s.lower())


def title_key(title: str) -> str | None:
    """버전 표기가 있으면 None, 아니면 괄호를 지운 정규화 제목"""
    for inside in re.findall(PARENS, title.lower()):
        if any(w in inside for w in SAME_SOUND_WORDS):
            continue
        if VERSION_RE.search(inside):
            return None
    return norm(re.sub(PARENS, "", title)) or None


# NOTE: 같은 아티스트의 여러 곡에 붙는 흔한 괄호 표기. 별칭으로 쓰면 다른 곡과 잘못 붙는다
GENERIC_ALIASES = {"intro", "outro", "interlude", "skit", "prologue", "epilogue", "bonustrack", "hiddentrack",
                   "titletrack", "solo", "freestyle", "stripped", "rearranged"}


NON_LATIN = re.compile(r"[가-힣ㄱ-ㅎ一-龥ぁ-んァ-ン]")


def title_aliases(title: str) -> list[str]:
    """제목 괄호 속 번역 이름: "작은 것들을 위한 시 (Boy With Luv)" → ["Boy With Luv"]
    멜론은 "한글 (English)", iTunes는 영어 제목만 적는 K-pop이 많다. feat.·버전·흔한 표기는 제외

    NOTE: 괄호 밖 제목에 한글·한자·일본어가 있을 때만 (= 번역 쌍). 괄호 밖이 영어면 괄호 속은 부제라
      ("Ribbons And Bows (From The Kacey Musgraves Christmas Show)") 같은 부제의 다른 곡과 잘못 붙는다
    """
    if not NON_LATIN.search(re.sub(PARENS, "", title)):
        return []
    insides = [x[1:-1].strip() for x in re.findall(PARENS, title)]
    # NOTE: "Boy With Luv"는 별칭이고 "with Zion.T"는 협업 표기 → 앞에 올 때만 협업으로 본다
    return [x for x in insides if norm(x) and not re.match(r"(feat|prod|with)\b", x.lower())
            and not VERSION_RE.search(x.lower()) and "reimagined" not in x.lower()
            and norm(x) not in GENERIC_ALIASES]


def artist_keys(artist: str) -> list[str]:
    """첫 번째 아티스트의 괄호 밖 이름, 괄호 안 이름 (예: 에픽하이 (EPIK HIGH) → [에픽하이, epikhigh])"""
    first = re.split(r",|&| feat\.? ", artist)[0]
    names = [re.sub(PARENS, "", first)] + [x[1:-1] for x in re.findall(PARENS, first)]
    return [k for k in map(norm, names) if k]
