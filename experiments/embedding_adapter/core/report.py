"""
결과 보고서: 화면에 출력하면서 같은 내용을 runs/{날짜}_{이름}/report.md에 남긴다
"""
from core.metrics import diff
from core.safe_io import new_run_dir, write_text

# 공통 지표 묶음의 열: (키, 머리글, 형식)
COLUMNS = [
    ("R@10", "R@10", ".4f"),
    ("MRR", "MRR", ".4f"),
    ("cold_R@10", "cold R@10", ".4f"),
    ("tail_warm_R@10", "비인기곡 warm R@10", ".4f"),
    ("overlap", "검색어 간 겹침", ".3f"),
    ("hub_top1", "허브 상위 1%", ".1%"),
    ("kpop", "top-10 kpop", ".1%"),
    ("coverage", "커버리지", ".1%"),
    ("cohesion_warm", "응집도 warm", ".4f"),
    ("cohesion_cold", "응집도 cold", ".4f"),
    ("sdd_R@10", "SDD R@10", ".2f"),
]


class Report:
    def __init__(self, name: str, title: str):
        self.name = name
        self.lines = [f"# {title}\n"]

    def add(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def table(self, rows: dict, columns=COLUMNS) -> None:
        """rows = {이름: evaluate_all/measure 결과}. 결과에 없는 열은 뺀다"""
        cols = [c for c in columns if any(c[0] in m for m in rows.values())]
        self.add("| 시스템 | " + " | ".join(h for _, h, _ in cols) + " |")
        self.add("|---|" + "---|" * len(cols))
        for name, m in rows.items():
            cells = [format(m[k], f) if k in m else "—" for k, _, f in cols]
            self.add(f"| {name} | " + " | ".join(cells) + " |")
        self.add()

    def ci(self, label: str, base: dict, other: dict, key: str = "R@10", digits: int = 4) -> tuple[float, float, float]:
        """other − base 의 paired bootstrap 95% 구간 한 줄. 하한 > 0 개선, 상한 < 0 악화, 그 밖 구분 불가"""
        d, lo, hi = diff(base, other, key)
        verdict = "개선" if lo > 0 else "악화" if hi < 0 else "구분 불가"
        self.add(f"- {label} {key}: {d:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] ({verdict})")
        return d, lo, hi

    def save(self) -> None:
        out = write_text(new_run_dir(self.name) / "report.md", "\n".join(self.lines) + "\n")
        print(f"\n저장: {out}")
