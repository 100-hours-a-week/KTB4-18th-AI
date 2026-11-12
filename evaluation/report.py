"""저장된 회귀 비교 결과를 Markdown으로 표시한다."""

import json

from .compare import LABELS, WAITS


def seconds(value):
    return f"{value['median']:.2f}초 (p90 {value['p90']:.2f}초)"


def usd(row):
    return (
        f"${row['per_request']:.6f}"
        if row["per_request"] is not None
        else f"누락 {row['missing']}건"
    )


def render(report):
    lines = []
    if report["incomplete"]:
        lines += [
            "## 확정 비교가 아니다. 완료된 공통 사례만 참고로 본다",
            *[f"- {item}" for item in report["incomplete"]],
            "",
        ]
    if report["conditions"]:
        lines += ["## 의도한 실험 변수", json.dumps(report["conditions"], ensure_ascii=False), ""]
    lines += [
        "## 완료 현황",
        "",
        "| 기능 | 기준 예정/실행/성공/채점 | 비교 예정/실행/성공/채점 |",
        "|---|---|---|",
    ]
    for kind, row in report["coverage"].items():
        cell = lambda r: (
            f"{r['planned']}/{r['run']}/{r['service_ok']}/{r['judged']}·{r['judge_total']}턴"
        )
        lines.append(f"| {LABELS[kind]} | {cell(row['base'])} | {cell(row['new'])} |")
    lines += [
        "",
        "## 사람이 보는 지표",
        "",
        "| 지표 | 기준 | 비교 | 바뀐 사례 |",
        "|---|---:|---:|---|",
    ]
    for kind, row in report["functional"].items():
        moved = (
            " ".join([f"+{i}" for i in row["improved"]] + [f"-{i}" for i in row["regressed"]])
            or "없음"
        )
        lines.append(
            f"| 기능 요구 충족 · {LABELS[kind]} | {row['base']}/{row['total']} | {row['new']}/{row['total']} | {moved} |"
        )
    for kind, row in report["faithfulness"].items():
        if row["total"]:
            moved = " ".join(dict.fromkeys(row["changed"])) or "없음"
            lines.append(
                f"| 안내 근거 충실도 · {LABELS[kind]} | {row['base']}/{row['total']} | {row['new']}/{row['total']} | {moved} |"
            )
    row = report["scene"]
    if report["quality_skipped"]:
        lines.append(
            "| 품질 지표 | 비교 안 함 | 비교 안 함 | 한쪽 또는 양쪽이 코드 검사만 실행했다 |"
        )
    elif row:
        moved = " ".join(row["changed"]) or "없음"
        lines.append(f"| 장면 해석 점수 (0~3) | {row['base']:.2f} | {row['new']:.2f} | {moved} |")
    else:
        lines.append("| 장면 해석 점수 (0~3) | 없음 | 없음 | 없음 |")
    repeat = report["song_repeat"]
    share = lambda value: "없음" if value is None else f"{value:.1%}"
    lines.append(
        f"| 곡 반복: 최다 곡이 나온 사례 비율 | {share(repeat['base'])} | {share(repeat['new'])} | - |"
    )
    for kind, rows in report["latency"].items():
        for key, label in WAITS:
            row = rows[key]
            if row:
                note = f"같은 턴 {row['n']}개" + (
                    f", 한쪽만 있는 턴 {row['excluded']}개 제외" if row["excluded"] else ""
                )
                lines.append(
                    f"| {label} · {LABELS[kind]} | {seconds(row['base'])} | {seconds(row['new'])} | {note} |"
                )
    for kind, row in report["request_cost"].items():
        lines.append(
            f"| 요청당 비용 · {LABELS[kind]} | {usd(row['base'])} | {usd(row['new'])} | 요청 {row['new']['requests']}건 |"
        )
    cost = report["service_cost"]
    lines.append(
        f"| 서비스 비용 합계, 재시도 포함 (USD) | {cost['base']['known']:.6f} | {cost['new']['known']:.6f} | - |"
    )
    lines += ["", "## 지표별 분모", "", "| 기능 | 검사 | 기준 | 비교 |", "|---|---|---:|---:|"]
    for kind, rows in report["denominators"].items():
        for key, row in rows.items():
            lines.append(
                f"| {LABELS[kind]} | {key} | {row['base'][0]:g}/{row['base'][1]} | {row['new'][0]:g}/{row['new'][1]} |"
            )
    retried = [
        f"- {LABELS[k]} {name}: 시도 {row['attempts']}회, 실패 {row['failed']}회, 재시도 사례 {' '.join(row['retried'])}"
        for k, sides in report["attempts"].items()
        for name, row in (("기준", sides["base"]), ("비교", sides["new"]))
        if row and row["retried"]
    ]
    if retried:
        lines += ["", "## 재시도 이력", "", "위 시간과 요청당 비용은 최종 시도 기준이다.", *retried]
    rescored_rows = [
        f"- {LABELS[k]} {side}: {' '.join(ids)}"
        for k, sides in report["rescored"].items()
        for side, ids in (("기준", sides["base"]), ("비교", sides["new"]))
        if ids
    ]
    if rescored_rows:
        lines += ["", "## 저장 당시와 현재 규칙의 판정이 다른 사례", "", *rescored_rows]
    lines += ["", "## 자동 경고", ""]
    lines += [f"- {item}" for item in report["warnings"]] or ["- 없음"]
    lines += ["", "## 비교 버전의 실패 사례", ""]
    lines += [f"- {case}: {', '.join(names)}" for case, names in report["failures"].items()] or [
        "- 없음"
    ]
    return "\n".join(lines)
