"""
4-5. 쿼리 어댑터 학습 (Low-Rank Residual + multi-positive InfoNCE)

  q_c = norm(q − μ_q)                    평균 제거 (고정)
  q_a = norm(q_c + q_c·A·B)              학습. B = 0에서 시작 → 처음엔 아무 변환도 안 함
  s′  = norm(s − μ_s)                    곡 (고정 → 서비스 DB를 다시 만들 필요가 없다)
  손실 = multi_positive_infonce(q_a · s′ᵀ, 정답 마스크, τ = 0.05)

입력      데이터셋(기본 service_format)의 train 검색어·정답 warm 곡, 카탈로그, val(학습 중 확인용)
            학습 검색어: 비인기곡 아닌 정답이 3곡 이상인 것 (17,802개 중 16,939개)
            오답 후보:  멜론 매칭 곡 − cold 곡. 멜론에 없는 곡은 라벨이 없을 뿐 오답이 아니다
                       (처음에 카탈로그 전체를 오답으로 썼더니 "멜론에 있는 곡인가"를 배우는 지름길이 생겼다: --gallery all)
출력      runs/{날짜}_train_query_adapter_{tag}/checkpoint.pt  {"A", "B", "mu_q", "mu_s", "config", "val", …}
          화면: 에폭마다 val R@10·R@50·cold, 끝에 학습 전(W = I)과 나란히 공통 지표
설정      rank 512, τ 0.05, AdamW lr 1e-3 wd 0.01, batch 256 (1에폭 = 67스텝), seed 0, 5에폭 고정·마지막 시점
          조기 종료를 쓰지 않는 이유: val cold R@50은 평가마다 ±0.004 흔들려 촘촘히 볼수록 잡음 중 최고를 고르게 된다
          NOTE: 같은 시드라도 MPS에서는 소수점 끝이 다를 수 있다. 실험 결과는 원래 체크포인트(config.CHECKPOINTS)로 재현한다

이 저장소의 체크포인트와 학습 옵션의 관계
  query_adapter_r{32,128,512,3072}   --rank 32 / 128 / 512 / 3072
  query_adapter_r512_tail_pos_only   --tail_pos_only                 (5-7 C)
  query_adapter_r512_melon_format    --dataset melon_format          (5-8)
  예전 형식 검색어 실험(5-2~5-4)의 옵션도 남겨 둔다: --lam(인기도 보정), --pos_weight, --select cold_R@50(조기 종료),
  --drop_tail_labels(5-7 대조군 B), --act gelu(비선형)

사용:
  python -m train.train_query_adapter --tag r512
  python -m train.train_query_adapter --rank 32 --tag r32
  python -m train.train_query_adapter --demo        # InfoNCE 숫자 예시 (문서 4-4)
"""
import argparse
import time

import numpy as np
import torch

import config
from core.adapters import LowRankAdapter, multi_positive_infonce
from core.data import load_bench
from core.metrics import evaluate, evaluate_all
from core.report import Report
from core.safe_io import guarded_output, new_run_dir
from core.systems import Space, center, means

DEFAULTS = {
    "dataset": "service_format",
    "train_gallery": "melon_minus_cold",   # 학습 오답 후보
    "select_metric": "last",               # 저장 시점 (last = 조기 종료 없음)
    "logq_lambda": 0.0,                    # 인기도 보정 세기 (0 = 없음)
    "pos_weight_power": 0.0,               # 정답 가중치 1/빈도^γ의 γ (0 = 없음)
    "drop_tail_labels": False,             # 대조군: 비인기곡을 학습 정답·오답 후보에서 뺀다
    "min_nontail_pos": 3,                  # 비인기곡 아닌 warm 정답이 이 개수 이상인 train 검색어만
    "tail_pos_only": False,                # 비인기곡을 정답으로만 쓰고 오답으로는 쓰지 않는다
    "act": None,                           # 어댑터 비선형 (None = 선형, "gelu")
    "seed": 0,
    "rank": 512,
    "tau": 0.05,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 256,
    "max_epochs": 5,
    "eval_every": None,                    # 갱신 몇 번마다 val 평가 (None = 1에폭마다)
    "patience": 5,                         # select_metric이 이 횟수만큼 안 나아지면 멈춘다 (last면 무관)
}


def demo_infonce() -> None:
    """곡 5개, 정답 A·C 예시로 τ와 마스킹의 효과를 숫자로 본다"""
    #                       A     B     C     D     E
    scores = torch.tensor([[0.9, 0.2, 0.5, 0.1, 0.3]])
    pos = torch.tensor([[True, False, True, False, False]])
    for tau in (1.0, 0.05):
        print(f"\nτ = {tau}")
        for name, i in (("A", 0), ("C", 2)):
            p = torch.softmax(scores[0, [i, 1, 3, 4]] / tau, dim=0)[0]
            print(f"  문제 '{name}를 고르시오' (보기 {name},B,D,E): 확률 {p:.3f} → 손실 {-torch.log(p):.3f}")
        print(f"  multi_positive_infonce = {multi_positive_infonce(scores, pos, tau):.3f}  (두 손실의 평균)")
    p_c_unmasked = torch.softmax(scores[0] / 0.05, dim=0)[2]
    print(f"\n마스킹 안 했을 때 C 확률 (τ=0.05, 보기에 A 포함): {p_c_unmasked:.4f}")


def select_train(records: list[dict], tail_rows: set, min_nontail_pos: int,
                 drop_tail: bool) -> tuple[list[int], list[dict]]:
    """학습 검색어 고르기 → (남긴 번호, 레코드)

    min_nontail_pos: 비인기곡이 아닌 warm 정답이 이 개수 이상인 검색어만. A·B(대조군) 모두 3을 주면 학습 검색어가 같아진다
    drop_tail: 정답에서 비인기곡을 뺀다 (B). 정답이 0곡이 되는 검색어는 뺀다
    """
    need = max(min_nontail_pos, 1 if drop_tail else 0)
    keep = [k for k, r in enumerate(records) if sum(j not in tail_rows for j in r["warm_rows"]) >= need]
    out = [{**records[k], "warm_rows": [j for j in records[k]["warm_rows"] if not (drop_tail and j in tail_rows)]}
           for k in keep]
    print(f"train 검색어 고르기: 비인기곡 아닌 정답 {need}곡 이상 → {len(records):,}개 중 {len(keep):,}개, "
          f"비인기곡 라벨 {'제외' if drop_tail else '유지'}")
    return keep, out


def val_metrics(adapter: LowRankAdapter, Q_val: torch.Tensor, S_np: np.ndarray, records: list[dict]) -> dict:
    """val 전체 정답 R@10/R@50, cold R@10/R@50 (카탈로그 전체에서 검색)"""
    with torch.no_grad():
        Q = adapter(Q_val).cpu().numpy()
    m_all = evaluate(Q, S_np, [r["warm_rows"] + r["cold_rows"] for r in records])
    m_cold = evaluate(Q, S_np, [r["cold_rows"] for r in records], ignore=[r["warm_rows"] for r in records])
    return {"R@10": m_all["R@10"], "R@50": m_all["R@50"], "cold_R@10": m_cold["R@10"], "cold_R@50": m_cold["R@50"]}


def log_line(epoch: float, step: int, loss: str, m: dict, ab_norm: float, tail: str = "") -> None:
    print(f"epoch {epoch:5.2f} (step {step:4d}) | loss {loss} | val R@10 {m['R@10']:.4f}  R@50 {m['R@50']:.4f}  "
          f"cold R@10 {m['cold_R@10']:.4f}  cold R@50 {m['cold_R@50']:.4f} | ‖AB‖ {ab_norm:.3f}{tail}")


def train(cfg: dict, tag: str) -> None:
    run = new_run_dir(f"train_query_adapter_{tag}")
    out = guarded_output(run / "checkpoint.pt")
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 1. 데이터: 평균 제거 공간. μ_q는 train 검색어 전부로 (검색어를 걸러도 같은 μ)
    bench = load_bench(cfg["dataset"], "val")
    mu_q, mu_s = means(bench.Qt_raw, bench.S_raw)
    S_np = center(bench.S_raw, mu_s)
    Q_train_np, Q_val_np = center(bench.Qt_raw, mu_q), center(bench.Q_raw, mu_q)
    train_records, val_records = bench.train_records, bench.records

    # 2. 학습 후보: 멜론 매칭 곡 − cold 곡
    melon_rows = sorted(bench.melon_rows)
    cold_rows = set(bench.cold)
    if cfg["drop_tail_labels"] or cfg["min_nontail_pos"]:
        assert bench.tail, "이 데이터셋 카탈로그에는 비인기곡이 없다"
        keep, train_records = select_train(train_records, bench.tail, cfg["min_nontail_pos"], cfg["drop_tail_labels"])
        Q_train_np = Q_train_np[keep]
        if cfg["drop_tail_labels"]:
            cold_rows |= bench.tail   # 비인기곡을 매칭 안 된 곡처럼 → 오답 후보에서도 뺀다
    gallery = list(range(len(S_np))) if cfg["train_gallery"] == "all" else [j for j in melon_rows if j not in cold_rows]
    col_of = {j: c for c, j in enumerate(gallery)}            # 카탈로그 행 → 학습 후보 열
    train_pos = [[col_of[j] for j in r["warm_rows"]] for r in train_records]

    neg_exclude = None
    if cfg["tail_pos_only"]:
        assert not cfg["drop_tail_labels"], "비인기곡 라벨을 뺐으면 정답으로만 쓸 수도 없다"
        neg_exclude = torch.tensor([j in bench.tail for j in gallery], device=device)
        print(f"비인기곡 정답 전용: 학습 후보 {len(gallery):,}곡 중 {int(neg_exclude.sum()):,}곡은 오답으로 안 씀")

    # 3. 인기도 보정 (5-3): 학습할 때만 인기곡 logit에 λ·log π. π = train 정답 등장 비율 (+1 스무딩)
    bias = None
    if cfg["logq_lambda"]:
        counts = np.bincount([c for pos in train_pos for c in pos], minlength=len(gallery)) + 1
        bias = torch.from_numpy(cfg["logq_lambda"] * np.log(counts / counts.sum()).astype(np.float32)).to(device)

    # 3-1. 정답 가중치 1/빈도^γ
    pos_weight = None
    if cfg["pos_weight_power"]:
        counts = np.bincount([c for pos in train_pos for c in pos], minlength=len(gallery)).clip(min=1)
        pos_weight = torch.from_numpy((counts ** -cfg["pos_weight_power"]).astype(np.float32)).to(device)

    S_train = torch.from_numpy(S_np).to(device)[gallery]
    Q_train = torch.from_numpy(Q_train_np).to(device)
    Q_val = torch.from_numpy(Q_val_np).to(device)
    steps_per_epoch = -(-len(train_pos) // cfg["batch_size"])
    eval_every = cfg["eval_every"] or steps_per_epoch
    print(f"설정: {cfg}")
    print(f"카탈로그 {len(S_np):,}곡 중 학습 후보 {len(gallery):,}곡 / train 검색어 {len(train_pos):,} / "
          f"val 검색어 {len(val_records):,} / {device}")

    # 4. 모델
    adapter = LowRankAdapter(S_np.shape[1], cfg["rank"], cfg["act"]).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # NOTE: 학습 전(W = I) 결과는 평균 제거(둘 다)와 같아야 시작점이 맞다
    best = val_metrics(adapter, Q_val, S_np, val_records)
    best_step, best_state = 0, {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    log_line(0.0, 0, "   -  ", best, 0.0)

    # 5. 학습: eval_every 스텝마다 val 평가
    step, stale, losses, t0 = 0, 0, [], time.time()
    for _ in range(cfg["max_epochs"]):
        order = rng.permutation(len(train_pos))
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            pos_mask = torch.zeros(len(idx), len(gallery), dtype=torch.bool, device=device)
            for k, i in enumerate(idx):
                pos_mask[k, train_pos[i]] = True

            loss = multi_positive_infonce(adapter(Q_train[idx]) @ S_train.T, pos_mask, cfg["tau"], bias, pos_weight,
                                          neg_exclude)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % eval_every:
                continue

            m = val_metrics(adapter, Q_val, S_np, val_records)
            ab_norm = torch.linalg.norm(adapter.A @ adapter.B).item()   # W가 I에서 얼마나 멀어졌나
            improved = True if cfg["select_metric"] == "last" else m[cfg["select_metric"]] > best[cfg["select_metric"]]
            log_line(step / steps_per_epoch, step, f"{np.mean(losses):.4f}", m, ab_norm,
                     f" | {time.time() - t0:.0f}s {'*' if improved else ''}")
            losses, t0 = [], time.time()
            if improved:
                best, best_step, stale = m, step, 0
                best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
            else:
                stale += 1
            if stale >= cfg["patience"]:
                break
        if stale >= cfg["patience"]:
            print(f"val {cfg['select_metric']}이 평가 {cfg['patience']}번 동안 안 나아져서 중단")
            break

    # 6. 저장 시점으로 되돌려 저장 + 학습 전과 나란히
    adapter.load_state_dict(best_state)
    best_epoch = round(best_step / steps_per_epoch, 2)
    torch.save({"config": {**cfg, "dataset_file": config.DATASETS[cfg["dataset"]][0]}, "best_step": best_step,
                "best_epoch": best_epoch, "val": best, "A": best_state["A"].cpu(), "B": best_state["B"].cpu(),
                "mu_q": torch.from_numpy(mu_q), "mu_s": torch.from_numpy(mu_s)}, out)
    print(f"\n저장: {out} (step {best_step}, epoch {best_epoch})\n")

    with torch.no_grad():
        Q_adapted = adapter(Q_val).cpu().numpy()
    rep = Report(f"train_query_adapter_{tag}_val", f"쿼리 어댑터 학습 결과 (val, α0 β0 순수 어댑터): {tag}")
    rep.table({"학습 전 (W = I, 평균 제거)": evaluate_all(Space(Q_val_np, S_np, S_np), bench),
               f"어댑터 rank {cfg['rank']}": evaluate_all(Space(Q_adapted, S_np, S_np), bench)})
    rep.add("α·β 지점은 python -m studies.5_5_alpha_beta_grid --checkpoint <이 checkpoint.pt> 로 고른다")
    rep.save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="InfoNCE 숫자 예시만 출력")
    ap.add_argument("--tag", default="run", help="runs/ 폴더 이름에 붙는 꼬리표")
    ap.add_argument("--dataset", choices=list(config.DATASETS))
    ap.add_argument("--gallery", choices=["melon_minus_cold", "all"], help="학습 오답 후보")
    ap.add_argument("--select", choices=["cold_R@50", "R@10", "last"], help="저장 시점 (last = 조기 종료 없음)")
    ap.add_argument("--epochs", type=int, help="최대 에폭 (--select last면 정확히 이만큼)")
    ap.add_argument("--lam", type=float, help="인기도 보정 세기 λ (5-3)")
    ap.add_argument("--pos_weight", type=float, help="정답 가중치 1/빈도^γ의 γ")
    ap.add_argument("--rank", type=int, help="어댑터 rank")
    ap.add_argument("--lr", type=float, help="학습률")
    ap.add_argument("--eval_every", type=int, help="갱신 몇 번마다 val 평가")
    ap.add_argument("--patience", type=int, help="평가 몇 번 동안 안 나아지면 멈출지")
    ap.add_argument("--drop_tail_labels", action="store_true", help="대조군: 비인기곡을 학습 정답·오답 후보에서 뺀다")
    ap.add_argument("--min_nontail_pos", type=int, help="비인기곡 아닌 정답이 이 개수 이상인 train 검색어만")
    ap.add_argument("--tail_pos_only", action="store_true", help="비인기곡을 정답으로만 쓰고 오답으로는 쓰지 않는다")
    ap.add_argument("--act", choices=["gelu"], help="어댑터 비선형 (기본 선형)")
    args = ap.parse_args()
    if args.demo:
        demo_infonce()
        return

    cfg = dict(DEFAULTS)
    for key, value in (("dataset", args.dataset), ("train_gallery", args.gallery), ("select_metric", args.select),
                       ("max_epochs", args.epochs), ("logq_lambda", args.lam), ("pos_weight_power", args.pos_weight),
                       ("rank", args.rank), ("lr", args.lr), ("eval_every", args.eval_every),
                       ("patience", args.patience), ("drop_tail_labels", args.drop_tail_labels),
                       ("min_nontail_pos", args.min_nontail_pos), ("tail_pos_only", args.tail_pos_only),
                       ("act", args.act)):
        if value is not None and value is not False:
            cfg[key] = value
    train(cfg, args.tag)


if __name__ == "__main__":
    main()
