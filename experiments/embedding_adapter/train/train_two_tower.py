"""
4-6. 두 타워 학습: 쿼리 어댑터 + 곡 어댑터를 같이

  q_a = norm(q_c + q_c·A_q·B_q)              쿼리 어댑터 (선형)
  s″  = norm(s′ + s′·A_s·B_s)                곡 어댑터 (선형 + 정규화)
    NOTE: 정규화가 있어서 쿼리 쪽으로 접히지 않는다. 둘 다 정규화 없는 선형이면 q·(W_q·W_sᵀ)·s = 쿼리 행렬 하나
  손실 = 텍스트 InfoNCE(q_a → s″, 쿼리 어댑터와 같음) + λ · 곡-곡 InfoNCE(s″끼리)
    곡-곡: 배치 재생목록 중 정답 4곡 이상을 매번 무작위로 반 나눠, 절반의 평균 방향으로 나머지 절반을 맞힌다

입력      데이터셋(기본 service_format), 학습 검색어·오답 후보는 쿼리 어댑터와 같다 (정답 3곡 이상, 멜론 곡 − cold)
출력      runs/{날짜}_train_two_tower_{tag}/checkpoint.pt  {"q_A", "q_B", "s_A", "s_B", "mu_q", "mu_s", "config"}
          화면: 에폭마다 val R@10·cold·겹침·허브·kpop·응집도, 끝에 α × β 격자와 규칙 지점
          NOTE: 끝의 격자는 r(s)를 학습 검색어(정답 3곡 이상, 16,939개)로 계산한다. 비교 스크립트(5-9a)는 train 전부로 계산해
                끝자리가 조금 다르다. 원래 실험은 여기서 고른 지점을 비교에 썼다
설정      쿼리·곡 모두 rank 512, B = 0에서 시작, τ 0.05, AdamW lr 1e-3 wd 0.01, batch 256, seed 0, 5에폭

이 저장소의 체크포인트와 학습 옵션의 관계
  two_tower_lam0 / lam025 / lam05 / lam1   --lam 0 / 0.25 / 0.5 / 1
  two_tower_lam0_init / lam05_init         + --init_query query_adapter_r512

사용:
  python -m train.train_two_tower --lam 0.5 --tag lam05
  python -m train.train_two_tower --lam 0.5 --init_query query_adapter_r512 --tag lam05_init
  python -m train.train_two_tower --smoke        # 동작 확인: 3스텝, 저장·격자 생략
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
from core.adapters import LowRankAdapter, multi_positive_infonce
from core.data import l2norm, load_bench
from core.metrics import COHESION_MIN_POS, cohesion, diversity, evaluate, kpop_share_top10
from core.safe_io import guarded_output, new_run_dir
from core.systems import center, hub_scores, means, with_hub_penalty
from train.train_query_adapter import select_train

CLUE_FILL = -1e4   # 곡-곡 손실에서 단서 곡 점수 (−inf면 정답 아닌 칸에서 −inf × 0 = nan)


def metrics(Q, S_x, bench) -> dict:
    m = evaluate(Q, S_x, bench.all_pos)
    c = evaluate(Q, S_x, bench.cold_pos, ignore=bench.warm_ign)
    d = diversity(Q, S_x, bench.melon_rows)
    return {"R@10": m["R@10"], "cold_R@10": c["R@10"], "overlap": d["overlap"], "hub_top1": d["hub_top1"],
            "kpop": kpop_share_top10(Q, S_x, bench.is_kpop)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="service_format", choices=list(config.DATASETS))
    ap.add_argument("--q_rank", type=int, default=512)
    ap.add_argument("--s_rank", type=int, default=512)
    ap.add_argument("--song_act", choices=["gelu"], help="곡 어댑터 비선형 (기본 선형 + 정규화)")
    ap.add_argument("--lam", type=float, default=0.0, help="곡-곡 손실 가중치 λ (0 = 텍스트 손실만)")
    ap.add_argument("--init_query", help="쿼리 어댑터를 이 체크포인트(이름표 이름 또는 .pt)의 A·B에서 시작")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--smoke", action="store_true", help="동작 확인: 3스텝, 저장·격자 생략")
    args = ap.parse_args()
    cfg = {**vars(args), "tau": 0.05, "lr": 1e-3, "weight_decay": 0.01, "batch_size": 256, "seed": 0,
           "min_nontail_pos": 3, "dataset_file": config.DATASETS[args.dataset][0]}

    out = None if args.smoke else guarded_output(new_run_dir(f"train_two_tower_{args.tag}") / "checkpoint.pt")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 1. 데이터: 쿼리 어댑터와 같은 평균 제거 공간, 같은 학습 검색어 고르기
    bench = load_bench(args.dataset, "val")
    mu_q, mu_s = means(bench.Qt_raw, bench.S_raw)
    S_np = center(bench.S_raw, mu_s)
    Qt_np, Qv_np = center(bench.Qt_raw, mu_q), center(bench.Q_raw, mu_q)
    keep, train_records = select_train(bench.train_records, bench.tail, cfg["min_nontail_pos"], False)
    Qt_np = Qt_np[keep]
    gallery = [j for j in sorted(bench.melon_rows) if j not in bench.cold]
    col_of = {j: c for c, j in enumerate(gallery)}
    train_pos = [[col_of[j] for j in r["warm_rows"]] for r in train_records]

    # 2. 모델: 둘 다 B = 0에서 시작 → 처음엔 평균 제거와 같다
    dim = S_np.shape[1]
    qa = LowRankAdapter(dim, args.q_rank).to(device)
    sa = LowRankAdapter(dim, args.s_rank, args.song_act).to(device)
    if args.init_query:
        ck = torch.load(config.checkpoint_path(args.init_query))
        assert np.allclose(ck["mu_q"].numpy(), mu_q, atol=1e-6), "체크포인트 μ_q가 이 데이터셋 train 평균과 다르다"
        qa.load_state_dict({"A": ck["A"], "B": ck["B"]})
    opt = torch.optim.AdamW(list(qa.parameters()) + list(sa.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    S_all, S_gal = torch.from_numpy(S_np).to(device), torch.from_numpy(S_np[gallery]).to(device)
    Qt, Qv = torch.from_numpy(Qt_np).to(device), torch.from_numpy(Qv_np).to(device)
    print(f"두 타워: {args.dataset}, 쿼리 r{args.q_rank} + 곡 r{args.s_rank} {args.song_act or '선형+정규화'}, λ={args.lam}, "
          f"쿼리 시작 {args.init_query or '0'} / train 검색어 {len(train_pos):,}, 학습 후보 {len(gallery):,}곡, {device}")

    def snapshot(label: str) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            S_new, Qv_new = sa(S_all).cpu().numpy(), qa(Qv).cpu().numpy()
        m, h = metrics(Qv_new, S_new, bench), cohesion(S_new, bench)
        print(f"  {label:10s} R@10 {m['R@10']:.4f}  cold R@10 {m['cold_R@10']:.4f} | 겹침 {m['overlap']:.3f}  "
              f"허브 상위1% {m['hub_top1']:.1%}  kpop {m['kpop']:.1%} | 응집도 warm {h['warm']:.4f} cold {h['cold']:.4f}")
        return S_new, Qv_new

    snapshot("epoch 0")
    steps_per_epoch = -(-len(train_pos) // cfg["batch_size"])
    for epoch in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        order = rng.permutation(len(train_pos))
        for step, start in enumerate(range(0, len(order), cfg["batch_size"])):
            if args.smoke and step >= 3:
                break
            idx = order[start:start + cfg["batch_size"]]
            pos_mask = torch.zeros(len(idx), len(gallery), dtype=torch.bool, device=device)
            for k, i in enumerate(idx):
                pos_mask[k, train_pos[i]] = True
            G = sa(S_gal)                                                   # (학습 후보, D) = s″
            loss = multi_positive_infonce(qa(Qt[idx]) @ G.T, pos_mask, cfg["tau"])

            if args.lam:
                # 곡-곡: 배치 재생목록 중 정답 4곡 이상, 매번 무작위 반 나눠 단서 평균 → 나머지
                rows = [train_pos[i] for i in idx if len(train_pos[i]) >= COHESION_MIN_POS]
                clue_w = torch.zeros(len(rows), len(gallery), device=device)
                tgt = torch.zeros(len(rows), len(gallery), dtype=torch.bool, device=device)
                for k, p in enumerate(rows):
                    perm = rng.permutation(p).tolist()
                    clue_w[k, perm[:len(p) // 2]] = 1 / (len(p) // 2)
                    tgt[k, perm[len(p) // 2:]] = True
                Qs = F.normalize(clue_w @ G, dim=-1)
                loss = loss + args.lam * multi_positive_infonce((Qs @ G.T).masked_fill(clue_w > 0, CLUE_FILL), tgt, cfg["tau"])

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch} loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s, {steps_per_epoch}스텝)")
        S_new, Qv_new = snapshot(f"epoch {epoch}")
        if args.smoke:
            print("동작 확인 끝 (저장·격자 생략)")
            return

    torch.save({"config": cfg, "q_A": qa.A.detach().cpu(), "q_B": qa.B.detach().cpu(),
                "s_A": sa.A.detach().cpu(), "s_B": sa.B.detach().cpu(),
                "mu_q": torch.from_numpy(mu_q), "mu_s": torch.from_numpy(mu_s)}, out)
    print(f"저장: {out}")

    # 3. α × β 격자 + 규칙 (기준 겹침 = 평균 제거: 원래 곡, 어댑터 없음)
    base = metrics(Qv_np, S_np, bench)
    with torch.no_grad():
        Qt_a = qa(Qt).cpu().numpy()
    results = {}
    for a in config.ALPHAS:
        r = hub_scores(S_new, l2norm(a * Qt_np + (1 - a) * Qt_a))
        Qv_mix = l2norm(a * Qv_np + (1 - a) * Qv_new)
        for b in config.BETAS:
            S_aug, Q_aug = with_hub_penalty(S_new, Qv_mix, r, b)
            results[a, b] = metrics(Q_aug, S_aug, bench)
    for key, fmt in (("R@10", ".4f"), ("cold_R@10", ".4f"), ("overlap", ".3f")):
        print(f"\n[{key}]\n| α \\ β | " + " | ".join(str(b) for b in config.BETAS) + " |\n|---|" + "---|" * len(config.BETAS))
        for a in config.ALPHAS:
            print(f"| {a} | " + " | ".join(f"{results[a, b][key]:{fmt}}" for b in config.BETAS) + " |")
    ok = [k for k, v in results.items() if v["overlap"] <= base["overlap"]]
    a, b = max(ok, key=lambda k: results[k]["R@10"])
    print(f"\n[선택 규칙] 겹침 ≤ {base['overlap']:.3f}곡(평균 제거)인 {len(ok)}개 조합 중 R@10 최대 → α={a}, β={b} "
          f"(R@10 {results[a, b]['R@10']:.4f}, cold {results[a, b]['cold_R@10']:.4f})")


if __name__ == "__main__":
    main()
