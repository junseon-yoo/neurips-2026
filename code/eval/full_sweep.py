#!/usr/bin/env python3
"""Full per-domain rank-K sweep for one (model, level) pair.

For each of 8 domains, samples 200 K papers (seed=42) from the cleaned pool,
GPU-computes top-K embedding neighbors, then for k ∈ {2,5,10,25,50,100} reports:
    - rank-k same-community rate (probability that the k-th rank neighbor
      shares the query paper's community) + enrichment over Σ p²
    - any-same / mean-same-count within top-k
    - unique-community count + analytical baseline Σ(1-(1-p)^k)
    - (mode community count = max same-comm cluster within top-k)

Inputs:
    $DATA_DIR/embeddings/<model>.parquet   (paper_id, embedding)
    $DATA_DIR/communities/<community_file>.parquet  (paper_id, community_id)
    $DATA_DIR/target/neurips_4m.parquet
    $DATA_DIR/citation_graph.parquet
    $DATA_DIR/dedup_v2_clusters.json (canonical filter)

Output:
    $DATA_DIR/full_sweep/full_sweep_<model>_<level>hier.json

CLI:
    python full_sweep.py --model qwen3 --level l1 \\
        --comm-path data/communities/hier_L1_flat.parquet
"""
import argparse
import gc
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAINS = ["biology", "biomedical", "chemistry", "computer_science",
           "engineering", "environmental_earth", "materials_science", "physics"]
K_VALUES = [2, 5, 10, 25, 50, 100]
SAMPLE_SIZE = 200_000
SEED = 42
K_MAX = 101

BOILERPLATE_TITLE = {"editorial board", "front cover", "back cover", "front matter",
                     "back matter", "table of contents", "author index", "subject index",
                     "retraction", "corrigendum", "erratum", "acknowledgments"}


def is_boilerplate(t: str | None, a: str | None) -> bool:
    tl = (t or "").strip().lower()
    al = (a or "").strip().lower()
    if not tl or len(al) < 200:
        return True
    for kw in BOILERPLATE_TITLE:
        if kw in tl:
            return True
    if re.search(r"\bvolume\s+\d+\s+issue\s+\d+", tl):
        return True
    return False


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stream_parquet_filtered(path: Path, keep_set: set[int]):
    pf = pq.ParquetFile(str(path))
    keep = np.fromiter(keep_set, dtype=np.int64)
    pids_d, X_d = [], []
    for batch in pf.iter_batches(batch_size=50000, columns=["paper_id", "embedding"]):
        b_pids = batch.column("paper_id").to_numpy(zero_copy_only=False)
        try:
            b_pids = b_pids.astype(np.int64)
        except Exception:
            b_pids = np.fromiter((int(x) for x in b_pids), dtype=np.int64, count=len(b_pids))
        mask = np.isin(b_pids, keep)
        if not mask.any():
            continue
        ec = batch.column("embedding")
        if pa.types.is_fixed_size_list(ec.type):
            D = ec.type.list_size
            flat = ec.values.to_numpy(zero_copy_only=False)
        else:
            off = ec.offsets.to_numpy()
            D = int(off[1] - off[0])
            flat = ec.values.to_numpy(zero_copy_only=False)
        X = flat.reshape(-1, D).astype(np.float32, copy=True)
        pids_d.append(b_pids[mask])
        X_d.append(X[mask])
    return np.concatenate(pids_d), np.concatenate(X_d, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model file basename (e.g. 'qwen3' → embeddings/qwen3_4m.parquet)")
    ap.add_argument("--level", required=True, choices=["l1", "l2"])
    ap.add_argument("--comm-path", required=True,
                    help="path to community parquet (paper_id, community_id)")
    ap.add_argument("--out-suffix", default=None)
    args = ap.parse_args()

    suffix = args.out_suffix or f"{args.level}hier"
    out_path = DATA_DIR / "full_sweep" / f"full_sweep_{args.model}_{suffix}.json"
    if out_path.exists():
        log(f"exists, skip: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log("Loading target + community + citation")
    t = pq.read_table(str(DATA_DIR / "target" / "neurips_4m.parquet"),
                      columns=["paper_id", "domain", "title", "abstract"])
    tgt_pids = t.column("paper_id").to_numpy().astype(np.int64)
    tgt_dom = t.column("domain").to_numpy()
    titles = t.column("title").to_pylist()
    abstracts = t.column("abstract").to_pylist()

    c = pq.read_table(args.comm_path, columns=["paper_id", "community_id"])
    cpid = c.column("paper_id").to_numpy().astype(np.int64)
    ccid = c.column("community_id").to_numpy().astype(np.int64)
    u, inv, cnt = np.unique(ccid, return_inverse=True, return_counts=True)
    size_arr = cnt[inv]
    comm_map = {int(p): (int(ccid[i]), int(size_arr[i])) for i, p in enumerate(cpid)}

    cg = pq.read_table(str(DATA_DIR / "citation_graph.parquet"))
    src = cg.column("paper_id").to_numpy()
    dst = cg.column("paper_reference_id").to_numpy()
    def to_i(a): return np.fromiter((int(x) for x in a), dtype=np.int64, count=len(a))
    cited = set(np.unique(np.concatenate([to_i(src), to_i(dst)])).tolist())

    with open(DATA_DIR / "dedup_v2_clusters.json") as f:
        clusters = json.load(f)
    non_canonical = set()
    for cl in clusters:
        canon = int(cl["canonical"])
        for p in cl["members"]:
            if int(p) != canon:
                non_canonical.add(int(p))

    dom_map = {}
    boiler = set()
    for i in range(len(tgt_pids)):
        p = int(tgt_pids[i])
        dom_map[p] = str(tgt_dom[i])
        if is_boilerplate(titles[i], abstracts[i]):
            boiler.add(p)

    log("Sampling per-domain pools")
    domain_samples = {}
    for dom in DOMAINS:
        pool = [p for p, d in dom_map.items()
                if d == dom and p in cited and p not in boiler and p not in non_canonical
                and p in comm_map and comm_map[p][1] >= 2]
        rng = np.random.default_rng(SEED)
        if len(pool) > SAMPLE_SIZE:
            sampled = rng.choice(np.array(pool, dtype=np.int64), size=SAMPLE_SIZE, replace=False)
        else:
            sampled = np.array(pool, dtype=np.int64)
        domain_samples[dom] = sampled
        log(f"  {dom}: pool={len(pool):,} → sampled={len(sampled):,}")

    union = set()
    for s in domain_samples.values():
        union.update(int(p) for p in s)
    paper_emb = DATA_DIR / "embeddings" / f"{args.model}_4m.parquet"
    log(f"Loading {paper_emb} (streaming filter, |union|={len(union):,})")
    pids_all, X_all = stream_parquet_filtered(paper_emb, union)

    out = {"model": args.model, "level": args.level, "domains": {}}
    for dom in DOMAINS:
        sampled = domain_samples[dom]
        mask = np.isin(pids_all, sampled)
        pids_d = pids_all[mask]
        X_d = X_all[mask]
        norms = np.linalg.norm(X_d, axis=1, keepdims=True).clip(min=1e-12)
        X_d = X_d / norms
        N = X_d.shape[0]
        comm_arr = np.array([comm_map[int(p)][0] for p in pids_d.tolist()], dtype=np.int64)
        u_c, cnt_c = np.unique(comm_arr, return_counts=True)
        pcs = cnt_c / N
        base_same = float(np.sum(pcs ** 2))
        log(f"=== {dom} === N={N:,} #comm={len(u_c):,}  baseline Σp²={base_same:.5f}")

        Xt = torch.from_numpy(X_d).to(DEVICE, dtype=torch.float16)
        K_local = min(K_MAX, N)
        D_arr = np.zeros((N, K_local), dtype=np.float32)
        I_arr = np.zeros((N, K_local), dtype=np.int64)
        chunk = 1024
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            sims = Xt[s:e] @ Xt.T
            sv, si = torch.topk(sims, K_local, dim=1, largest=True, sorted=True)
            D_arr[s:e, :sv.shape[1]] = sv.float().cpu().numpy()
            I_arr[s:e, :si.shape[1]] = si.cpu().numpy()
        del Xt
        torch.cuda.empty_cache()

        dom_out = {
            "pool_N": int(N),
            "baseline_same_prob": base_same,
            "n_communities_in_pool": int(len(u_c)),
            "max_community_size_in_pool": int(cnt_c.max()),
            "max_community_share": float(cnt_c.max() / N),
            "by_k": {},
        }
        for k in K_VALUES:
            if k > K_local:
                continue
            group_idx = I_arr[:, 0:k]
            group_comm = comm_arr[group_idx]
            if k >= 2:
                neigh_at_rank = I_arr[:, k - 1]
                same_at_rank = (comm_arr == comm_arr[neigh_at_rank]).astype(np.int32)
                rank_k_same = float(same_at_rank.mean())
            else:
                rank_k_same = 1.0
            sorted_gc = np.sort(group_comm, axis=1)
            diffs = sorted_gc[:, 1:] != sorted_gc[:, :-1]
            unique_counts = diffs.sum(axis=1).astype(np.int32) + 1
            u_mean = float(unique_counts.mean())
            u_base = float(np.sum(1.0 - (1.0 - pcs) ** k))
            self_col = comm_arr[:, None]
            if k >= 2:
                neigh = group_comm[:, 1:]
                any_same = float((neigh == self_col).any(axis=1).mean())
                same_mean = float((neigh == self_col).sum(axis=1).mean())
            else:
                any_same = 1.0
                same_mean = 1.0
            dom_out["by_k"][str(k)] = {
                "mean_sim_at_rank": float(D_arr[:, k - 1].mean()) if k - 1 < K_local else None,
                "rank_k_same_rate": rank_k_same,
                "rank_k_same_enrichment": rank_k_same / base_same if base_same > 0 else None,
                "any_same_in_topk": any_same,
                "mean_same_count_in_topk": same_mean,
                "unique_community_count": {
                    "observed_mean": u_mean,
                    "baseline_mean": u_base,
                    "enrichment": u_mean / u_base if u_base > 0 else None,
                },
            }
        out["domains"][dom] = dom_out

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
