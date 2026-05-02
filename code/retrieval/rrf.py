#!/usr/bin/env python3
"""Reciprocal Rank Fusion (RRF) over an arbitrary set of ranking lists.

RRF: score(d) = Σ_i 1 / (k + rank_i(d))   (default k=60, parameter-free in IR)

Two ranking lists are computed per agenda from the candidate pool (top-1000
embedding candidates):
    - Embedding rank: position in cosine top-1000
    - Citation rank:  rank by induced-subgraph in-degree (ties broken by sim)

Optionally combine with BM25 ranks (3-way) by passing --bm25-source.

Inputs:
    $DATA_DIR/citation_graph.parquet
    $DATA_DIR/agenda_topk/topk_<dense>.json    (dense = gemini / qwen3-8b / ...)
    [optional] $DATA_DIR/agenda_topk/topk_bm25.json
    $DATA_DIR/communities/hier_l1_1e-04_l2_1e-02.parquet
Output:
    $DATA_DIR/analysis/rrf_<dense>[_<bm25>].json
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
RRF_K = 60


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def induced_in_degree(cands: list[int], out_edges: dict[int, list[int]]) -> Counter:
    cset = set(cands)
    deg: Counter = Counter()
    for s in cands:
        if s not in out_edges:
            continue
        for d in out_edges[s]:
            if d in cset:
                deg[d] += 1
    return deg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-source", required=True,
                    help="topk file basename for dense (e.g. 'gemini')")
    ap.add_argument("--bm25-source", default=None,
                    help="optional 3rd ranking from BM25 topk file")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    src_dense = DATA_DIR / "agenda_topk" / f"topk_{args.dense_source}.json"
    out_name = f"rrf_{args.dense_source}"
    if args.bm25_source:
        out_name += f"_bm25_{args.bm25_source}"
    out_path = DATA_DIR / "analysis" / f"{out_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log("Loading citation graph")
    cg = pq.read_table(str(DATA_DIR / "citation_graph.parquet"))
    sa = cg.column("paper_id").to_numpy()
    da = cg.column("paper_reference_id").to_numpy()
    def to_i(x):
        try: return x.astype(np.int64)
        except: return np.fromiter((int(v) for v in x), dtype=np.int64, count=len(x))
    sa, da = to_i(sa), to_i(da)
    out_edges: dict[int, list[int]] = defaultdict(list)
    for s, d in zip(sa.tolist(), da.tolist()):
        out_edges[s].append(d)

    log("Loading hier")
    h = pq.read_table(str(DATA_DIR / "communities" / "hier_l1_1e-04_l2_1e-02.parquet"))
    hp = h.column("paper_id").to_numpy().astype(np.int64)
    hl1 = h.column("level1_comm").to_numpy().astype(np.int64)
    hl2 = h.column("level2_comm").to_numpy().astype(np.int64)
    l1_of = dict(zip(hp.tolist(), hl1.tolist()))
    l2_of = dict(zip(hp.tolist(), hl2.tolist()))

    log(f"Loading dense top-1000: {src_dense}")
    with open(src_dense) as f:
        dense = json.load(f)

    bm25 = None
    if args.bm25_source:
        with open(DATA_DIR / "agenda_topk" / f"topk_{args.bm25_source}.json") as f:
            bm25 = {(x.get("category"), x.get("no")): x for x in json.load(f)}

    results = []
    for ag in dense:
        cand_pids = [p["paper_id"] for p in ag["top_k"]]
        sim_of = {p["paper_id"]: p.get("sim", 0) for p in ag["top_k"]}
        rank_dense = {p: i + 1 for i, p in enumerate(cand_pids)}

        deg = induced_in_degree(cand_pids, out_edges)
        cite_sorted = sorted(cand_pids, key=lambda p: (-deg.get(p, 0), -sim_of.get(p, 0)))
        rank_cite = {p: i + 1 for i, p in enumerate(cite_sorted)}

        # RRF score per candidate
        score: dict[int, float] = {}
        for p in cand_pids:
            score[p] = 1.0 / (RRF_K + rank_dense[p]) + 1.0 / (RRF_K + rank_cite[p])
        if bm25 is not None:
            ag_bm = bm25.get((ag.get("category"), ag.get("no")), {})
            bm_pids = [p["paper_id"] for p in ag_bm.get("top_k", [])]
            rank_bm = {p: i + 1 for i, p in enumerate(bm_pids)}
            # union of all candidates
            union = list(set(cand_pids) | set(bm_pids))
            score = {}
            deg_u = induced_in_degree(union, out_edges)
            cite_sorted_u = sorted(union, key=lambda p: (-deg_u.get(p, 0),))
            rank_cite_u = {p: i + 1 for i, p in enumerate(cite_sorted_u)}
            for p in union:
                rd = rank_dense.get(p, 1001)
                rb = rank_bm.get(p, 1001)
                rc = rank_cite_u.get(p, 1001)
                score[p] = 1.0 / (RRF_K + rd) + 1.0 / (RRF_K + rb) + 1.0 / (RRF_K + rc)
            cand_pids = union

        ranked = sorted(score.keys(), key=lambda p: -score[p])
        top = ranked[:args.top_n]

        l1s = [l1_of.get(p) for p in top if p in l1_of]
        l2s = [l2_of.get(p) for p in top if p in l2_of]
        n_match = len(l1s)
        c1, c2 = Counter(l1s), Counter(l2s)
        results.append({
            "category": ag.get("category"),
            "no": ag.get("no"),
            "agenda": ag["agenda"],
            "top_n_paper_ids": top,
            "top_n_l1": [l1_of.get(p, -1) for p in top],
            "top_n_l2": [l2_of.get(p, -1) for p in top],
            "n_match_in_hier": n_match,
            "unique_l1": len(c1),
            "unique_l2": len(c2),
            "top1_l1_share": max(c1.values()) / n_match if n_match else 0,
            "top1_l2_share": max(c2.values()) / n_match if n_match else 0,
        })

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
