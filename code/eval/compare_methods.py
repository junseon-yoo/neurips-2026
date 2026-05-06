#!/usr/bin/env python3
"""Compare all retrieval methods at k=10 on the 80-agenda benchmark.

Methods covered (each consumes its corresponding topk_<method>.json file):
    - graph_search   : community_papers from agent's keyword search (input format
                        provided as 'queries_with_graph_search.json')
    - bm25            : sparse top-K
    - specter2 / qwen3 / qwen3_8b / gemini : dense top-K
    - rerank_citation_<dense> : top-K candidates → induced subgraph in-degree top-N
    - rrf_<dense>[_bm25_<bm25>] : reciprocal rank fusion

For each method, computes per-domain and overall mean of:
    - top1_L1 share (% of top-K in dominant L1 community)
    - top1_L2 share
    - #unique L1, #unique L2

Inputs:
    $DATA_DIR/communities/hier_l1_1e-04_l2_1e-02.parquet
    $DATA_DIR/agenda_topk/topk_*.json
    $DATA_DIR/analysis/{rerank_citation_*,rrf_*}.json
    $DATA_DIR/queries_80.json (for graph_search results stored under 'community_papers')

Output:
    $DATA_DIR/analysis/method_comparison_k10.json
"""
import argparse
import glob
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DOMAINS = ["biology", "biomedical", "chemistry", "cs", "engineering",
           "environmental_science", "materials_science", "physics"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stats_for_pids(pids: list[int], k: int, l1_of: dict, l2_of: dict) -> dict:
    sub = pids[:k]
    l1s = [l1_of.get(p) for p in sub if p in l1_of]
    l2s = [l2_of.get(p) for p in sub if p in l2_of]
    n = len(l1s)
    if n == 0:
        return {"n_match": 0, "top1_l1": 0, "top1_l2": 0, "unique_l1": 0, "unique_l2": 0}
    c1, c2 = Counter(l1s), Counter(l2s)
    return {
        "n_match": n,
        "top1_l1": max(c1.values()) / n,
        "top1_l2": max(c2.values()) / n,
        "unique_l1": len(c1),
        "unique_l2": len(c2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    log("Loading hier")
    h = pq.read_table(str(DATA_DIR / "communities" / "hier_l1_1e-04_l2_1e-02.parquet"))
    hp = h.column("paper_id").to_numpy().astype(np.int64)
    hl1 = h.column("level1_comm").to_numpy().astype(np.int64)
    hl2 = h.column("level2_comm").to_numpy().astype(np.int64)
    l1_of = dict(zip(hp.tolist(), hl1.tolist()))
    l2_of = dict(zip(hp.tolist(), hl2.tolist()))

    method_results: dict[str, list[dict]] = defaultdict(list)

    # Direct topk_*.json files (BM25 + dense)
    for path in sorted(glob.glob(str(DATA_DIR / "agenda_topk" / "topk_*.json"))):
        method = Path(path).stem.replace("topk_", "")
        with open(path) as f:
            data = json.load(f)
        for ag in data:
            pids = [p["paper_id"] for p in ag["top_k"]]
            method_results[method].append(
                {"category": ag.get("category"), "no": ag.get("no"),
                 **stats_for_pids(pids, args.top_n, l1_of, l2_of)}
            )

    # Graph search (community_papers from agent search output)
    qp = DATA_DIR / "queries_80_with_graph_search.json"
    if qp.exists():
        with open(qp) as f:
            qd = json.load(f)
        agendas_list = qd.get("results", qd.get("agendas", []))
        for ag in agendas_list:
            cps = ag.get("community_papers")
            if not cps:
                continue
            pids = [int(p["paper_id"]) for p in cps]
            method_results["graph_search"].append(
                {"category": ag.get("category"), "no": ag.get("no"),
                 **stats_for_pids(pids, args.top_n, l1_of, l2_of)}
            )

    # Hybrid analysis files (rerank, rrf)
    for path in sorted(glob.glob(str(DATA_DIR / "analysis" / "rerank_citation_*.json"))):
        method = "rerank_" + Path(path).stem.replace("rerank_citation_", "")
        with open(path) as f:
            data = json.load(f)
        for ag in data:
            method_results[method].append({
                "category": ag.get("category"), "no": ag.get("no"),
                "n_match": ag.get("n_match_in_hier", 0),
                "top1_l1": ag.get("top1_l1_share", 0),
                "top1_l2": ag.get("top1_l2_share", 0),
                "unique_l1": ag.get("unique_l1", 0),
                "unique_l2": ag.get("unique_l2", 0),
            })
    for path in sorted(glob.glob(str(DATA_DIR / "analysis" / "rrf_*.json"))):
        method = Path(path).stem
        with open(path) as f:
            data = json.load(f)
        for ag in data:
            method_results[method].append({
                "category": ag.get("category"), "no": ag.get("no"),
                "n_match": ag.get("n_match_in_hier", 0),
                "top1_l1": ag.get("top1_l1_share", 0),
                "top1_l2": ag.get("top1_l2_share", 0),
                "unique_l1": ag.get("unique_l1", 0),
                "unique_l2": ag.get("unique_l2", 0),
            })

    # Aggregate
    summary = {}
    for method, rows in method_results.items():
        per_cat = defaultdict(list)
        for r in rows:
            per_cat[r["category"]].append(r)
        cat_means = {}
        for cat in DOMAINS:
            rs = per_cat.get(cat, [])
            if not rs:
                cat_means[cat] = None
                continue
            cat_means[cat] = {
                "top1_l2": float(np.mean([r["top1_l2"] for r in rs])),
                "top1_l1": float(np.mean([r["top1_l1"] for r in rs])),
                "unique_l2": float(np.mean([r["unique_l2"] for r in rs])),
                "unique_l1": float(np.mean([r["unique_l1"] for r in rs])),
                "n_agendas": len(rs),
            }
        summary[method] = {
            "per_category": cat_means,
            "overall_top1_l2": float(np.mean([r["top1_l2"] for r in rows])) if rows else 0,
            "overall_top1_l1": float(np.mean([r["top1_l1"] for r in rows])) if rows else 0,
            "overall_unique_l2": float(np.mean([r["unique_l2"] for r in rows])) if rows else 0,
            "n_total": len(rows),
        }

    out = DATA_DIR / "analysis" / f"method_comparison_k{args.top_n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Wrote {out}")

    # Print headline
    print()
    print("Top-N =", args.top_n, "— overall mean per method:")
    print(f'{"method":<35} {"top1_L1":>8} {"top1_L2":>8} {"#L2":>6}')
    for method, s in sorted(summary.items(), key=lambda x: -x[1]["overall_top1_l2"]):
        print(f'{method:<35} {s["overall_top1_l1"]*100:>7.1f}% '
              f'{s["overall_top1_l2"]*100:>7.1f}% {s["overall_unique_l2"]:>6.2f}')


if __name__ == "__main__":
    main()
