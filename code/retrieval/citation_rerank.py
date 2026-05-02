#!/usr/bin/env python3
"""Rerank top-K candidates by internal in-degree (induced citation subgraph).

Pipeline: for each agenda's top-K candidates, build the citation subgraph among
those K papers (only edges where both endpoints are in the candidate set),
count in-degree per candidate, return the top-N (default 10) by in-degree.

Inputs:
    $DATA_DIR/citation_graph.parquet              (paper_id, paper_reference_id)
    $DATA_DIR/agenda_topk/topk_<source>.json
    $DATA_DIR/communities/hier_l1_1e-04_l2_1e-02.parquet  (for L1/L2 stats output)

Output:
    $DATA_DIR/analysis/rerank_citation_<source>.json
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


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="topk file basename, e.g. 'gemini' for topk_gemini.json")
    ap.add_argument("--top-n", type=int, default=10, help="final top-N after rerank")
    ap.add_argument("--candidate-k", type=int, default=1000,
                    help="how many embedding candidates to consider")
    args = ap.parse_args()

    src = DATA_DIR / "agenda_topk" / f"topk_{args.source}.json"
    out = DATA_DIR / "analysis" / f"rerank_citation_{args.source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    log("Loading citation graph")
    cg = pq.read_table(str(DATA_DIR / "citation_graph.parquet"))
    sa = cg.column("paper_id").to_numpy()
    da = cg.column("paper_reference_id").to_numpy()
    def to_i(x):
        try: return x.astype(np.int64)
        except: return np.fromiter((int(v) for v in x), dtype=np.int64, count=len(x))
    sa, da = to_i(sa), to_i(da)
    out_edges = defaultdict(list)
    for s, d in zip(sa.tolist(), da.tolist()):
        out_edges[s].append(d)
    log(f"  {len(sa):,} edges")

    log("Loading hier L1/L2")
    h = pq.read_table(str(DATA_DIR / "communities" / "hier_l1_1e-04_l2_1e-02.parquet"))
    hp = h.column("paper_id").to_numpy().astype(np.int64)
    hl1 = h.column("level1_comm").to_numpy().astype(np.int64)
    hl2 = h.column("level2_comm").to_numpy().astype(np.int64)
    l1_of = dict(zip(hp.tolist(), hl1.tolist()))
    l2_of = dict(zip(hp.tolist(), hl2.tolist()))

    log(f"Loading candidates: {src}")
    with open(src) as f:
        agendas = json.load(f)

    results = []
    for ag in agendas:
        cand_pids = [p["paper_id"] for p in ag["top_k"][:args.candidate_k]]
        cand_set = set(cand_pids)
        cand_score = {p["paper_id"]: p.get("sim", p.get("score", 0)) for p in ag["top_k"][:args.candidate_k]}

        # Induced subgraph in-degree
        in_deg: Counter = Counter()
        n_edges = 0
        for s in cand_pids:
            if s not in out_edges:
                continue
            for d in out_edges[s]:
                if d in cand_set:
                    in_deg[d] += 1
                    n_edges += 1
        ranked = sorted(cand_pids, key=lambda p: (-in_deg.get(p, 0), -cand_score.get(p, 0)))
        top = ranked[:args.top_n]

        l1s = [l1_of.get(p) for p in top if p in l1_of]
        l2s = [l2_of.get(p) for p in top if p in l2_of]
        n_match = len(l1s)
        c1, c2 = Counter(l1s), Counter(l2s)
        results.append({
            "category": ag.get("category"),
            "no": ag.get("no"),
            "agenda": ag["agenda"],
            "n_internal_edges": n_edges,
            "top_n_paper_ids": top,
            "top_n_in_degree": [in_deg.get(p, 0) for p in top],
            "top_n_l1": [l1_of.get(p, -1) for p in top],
            "top_n_l2": [l2_of.get(p, -1) for p in top],
            "n_match_in_hier": n_match,
            "unique_l1": len(c1),
            "unique_l2": len(c2),
            "top1_l1_share": max(c1.values()) / n_match if n_match else 0,
            "top1_l2_share": max(c2.values()) / n_match if n_match else 0,
        })

    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
