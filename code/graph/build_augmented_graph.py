#!/usr/bin/env python3
"""Combine direct citation + BC + CC into a single weighted undirected graph.

Weight scheme:
    - Direct citation: weight floor = 1.0
    - BC edge:  weight += BC_cos  (Salton cosine, [0, 1])
    - CC edge:  weight += CC_cos
    - Same pair across layers: sum (capped only by data)

Inputs:
    $DATA_DIR/citation_graph.parquet  (paper_id, paper_reference_id) — target→target
    $DATA_DIR/bc_edges_full.parquet   (a, b, shared, cos)
    $DATA_DIR/cc_edges_full.parquet

Output:
    $DATA_DIR/augmented_graph_v2.parquet (a, b, weight float32)
"""
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
OUT = DATA_DIR / "augmented_graph_v2.parquet"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    edge_w: dict[tuple[int, int], float] = defaultdict(float)

    log("Loading direct citation edges")
    cg = pq.read_table(str(DATA_DIR / "citation_graph.parquet"))
    src = cg.column("paper_id").to_numpy().astype(np.int64)
    dst = cg.column("paper_reference_id").to_numpy().astype(np.int64)
    n_direct = 0
    for s, d in zip(src.tolist(), dst.tolist()):
        if s == d:
            continue
        key = (s, d) if s < d else (d, s)
        if key not in edge_w:
            n_direct += 1
        edge_w[key] = max(edge_w[key], 1.0)
    log(f"  direct pairs (undirected dedup): {n_direct:,}")

    log("Loading BC edges (full)")
    bc = pq.read_table(str(DATA_DIR / "bc_edges_full.parquet"))
    bc_a = bc.column("a").to_numpy().astype(np.int64).tolist()
    bc_b = bc.column("b").to_numpy().astype(np.int64).tolist()
    bc_cos = bc.column("cos").to_numpy().tolist()
    n_new_bc = 0
    for a, b, w in zip(bc_a, bc_b, bc_cos):
        key = (a, b) if a < b else (b, a)
        if key not in edge_w:
            n_new_bc += 1
        edge_w[key] += float(w)
    log(f"  BC edges: {len(bc_a):,}  new pairs added: {n_new_bc:,}")
    del bc, bc_a, bc_b, bc_cos

    log("Loading CC edges (full)")
    cc = pq.read_table(str(DATA_DIR / "cc_edges_full.parquet"))
    cc_a = cc.column("a").to_numpy().astype(np.int64).tolist()
    cc_b = cc.column("b").to_numpy().astype(np.int64).tolist()
    cc_cos = cc.column("cos").to_numpy().tolist()
    n_new_cc = 0
    for a, b, w in zip(cc_a, cc_b, cc_cos):
        key = (a, b) if a < b else (b, a)
        if key not in edge_w:
            n_new_cc += 1
        edge_w[key] += float(w)
    log(f"  CC edges: {len(cc_a):,}  new pairs added: {n_new_cc:,}")
    del cc, cc_a, cc_b, cc_cos

    log(f"Total unique undirected edges: {len(edge_w):,}")
    weights = np.array(list(edge_w.values()), dtype=np.float32)
    log(f"  weight q5/25/50/75/95: {np.percentile(weights, [5,25,50,75,95]).round(3).tolist()}")
    log(f"  edges w>=1 (direct): {int((weights >= 1.0).sum()):,}")
    log(f"  edges 0<w<1 (BC/CC only): {int((weights < 1.0).sum()):,}")

    log("Writing parquet")
    a_arr = np.fromiter((k[0] for k in edge_w.keys()), dtype=np.int64, count=len(edge_w))
    b_arr = np.fromiter((k[1] for k in edge_w.keys()), dtype=np.int64, count=len(edge_w))
    tbl = pa.table({"a": a_arr, "b": b_arr, "weight": weights})
    pq.write_table(tbl, str(OUT), compression="zstd")
    log(f"Wrote {OUT}")

    nodes = np.unique(np.concatenate([a_arr, b_arr]))
    log(f"Total unique nodes: {len(nodes):,}")


if __name__ == "__main__":
    main()
