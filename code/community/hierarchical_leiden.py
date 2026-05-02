#!/usr/bin/env python3
"""Hierarchical Leiden CPM: split each Level-1 community (γ=1e-4) into Level-2
sub-communities by running Leiden on its INDUCED SUBGRAPH at higher γ (=1e-2).

Why hierarchical: running γ=1e-2 globally atomizes 100K+ papers because
cross-subfield edges are too weak. Running it inside each L1 subgraph keeps
papers anchored to their sub-field while resolving narrow research agendas.

Inputs:
    $DATA_DIR/augmented_graph_v2.parquet
    $DATA_DIR/communities_augmented_v2/leiden_cpm_g1e-04.parquet  (Level 1)

Outputs:
    $DATA_DIR/communities_augmented_v2/hier_l1_1e-04_l2_1e-02.parquet
        (paper_id, level1_comm, level2_comm)
"""
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
GRAPH_PATH = DATA_DIR / "augmented_graph_v2.parquet"
L1_PATH = DATA_DIR / "communities_augmented_v2" / "leiden_cpm_g1e-04.parquet"
OUT_DIR = DATA_DIR / "communities_augmented_v2"
OUT = OUT_DIR / "hier_l1_1e-04_l2_1e-02.parquet"
STATS = OUT_DIR / "hier_stats.json"

MIN_SIZE = 200      # only refine L1 communities with >=200 papers
L2_GAMMA = 1e-2     # high resolution within each L1 subgraph
N_WORKERS = int(os.environ.get("LEIDEN_WORKERS", "6"))


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def process_community(args):
    cid, pids, src, dst, wts = args
    import igraph as ig
    import leidenalg as la

    g = ig.Graph(n=len(pids), edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.es["weight"] = wts.tolist()
    part = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=L2_GAMMA,
        seed=42,
    )
    sub = np.asarray(part.membership, dtype=np.int32)
    return cid, pids, sub


def main() -> None:
    log("Loading augmented graph")
    edges = pq.read_table(str(GRAPH_PATH))
    a = edges.column("a").to_numpy().astype(np.int64)
    b = edges.column("b").to_numpy().astype(np.int64)
    w = edges.column("weight").to_numpy().astype(np.float32)
    log(f"  edges: {len(a):,}")

    log("Loading Level 1 community assignment")
    l1 = pq.read_table(str(L1_PATH))
    l1_pids = l1.column("paper_id").to_numpy().astype(np.int64)
    l1_cids = l1.column("community_id").to_numpy().astype(np.int64)
    comm_of = dict(zip(l1_pids.tolist(), l1_cids.tolist()))

    sizes: dict[int, int] = defaultdict(int)
    for c in l1_cids.tolist():
        sizes[int(c)] += 1
    large_comms = set(c for c, s in sizes.items() if s >= MIN_SIZE)
    log(f"  L1 communities >=({MIN_SIZE}): {len(large_comms):,}")

    papers_by_comm: dict[int, list[int]] = defaultdict(list)
    for p, c in comm_of.items():
        if c in large_comms:
            papers_by_comm[c].append(p)

    log("Filtering induced edges (vectorized)")
    a_list = a.tolist()
    b_list = b.tolist()
    ca = np.fromiter((comm_of.get(x, -1) for x in a_list), dtype=np.int64, count=len(a))
    cb = np.fromiter((comm_of.get(x, -1) for x in b_list), dtype=np.int64, count=len(b))
    mask = (ca == cb) & (ca >= 0) & np.isin(ca, np.fromiter(large_comms, dtype=np.int64))
    fa = a[mask]
    fb = b[mask]
    fw = w[mask]
    fc = ca[mask]
    log(f"  induced edges: {len(fa):,}")

    order = np.argsort(fc, kind="stable")
    fa, fb, fw, fc = fa[order], fb[order], fw[order], fc[order]
    unique_c, start_idx = np.unique(fc, return_index=True)
    end_idx = np.concatenate([start_idx[1:], [len(fc)]])

    log("Preparing tasks (one per L1 community)")
    tasks = []
    for cid, s, e in zip(unique_c.tolist(), start_idx.tolist(), end_idx.tolist()):
        pids_list = papers_by_comm.get(cid, [])
        if not pids_list or e - s == 0:
            continue
        pids = np.array(sorted(pids_list), dtype=np.int64)
        pid_to_idx = {int(p): i for i, p in enumerate(pids.tolist())}
        sub_a = np.fromiter((pid_to_idx[int(x)] for x in fa[s:e].tolist()), dtype=np.int32, count=e - s)
        sub_b = np.fromiter((pid_to_idx[int(x)] for x in fb[s:e].tolist()), dtype=np.int32, count=e - s)
        tasks.append((cid, pids, sub_a, sub_b, fw[s:e]))
    log(f"  tasks: {len(tasks):,}")

    log(f"Running parallel Leiden ({N_WORKERS} workers, γ={L2_GAMMA})")
    tasks.sort(key=lambda t: -len(t[1]))
    ctx = mp.get_context("spawn")
    t0 = time.time()
    results = []
    with ctx.Pool(N_WORKERS) as pool:
        done = 0
        for r in pool.imap_unordered(process_community, tasks, chunksize=1):
            results.append(r)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                log(f"  progress: {done}/{len(tasks)}  ({time.time()-t0:.0f}s)")
    log(f"Leiden done in {time.time()-t0:.0f}s")

    log("Aggregating")
    out_pids: list[int] = []
    out_l1: list[int] = []
    out_l2: list[int] = []
    processed_l1 = set()
    for cid, pids, sub in results:
        processed_l1.add(cid)
        for pid, sc in zip(pids.tolist(), sub.tolist()):
            out_pids.append(pid)
            out_l1.append(cid)
            out_l2.append(cid * 1_000_000 + int(sc))
    # Add unsplit L1 communities (small or no induced edges)
    for p, c in comm_of.items():
        if c in large_comms and c in processed_l1:
            continue
        out_pids.append(p)
        out_l1.append(c)
        out_l2.append(c * 1_000_000)

    tbl = pa.table(
        {
            "paper_id": np.array(out_pids, dtype=np.int64),
            "level1_comm": np.array(out_l1, dtype=np.int64),
            "level2_comm": np.array(out_l2, dtype=np.int64),
        }
    )
    pq.write_table(tbl, str(OUT), compression="zstd")
    log(f"Wrote {OUT}")

    l2_arr = np.array(out_l2, dtype=np.int64)
    _, l2_cnt = np.unique(l2_arr, return_counts=True)
    stats = {
        "total_l1_communities": len(set(out_l1)),
        "total_l2_communities": len(set(out_l2)),
        "tasks_processed": len(results),
        "l2_size_stats": {
            "max": int(l2_cnt.max()),
            "median": int(np.median(l2_cnt)),
            "singletons": int((l2_cnt == 1).sum()),
        },
        "l2_size_bins": {
            "1": int((l2_cnt == 1).sum()),
            "2-9": int(((l2_cnt >= 2) & (l2_cnt <= 9)).sum()),
            "10-99": int(((l2_cnt >= 10) & (l2_cnt <= 99)).sum()),
            "100-999": int(((l2_cnt >= 100) & (l2_cnt <= 999)).sum()),
            "1k-9999": int(((l2_cnt >= 1000) & (l2_cnt <= 9999)).sum()),
            "10k+": int((l2_cnt >= 10000).sum()),
        },
    }
    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"Stats: {stats}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
