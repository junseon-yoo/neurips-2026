#!/usr/bin/env python3
"""Parallel Leiden CPM γ sweep on the augmented graph.

Spawns N worker processes; each worker loads the graph once, then pulls γ values
from a shared queue and writes a per-γ community parquet.

Inputs:
    $DATA_DIR/augmented_graph_v2.parquet (a, b, weight)

Outputs:
    $DATA_DIR/communities_augmented_v2/leiden_cpm_g{γ}.parquet (paper_id, community_id)
    $DATA_DIR/leiden_cpm_sweep_v2_stats.json
"""
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
GRAPH_PATH = DATA_DIR / "augmented_graph_v2.parquet"
OUT_DIR = DATA_DIR / "communities_augmented_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_PATH = DATA_DIR / "leiden_cpm_sweep_v2_stats.json"

GAMMAS = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
N_WORKERS = int(os.environ.get("LEIDEN_WORKERS", "4"))


def log(m: str, who: str = "main") -> None:
    print(f"[{time.strftime('%H:%M:%S')}][{who}] {m}", flush=True)


def worker(worker_id: int, gamma_queue: mp.Queue, result_queue: mp.Queue) -> None:
    import igraph as ig
    import leidenalg as la

    who = f"w{worker_id}"
    log("loading graph", who)
    edges = pq.read_table(str(GRAPH_PATH))
    a = edges.column("a").to_numpy().astype(np.int64)
    b = edges.column("b").to_numpy().astype(np.int64)
    w = edges.column("weight").to_numpy().astype(np.float32)
    all_ids = np.unique(np.concatenate([a, b]))
    id_map = {int(nid): i for i, nid in enumerate(all_ids.tolist())}
    inv_ids = all_ids
    N = len(all_ids)

    ea = np.fromiter((id_map[int(x)] for x in a.tolist()), dtype=np.int64, count=len(a))
    eb = np.fromiter((id_map[int(x)] for x in b.tolist()), dtype=np.int64, count=len(b))
    g = ig.Graph(n=N, edges=list(zip(ea.tolist(), eb.tolist())), directed=False)
    g.es["weight"] = w.tolist()
    log(f"igraph |V|={N:,} |E|={len(a):,}", who)
    del a, b, ea, eb, edges

    while True:
        try:
            gamma = gamma_queue.get(timeout=2)
        except Exception:
            return
        if gamma is None:
            return
        tag = f"g{gamma:.0e}"
        out_path = OUT_DIR / f"leiden_cpm_{tag}.parquet"
        log(f"=== γ={gamma:.0e} START ===", who)
        if out_path.exists():
            result_queue.put((tag, {"skip": True}))
            continue
        t0 = time.time()
        try:
            part = la.find_partition(
                g,
                la.CPMVertexPartition,
                weights="weight",
                resolution_parameter=gamma,
                seed=42,
            )
        except Exception as e:
            result_queue.put((tag, {"error": str(e)}))
            continue
        mem = np.asarray(part.membership, dtype=np.int64)
        elapsed = time.time() - t0
        u, cnt = np.unique(mem, return_counts=True)
        stats = {
            "gamma": gamma,
            "n_communities": int(len(u)),
            "max_size": int(cnt.max()),
            "median_nonsingleton": int(np.median(cnt[cnt > 1])) if (cnt > 1).any() else 0,
            "quality": float(part.quality()),
            "time_sec": elapsed,
            "size_bins": {
                "1": int((cnt == 1).sum()),
                "2-9": int(((cnt >= 2) & (cnt <= 9)).sum()),
                "10-99": int(((cnt >= 10) & (cnt <= 99)).sum()),
                "100-999": int(((cnt >= 100) & (cnt <= 999)).sum()),
                "1k-9999": int(((cnt >= 1000) & (cnt <= 9999)).sum()),
                "10k+": int((cnt >= 10000).sum()),
            },
            "top10_sizes": sorted(cnt.tolist(), reverse=True)[:10],
        }
        log(f"γ={gamma:.0e} n={len(u):,} max={cnt.max():,} t={elapsed:.0f}s", who)
        tbl = pa.table({"paper_id": inv_ids.astype(np.int64), "community_id": mem.astype(np.int64)})
        pq.write_table(tbl, str(out_path), compression="zstd")
        result_queue.put((tag, stats))


def main() -> None:
    mp.set_start_method("spawn", force=True)
    log(f"Launching {N_WORKERS} workers for {len(GAMMAS)} γ values: {GAMMAS}")

    gamma_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    for g in GAMMAS:
        gamma_queue.put(g)
    for _ in range(N_WORKERS):
        gamma_queue.put(None)

    workers = []
    for i in range(N_WORKERS):
        p = mp.Process(target=worker, args=(i, gamma_queue, result_queue))
        p.start()
        workers.append(p)

    all_stats: dict = {}
    received = 0
    while received < len(GAMMAS):
        try:
            tag, stats = result_queue.get(timeout=60)
        except Exception:
            if not any(p.is_alive() for p in workers):
                break
            continue
        all_stats[tag] = stats
        received += 1
        with open(STATS_PATH, "w") as f:
            json.dump(all_stats, f, indent=2)
        log(f"Received {tag} [{received}/{len(GAMMAS)}]")

    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    log("Done.")


if __name__ == "__main__":
    main()
