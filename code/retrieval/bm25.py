#!/usr/bin/env python3
"""BM25 sparse retrieval baseline.

Indexes the 4M target corpus (title + abstract) with bm25s, then retrieves
top-K papers per agenda query.

Inputs:
    $DATA_DIR/target/neurips_4m.parquet
    $DATA_DIR/queries_80.json (or supply --queries-path)

Outputs:
    $DATA_DIR/bm25_index/   (saved index for reuse)
    $DATA_DIR/agenda_topk/topk_bm25.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import bm25s
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_or_load_index(target_path: Path, index_dir: Path):
    if index_dir.exists() and (index_dir / "params.index.json").exists():
        log(f"Loading existing index from {index_dir}")
        retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
        # paper_ids must be loaded separately
        pids = pq.read_table(str(target_path), columns=["paper_id"]).column("paper_id").to_numpy().astype(np.int64)
        return retriever, pids

    log(f"Building BM25 index from {target_path}")
    t = pq.read_table(str(target_path), columns=["paper_id", "title", "abstract"])
    pids = t.column("paper_id").to_numpy().astype(np.int64)
    titles = t.column("title").to_pylist()
    abstracts = t.column("abstract").to_pylist()
    corpus = [(titles[i] or "") + " " + (abstracts[i] or "") for i in range(len(pids))]
    log(f"  {len(corpus):,} docs")

    log("Tokenizing")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
    log(f"  done in {time.time()-t0:.0f}s")

    log("Indexing")
    t0 = time.time()
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    log(f"  done in {time.time()-t0:.0f}s")

    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(index_dir))
    log(f"  saved to {index_dir}")
    return retriever, pids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-path", default=str(DATA_DIR / "queries_80.json"))
    ap.add_argument("--out-path", default=str(DATA_DIR / "agenda_topk" / "topk_bm25.json"))
    ap.add_argument("--top-k", type=int, default=1000)
    args = ap.parse_args()

    target_path = DATA_DIR / "target" / "neurips_4m.parquet"
    index_dir = DATA_DIR / "bm25_index"
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    retriever, pids = build_or_load_index(target_path, index_dir)

    log(f"Loading queries: {args.queries_path}")
    with open(args.queries_path) as f:
        agendas = json.load(f).get("agendas", json.load(open(args.queries_path)))

    queries = [a["agenda"] for a in agendas]
    log(f"  {len(queries)} queries")

    log(f"Searching top-{args.top_k}")
    t0 = time.time()
    q_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    results, scores = retriever.retrieve(q_tokens, k=args.top_k, show_progress=False)
    log(f"  done in {time.time()-t0:.0f}s")

    out = []
    for i, a in enumerate(agendas):
        top = []
        for j in range(args.top_k):
            idx = int(results[i][j])
            if 0 <= idx < len(pids):
                top.append({"paper_id": int(pids[idx]), "score": round(float(scores[i][j]), 4)})
        out.append({
            "category": a.get("category"),
            "no": a.get("no"),
            "agenda": a["agenda"],
            "top_k": top,
        })
    with open(args.out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
