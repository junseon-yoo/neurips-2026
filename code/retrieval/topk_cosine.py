#!/usr/bin/env python3
"""Dense retrieval: embed agenda titles, return top-K papers by cosine.

Supports qwen3 (0.6B / 8B), specter2, gemini.

Inputs:
    $DATA_DIR/embeddings/<model>_4m.parquet  (paper_id, embedding)
    $DATA_DIR/queries_80.json
Output:
    $DATA_DIR/agenda_topk/topk_<model>.json
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def last_token_pool(h, mask):
    left_padding = (mask[:, -1].sum() == mask.shape[0])
    if left_padding:
        return h[:, -1]
    seq_lens = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), seq_lens]


def embed_query_qwen3(model_size: str, queries: list[str]) -> np.ndarray:
    name = f"Qwen/Qwen3-Embedding-{model_size.upper()}"
    log(f"Loading {name}")
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(queries), 16):
            bd = tok(queries[i:i + 16], padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to("cuda")
            out = model(**bd)
            emb = last_token_pool(out.last_hidden_state, bd["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            embs.append(emb.float().cpu().numpy())
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(embs, axis=0).astype(np.float32)


def embed_query_specter2(queries: list[str]) -> np.ndarray:
    name = "allenai/specter2_base"
    log(f"Loading {name}")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).cuda().eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(queries), 32):
            bd = tok(queries[i:i + 32], padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to("cuda")
            out = model(**bd)
            emb = out.last_hidden_state[:, 0, :]
            embs.append(emb.cpu().numpy())
    embs = np.concatenate(embs, axis=0).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-12)
    return embs / norms


def embed_query_gemini(queries: list[str]) -> np.ndarray:
    from google import genai
    from google.genai import types
    project = os.environ.get("GCP_PROJECT")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not set")
    client = genai.Client(vertexai=True, project=project, location=location)
    embs = []
    for i, q in enumerate(queries):
        r = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[q],
            config=types.EmbedContentConfig(output_dimensionality=1024, task_type="RETRIEVAL_QUERY"),
        )
        embs.append(r.embeddings[0].values)
        if (i + 1) % 20 == 0:
            log(f"  {i+1}/{len(queries)}")
    arr = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
    return arr / norms


def stream_paper_embedding(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    log(f"Loading paper embedding {parquet_path}")
    pf = pq.ParquetFile(str(parquet_path))
    pids_list, X_list = [], []
    for batch in pf.iter_batches(batch_size=100000, columns=["paper_id", "embedding"]):
        b_pids = batch.column("paper_id").to_numpy(zero_copy_only=False)
        try:
            b_pids = b_pids.astype(np.int64)
        except Exception:
            b_pids = np.fromiter((int(x) for x in b_pids), dtype=np.int64, count=len(b_pids))
        ec = batch.column("embedding")
        if pa.types.is_fixed_size_list(ec.type):
            D = ec.type.list_size
            flat = ec.values.to_numpy(zero_copy_only=False)
        else:
            off = ec.offsets.to_numpy()
            D = int(off[1] - off[0])
            flat = ec.values.to_numpy(zero_copy_only=False)
        X = flat.reshape(-1, D).astype(np.float32, copy=True)
        pids_list.append(b_pids)
        X_list.append(X)
    pids = np.concatenate(pids_list)
    X = np.concatenate(X_list, axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    X = X / norms
    return pids, X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-0.6b", "qwen3-8b", "specter2", "gemini"])
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--queries-path", default=str(DATA_DIR / "queries_80.json"))
    args = ap.parse_args()

    log(f"Loading queries from {args.queries_path}")
    with open(args.queries_path) as f:
        agendas = json.load(f).get("agendas", json.load(open(args.queries_path)))
    queries = [a["agenda"] for a in agendas]
    log(f"  {len(queries)} queries")

    if args.model == "qwen3-0.6b":
        qvecs = embed_query_qwen3("0.6b", queries)
        paper_emb = DATA_DIR / "embeddings" / "qwen3_0.6b_4m.parquet"
    elif args.model == "qwen3-8b":
        qvecs = embed_query_qwen3("8b", queries)
        paper_emb = DATA_DIR / "embeddings" / "qwen3_8b_4m.parquet"
    elif args.model == "specter2":
        qvecs = embed_query_specter2(queries)
        paper_emb = DATA_DIR / "embeddings" / "specter2_4m.parquet"
    else:
        qvecs = embed_query_gemini(queries)
        paper_emb = DATA_DIR / "embeddings" / "gemini_4m.parquet"

    pids, X = stream_paper_embedding(paper_emb)
    log(f"papers: {len(pids):,} × {X.shape[1]}")

    # Streaming top-K via numpy (works on CPU; for GPU use torch.matmul)
    n_q = len(queries)
    top_sims = np.full((n_q, args.top_k), -np.inf, dtype=np.float32)
    top_pids = np.zeros((n_q, args.top_k), dtype=np.int64)
    chunk = 50000
    for s in range(0, len(pids), chunk):
        e = min(len(pids), s + chunk)
        sims = qvecs @ X[s:e].T
        for qi in range(n_q):
            cs = sims[qi]
            cand = np.concatenate([top_sims[qi], cs])
            cand_pids = np.concatenate([top_pids[qi], pids[s:e]])
            idx = np.argpartition(-cand, args.top_k)[:args.top_k]
            order = idx[np.argsort(-cand[idx])]
            top_sims[qi] = cand[order]
            top_pids[qi] = cand_pids[order]

    out_path = DATA_DIR / "agenda_topk" / f"topk_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for i, a in enumerate(agendas):
        out.append({
            "category": a.get("category"),
            "no": a.get("no"),
            "agenda": a["agenda"],
            "top_k": [{"paper_id": int(top_pids[i][j]), "sim": round(float(top_sims[i][j]), 5)}
                       for j in range(args.top_k)],
        })
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
