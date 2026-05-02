#!/usr/bin/env python3
"""Embed target corpus with Qwen3-Embedding (0.6B or 8B).

Pass --model 0.6b or --model 8b. For the 8B model use a 24GB+ GPU.

Inputs:  $DATA_DIR/target/neurips_4m.parquet (paper_id, title, abstract, domain)
Output:  $DATA_DIR/embeddings/qwen3_<size>_4m.parquet (paper_id, embedding)
         (or per-domain shards for the 8B variant)
"""
import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
TARGET = DATA_DIR / "target" / "neurips_4m.parquet"
OUT_DIR = DATA_DIR / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "8b": "Qwen/Qwen3-Embedding-8B",
}
EMB_DIMS = {"0.6b": 1024, "8b": 4096}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def last_token_pool(h, mask):
    left_padding = (mask[:, -1].sum() == mask.shape[0])
    if left_padding:
        return h[:, -1]
    seq_lens = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), seq_lens]


def embed(model_size: str, batch: int, max_len: int = 512, per_domain: bool = False) -> None:
    name = MODEL_NAMES[model_size]
    log(f"Loading {name}")
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()

    df = pq.read_table(str(TARGET), columns=["paper_id", "title", "abstract", "domain"]).to_pandas()
    log(f"  {len(df):,} papers")

    domains = df["domain"].unique().tolist() if per_domain else [None]
    for dom in domains:
        sub = df if dom is None else df[df["domain"] == dom].reset_index(drop=True)
        if len(sub) == 0:
            continue
        suffix = f"_{dom}" if dom else "_4m"
        out = OUT_DIR / f"qwen3_{model_size}{suffix}.parquet"
        if out.exists():
            log(f"  exists, skip: {out.name}")
            continue

        texts = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).tolist()
        n = len(texts)
        log(f"{dom or 'all'}: encoding {n:,}")

        all_embs = []
        for i in range(0, n, batch):
            with torch.no_grad():
                bd = tok(
                    texts[i:i + batch],
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to("cuda")
                emb = model(**bd).last_hidden_state
                emb = last_token_pool(emb, bd["attention_mask"])
                emb = F.normalize(emb, p=2, dim=1)
                all_embs.append(emb.cpu().numpy())
            if (i // batch) % 100 == 0 and i > 0:
                log(f"  {i:,}/{n:,}")
        embs = np.concatenate(all_embs, axis=0)
        log(f"  embeddings: {embs.shape}")
        out_df = pd.DataFrame(
            {"paper_id": sub["paper_id"].astype(str).tolist(), "embedding": [e.tolist() for e in embs]}
        )
        out_df.to_parquet(out, compression="zstd")
        log(f"  wrote {out}")
        del all_embs, embs, out_df
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["0.6b", "8b"], default="0.6b")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--per-domain", action="store_true",
                   help="(8b only) write per-domain parquets to limit RAM/disk")
    args = p.parse_args()
    embed(args.model, args.batch, per_domain=args.per_domain)


if __name__ == "__main__":
    main()
