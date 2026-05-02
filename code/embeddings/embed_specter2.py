#!/usr/bin/env python3
"""Embed target corpus with SPECTER2 (allenai/specter2_base).

Inputs:  $DATA_DIR/target/neurips_4m.parquet
Output:  $DATA_DIR/embeddings/specter2_4m.parquet (paper_id, embedding 768-d)
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
TARGET = DATA_DIR / "target" / "neurips_4m.parquet"
OUT = DATA_DIR / "embeddings" / "specter2_4m.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "allenai/specter2_base"
BATCH = int(os.environ.get("SPECTER2_BATCH", "256"))
MAX_LEN = 512


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    log(f"Loading {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).cuda().eval()

    log("Loading target")
    df = pq.read_table(str(TARGET), columns=["paper_id", "title", "abstract"]).to_pandas()
    log(f"  {len(df):,} papers")

    # SPECTER2 expects: title + [SEP] + abstract
    sep = tok.sep_token or " "
    texts = (df["title"].fillna("") + f" {sep} " + df["abstract"].fillna("")).tolist()
    paper_ids = df["paper_id"].astype(str).tolist()
    n = len(texts)
    all_embs = []
    for i in range(0, n, BATCH):
        with torch.no_grad():
            bd = tok(
                texts[i:i + BATCH],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to("cuda")
            out = model(**bd)
            emb = out.last_hidden_state[:, 0, :]   # [CLS]
            all_embs.append(emb.cpu().numpy())
        if (i // BATCH) % 50 == 0 and i > 0:
            log(f"  {i:,}/{n:,}")
    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-12)
    embs = embs / norms
    log(f"  embeddings: {embs.shape}")
    out_df = pd.DataFrame({"paper_id": paper_ids, "embedding": [e.tolist() for e in embs]})
    out_df.to_parquet(OUT, compression="zstd")
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
