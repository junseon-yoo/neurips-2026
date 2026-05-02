#!/usr/bin/env python3
"""Embed target corpus with Gemini text-embedding (Vertex AI).

Requires gcloud ADC + a Google Cloud project. Set:
    GCP_PROJECT=<project-id>
    GCP_LOCATION=us-central1

Output: $DATA_DIR/embeddings/gemini_4m.parquet (paper_id, embedding 1024-d)

Vertex AI batch API is recommended for the full 4M corpus; this script does
synchronous calls in batches and is provided for reproducibility / sub-corpus
runs. For the full 4M, expect non-trivial cost and use the batch API.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from google import genai
from google.genai import types

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
TARGET = DATA_DIR / "target" / "neurips_4m.parquet"
OUT = DATA_DIR / "embeddings" / "gemini_4m.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

PROJECT = os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL = "gemini-embedding-001"
OUTPUT_DIM = 1024
BATCH = 100   # Vertex API batch size limit


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    if not PROJECT:
        raise RuntimeError("Set GCP_PROJECT env variable")
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    df = pq.read_table(str(TARGET), columns=["paper_id", "title", "abstract"]).to_pandas()
    log(f"{len(df):,} papers")

    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    paper_ids = df["paper_id"].astype(str).tolist()
    n = len(texts)
    all_embs: list[list[float]] = []

    for i in range(0, n, BATCH):
        chunk = texts[i:i + BATCH]
        for attempt in range(3):
            try:
                rs = client.models.embed_content(
                    model=MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(
                        output_dimensionality=OUTPUT_DIM,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                for emb in rs.embeddings:
                    all_embs.append(emb.values)
                break
            except Exception as e:
                log(f"  retry {attempt+1} after error: {e}")
                time.sleep(2 ** attempt)
        if (i // BATCH) % 50 == 0 and i > 0:
            log(f"  {i:,}/{n:,}")

    embs = np.array(all_embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-12)
    embs = embs / norms
    log(f"embeddings: {embs.shape}")

    out_df = pd.DataFrame({"paper_id": paper_ids, "embedding": [e.tolist() for e in embs]})
    out_df.to_parquet(OUT, compression="zstd")
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
