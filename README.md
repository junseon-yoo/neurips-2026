# Embedding vs Citation Graph for Research-Agenda Retrieval (NeurIPS 2026)

Code, retrieval results, and reproducibility scripts for the paper.

We compare **four retrieval methods** for finding papers that match a curated research agenda, evaluated on **80 agenda queries × 8 scientific domains** with citation-graph community labels (L1 sub-field / L2 agenda) as ground truth.

**Methods**:
- Sparse: **BM25** (Lucene-style)
- Dense: **SPECTER2 / qwen3 0.6B / qwen3 8B / Gemini text-embedding**
- Structural: **Augmented citation graph** (direct + bibliographic coupling + co-citation) → keyword-filtered + Leiden CPM communities
- **Hybrid**: top-1000 candidates → citation rerank / RRF fusion

## Quick Start

```bash
pip install -r requirements.txt

# 1. Build the augmented citation graph (uses full reference table from Zenodo)
python code/graph/duckdb_bc_cc.py
python code/graph/build_augmented_graph.py

# 2. Hierarchical community detection (Leiden CPM)
python code/community/leiden_cpm_parallel.py        # L1 sweep over γ
python code/community/hierarchical_leiden.py        # L2 within each L1

# 3. Embed the 4M target corpus (per model)
python code/embeddings/embed_qwen3.py
python code/embeddings/embed_specter2.py
python code/embeddings/embed_gemini.py

# 4. Run retrievers + hybrid + RRF for the 80-query benchmark
python code/retrieval/bm25.py
python code/retrieval/topk_cosine.py --model gemini
python code/retrieval/citation_rerank.py
python code/retrieval/rrf.py

# 5. Reproduce paper tables/figures
python code/eval/agenda_l1l2_analysis.py
python code/eval/compare_methods.py
```

## Data

Small artifacts are committed in `data/`. The large augmented graph + community parquets are deposited on Zenodo:

- **DOI**: [10.5281/zenodo.20046263](https://doi.org/10.5281/zenodo.20046263) (5.6 GB zip, CC-BY-4.0)

See `EXTERNAL_DATA.md` for the full file inventory and schemas.

| Path | Size | Content |
|---|---|---|
| `data/queries_80.json` | 42 KB | 80 agenda queries × keywords × representative paper IDs |
| `data/full_sweep/full_sweep_<model>_<l1\|l2>hier.json` | 8 × ~45 KB | 4 models × 2 levels × 8 domains × k = {2,5,10,25,50,100} |
| `data/agenda_topk/topk_<method>.json` | ~600 KB each | top-100 / top-1000 retrieval per method |
| `data/analysis/*.json` | small | per-agenda L1/L2 distribution + method comparison |
| `data/stats/*.json` | small | community sweep / discordance / validation stats |

External (Zenodo, see `EXTERNAL_DATA.md`):
- `augmented_graph_v2.parquet` (1.9 GB) — final 153 M-edge graph
- `bc_edges_full.parquet` (1.3 GB) — bibliographic coupling edges
- `cc_edges_full.parquet` (256 MB) — co-citation edges
- `communities_augmented_v2/hier_l1_1e-04_l2_1e-02.parquet` (29 MB) — final L1/L2 community map

## Reproducibility

### Inputs

The fastest path is to download the Zenodo deposit (above) and run the retrieval / evaluation scripts directly — no S3 or paper-table rebuild required.

Rebuilding the augmented graph from scratch needs a `paper_reference` table. We used a private snapshot, but the equivalent is publicly available from the **OpenAlex** bulk export (`works.referenced_works`, 2026-03-31 snapshot used in the paper). The expected schema is `(paper_id string, paper_reference_id string)`; see `EXTERNAL_DATA.md`.

### Optional environment variables

Only needed if you want to read inputs from / write outputs to S3-compatible object storage instead of local disk. All scripts also accept a local `DATA_DIR`.

```bash
export DATA_DIR="./data"          # local fallback, used by default
export S3_BUCKET="..."            # optional — only if rebuilding from S3
export S3_ACCESS_KEY="..."
export S3_SECRET_KEY="..."
export S3_ENDPOINT_URL="..."
export GCP_PROJECT="..."          # only for code/embeddings/embed_gemini.py
export GCP_LOCATION="us-central1"
```

### Compute

Embeddings were generated on a single RTX 4070S (12 GB VRAM) for BM25 / SPECTER2 / qwen3-0.6B, and a single RTX 3090 (24 GB VRAM) for qwen3-8B at fp16. Graph construction and Leiden CPM run on CPU (DuckDB + igraph; ~22 vCPU, ≤32 GB RAM). End-to-end wall-clock is ≈ 12 hours.

## Repo Layout

```
.
├── code/
│   ├── graph/         # augmented graph construction (BC, CC via DuckDB)
│   ├── community/     # Leiden CPM (γ sweep + L1/L2 hierarchical)
│   ├── embeddings/    # 4-model paper embedding generation
│   ├── retrieval/     # BM25, dense top-K, citation rerank, RRF
│   ├── eval/          # discordance, sweep, hybrid comparison
│   └── plot/          # figure/table generators
├── data/              # small JSON artifacts (full results)
├── prompts/           # LLM agent prompts (search-strategist, research-analyst)
├── docs/              # methodology + results inventory
└── EXTERNAL_DATA.md   # links to large parquets
```

## License

- **Code** (`code/`, scripts) — MIT, see `LICENSE`.
- **Data** (`data/` and the linked Zenodo deposit) — CC-BY-4.0, see `LICENSE-DATA`.
