# Embedding vs Citation Graph for Research-Agenda Retrieval (NeurIPS 2026)

Code, retrieval results, and reproducibility scripts for the paper.

We compare **four retrieval methods** for finding papers that match a curated research agenda, evaluated on **80 agenda queries × 8 scientific domains** with citation-graph community labels (L1 sub-field / L2 agenda) as ground truth.

**Methods**:
- Sparse: **BM25** (Lucene-style)
- Dense: **SPECTER2 / qwen3 0.6B / qwen3 8B / Gemini text-embedding**
- Structural: **Augmented citation graph** (direct + bibliographic coupling + co-citation) → keyword-filtered + Leiden CPM communities
- **Hybrid**: top-1000 candidates → citation rerank / RRF fusion

## Quick Start

The paper's tables and figures can be reproduced from the JSONs already in `data/` — no GPU, no large download.

```bash
git clone <this repo> && cd neurips-2026
pip install -r requirements.txt

# Reproduce paper tables/figures from committed retrieval outputs
python code/eval/agenda_l1l2_analysis.py
python code/eval/compare_methods.py
```

To re-run the retrieval / community-detection pipeline yourself, fetch the large parquet artifacts from Zenodo first (see [Reproducibility](#reproducibility)).

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

There are three reproduction tiers. Pick the one that matches what you want to verify.

### Tier 1 — paper tables/figures only (no GPU, no download)

All retrieval outputs (`data/agenda_topk/topk_<method>.json`) and per-agenda L1/L2 analysis (`data/analysis/*.json`) are committed in this repo. The eval scripts read them directly:

```bash
pip install -r requirements.txt
python code/eval/agenda_l1l2_analysis.py
python code/eval/compare_methods.py
```

### Tier 2 — re-run retrieval against the committed graph & community labels

Download the Zenodo zip, unzip into `data/`, and run the retrieval pipeline. This regenerates the `agenda_topk/*.json` files that Tier 1 consumes.

```bash
# 1. Download + unpack Zenodo deposit (5.6 GB zip)
#    DOI: 10.5281/zenodo.20046263 — https://zenodo.org/records/20046263
mkdir -p data && cd data
curl -L -o zenodo.zip "https://zenodo.org/records/20046263/files/neurips-2026-data.zip"
unzip zenodo.zip && cd ..

# 2. Generate the four sets of paper embeddings (GPU required)
python code/embeddings/embed_specter2.py
python code/embeddings/embed_qwen3.py --model 0.6B   # 12 GB VRAM
python code/embeddings/embed_qwen3.py --model 8B     # 24 GB VRAM
python code/embeddings/embed_gemini.py               # Vertex AI ADC required

# 3. Run retrievers (BM25, dense top-K, citation rerank, RRF fusion)
python code/retrieval/bm25.py
python code/retrieval/topk_cosine.py --model specter2
python code/retrieval/topk_cosine.py --model qwen3-0.6b
python code/retrieval/topk_cosine.py --model qwen3-8b
python code/retrieval/topk_cosine.py --model gemini
python code/retrieval/citation_rerank.py
python code/retrieval/rrf.py
```

By default all paths resolve to `DATA_DIR=./data`; override with `export DATA_DIR=...` if you keep artifacts elsewhere. Regenerating Gemini embeddings additionally needs a Google Cloud project (Vertex AI ADC) — see the docstring of `code/embeddings/embed_gemini.py`.

### Tier 3 — rebuild the citation graph and communities from scratch

The augmented graph is derivable from any `paper_reference` table with schema `(paper_id string, paper_reference_id string)`. The paper used a 2026-03-31 snapshot equivalent to **OpenAlex** `works.referenced_works`. After preparing `paper_reference.parquet` under `$DATA_DIR/`:

```bash
python code/graph/duckdb_bc_cc.py            # BC + CC via DuckDB out-of-core (~3 min)
python code/graph/build_augmented_graph.py   # combine direct + BC + CC into one weighted graph
python code/community/leiden_cpm_parallel.py # L1 γ sweep
python code/community/hierarchical_leiden.py # L2 inside each L1 induced subgraph
```

This regenerates `augmented_graph_v2.parquet` and the `communities_augmented_v2/` parquets that ship in the Zenodo deposit.

### Compute used in the paper

- **Embeddings**: RTX 4070S (12 GB) for BM25 / SPECTER2 / qwen3-0.6B; RTX 3090 (24 GB) for qwen3-8B at fp16.
- **Graph + Leiden CPM**: CPU only — 22 vCPU, ≤32 GB RAM (DuckDB + igraph).
- **Wall-clock end-to-end**: ≈ 12 h (excluding Gemini API quota).

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
