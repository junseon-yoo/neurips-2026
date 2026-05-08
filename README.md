# Embedding vs Citation Graph for Research-Agenda Retrieval

Code, retrieval results, and reproducibility scripts for the accompanying paper.

We compare **four retrieval methods** for finding papers that match a curated research agenda, evaluated on **80 agenda queries × 8 scientific domains** with citation-graph community labels (L1 sub-field / L2 agenda) as ground truth.

**Methods**:
- Sparse: **BM25** (Lucene-style)
- Dense: **SPECTER2 / qwen3 0.6B / qwen3 8B / Gemini text-embedding**
- Structural: **Augmented citation graph** (direct + bibliographic coupling + co-citation) → keyword-filtered + Leiden CPM communities
- **Hybrid**: top-1000 candidates → citation rerank / RRF fusion

## Quick Start

The paper's analysis outputs are committed under `data/analysis/` and `data/full_sweep/` — open them directly to inspect every per-method, per-domain, per-k number reported in the paper. To re-run the retrieval pipeline end-to-end, see [Reproducibility](#reproducibility).

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

### Inspect-only (no download, no GPU)

Open the committed JSON outputs:

- `data/agenda_topk/topk_<method>.json` — top-K paper IDs per agenda, per method
- `data/full_sweep/full_sweep_<model>_<l1|l2>hier.json` — per-domain rank-K sweep (k ∈ {2,5,10,25,50,100})
- `data/analysis/four_model_comparison_k10.json` — main result table
- `data/analysis/{rerank_citation,rrf,bm25_hybrid,lexical_divergence,...}.json` — every other figure/table

These are the source of truth for every number cited in the paper.

### Re-run the retrieval pipeline (Zenodo + GPU)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download + unpack the Zenodo deposit (5.6 GB zip; ~6.7 GB unpacked)
#    DOI: 10.5281/zenodo.20046263 — https://zenodo.org/records/20046263
mkdir -p data && cd data
curl -L -o neurips-2026-data.zip \
  "https://zenodo.org/records/20046263/files/neurips-2026-data.zip?download=1"
unzip -n neurips-2026-data.zip                # -n = never overwrite committed files (e.g. data/README.md)
mv communities_augmented_v2 communities       # retrieval/eval scripts expect $DATA_DIR/communities/
cd ..

# 3. Generate the four sets of paper embeddings on the 4 M target corpus.
#    Outputs land in data/embeddings/<name>_4m.parquet — keep these names; downstream
#    scripts assume a single 4 M parquet per model (do NOT pass --per-domain).
python code/embeddings/embed_specter2.py                # SPECTER2 base, 768-d, ~12 GB VRAM
python code/embeddings/embed_qwen3.py --model 0.6b      # ~12 GB VRAM
python code/embeddings/embed_qwen3.py --model 8b        # ~24 GB VRAM (fp16); needs ~256 GB host RAM at the .tolist()→parquet step,
                                                        # or pass --per-domain and concatenate the 8 shards (see Notes)
GCP_PROJECT=<your-project> python code/embeddings/embed_gemini.py   # see Notes below — sync mode is sub-corpus only

# 4. Retrievers (BM25 sparse + 4 dense + citation rerank + RRF fusion).
#    The gemini call below also needs GCP_PROJECT in the env, since topk_cosine
#    embeds the 80 agenda queries via Vertex AI before the corpus dot-product.
python code/retrieval/bm25.py
python code/retrieval/topk_cosine.py --model specter2
python code/retrieval/topk_cosine.py --model qwen3-0.6b
python code/retrieval/topk_cosine.py --model qwen3-8b
GCP_PROJECT=<your-project> python code/retrieval/topk_cosine.py --model gemini
for src in specter2 qwen3-0.6b qwen3-8b gemini bm25; do
  python code/retrieval/citation_rerank.py --source "$src"
done
for src in specter2 qwen3-0.6b qwen3-8b gemini; do
  python code/retrieval/rrf.py --dense-source "$src"          # 2-way RRF (embedding + citation); add --bm25-source bm25 for 3-way
done

# 5. Eval — full_sweep is one (model, level) pair per call (4 models × 2 levels = 8 calls).
#    Use the underscored model names here (matches the embedding filename stem).
#    full_sweep.py skips if its output file already exists. The repo ships the
#    paper's reference sweep JSONs, so move them aside first to actually regenerate:
mkdir -p data/full_sweep_ref && mv data/full_sweep/*.json data/full_sweep_ref/ 2>/dev/null || true
python code/eval/full_sweep.py --model specter2   --level l1 --comm-path data/communities/hier_L1_flat.parquet
python code/eval/full_sweep.py --model specter2   --level l2 --comm-path data/communities/hier_L2_flat.parquet
python code/eval/full_sweep.py --model qwen3_0.6b --level l1 --comm-path data/communities/hier_L1_flat.parquet
python code/eval/full_sweep.py --model qwen3_0.6b --level l2 --comm-path data/communities/hier_L2_flat.parquet
python code/eval/full_sweep.py --model qwen3_8b   --level l1 --comm-path data/communities/hier_L1_flat.parquet
python code/eval/full_sweep.py --model qwen3_8b   --level l2 --comm-path data/communities/hier_L2_flat.parquet
python code/eval/full_sweep.py --model gemini     --level l1 --comm-path data/communities/hier_L1_flat.parquet
python code/eval/full_sweep.py --model gemini     --level l2 --comm-path data/communities/hier_L2_flat.parquet

python code/eval/compare_methods.py
python code/eval/lexical_divergence.py
```

**Notes:**

- `DATA_DIR` defaults to `./data`; override via `export DATA_DIR=...`.
- **Gemini embeddings.** `embed_gemini.py` issues *synchronous* Vertex AI calls. For the full 4 M corpus this is impractical on cost and rate limits — the paper's `gemini_4m.parquet` was generated via the **Vertex AI batch prediction API**. The synchronous script is provided only for sub-corpus / agenda-side runs; full-corpus reproduction requires running the same model through the batch API. See the script's docstring.
- **Optional disk savings.** `bc_edges_full.parquet` (1.3 GB), `cc_edges_full.parquet` (256 MB), and `augmented_graph_v2.parquet` (1.9 GB) are included for transparency / sensitivity analysis but are **not read** by any retrieval or eval script. You may delete them after unzip if disk is tight (~3.5 GB saved).
- **Argument naming reminder.** `topk_cosine.py` / `citation_rerank.py` use hyphenated identifiers (`qwen3-0.6b`, `qwen3-8b`); `full_sweep.py --model` is interpolated directly into the embedding filename (`{model}_4m.parquet`), so pass `qwen3_0.6b` / `qwen3_8b` (underscored) / `specter2` / `gemini` there.
- **Output filenames may differ slightly from committed.** The committed `data/full_sweep/full_sweep_qwen3_*.json` (without size suffix) was generated by an earlier naming convention; a fresh re-run produces `full_sweep_qwen3_0.6b_*.json` and `full_sweep_qwen3_8b_*.json`. Same content, different stem. Likewise `compare_methods.py` writes `method_comparison_k10.json` while the committed paper-cited version is `four_model_comparison_k10.json` — the new file is the reproducer's verifying copy.
- **qwen3-8B host-RAM workaround.** `embed_qwen3.py` materialises every 4096-d row as a Python list at the parquet-write step, which peaks at ~256 GB RAM for the full 4 M corpus. On smaller boxes, generate per-domain shards and concatenate:
  ```bash
  python code/embeddings/embed_qwen3.py --model 8b --per-domain
  python -c "import pyarrow.parquet as pq, pyarrow as pa, glob; \
             ts=[pq.read_table(p) for p in sorted(glob.glob('data/embeddings/qwen3_8b_*.parquet')) if '_4m.parquet' not in p]; \
             pq.write_table(pa.concat_tables(ts), 'data/embeddings/qwen3_8b_4m.parquet', compression='zstd')"
  ```
  All downstream scripts (`topk_cosine`, `full_sweep`) only read `qwen3_8b_4m.parquet`.

The graph-construction and community-detection scripts under `code/graph/` and `code/community/` are included for transparency, but rebuilding the citation graph from raw inputs requires private snapshots of multiple bibliographic sources (OpenAlex, Semantic Scholar, etc.) that we cannot redistribute. The Zenodo deposit ships the resulting `augmented_graph_v2.parquet`, `bc_edges_full.parquet`, `cc_edges_full.parquet`, `citation_graph.parquet`, and the full `communities_augmented_v2/` set, so this stage does not need to be re-run.

### Hardware used in the paper

| Stage | GPU | CPU / RAM | Disk | Wall-clock |
|---|---|---|---|---|
| `embed_specter2.py` | 1 × RTX 4070S, 12 GB VRAM | — | +12 GB output | ~1 h |
| `embed_qwen3.py --model 0.6b` | 1 × RTX 4070S, 12 GB VRAM | — | +16 GB output | ~2 h |
| `embed_qwen3.py --model 8b` (fp16) | 1 × RTX 3090, 24 GB VRAM | 64 GB RAM (large df materialization) | +64 GB output | ~6 h |
| `embed_gemini.py` (synchronous) | — | minimal | +16 GB output | hours–days for 4 M; the paper used the Vertex **batch prediction API** instead |
| `bm25.py` | — | 16 vCPU, ~24 GB RAM (bm25s in-memory index) | +2 GB index | ~10 min |
| `topk_cosine.py --model qwen3-8b` | optional (query embed) | **64 GB RAM** for the 4 M × 4096-d float32 matrix | — | 30 – 60 min / model |
| `topk_cosine.py --model {qwen3-0.6b,specter2,gemini}` | optional (query embed) | 16 – 24 GB RAM | — | 10 – 30 min / model |
| `citation_rerank.py`, `rrf.py` | — | <8 GB RAM | — | < 1 min / call |
| `full_sweep.py` | 1 × RTX 4070S (12 GB) suffices for 1024-d models; **24 GB VRAM for qwen3-8B** | 32 GB RAM | — | 5 – 10 min / (model, level) |
| `compare_methods.py`, `lexical_divergence.py` | — | 16 GB RAM | — | <5 min total |
| (Reference) Leiden CPM γ-sweep + hierarchical L2 | — | 22 vCPU, ≤32 GB RAM | — | ~3 h (already done; ships in Zenodo) |

**Approximate totals** for the full retrieval / eval pipeline (Steps 2 – 5):

- Disk: ~7 GB Zenodo unpacked + ~108 GB embeddings (12 SPECTER2 + 16 qwen3-0.6B + 64 qwen3-8B + 16 Gemini) + 2 GB BM25 index ≈ **120 GB free space recommended** (or ~55 GB if you skip qwen3-8B).
- RAM: 24 GB suffices for everything except qwen3-8B; that model needs ~64 GB to *run* the dense top-K (`topk_cosine`) and ~256 GB to *write* the embedding parquet in a single pass — see the per-domain workaround in the Notes above.
- Wall-clock: **~12 h** end-to-end on a 24 GB-VRAM + 64 GB-RAM box if you defer qwen3-8B; ~24 h with qwen3-8B included. Excludes Gemini batch-API runtime.

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
