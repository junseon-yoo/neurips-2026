# External Data

The augmented citation graph and community parquets are too large for GitHub. They will be hosted on Zenodo (anonymous deposit) at submission time.

## Files (will be uploaded as a single zip)

| File | Size | Schema |
|---|---|---|
| `augmented_graph_v2.parquet` | 1.9 GB | `(a int64, b int64, weight float32)` — 153 M undirected edges over 3.58 M papers |
| `bc_edges_full.parquet` | 1.3 GB | `(a, b, shared int32, cos float32)` — 132 M bibliographic-coupling edges (shared ≥ 3) |
| `cc_edges_full.parquet` | 256 MB | `(a, b, shared, cos)` — 21 M co-citation edges |
| `communities_augmented_v2/leiden_cpm_g1e-04.parquet` | 24 MB | `(paper_id, community_id)` — Level 1 (sub-field) at γ = 1e-4, 73 K communities |
| `communities_augmented_v2/hier_l1_1e-04_l2_1e-02.parquet` | 29 MB | `(paper_id, level1_comm, level2_comm)` — final hierarchical map, 329 K L2 communities |
| `communities_augmented_v2/leiden_cpm_g{1e-6 .. 1e-2}.parquet` | ~25 MB each | full γ sweep (8 files) for sensitivity analysis |

Once uploaded, the README will be updated with:

```
DOI: 10.5281/zenodo.<XXXXXXX>
URL: https://zenodo.org/records/<XXXXXXX>
```

## Direct citation table (input to graph build)

`paper_reference` table over the full ~150 M paper corpus is required for BC/CC computation. We used the Hetzner Object Storage snapshot at `dataset/20260331/core_tables/paper_reference/` (200 zstd-compressed parquet shards, 14.5 GB).

For external reproducibility, equivalent data can be derived from:
- **OpenAlex** bulk export (`works/referenced_works`)
- **Semantic Scholar** Citation Graph

Schema expected by `code/graph/duckdb_bc_cc.py`:

```
paper_reference: (paper_id string, paper_reference_id string)
```

## Embeddings

Pre-computed paper embeddings (4 M papers × 4 models, ~50 GB total) are NOT redistributed:

- **qwen3-Embedding-0.6B** / **qwen3-Embedding-8B** / **SPECTER2** are loadable from Hugging Face — re-run `code/embeddings/embed_*.py` to regenerate.
- **Gemini text-embedding-001** is API-only. Use `code/embeddings/embed_gemini.py` with a Google Cloud project.

Time + compute estimates: see paper Appendix.
