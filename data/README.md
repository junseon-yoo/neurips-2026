# Data — schema reference

Small artifacts only. Large parquets (graph, communities, embeddings) live on Zenodo (see `../EXTERNAL_DATA.md`).

## `queries_80.json`

Eighty research-agenda queries used as the retrieval benchmark, balanced 10 per domain × 8 domains.

```jsonc
{
  "summary": {"total_agendas": 80, "per_category": 10},
  "agendas": [
    {
      "category": "physics",
      "no": 19,
      "agenda": "Hubble Tension & Cosmological Solutions",
      "keywords": ["H0 tension", "Hubble parameter", ...],
      "n_keywords": 20,
      "community_count": 366       // # papers in agent's keyword search ∩ L1 community
    },
    ...
  ]
}
```

## `queries_80_with_graph_search.json`

Same 80 agendas plus the **graph-search baseline output**: each agenda was sent through the LLM agent's keyword search → top 1000 dev-API results → filtered to papers with a community label. The `community_papers` field is the graph-search retrieval result used as one of the comparison methods.

```jsonc
{
  "summary": {...},
  "results": [
    {
      "category": "physics", "no": 19,
      "agenda": "Hubble Tension & Cosmological Solutions",
      "condition": {...},                   // boolean keyword query the agent generated
      "top_papers": [...],                  // dev-API top 1000 (paper_id, title, year, doi, citedCount)
      "community_papers": [                 // top_papers ∩ in-community → graph-search result set
        {"paper_id": "...", "title": "...", "domain": "...", "community_id": ...},
        ...
      ],
      "community_count": 366
    },
    ...
  ]
}
```

## `subfield_check.json`

For each of 80 agendas, the top-100 dense neighbors with full **abstracts** included. Used for qualitative validation / case study reproduction. Each query also carries the agent-issued `query_title` and `neighbors` array with `(rank, sim, paper_id, title, abstract_head, year, ...)`.

## `agenda_topk/topk_<method>.json`

Per-agenda top-K retrieval result for one method.

```jsonc
[
  {
    "category": "physics", "no": 19,
    "agenda": "Hubble Tension & Cosmological Solutions",
    "top_k": [
      {"paper_id": 4367727818, "sim": 0.7942},   // dense methods → "sim"
      {"paper_id": 3042522015, "score": 12.7371},  // bm25 → "score"
      ...
    ]
  },
  ...
]
```

`topk_<method>.json` has K = 100; `topk_<method>_top1000.json` has K = 1000.

Methods provided:
- `bm25` — sparse term-based retrieval
- `specter2` / `qwen3-0.6b` / `qwen3-8b` / `gemini` — dense cosine retrieval

## `full_sweep/full_sweep_<model>_<l1|l2>hier.json`

Per-model rank-K sweep over 200 K papers × 8 domains.

```jsonc
{
  "model": "qwen3", "level": "l1",
  "domains": {
    "physics": {
      "pool_N": 200000,
      "baseline_same_prob": 0.0042,
      "n_communities_in_pool": 1234,
      "max_community_size_in_pool": 23456,
      "max_community_share": 0.117,
      "by_k": {
        "10": {
          "rank_k_same_rate": 0.535,
          "rank_k_same_enrichment": 128.4,
          "any_same_in_topk": 0.85,
          "mean_same_count_in_topk": 5.42,
          "unique_community_count": {"observed_mean": 3.97, "baseline_mean": 7.84, "enrichment": 0.51}
        },
        ...
      }
    },
    ...
  }
}
```

## `analysis/`

| File | Content |
|---|---|
| `four_model_comparison_k10.json` | Per-domain top1_L1/L2 per (model, level) at k=10 |
| `compare_embedding_vs_graph_topk.json` | Per-agenda raw stats: graph_search vs each dense at k=10/50/100 |
| `agenda_l1l2_distribution.json` | Each agenda's `community_papers` (graph search) → L1/L2 counts |
| `agenda_topk_l1l2_distribution.json` | Each agenda's embedding top-K → L1/L2 counts (4 models) |
| `bm25_hybrid.json` | BM25 alone + 4 hybrid variants per agenda |
| `rerank_citation_gemini.json` | Hybrid: gemini top-1000 + induced subgraph in-degree → top-10 |
| `rrf_gemini.json` | RRF: gemini rank + citation rank → top-10 |
| `lexical_divergence.json` | Per-domain JSD + TF-IDF JSD + discriminative-vocab fraction |

## `stats/`

| File | Content |
|---|---|
| `knn_hier_discordance.json` | 200K random + 25K/domain k-NN same-community rate at L1/L2 |
| `subfield_rerun_hier.json` | 80-query manual benchmark per agenda × method (L1/L2 breakdown) |
| `hier_stats.json` | Hierarchical-leiden output stats (community count + size distribution) |
