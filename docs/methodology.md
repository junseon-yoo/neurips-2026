# Methodology — Augmented Citation Graph + Hierarchical Communities

End-to-end summary of the data pipeline, from raw reference table to the L1/L2 community labels used as ground truth in the retrieval benchmark.

## 1. Augmented Graph Construction

Three layers of evidence are combined into a single weighted undirected graph over the 4M target corpus.

### 1.1 Direct citation (target → target)

Source: `citation_graph.parquet` `(paper_id, paper_reference_id)` filtered to rows where both endpoints are in the target corpus. Deduplicated to undirected pairs.

- Edges: 14.56 M
- Mean degree: 8.8 (sparse — 17 % of nodes have no in-corpus citation)
- Weight: floor 1.0 (strongest evidence)

### 1.2 Bibliographic coupling (BC)

Two papers A, B share ≥ 3 references (references can be any paper, not just target).

$$\mathrm{BC}_{cos}(A, B) = \frac{|R(A) \cap R(B)|}{\sqrt{|R(A)| \cdot |R(B)|}}$$

Implemented as a DuckDB self-join on the full reference table:

```sql
SELECT LEAST(a.pid, b.pid), GREATEST(a.pid, b.pid), COUNT(*) shared
FROM target_refs a JOIN target_refs b
  ON a.rid = b.rid AND a.pid < b.pid
GROUP BY 1, 2 HAVING COUNT(*) >= 3
```

- Hot-ref cap: drop refs cited by > 500 target papers (universal references)
- Output: 132.19 M edges, mean cos ≈ 0.07

### 1.3 Co-citation (CC)

Two target papers cited together by ≥ 3 common citers.

$$\mathrm{CC}_{cos}(A, B) = \frac{|C(A) \cap C(B)|}{\sqrt{|C(A)| \cdot |C(B)|}}$$

- Survey cap: drop citers that cite > 200 target papers
- Output: 21.22 M edges

### 1.4 Combined graph

Weights accumulate: `direct floor 1.0 + Σ BC_cos + Σ CC_cos`. Result:

- 153.18 M undirected edges
- 3.58 M nodes (89.5 % of target)
- Mean degree 85.5 (× 9.7 vs target-only)
- 14.6 M edges with w ≥ 1 (direct present), 138.6 M with 0 < w < 1 (BC/CC only)

DuckDB out-of-core runtime: ~3 minutes for both BC and CC.

## 2. Level-1 Communities (sub-fields)

Single Leiden CPM run over the full augmented graph.

`leidenalg.find_partition(g, la.CPMVertexPartition, weights='weight', resolution_parameter=γ, seed=42)`

We sweep γ ∈ {1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2}. The selected resolution is **γ = 1e-4**, where:

| γ | n_comm | max | 10 K+ | 1 K-10 K | singletons |
|---|---|---|---|---|---|
| 1e-6 | 85 K | 907 K | 14 | 14 | 60 K |
| 1e-5 | 76 K | 144 K | 75 | 116 | 55 K |
| 5e-5 | 71 K | 80 K | 86 | 494 | 50 K |
| **1e-4** | **73 K** | **53 K** | **40** | **740** | 51 K |
| 5e-4 | 89 K | 19 K | 2 | 778 | 55 K |

Top-10 sizes at γ=1e-4 (17 K-53 K) match named sub-fields ("Herbig Ae/Be stars", "Ti alloy deformation", "fault tree analysis", ...).

Per-domain L1 size distribution still has artefacts in physics (one cluster holding 11.6 % of the domain), motivating the hierarchical split.

## 3. Level-2 Communities (research agendas)

For every L1 community with ≥ 200 papers (1,896 of them):

1. Induce the subgraph (only edges where both endpoints lie in the L1 community)
2. Run Leiden CPM on the subgraph with γ = 1e-2
3. Assign global L2 IDs as `L2_id = L1_id × 1,000,000 + local_sub_id`

Why hierarchical: running γ = 1e-2 globally atomizes ~3 % of papers because cross-subfield edges are too weak. Inside an L1 subgraph the internal density is high (BC/CC concentrated), so a higher resolution finds research-agenda clusters without atomizing.

| Metric | L1 (γ=1e-4) | L2 (γ=1e-4 × γ=1e-2) |
|---|---|---|
| # communities | 73 K | **329 K** |
| max size | 53 K | **1,712** |
| singletons (% comm / % papers) | 69 % / 1.4 % | 46 % / 4.3 % |
| non-singleton mean size | — | 19.4 (median 7) |
| 100-999 papers (≈ agenda) | 1,632 | **6,010** |
| 10-99 papers | — | 69 K |

51 % of papers sit in 10-99 sized clusters; 33 % in 100-999. The typical paper's L2 community is the size of "a narrow research thread within a sub-field" (10s-100s of papers across 1-5 labs).

## 4. Caveats

- 10.5 % of target papers have neither direct citation nor BC/CC tie to any other target paper → excluded from the graph
- 4.3 % of papers end up as L2 singletons (no narrow agenda neighbor). They are kept in L1 but lose discriminative L2 signal
- L2 size cap = 1,712 — a few large L2 clusters survive (mostly ~1 K-paper sub-areas of physics/cosmology)
- BC `max_citers = 500` and CC `max_refs = 200` are heuristic noise filters; not swept in this work

Full reproducibility scripts:
- `code/graph/duckdb_bc_cc.py`
- `code/graph/build_augmented_graph.py`
- `code/community/leiden_cpm_parallel.py`
- `code/community/hierarchical_leiden.py`
