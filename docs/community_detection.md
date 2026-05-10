# Hierarchical Community Detection — Step-by-Step

A procedural walkthrough of how the L1 (sub-field) and L2 (research-agenda) community labels in `communities_augmented_v2/hier_l1_1e-04_l2_1e-02.parquet` are built. Complements the high-level summary in `methodology.md`.

The pipeline has four stages:

```
                      ┌─────────────────────────────────────────────────────┐
  paper_reference  ─► │ Stage A  Build BC and CC edges (DuckDB self-joins)  │
                      └─────────────────────────────────────────────────────┘
                                            │
                              bc_edges_full + cc_edges_full
                                            ▼
                      ┌─────────────────────────────────────────────────────┐
  citation_graph  ──► │ Stage B  Combine direct + BC + CC → augmented graph │
                      └─────────────────────────────────────────────────────┘
                                            │
                              augmented_graph_v2.parquet
                                            ▼
                      ┌─────────────────────────────────────────────────────┐
                      │ Stage C  Level-1 Leiden CPM γ sweep (one global run │
                      │          per γ; pick γ = 1e-4 as the L1 operating   │
                      │          point)                                     │
                      └─────────────────────────────────────────────────────┘
                                            │
                              leiden_cpm_g1e-04.parquet
                                            ▼
                      ┌─────────────────────────────────────────────────────┐
                      │ Stage D  Level-2 Leiden CPM, one run per L1         │
                      │          community on its induced subgraph at γ =   │
                      │          1e-2; compose global L2 IDs                │
                      └─────────────────────────────────────────────────────┘
                                            │
                            hier_l1_1e-04_l2_1e-02.parquet
```

All inputs and outputs are int64 paper IDs. Random seed = 42 throughout.

---

## Stage A. BC and CC edges from the reference table

**Script:** `code/graph/duckdb_bc_cc.py`

**Inputs**

| Path | Schema | Notes |
|---|---|---|
| `$DATA_DIR/raw/paper_reference/*.parquet` | `(paper_id string, paper_reference_id string)` | Full ~150 M-paper reference table. Cannot be redistributed; reproduce from OpenAlex `works.referenced_works`. |
| `$DATA_DIR/target/neurips_4m.parquet` | `paper_id` column | The 4 M-paper target corpus. |

**Outputs**

| Path | Schema |
|---|---|
| `$DATA_DIR/bc_edges_full.parquet` | `(a bigint, b bigint, shared int32, cos float32)` |
| `$DATA_DIR/cc_edges_full.parquet` | same |

**Hyperparameters** (constants near the top of the script)

```python
SHARED_MIN      = 3       # minimum shared references / citers per edge
BC_MAX_CITERS   = 500     # drop refs cited by >500 target papers (universal references)
CC_MAX_REFS     = 200     # drop survey-like citers citing >200 target papers
DUCKDB_MEMORY_LIMIT = "80GB"
DUCKDB_THREADS  = 16
```

**Bibliographic coupling (BC).** A pair of target papers (A, B) is connected if they share at least 3 references. Salton cosine:

$$\mathrm{BC}_{cos}(A, B) = \frac{|R(A) \cap R(B)|}{\sqrt{|R(A)| \cdot |R(B)|}}$$

DuckDB executes a self-join on the reference table after filtering to target paper IDs and stripping "hot" references (papers cited by > 500 target papers — these are vocabulary words like seminal textbooks and dilute the signal):

```sql
-- after building target_refs(pid bigint, rid bigint) and dropping hot rids
SELECT LEAST(a.pid, b.pid) AS a,
       GREATEST(a.pid, b.pid) AS b,
       COUNT(*) AS shared
FROM target_refs a
JOIN target_refs b ON a.rid = b.rid AND a.pid < b.pid
GROUP BY 1, 2
HAVING COUNT(*) >= 3
```

`cos` is computed in a follow-up pass using each paper's reference count.

**Co-citation (CC).** Symmetric: a pair (A, B) is connected if at least 3 common citers cite both. Survey-like citers (citing > 200 target papers) are dropped.

```sql
SELECT LEAST(a.rid, b.rid) AS a,
       GREATEST(a.rid, b.rid) AS b,
       COUNT(*) AS shared
FROM target_refs a
JOIN target_refs b ON a.pid = b.pid AND a.rid < b.rid
GROUP BY 1, 2
HAVING COUNT(*) >= 3
```

**Runtime.** ~3 minutes total (BC + CC) on a 22-vCPU box with 80 GB DuckDB memory cap. Disk-spilled via `temp_directory`.

**Output sizes**

| Layer | # edges | mean cos |
|---|---|---|
| BC | 132.19 M | ≈ 0.07 |
| CC | 21.22 M | ≈ 0.05 |

---

## Stage B. Combine layers into the augmented graph

**Script:** `code/graph/build_augmented_graph.py`

**Weight scheme.** For each undirected pair (A, B):

```
weight(A, B) = 1.0          if (A, B) appears in citation_graph (direct)
             + BC_cos(A, B) if (A, B) ∈ BC
             + CC_cos(A, B) if (A, B) ∈ CC
```

A pair appearing in multiple layers sums all three terms. Direct citation gets a flat 1.0 floor because it is the strongest single signal; BC/CC contribute fractional cosines on top.

**Output:** `$DATA_DIR/augmented_graph_v2.parquet (a, b, weight float32)` — 153.18 M undirected edges over 3.58 M of 4 M target papers (10.5 % of papers have no in-corpus tie and are dropped from the graph; they will end up unassigned).

---

## Stage C. Level-1 Leiden CPM γ sweep

**Script:** `code/community/leiden_cpm_parallel.py`

**What it does.** Spawns N worker processes. Each worker loads `augmented_graph_v2.parquet` into an `igraph.Graph` once, then pulls γ values from a queue and runs Leiden CPM independently for each γ, writing one parquet per γ.

**Hyperparameters**

```python
GAMMAS = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
N_WORKERS = 4  # env: LEIDEN_WORKERS
SEED = 42
```

The Leiden call:

```python
import leidenalg as la
part = la.find_partition(
    g,
    la.CPMVertexPartition,
    weights="weight",
    resolution_parameter=gamma,
    seed=42,
)
```

CPM (Constant Potts Model) was chosen over modularity because:

1. It has a resolution parameter γ with an explicit interpretation (an edge must have weight > γ to "matter"), letting us sweep the granularity.
2. It is **resolution-limit free**: the giant catch-all communities that modularity tends to produce on heterogeneous graphs don't appear here as readily.

**Output (per γ):** `$DATA_DIR/communities_augmented_v2/leiden_cpm_g{γ}.parquet` — `(paper_id int64, community_id int64)`. Plus an aggregated `leiden_cpm_sweep_v2_stats.json` with size distributions.

**Sweep summary** (selected from the 8 γ values; full numbers are in `data/stats/`):

| γ | # comm | max size | 100-999 sized | 10 K+ sized |
|---|---|---|---|---|
| 1e-6 | 85 K | 907 K | 1.5 K | 14 |
| 1e-5 | 76 K | 144 K | 1.6 K | 75 |
| 5e-5 | 71 K | 80 K | 1.6 K | 86 |
| **1e-4** | **73 K** | **53 K** | **1.6 K** | **40** |
| 5e-4 | 89 K | 19 K | 0.8 K | 2 |

**Operating point γ = 1e-4** chosen because:

- Top sizes (10 K-53 K) correspond to recognisable named sub-fields when audited against paper titles (e.g. "Herbig Ae/Be stars", "Ti-alloy plastic deformation", "fault tree analysis in safety engineering").
- It leaves a healthy 1,632-strong 100-999 size bin to feed into the L2 step.
- Singleton rate (51 K of 73 K communities have size 1, but they collectively cover only 1.4 % of papers) is acceptable.

**Runtime.** ~20 minutes per γ on a 22-vCPU box; the 8-γ sweep runs in ~3 hours wall-clock with 4 workers (one γ per worker at a time).

---

## Stage D. Level-2 hierarchical refinement

**Script:** `code/community/hierarchical_leiden.py`

**Inputs**

```
$DATA_DIR/augmented_graph_v2.parquet
$DATA_DIR/communities_augmented_v2/leiden_cpm_g1e-04.parquet   # Level-1 result
```

**Hyperparameters**

```python
MIN_SIZE  = 200       # only refine L1 communities with >=200 papers
L2_GAMMA  = 1e-2      # much higher resolution than L1
N_WORKERS = 6
SEED = 42             # passed into each per-L1 Leiden call
```

**The induced-subgraph trick.** Running γ = 1e-2 globally on the full augmented graph would over-fragment everything because cross-sub-field BC/CC ties have weights far below 1e-2. Inside a single L1 community, however, internal density is high (BC/CC mass concentrates within the sub-field), so γ = 1e-2 finds research-agenda clusters without atomising.

Per L1 community c with `|c| ≥ 200`:

1. **Filter edges**: keep edge (a, b) only if `L1(a) == L1(b) == c`. Vectorised once across all 153 M edges using NumPy `np.isin` over the L1 large-community set — total filter time < 1 minute.
2. **Build a local igraph** with nodes 0..|c|-1 (compact reindex).
3. **Run Leiden CPM** on this subgraph at γ = 1e-2 with `seed=42`.
4. **Compose global L2 IDs**: `L2_id = c × 1_000_000 + local_sub_id`. The factor 10⁶ is more than enough — the largest L1 has 53 K papers, well below 10⁶ sub-IDs.

Steps 2-3 are run in parallel via `multiprocessing.Pool(6)`, tasks sorted by L1 size (largest first) for load balancing.

**Output:** `$DATA_DIR/communities_augmented_v2/hier_l1_1e-04_l2_1e-02.parquet` with schema `(paper_id int64, level1_comm int64, level2_comm int64)`. Plus `hier_stats.json` with the per-L1 split summary.

**Unsplit L1s.** Two cases produce L1 communities that aren't refined:

- L1 size < `MIN_SIZE = 200` (mostly singletons and tiny clusters).
- L1 with no surviving internal edges (rare).

For these, the script writes `level2_comm = L1_id × 1_000_000 + 0`, i.e. one trivial L2 per L1.

**Runtime.** ~4 minutes wall-clock on the 22-vCPU box with 6 workers. 1,896 of the 73 K L1 communities qualify for refinement; together they cover the long tail of papers that L1 alone leaves under-resolved.

---

## Final output and what it gets used for

```
hier_l1_1e-04_l2_1e-02.parquet
├── paper_id      int64
├── level1_comm   int64   # ~73 K sub-field IDs (Stage C)
└── level2_comm   int64   # ~329 K research-agenda IDs (Stage D)
```

- 329 K L2 communities, median non-singleton size 7, max 1,712.
- 46 % of L2 IDs are singletons but they only contain 4.3 % of papers; the bulk of papers sit in 10–999-sized L2 communities.

This file is the ground-truth label source for every retrieval-evaluation table in the paper: `top1_L2`, `#unique_L2`, `top1_L1` are all computed by looking up each retrieved paper's `level1_comm` / `level2_comm` here.

Convenience projections (also shipped in the Zenodo deposit):
- `hier_L1_flat.parquet` — `(paper_id, community_id)` where `community_id = level1_comm`
- `hier_L2_flat.parquet` — same projection for L2

`full_sweep.py` and `compare_methods.py` consume the flat L1/L2 parquets; `citation_rerank.py` and `rrf.py` read the full `hier_l1_1e-04_l2_1e-02.parquet` so they can report both levels per agenda.

## Reproducing this stage from scratch

Given the Zenodo deposit unpacked under `data/` (so `data/augmented_graph_v2.parquet` exists), only Stages C and D are reproducible without the private reference table:

```bash
# Stage C — γ sweep (~3 h on 22 vCPU, 4 workers)
LEIDEN_WORKERS=4 python code/community/leiden_cpm_parallel.py

# Stage D — hierarchical L2 (~4 min)
LEIDEN_WORKERS=6 python code/community/hierarchical_leiden.py
```

Stages A and B can only be re-run if you supply your own `paper_reference` table (e.g. from OpenAlex `works.referenced_works`) under `data/raw/paper_reference/*.parquet`. The Zenodo deposit ships the resulting `bc_edges_full.parquet`, `cc_edges_full.parquet`, and `augmented_graph_v2.parquet` so re-running Stages A/B is not required to get the same L1/L2 labels.
