#!/usr/bin/env python3
"""Compute Bibliographic Coupling (BC) and Co-citation (CC) edges
from a full paper-reference table using DuckDB (out-of-core, disk-spill).

Inputs (set via env or edit DATA_DIR):
  - $DATA_DIR/raw/paper_reference/*.parquet   (full ~150M-paper reference table,
                                                schema: paper_id string, paper_reference_id string)
  - $DATA_DIR/target/neurips_4m.parquet       (target corpus, paper_id column)

Outputs:
  - $DATA_DIR/bc_edges_full.parquet  (a, b BIGINT, shared INT, cos FLOAT)
  - $DATA_DIR/cc_edges_full.parquet  (a, b BIGINT, shared INT, cos FLOAT)

BC edge: pair of target papers sharing >= SHARED_MIN external/internal references
CC edge: pair of target papers cited together by >= SHARED_MIN common citers
Salton cosine: shared / sqrt(|refs(A)| * |refs(B)|)
Caps: drop refs cited by >BC_MAX_CITERS targets / citers citing >CC_MAX_REFS targets
"""
import os
import time
from pathlib import Path
import duckdb

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
PR_GLOB = str(DATA_DIR / "raw" / "paper_reference" / "*.parquet")
TARGET_PATH = str(DATA_DIR / "target" / "neurips_4m.parquet")
OUT_BC = str(DATA_DIR / "bc_edges_full.parquet")
OUT_CC = str(DATA_DIR / "cc_edges_full.parquet")

MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "80GB")
TMP_DIR = os.environ.get("DUCKDB_TMP_DIR", "./duckdb_tmp")
THREADS = int(os.environ.get("DUCKDB_THREADS", "16"))

BC_MAX_CITERS = 500
CC_MAX_REFS = 200
SHARED_MIN = 3


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    os.makedirs(TMP_DIR, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET temp_directory='{TMP_DIR}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("PRAGMA enable_progress_bar=false")
    log(f"DuckDB ready: mem={MEMORY_LIMIT} threads={THREADS} tmp={TMP_DIR}")

    log("Loading target paper ids")
    con.execute(f"""
        CREATE TEMP TABLE target_ids AS
        SELECT DISTINCT paper_id FROM read_parquet('{TARGET_PATH}')
    """)
    n_target = con.execute("SELECT COUNT(*) FROM target_ids").fetchone()[0]
    log(f"  target ids: {n_target:,}")

    log("Building target_refs (filter paper_reference by target paper_id)")
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE target_refs AS
        SELECT CAST(pr.paper_id AS BIGINT) AS pid,
               CAST(pr.paper_reference_id AS BIGINT) AS rid
        FROM read_parquet('{PR_GLOB}') pr
        SEMI JOIN target_ids ti ON pr.paper_id = ti.paper_id
    """)
    n_tref = con.execute("SELECT COUNT(*) FROM target_refs").fetchone()[0]
    log(f"  target_refs rows: {n_tref:,}  ({time.time()-t0:.1f}s)")

    log("Compute |refs(paper)| (out-degree)")
    con.execute("""
        CREATE TEMP TABLE tref_outdeg AS
        SELECT pid, COUNT(*) AS n FROM target_refs GROUP BY pid
    """)

    log(f"Drop hot refs cited by >{BC_MAX_CITERS} target papers")
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE hot_refs AS
        SELECT rid FROM (
            SELECT rid, COUNT(*) AS n FROM target_refs GROUP BY rid
        ) WHERE n > {BC_MAX_CITERS}
    """)
    n_hot = con.execute("SELECT COUNT(*) FROM hot_refs").fetchone()[0]
    log(f"  hot refs: {n_hot:,}  ({time.time()-t0:.1f}s)")

    con.execute("""
        CREATE TEMP TABLE target_refs_f AS
        SELECT pid, rid FROM target_refs
        ANTI JOIN hot_refs ON target_refs.rid = hot_refs.rid
    """)
    n_refs_f = con.execute("SELECT COUNT(*) FROM target_refs_f").fetchone()[0]
    log(f"  target_refs_f rows (after filter): {n_refs_f:,}")
    con.execute("DROP TABLE target_refs")

    log(f"\nCompute BC pairs (shared >= {SHARED_MIN}) — Salton cosine")
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT p1 AS a, p2 AS b, shared,
                   CAST(shared / SQRT(CAST(c1.n AS DOUBLE) * c2.n) AS FLOAT) AS cos
            FROM (
                SELECT LEAST(a.pid, b.pid) AS p1,
                       GREATEST(a.pid, b.pid) AS p2,
                       COUNT(*) AS shared
                FROM target_refs_f a
                JOIN target_refs_f b ON a.rid = b.rid AND a.pid < b.pid
                GROUP BY 1, 2
                HAVING COUNT(*) >= {SHARED_MIN}
            ) raw
            JOIN tref_outdeg c1 ON c1.pid = p1
            JOIN tref_outdeg c2 ON c2.pid = p2
        ) TO '{OUT_BC}' (FORMAT 'parquet', COMPRESSION 'zstd')
    """)
    log(f"  wrote {OUT_BC}  ({time.time()-t0:.1f}s)")

    con.execute("DROP TABLE target_refs_f")
    con.execute("DROP TABLE hot_refs")
    con.execute("DROP TABLE tref_outdeg")

    log("\nBuild target_citers (filter paper_reference by target paper_reference_id)")
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE target_citers AS
        SELECT CAST(pr.paper_id AS BIGINT) AS cid,
               CAST(pr.paper_reference_id AS BIGINT) AS tid
        FROM read_parquet('{PR_GLOB}') pr
        SEMI JOIN target_ids ti ON pr.paper_reference_id = ti.paper_id
    """)
    n_tc = con.execute("SELECT COUNT(*) FROM target_citers").fetchone()[0]
    log(f"  target_citers rows: {n_tc:,}  ({time.time()-t0:.1f}s)")

    log("Compute |citers(target)| (in-degree)")
    con.execute("""
        CREATE TEMP TABLE tgt_indeg AS
        SELECT tid AS pid, COUNT(*) AS n FROM target_citers GROUP BY tid
    """)

    log(f"Drop survey-like citers (cite >{CC_MAX_REFS} target papers)")
    con.execute(f"""
        CREATE TEMP TABLE survey_citers AS
        SELECT cid FROM (
            SELECT cid, COUNT(*) AS n FROM target_citers GROUP BY cid
        ) WHERE n > {CC_MAX_REFS}
    """)
    con.execute("""
        CREATE TEMP TABLE target_citers_f AS
        SELECT cid, tid FROM target_citers
        ANTI JOIN survey_citers ON target_citers.cid = survey_citers.cid
    """)
    con.execute("DROP TABLE target_citers")
    con.execute("DROP TABLE survey_citers")

    log(f"\nCompute CC pairs (shared >= {SHARED_MIN})")
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT t1 AS a, t2 AS b, shared,
                   CAST(shared / SQRT(CAST(c1.n AS DOUBLE) * c2.n) AS FLOAT) AS cos
            FROM (
                SELECT LEAST(a.tid, b.tid) AS t1,
                       GREATEST(a.tid, b.tid) AS t2,
                       COUNT(*) AS shared
                FROM target_citers_f a
                JOIN target_citers_f b ON a.cid = b.cid AND a.tid < b.tid
                GROUP BY 1, 2
                HAVING COUNT(*) >= {SHARED_MIN}
            ) raw
            JOIN tgt_indeg c1 ON c1.pid = t1
            JOIN tgt_indeg c2 ON c2.pid = t2
        ) TO '{OUT_CC}' (FORMAT 'parquet', COMPRESSION 'zstd')
    """)
    log(f"  wrote {OUT_CC}  ({time.time()-t0:.1f}s)")
    log("Done.")


if __name__ == "__main__":
    main()
