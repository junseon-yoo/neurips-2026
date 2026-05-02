#!/usr/bin/env python3
"""TF-IDF-weighted L2-pair lexical divergence per domain.

For each L1 sub-field with ≥2 valid L2 agendas, compute pairwise:
    - Raw unigram JSD between L2s' abstract token distributions
    - TF-IDF-weighted JSD (downweights vocabulary common across same-L1 L2s)
    - Top-K TF-IDF Jaccard distance
    - Discriminative-word fraction (% of tokens unique to a single L2 in L1)
Average per domain.

Inputs:
    $DATA_DIR/target/neurips_4m.parquet
    $DATA_DIR/communities/hier_l1_1e-04_l2_1e-02.parquet
Output:
    $DATA_DIR/analysis/lexical_divergence.json
"""
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

MIN_L2_SIZE = 20
MAX_L2_PER_L1 = 30
MAX_PAIRS_PER_L1 = 200
TOP_K = 50
SEED = 42

DOMAINS = ["biology", "biomedical", "chemistry", "computer_science",
           "engineering", "environmental_earth", "materials_science", "physics"]
STOPWORDS = set("""a about above after again against all am an and any are aren as at be because been before being
below between both but by could did do does doing don down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own re s same she should so some such t than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves
abstract paper study research method approach result finding novel propose proposed using used analysis based show shown
demonstrate present new however these those one two three may can has have are is was were been being thus moreover
furthermore therefore here we our this study results show shown demonstrate suggest indicate found observed obtained
provided provide ie eg al et fig figure table""".split())

TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) >= 3 and t not in STOPWORDS]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def jsd_dict(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    P = np.array([p.get(k, 0) for k in keys], dtype=np.float64) + 1e-12
    Q = np.array([q.get(k, 0) for k in keys], dtype=np.float64) + 1e-12
    P /= P.sum(); Q /= Q.sum()
    M = 0.5 * (P + Q)
    return float(0.5 * (np.sum(P * np.log2(P / M)) + np.sum(Q * np.log2(Q / M))))


def main() -> None:
    random.seed(SEED)

    log("Loading target")
    t = pq.read_table(str(DATA_DIR / "target" / "neurips_4m.parquet"),
                      columns=["paper_id", "domain", "title", "abstract"])
    pids = t.column("paper_id").to_numpy().astype(np.int64)
    doms = t.column("domain").to_numpy()
    titles = t.column("title").to_pylist()
    abstracts = t.column("abstract").to_pylist()
    text_of, dom_of = {}, {}
    for i, p in enumerate(pids.tolist()):
        text_of[int(p)] = (titles[i] or "") + " " + (abstracts[i] or "")
        dom_of[int(p)] = str(doms[i])

    log("Loading hier")
    h = pq.read_table(str(DATA_DIR / "communities" / "hier_l1_1e-04_l2_1e-02.parquet"))
    hp = h.column("paper_id").to_numpy().astype(np.int64)
    hl1 = h.column("level1_comm").to_numpy().astype(np.int64)
    hl2 = h.column("level2_comm").to_numpy().astype(np.int64)
    l1_of, l2_of = {}, {}
    for p, a, b in zip(hp.tolist(), hl1.tolist(), hl2.tolist()):
        l1_of[int(p)] = int(a)
        l2_of[int(p)] = int(b)

    l1_l2_papers = defaultdict(lambda: defaultdict(list))
    for p, l1 in l1_of.items():
        if p in l2_of:
            l1_l2_papers[l1][l2_of[p]].append(p)

    def l1_domain(l1: int) -> tuple[str | None, float]:
        cnt: Counter = Counter()
        for _, papers in l1_l2_papers[l1].items():
            for p in papers:
                d = dom_of.get(p)
                if d:
                    cnt[d] += 1
        if not cnt:
            return None, 0.0
        top = cnt.most_common(1)[0]
        return top[0], top[1] / sum(cnt.values())

    valid: dict[str, list[tuple[int, list[tuple[int, list[int]]]]]] = defaultdict(list)
    for l1, l2dict in l1_l2_papers.items():
        valid_l2 = [(l2, papers) for l2, papers in l2dict.items() if len(papers) >= MIN_L2_SIZE]
        if len(valid_l2) < 2:
            continue
        dom, share = l1_domain(l1)
        if dom not in DOMAINS or share < 0.8:
            continue
        if len(valid_l2) > MAX_L2_PER_L1:
            valid_l2 = random.sample(valid_l2, MAX_L2_PER_L1)
        valid[dom].append((l1, valid_l2))

    log("Tokenizing per L2")
    l2_tokens: dict[int, Counter] = {}
    for dom, l1_list in valid.items():
        for _, l2s in l1_list:
            for l2, papers in l2s:
                if l2 in l2_tokens:
                    continue
                cnt: Counter = Counter()
                for p in papers:
                    cnt.update(tokenize(text_of.get(p, "")))
                l2_tokens[l2] = cnt
    log(f"  tokenized {len(l2_tokens)} L2 communities")

    log("Computing TF-IDF JSD + Jaccard + discriminative fraction")
    domain_results = {}
    for dom, l1_list in valid.items():
        raw_jsds, tfidf_jsds, jaccards, discr_fracs = [], [], [], []
        for l1, l2s in l1_list:
            ids = [l2 for l2, _ in l2s]
            if len(ids) < 2:
                continue
            df: Counter = Counter()
            for l2 in ids:
                for w in l2_tokens[l2]:
                    df[w] += 1
            n_l2 = len(ids)
            idf = {w: math.log(n_l2 / d) for w, d in df.items()}
            tfidf, topk_terms = {}, {}
            for l2 in ids:
                cnt = l2_tokens[l2]
                total = sum(cnt.values())
                if total == 0:
                    tfidf[l2] = {}
                    topk_terms[l2] = set()
                    continue
                ti = {w: (c / total) * idf[w] for w, c in cnt.items()}
                tfidf[l2] = ti
                topk_terms[l2] = set(w for w, _ in sorted(ti.items(), key=lambda x: -x[1])[:TOP_K])

            for l2 in ids:
                cnt = l2_tokens[l2]
                total = sum(cnt.values())
                if total == 0:
                    continue
                u_count = sum(c for w, c in cnt.items() if df[w] == 1)
                discr_fracs.append(u_count / total)

            pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
            if len(pairs) > MAX_PAIRS_PER_L1:
                pairs = random.sample(pairs, MAX_PAIRS_PER_L1)
            for a, b in pairs:
                ca, cb = l2_tokens[a], l2_tokens[b]
                sa, sb = sum(ca.values()), sum(cb.values())
                if sa == 0 or sb == 0:
                    continue
                pa = {w: c / sa for w, c in ca.items()}
                pb = {w: c / sb for w, c in cb.items()}
                raw_jsds.append(jsd_dict(pa, pb))
                tfidf_jsds.append(jsd_dict(tfidf[a], tfidf[b]))
                ta, tb = topk_terms[a], topk_terms[b]
                inter = len(ta & tb)
                union = len(ta | tb)
                jaccards.append(1 - inter / union if union else 0)

        domain_results[dom] = {
            "n_l1_evaluated": len(l1_list),
            "mean_raw_jsd": float(np.mean(raw_jsds)) if raw_jsds else 0,
            "mean_tfidf_jsd": float(np.mean(tfidf_jsds)) if tfidf_jsds else 0,
            "mean_tfidf_topk_jaccard_dist": float(np.mean(jaccards)) if jaccards else 0,
            "mean_discriminative_fraction": float(np.mean(discr_fracs)) if discr_fracs else 0,
        }

    out = DATA_DIR / "analysis" / "lexical_divergence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(domain_results, f, indent=2)
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
