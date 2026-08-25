from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sqlite3
from collections import defaultdict
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")

    x = (len(values) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    frac = x - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode()).hexdigest()[:8]
    return base_seed + int(digest, 16)


def clustered_difference(
    rows,
    luna_field,
    ox_field,
    label,
    bootstrap_reps,
    base_seed,
):
    """
    Paired risk difference with WORLD as the resampling cluster.

    Each bootstrap replicate samples complete worlds with replacement and
    recomputes pooled Luna and Ox accuracy over the sampled clusters.
    """

    clusters = defaultdict(lambda: [0.0, 0.0, 0])

    luna_total = 0.0
    ox_total = 0.0
    n_total = 0

    for row in rows:
        l = float(row[luna_field])
        o = float(row[ox_field])

        clusters[row["world_id"]][0] += l
        clusters[row["world_id"]][1] += o
        clusters[row["world_id"]][2] += 1

        luna_total += l
        ox_total += o
        n_total += 1

    worlds = sorted(clusters)

    luna_acc = 100.0 * luna_total / n_total
    ox_acc = 100.0 * ox_total / n_total
    point = ox_acc - luna_acc

    rng = random.Random(stable_seed(base_seed, label))
    boot = []

    for _ in range(bootstrap_reps):
        bl = 0.0
        bo = 0.0
        bn = 0

        for _ in range(len(worlds)):
            world = worlds[rng.randrange(len(worlds))]
            l, o, n = clusters[world]

            bl += l
            bo += o
            bn += n

        boot.append(100.0 * (bo - bl) / bn)

    low = percentile(boot, 0.025)
    high = percentile(boot, 0.975)

    return {
        "label": label,
        "n": n_total,
        "worlds": len(worlds),
        "luna": luna_acc,
        "ox": ox_acc,
        "delta": point,
        "ci_low": low,
        "ci_high": high,
    }


def clustered_format_gap(rows, label, bootstrap_reps, base_seed):
    """
    Difference in semantic-vs-exact rescue gain:

        (Ox semantic - Ox exact)
          -
        (Luna semantic - Luna exact)

    Positive means Ox is penalized more heavily by exact/protocol scoring.
    """

    clusters = defaultdict(lambda: [0.0, 0.0, 0])

    luna_total = 0.0
    ox_total = 0.0
    n_total = 0

    for row in rows:
        luna_gain = (
            float(row["luna_semantic_v3"])
            - float(row["luna_exact"])
        )
        ox_gain = (
            float(row["ox_semantic_v3"])
            - float(row["ox_exact"])
        )

        clusters[row["world_id"]][0] += luna_gain
        clusters[row["world_id"]][1] += ox_gain
        clusters[row["world_id"]][2] += 1

        luna_total += luna_gain
        ox_total += ox_gain
        n_total += 1

    worlds = sorted(clusters)

    luna_gain_pp = 100.0 * luna_total / n_total
    ox_gain_pp = 100.0 * ox_total / n_total
    point = ox_gain_pp - luna_gain_pp

    rng = random.Random(stable_seed(base_seed, label))
    boot = []

    for _ in range(bootstrap_reps):
        bl = 0.0
        bo = 0.0
        bn = 0

        for _ in range(len(worlds)):
            world = worlds[rng.randrange(len(worlds))]
            l, o, n = clusters[world]

            bl += l
            bo += o
            bn += n

        boot.append(100.0 * (bo - bl) / bn)

    return {
        "label": label,
        "n": n_total,
        "worlds": len(worlds),
        "luna": luna_gain_pp,
        "ox": ox_gain_pp,
        "delta": point,
        "ci_low": percentile(boot, 0.025),
        "ci_high": percentile(boot, 0.975),
    }


def show(result):
    print(
        f"{result['label']:<38} "
        f"n={result['n']:<4} "
        f"Luna={result['luna']:6.2f}  "
        f"Ox={result['ox']:6.2f}  "
        f"Δ={result['delta']:+6.2f} pp  "
        f"95% CI [{result['ci_low']:+6.2f}, "
        f"{result['ci_high']:+6.2f}]"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default="artifacts/ox_zen_replay/semantic_rescore_final.sqlite",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260824,
    )
    parser.add_argument(
        "--csv",
        default="artifacts/ox_zen_replay/"
                "reader_bootstrap_summary.csv",
    )

    args = parser.parse_args()

    db_path = Path(args.db)

    if not db_path.exists():
        raise SystemExit(f"Missing database: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT *
        FROM semantic_results_v3
        ORDER BY world_id, task_id, architecture
    """).fetchall()

    # Exclude invalid benchmark task: its required tool name was
    # never supplied in the original memory evidence.
    rows = [
        r for r in rows
        if r["task_type"] != "tool_argument_grounding"
    ]

    worlds = sorted({r["world_id"] for r in rows})

    print()
    print("FIXED-CONTEXT READER REPLICATION")
    print("--------------------------------")
    print(f"Valid paired rows: {len(rows)}")
    print(f"Worlds: {len(worlds)}")
    print(f"Cluster bootstrap replicates: {args.bootstrap_reps}")

    results = []

    #
    # Primary analysis:
    # memory-bearing architectures only.
    #
    memory_rows = [
        r for r in rows
        if r["architecture"] != "query_only"
    ]

    print("\nPRIMARY: SEMANTIC ACCURACY, MEMORY-BEARING ARCHITECTURES")

    x = clustered_difference(
        memory_rows,
        "luna_semantic_v3",
        "ox_semantic_v3",
        "semantic_memory_only",
        args.bootstrap_reps,
        args.seed,
    )
    results.append(x)
    show(x)

    #
    # All architectures including query-only baseline.
    #
    print("\nSECONDARY: SEMANTIC ACCURACY, ALL ARCHITECTURES")

    x = clustered_difference(
        rows,
        "luna_semantic_v3",
        "ox_semantic_v3",
        "semantic_all",
        args.bootstrap_reps,
        args.seed,
    )
    results.append(x)
    show(x)

    #
    # Original exact/protocol score.
    #
    print("\nEXACT / PROTOCOL SCORE")

    x = clustered_difference(
        rows,
        "luna_exact",
        "ox_exact",
        "exact_all",
        args.bootstrap_reps,
        args.seed,
    )
    results.append(x)
    show(x)

    #
    # How much more exact-match scoring penalizes Ox.
    #
    print("\nSEMANTIC RESCUE GAIN")

    x = clustered_format_gap(
        rows,
        "semantic_minus_exact_gain",
        args.bootstrap_reps,
        args.seed,
    )
    results.append(x)
    show(x)

    #
    # Architecture-level paired effects.
    #
    print("\nSEMANTIC ACCURACY BY ARCHITECTURE")

    architectures = sorted({
        r["architecture"] for r in rows
    })

    for architecture in architectures:
        subset = [
            r for r in rows
            if r["architecture"] == architecture
        ]

        x = clustered_difference(
            subset,
            "luna_semantic_v3",
            "ox_semantic_v3",
            f"architecture:{architecture}",
            args.bootstrap_reps,
            args.seed,
        )

        results.append(x)
        show(x)

    #
    # Task-level effects among architectures that actually receive memory.
    #
    print("\nSEMANTIC ACCURACY BY TASK — MEMORY-BEARING ONLY")

    task_types = sorted({
        r["task_type"] for r in memory_rows
    })

    for task_type in task_types:
        subset = [
            r for r in memory_rows
            if r["task_type"] == task_type
        ]

        x = clustered_difference(
            subset,
            "luna_semantic_v3",
            "ox_semantic_v3",
            f"task:{task_type}",
            args.bootstrap_reps,
            args.seed,
        )

        results.append(x)
        show(x)

    #
    # Save publication-analysis summary.
    #
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "n",
                "worlds",
                "luna",
                "ox",
                "delta",
                "ci_low",
                "ci_high",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved: {csv_path}")

    conn.close()


if __name__ == "__main__":
    main()
