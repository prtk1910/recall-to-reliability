from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import math
import random
import sqlite3
import statistics

from .util import atomic_write_text, canonical_json


def load_primary_rows(db: sqlite3.Connection, stages: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    db.row_factory = sqlite3.Row
    where = (
        "WHERE r.repeat_index=0 "
        "AND t.task_type != 'tool_argument_grounding'"
    )
    params: list[Any] = []
    if stages:
        where += f" AND r.stage IN ({','.join('?' for _ in stages)})"
        params.extend(stages)
    query = f"""
      SELECT r.*, t.task_type,t.hop_count,t.evidence_turn_ids,t.valid_tool_call,
             w.temporal_distance,w.interference,w.contradictions,
             d.primary_cause,d.oracle_success,d.gold_only_success,d.evidence_selected
      FROM results r JOIN tasks t USING(task_id) JOIN worlds w USING(world_id)
      LEFT JOIN diagnostics d USING(result_id) {where}
    """
    return [dict(row) for row in db.execute(query, params)]


def analyze(db_path: Path, tables_dir: Path, seed: int) -> dict[str, Any]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    included_stages = ("screening",)
    rows = load_primary_rows(db, included_stages)
    expansion_rows = load_primary_rows(db, ("expansion",))
    if not rows:
        raise ValueError("no empirical results available")
    completed = db.execute("SELECT COUNT(*) FROM runs WHERE completed_at IS NULL").fetchone()[0]
    if completed:
        raise ValueError("cannot analyze incomplete runs")

    ingestion = ingestion_costs(db)
    architecture_summary: list[dict[str, Any]] = []
    for architecture, group in grouped(rows, "architecture").items():
        worlds = {row["world_id"] for row in group}
        correct = sum(row["success"] for row in group)
        answer_cost = sum(row["answer_cost_usd"] for row in group)
        ingest_cost = sum(ingestion.get((world, architecture), 0.0) for world in worlds)
        latencies = [row["latency_ms"] for row in group]
        architecture_summary.append({
            "architecture": architecture, "n": len(group), "worlds": len(worlds),
            "accuracy": correct / len(group),
            "accuracy_ci_low": clustered_accuracy_ci(group, seed)[0],
            "accuracy_ci_high": clustered_accuracy_ci(group, seed)[1],
            "answer_cost_usd": answer_cost, "ingestion_cost_usd": ingest_cost,
            "operational_cost_usd": answer_cost + ingest_cost,
            "cost_per_question_usd": (answer_cost + ingest_cost) / len(group),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": percentile(latencies, 0.95),
            "input_tokens": sum(row["input_tokens"] for row in group),
            "cached_tokens": sum(row["cached_tokens"] for row in group),
            "cache_write_tokens": sum(row["cache_write_tokens"] for row in group),
            "output_tokens": sum(row["output_tokens"] for row in group),
            "reasoning_tokens": sum(row["reasoning_tokens"] for row in group),
        })
    architecture_summary.sort(key=lambda row: row["architecture"])
    write_csv(tables_dir / "architecture_summary.csv", architecture_summary)

    architectures = sorted({row["architecture"] for row in rows})
    comparisons = []
    raw_ps = []
    for architecture in architectures:
        if architecture == "recency":
            continue
        comp = paired_cluster_bootstrap(rows, architecture, "recency", seed)
        comparisons.append(comp)
        raw_ps.append(comp["p_value"])
    adjusted = holm_adjust(raw_ps)
    for row, value in zip(comparisons, adjusted):
        row["holm_p_value"] = value
    write_csv(tables_dir / "paired_comparisons.csv", comparisons)

    failures = Counter(row["primary_cause"] for row in rows if not row["success"])
    failure_rows = [{"primary_cause": cause, "count": count,
                     "fraction": count / max(1, sum(failures.values()))}
                    for cause, count in sorted(failures.items())]
    write_csv(tables_dir / "failure_modes.csv", failure_rows)

    tool_rows = [row for row in rows if row["valid_tool_call"]]
    funnel = {
        "tool_questions": len(tool_rows),
        "correct_evidence_retrieval": sum(bool(row["evidence_selected"]) for row in tool_rows),
        "correct_tool_execution": sum(row["success"] for row in tool_rows),
    }
    write_csv(tables_dir / "tool_funnel.csv", [funnel])

    oracle_failures = [row for row in rows if not row["success"] and row["oracle_success"] is not None]
    oracle_rescue = sum(row["oracle_success"] for row in oracle_failures) / max(1, len(oracle_failures))
    non_retrieval = sum(row["primary_cause"] not in ("retrieval_miss", "retrieval_contamination")
                        for row in oracle_failures) / max(1, len(oracle_failures))

    gee = fit_working_independence_gee(rows)
    write_csv(tables_dir / "gee_coefficients.csv", gee)
    stress = stress_curves(rows)
    write_csv(tables_dir / "stress_curves.csv", stress)
    interaction = select_interaction([row for row in rows if row["stage"] == "screening"], seed)
    write_csv(tables_dir / "selected_interaction.csv", [interaction])
    interaction_gee = fit_selected_interaction_gee(expansion_rows, interaction)
    write_csv(tables_dir / "selected_interaction_coefficients.csv", interaction_gee)

    repeats = repeat_variance(db)
    write_csv(tables_dir / "repeat_variance.csv", repeats)
    analysis = {
        "included_stages": list(included_stages),
        "row_count": len(rows), "world_count": len({r["world_id"] for r in rows}),
        "expansion_row_count": len(expansion_rows),
        "architecture_summary": architecture_summary, "paired_comparisons": comparisons,
        "failure_modes": failure_rows, "oracle_rescue_rate": oracle_rescue,
        "non_retrieval_failure_fraction": non_retrieval, "tool_funnel": funnel,
        "selected_interaction": interaction, "gee_coefficients": gee,
        "selected_interaction_coefficients": interaction_gee,
        "repeat_variance": repeats,
    }
    atomic_write_text(tables_dir / "analysis.json", json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    return analysis


def ingestion_costs(db: sqlite3.Connection) -> dict[tuple[str, str], float]:
    costs: dict[tuple[str, str], float] = defaultdict(float)
    worlds = [row[0] for row in db.execute("SELECT world_id FROM worlds")]
    for purpose, cost in db.execute(
            "SELECT purpose,actual_cost_usd FROM api_calls WHERE status='complete' AND "
            "(purpose LIKE 'summary:%' OR purpose LIKE 'fact-extraction:%' OR "
            "purpose LIKE 'embedding:%' OR purpose LIKE 'query-embedding:%')"):
        architecture = ("hierarchical_summary" if purpose.startswith("summary:") else
                        "structured_temporal" if purpose.startswith("fact-extraction:") else "vector_rag")
        matches = [world for world in worlds if world in purpose]
        if matches:
            costs[(max(matches, key=len), architecture)] += float(cost or 0)
    return costs


def grouped(rows: Iterable[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    result: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return result


def clustered_accuracy_ci(rows: list[dict[str, Any]], seed: int,
                          replicates: int = 2_000) -> tuple[float, float]:
    by_world = grouped(rows, "world_id")
    worlds = sorted(by_world)
    rng = random.Random(seed + len(rows))
    values = []
    for _ in range(replicates):
        sample = [rng.choice(worlds) for _ in worlds]
        chosen = [row for world in sample for row in by_world[world]]
        values.append(sum(row["success"] for row in chosen) / len(chosen))
    return percentile(values, 0.025), percentile(values, 0.975)


def paired_cluster_bootstrap(rows: list[dict[str, Any]], left: str, right: str,
                             seed: int, replicates: int = 4_000) -> dict[str, Any]:
    pairs: dict[str, list[float]] = defaultdict(list)
    lookup = {(row["world_id"], row["task_id"], row["architecture"]): row["success"] for row in rows}
    for world, task, architecture in lookup:
        if architecture == left and (world, task, right) in lookup:
            pairs[world].append(lookup[(world, task, left)] - lookup[(world, task, right)])
    worlds = sorted(pairs)
    if not worlds:
        return {"architecture": left, "reference": right, "risk_difference": math.nan,
                "ci_low": math.nan, "ci_high": math.nan, "p_value": math.nan}
    observed = sum(sum(pairs[w]) for w in worlds) / sum(len(pairs[w]) for w in worlds)
    rng = random.Random(seed + sum(map(ord, left)))
    samples = []
    for _ in range(replicates):
        drawn = [rng.choice(worlds) for _ in worlds]
        samples.append(sum(sum(pairs[w]) for w in drawn) / sum(len(pairs[w]) for w in drawn))
    less = sum(value <= 0 for value in samples) / replicates
    greater = sum(value >= 0 for value in samples) / replicates
    return {"architecture": left, "reference": right, "risk_difference": observed,
            "ci_low": percentile(samples, 0.025), "ci_high": percentile(samples, 0.975),
            "p_value": min(1.0, 2 * min(less, greater))}


def holm_adjust(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [math.nan] * len(values)
    running = 0.0
    count = len(values)
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (count - rank) * value))
        result[index] = running
    return result


def stress_curves(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for architecture in sorted({row["architecture"] for row in rows}):
        subset = [row for row in rows if row["architecture"] == architecture]
        for factor in ("temporal_distance", "interference", "contradictions"):
            for level, group in grouped(subset, factor).items():
                output.append({"architecture": architecture, "factor": factor, "level": level,
                               "n": len(group), "accuracy": sum(r["success"] for r in group) / len(group)})
    return output


def repeat_variance(db: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = [dict(row) for row in db.execute(
        "SELECT r.task_id,r.architecture,COUNT(*) n,AVG(r.success) mean_success "
        "FROM results r JOIN tasks t USING(task_id) "
        "WHERE t.task_type != 'tool_argument_grounding' "
        "GROUP BY r.task_id,r.architecture HAVING COUNT(*)>1")]
    for row in rows:
        p = row["mean_success"]
        row["bernoulli_variance"] = p * (1 - p)
    return rows


def select_interaction(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    factors = ("temporal_distance", "interference", "contradictions")
    if not rows:
        return {"factor_a": "temporal_distance", "factor_b": "contradictions",
                "cv_mse_improvement": 0.0, "selection": "default_no_screening_data"}
    pairs = ((factors[0], factors[1]), (factors[0], factors[2]), (factors[1], factors[2]))
    improvements = []
    for left, right in pairs:
        errors_base, errors_interaction = [], []
        worlds = sorted({row["world_id"] for row in rows})
        for held_out in worlds:
            train = [row for row in rows if row["world_id"] != held_out]
            test = [row for row in rows if row["world_id"] == held_out]
            if not train or not test:
                continue
            base_names = regression_feature_names(train, None)
            int_names = regression_feature_names(train, (left, right))
            base_beta = linear_fit([regression_features(r, base_names, None) for r in train],
                                   [r["success"] for r in train])
            int_beta = linear_fit([regression_features(r, int_names, (left, right)) for r in train],
                                  [r["success"] for r in train])
            errors_base.extend((r["success"] - dot(regression_features(r, base_names, None), base_beta)) ** 2
                               for r in test)
            errors_interaction.extend((r["success"] - dot(regression_features(r, int_names, (left, right)), int_beta)) ** 2
                                      for r in test)
        improvement = (statistics.mean(errors_base) - statistics.mean(errors_interaction)
                       if errors_base and errors_interaction else 0.0)
        improvements.append((improvement, left, right))
    improvement, left, right = max(improvements)
    if improvement <= 1e-8:
        return {"factor_a": "temporal_distance", "factor_b": "contradictions",
                "cv_mse_improvement": improvement, "selection": "default_flat_effects"}
    return {"factor_a": left, "factor_b": right, "cv_mse_improvement": improvement,
            "selection": "largest_leave_one_world_out_improvement"}


def regression_feature_names(rows: list[dict[str, Any]], interaction: tuple[str, str] | None) -> list[str]:
    names = ["intercept", "temporal_distance", "interference", "contradictions"]
    names += [f"arch={a}" for a in sorted({r["architecture"] for r in rows})[1:]]
    if interaction:
        names.append(f"interaction={interaction[0]}*{interaction[1]}")
    return names


def level_code(factor: str, value: int) -> float:
    levels = {"temporal_distance": (10, 100, 1000), "interference": (10, 100, 1000),
              "contradictions": (0, 1, 3)}[factor]
    return float(levels.index(value) - 1)


def regression_features(row: dict[str, Any], names: list[str],
                        interaction: tuple[str, str] | None) -> list[float]:
    result = []
    for name in names:
        if name == "intercept": result.append(1.0)
        elif name.startswith("arch="): result.append(float(row["architecture"] == name[5:]))
        elif name.startswith("interaction="):
            assert interaction
            result.append(level_code(interaction[0], row[interaction[0]]) *
                          level_code(interaction[1], row[interaction[1]]))
        else: result.append(level_code(name, row[name]))
    return result


def fit_working_independence_gee(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Logistic GEE with working independence and world-clustered sandwich covariance.
    architectures = sorted({r["architecture"] for r in rows})
    tasks = sorted({r["task_type"] for r in rows})
    names = ["intercept"]
    names += [f"arch={a}" for a in architectures[1:]]
    names += ["distance", "interference", "contradictions"]
    names += [f"arch={a}*{f}" for a in architectures[1:] for f in ("distance", "interference", "contradictions")]
    names += [f"task={t}" for t in tasks[1:]] + ["hop_count"]

    def vector(row: dict[str, Any]) -> list[float]:
        d = level_code("temporal_distance", row["temporal_distance"])
        i = level_code("interference", row["interference"])
        c = level_code("contradictions", row["contradictions"])
        base = [1.0] + [float(row["architecture"] == a) for a in architectures[1:]] + [d, i, c]
        base += [float(row["architecture"] == a) * value for a in architectures[1:] for value in (d, i, c)]
        base += [float(row["task_type"] == task) for task in tasks[1:]] + [float(row["hop_count"])]
        return base
    return clustered_logit(rows, names, vector)


def fit_selected_interaction_gee(rows: list[dict[str, Any]],
                                 selected: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows:
        return []
    factor_a, factor_b = selected["factor_a"], selected["factor_b"]
    architectures = sorted({r["architecture"] for r in rows})
    tasks = sorted({r["task_type"] for r in rows})
    names = ["intercept"] + [f"arch={a}" for a in architectures[1:]]
    names += [factor_a, factor_b, f"{factor_a}*{factor_b}"]
    names += [f"task={task}" for task in tasks[1:]] + ["hop_count"]

    def vector(row: dict[str, Any]) -> list[float]:
        left = level_code(factor_a, row[factor_a])
        right = level_code(factor_b, row[factor_b])
        return ([1.0] + [float(row["architecture"] == a) for a in architectures[1:]]
                + [left, right, left * right]
                + [float(row["task_type"] == task) for task in tasks[1:]]
                + [float(row["hop_count"])])
    return clustered_logit(rows, names, vector)


def clustered_logit(rows: list[dict[str, Any]], names: list[str],
                     vector: Any) -> list[dict[str, Any]]:
    x = [vector(row) for row in rows]
    y = [float(row["success"]) for row in rows]
    beta = [0.0] * len(names)
    for _ in range(40):
        eta = [max(-20.0, min(20.0, dot(row, beta))) for row in x]
        mu = [1 / (1 + math.exp(-value)) for value in eta]
        weights = [max(1e-6, p * (1 - p)) for p in mu]
        z = [e + (target - p) / w for e, target, p, w in zip(eta, y, mu, weights)]
        xtwx = crossprod(x, weights, x, ridge=1e-5)
        xtwz = [sum(row[j] * w * value for row, w, value in zip(x, weights, z)) for j in range(len(names))]
        new = solve(xtwx, xtwz)
        if max(abs(a - b) for a, b in zip(beta, new)) < 1e-7:
            beta = new
            break
        beta = new
    mu = [1 / (1 + math.exp(-max(-20, min(20, dot(row, beta))))) for row in x]
    weights = [max(1e-6, p * (1 - p)) for p in mu]
    bread = inverse(crossprod(x, weights, x, ridge=1e-5))
    scores: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(names))
    for row, features, target, predicted in zip(rows, x, y, mu):
        for j, value in enumerate(features):
            scores[row["world_id"]][j] += value * (target - predicted)
    meat = [[0.0] * len(names) for _ in names]
    for score in scores.values():
        for i in range(len(names)):
            for j in range(len(names)):
                meat[i][j] += score[i] * score[j]
    covariance = matmul(matmul(bread, meat), bread)
    output = []
    for index, name in enumerate(names):
        se = math.sqrt(max(0.0, covariance[index][index]))
        z_value = beta[index] / se if se else math.nan
        p_value = math.erfc(abs(z_value) / math.sqrt(2)) if math.isfinite(z_value) else math.nan
        output.append({"term": name, "coefficient": beta[index], "cluster_robust_se": se,
                       "z": z_value, "p_value": p_value,
                       "odds_ratio": math.exp(max(-20, min(20, beta[index])))})
    return output


def linear_fit(x: list[list[float]], y: list[float]) -> list[float]:
    if not x:
        return []
    return solve(crossprod(x, [1.0] * len(x), x, ridge=1e-5),
                 [sum(row[j] * value for row, value in zip(x, y)) for j in range(len(x[0]))])


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def crossprod(left: list[list[float]], weights: list[float], right: list[list[float]],
              ridge: float = 0.0) -> list[list[float]]:
    p, q = len(left[0]), len(right[0])
    result = [[sum(l[i] * w * r[j] for l, w, r in zip(left, weights, right))
               for j in range(q)] for i in range(p)]
    for i in range(min(p, q)):
        result[i][i] += ridge
    return result


def solve(matrix: list[list[float]], values: list[float]) -> list[float]:
    inv = inverse(matrix)
    return [sum(inv[i][j] * values[j] for j in range(len(values))) for i in range(len(values))]


def inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    augmented = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] = 1e-12
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [value - factor * pivot_value
                              for value, pivot_value in zip(augmented[row], augmented[column])]
    return [row[n:] for row in augmented]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    fields = list(rows[0])
    lines = []
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
