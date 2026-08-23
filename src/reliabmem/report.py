from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import html
import json
import math
import sqlite3

from .analysis import load_primary_rows
from .config import ExperimentConfig
from .util import atomic_write_text, code_revision, sha256_file


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2")


def generate_report(root: Path, db_path: Path, analysis: dict[str, Any],
                    config: ExperimentConfig, run_id: str) -> None:
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    validate_report_inputs(db, analysis, config)
    figures = root / "artifacts" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = load_primary_rows(db, tuple(analysis["included_stages"]))
    expansion_rows = load_primary_rows(db, ("expansion",))
    create_stress_figure(figures / "01_stress_curves.svg", rows)
    create_interaction_figure(figures / "02_interaction_heatmaps.svg", expansion_rows,
                              analysis["selected_interaction"])
    create_failure_figure(figures / "03_failure_modes.svg", analysis)
    create_funnel_figure(figures / "04_retrieval_tool_funnel.svg", analysis["tool_funnel"])
    create_pareto_figure(figures / "05_accuracy_cost_pareto.svg", analysis["architecture_summary"])
    create_token_figure(figures / "06_token_cost_breakdown.svg", analysis["architecture_summary"])

    actual_spend = db.execute(
        "SELECT COALESCE(SUM(actual_cost_usd),0) FROM api_calls WHERE status='complete'"
    ).fetchone()[0]
    operational = sum(row["operational_cost_usd"] for row in analysis["architecture_summary"])
    diagnostic = db.execute(
        "SELECT COALESCE(SUM(actual_cost_usd),0) FROM api_calls WHERE status='complete' "
        "AND (purpose LIKE 'oracle:%' OR purpose LIKE 'gold-only:%')"
    ).fetchone()[0]
    compatibility_purposes = [
        row[0] for row in db.execute(
            "SELECT DISTINCT purpose FROM api_calls WHERE status='complete' AND purpose LIKE '%:compat:%'"
        )
    ]
    omitted_temperature = any('no-temperature' in purpose for purpose in compatibility_purposes)
    omitted_reasoning_context = any(
        'no-reasoning-context' in purpose for purpose in compatibility_purposes
    )
    temperature_method = (
        'provider-default temperature after the API explicitly rejected temperature=0'
        if omitted_temperature else 'temperature zero'
    )
    reasoning_context_method = (
        'stateless requests without the rejected reasoning.context field'
        if omitted_reasoning_context else 'reasoning.context=current_turn'
    )
    best = max(analysis["architecture_summary"], key=lambda row: row["accuracy"])
    failures = sum(row["count"] for row in analysis["failure_modes"])
    interaction = analysis["selected_interaction"]
    decisions_path = root / "artifacts" / "stage_decisions.json"
    decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    variance_note = "executed" if decisions.get("variance_audit") else "omitted"
    terra_note = "executed" if decisions.get("terra_check") else "omitted"
    terra_reason = decisions.get("terra_reason", "not scheduled in this run")
    manifest = build_manifest(root)
    manifest_runs = [row[0] for row in db.execute("SELECT run_id FROM runs WHERE completed_at IS NOT NULL")]
    manifest_runs.append(run_id)
    with db:
        for manifest_run in manifest_runs:
            for path, checksum in manifest.items():
                db.execute("INSERT OR REPLACE INTO manifests VALUES(?,?,?)",
                           (manifest_run, path, checksum))

    comparisons = "\n".join(
        f"| {r['architecture']} vs {r['reference']} | {pct(r['risk_difference'])} | "
        f"[{pct(r['ci_low'])}, {pct(r['ci_high'])}] | {r['holm_p_value']:.4f} |"
        for r in analysis["paired_comparisons"]
    )
    summaries = "\n".join(
        f"| {r['architecture']} | {pct(r['accuracy'])} [{pct(r['accuracy_ci_low'])}, {pct(r['accuracy_ci_high'])}] | "
        f"${r['cost_per_question_usd']:.6f} | {r['median_latency_ms']:.0f} | {r['p95_latency_ms']:.0f} |"
        for r in analysis["architecture_summary"]
    )
    failure_table = "\n".join(
        f"| {r['primary_cause']} | {r['count']} | {pct(r['fraction'])} |"
        for r in analysis["failure_modes"]
    ) or "| None observed | 0 | 0.0% |"
    hypotheses = hypothesis_text(rows, analysis)
    readme = f"""# From Recall to Reliability

## Abstract

This reproducible study evaluates six memory conditions on {analysis['row_count']:,} model decisions nested in {analysis['world_count']:,} deterministic synthetic worlds. It maps temporal distance, semantic interference, and superseding contradictions; intervenes on retrieval with source-turn oracles; and accounts for ingestion, retrieval, answering, latency, and cache use. The best observed architecture was **{best['architecture']}** at {pct(best['accuracy'])} accuracy (world-clustered 95% CI {pct(best['accuracy_ci_low'])}–{pct(best['accuracy_ci_high'])}). These findings describe this model, benchmark, and stress design only.

## Research questions and hypotheses

The study asks when long-term assistant memory stops being reliable, whether failures occur outside retrieval, how selective architectures compare with uncompressed context, whether degradation is nonlinear, and how retrieved evidence translates into valid local tool execution.

{hypotheses}

## Relation to prior work

This is a controlled extension of [MemFail](https://arxiv.org/abs/2605.26667), [MemOps](https://arxiv.org/abs/2607.12893), and [Mem2ActBench](https://arxiv.org/abs/2601.19935). The contribution is the joint stress-regime map, causal source-turn interventions, and cost–accuracy analysis; it is not a claim to the first memory-failure taxonomy. [LongMemEval](https://arxiv.org/abs/2410.10813) is treated as an optional secondary sanity check because public questions may overlap training and lack this benchmark's internal causal traces.

## Test bed

Conversations are generated from latent world state without an LLM. Each world has exact factor levels for final-evidence distance (10/100/1,000 turns), semantically similar interference (10/100/1,000 facts), and contradictions (0/1/3 updates). Its twelve tasks cover atomic, current, historical, coexisting, conditional, ordered, two-hop, four-hop, forgetting, abstention, tool-argument, and tool-selection behavior. Validators enforce source lineage, stress counts, no answer leakage, consistent transitions, and a 200,000-token ceiling.

All selective memories receive an 8,000-token context budget. Full context is the uncompressed ceiling; recency packs newest turns; vector RAG embeds bounded chunks and chronologically orders cosine-selected chunks; hierarchical memory summarizes ten-turn leaves with fan-out ten; structured temporal memory extracts versioned facts into SQLite; query-only establishes Memory Gain. Backends expose only `observe`, `build_context`, `snapshot`, and `reset`; no gold fact enters a memory operation.

The primary requested model was `{config.primary_model}` with low reasoning effort, {temperature_method}, `store=false`, and {reasoning_context_method}. Embeddings used `{config.embedding_model}`. Compatibility fields were removed only after a parameter-specific HTTP 400 and are visible in ledger purpose suffixes. Every request was pre-reserved against a ${config.spend_cap_usd:.2f} cap and keyed by its canonical payload and purpose for interrupted-run recovery. Returned and requested models, response IDs, retry counts, latency, input/cache-write/cached/output/reasoning tokens, hashes, and timestamps are retained in SQLite. Prices are the pinned snapshot recorded in configuration; billed-token fields are taken from API responses.

## Experimental chronology

The runner executed a schema/cost pilot, the fixed L9 screening core, and a fresh complete 3×3 expansion slice for **{interaction['factor_a']} × {interaction['factor_b']}**. The repeat audit was {variance_note}; Terra confirmation was {terra_note} ({terra_reason}). The interaction was selected by leave-one-world-out predictive improvement (`{interaction['selection']}`, ΔMSE {interaction['cv_mse_improvement']:.6f}). Runs are marked complete transactionally before analysis; report generation refuses incomplete runs.

## Results

| Architecture | Accuracy (world-clustered 95% CI) | Operational cost/question | Median latency ms | P95 latency ms |
|---|---:|---:|---:|---:|
{summaries}

![Three-panel memory stress curves](artifacts/figures/01_stress_curves.svg)

The planned paired comparisons use world-clustered bootstrap risk differences and Holm correction:

| Comparison | Risk difference | 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
{comparisons}

![Selected interaction operating regimes](artifacts/figures/02_interaction_heatmaps.svg)

The working-independence logistic GEE uses only the L9 screening worlds, with world-clustered sandwich standard errors and the specified architecture-by-stressor terms, task indicators, and hop count. Main-effect coefficients are in [`gee_coefficients.csv`](artifacts/tables/gee_coefficients.csv). The selected interaction is estimated only on {analysis['expansion_row_count']:,} decisions from fresh expansion worlds in [`selected_interaction_coefficients.csv`](artifacts/tables/selected_interaction_coefficients.csv), avoiding reuse of screening observations for confirmation.

## Mechanistic failure analysis

Each of {failures} primary failures received two paid counterfactuals: normal context augmented with original evidence, and minimal gold-only evidence. The oracle rescued {pct(analysis['oracle_rescue_rate'])}; {pct(analysis['non_retrieval_failure_fraction'])} of failures were assigned a non-retrieval primary cause.

| Primary cause | Count | Fraction of failures |
|---|---:|---:|
{failure_table}

![Failure distribution and oracle rescue](artifacts/figures/03_failure_modes.svg)

For local tool tasks, correct evidence appeared in {analysis['tool_funnel']['correct_evidence_retrieval']} of {analysis['tool_funnel']['tool_questions']} contexts and yielded {analysis['tool_funnel']['correct_tool_execution']} exact, locally simulated executions. No real message, purchase, booking, or external action was available.

![Retrieval-to-tool execution funnel](artifacts/figures/04_retrieval_tool_funnel.svg)

## Cost, latency, and variance

Recorded API spend was ${actual_spend:.4f}: ${operational:.4f} attributed to operational ingestion/answering in the primary analysis and ${diagnostic:.4f} to oracle diagnosis. Benchmark generation itself used no API. Costs are empirical for the pinned prices and returned usage, not a general pricing forecast.

![Accuracy-cost Pareto view](artifacts/figures/05_accuracy_cost_pareto.svg)

![Token, cache, and cost breakdown](artifacts/figures/06_token_cost_breakdown.svg)

Repeat-call variance is reported separately in [`repeat_variance.csv`](artifacts/tables/repeat_variance.csv), while all primary uncertainty resamples worlds rather than treating the twelve questions per world as independent.

## Reproducibility

Run `OPENAI_API_KEY=... make paper` from this directory. Completed paid calls are not repeated. Raw event records are in [`results.jsonl`](artifacts/raw/results.jsonl), normalized data and the request ledger are in `artifacts/results.sqlite`, derived tables are in `artifacts/tables`, and the frozen protocol is in [`PLAN.md`](PLAN.md). Configuration hash: `{config.hash()}`. Code revision: `{code_revision(root)}`. The database manifest contains {len(manifest)} pre-report artifact checksums.

## Limitations

The benchmark is synthetic, English, single-provider, and centered on assistant-style facts. Extraction and summarization use the same primary model as answering, so their errors are not independent. Temperature zero does not guarantee identical outputs, and the variance audit is limited. GEE estimates can be unstable for rare outcomes or separation. The selected expansion interaction is confirmatory only for fresh worlds in its fixed third-factor slice. Public-benchmark evidence, when present, is supplementary. Results do not establish broad cross-model or real-user generalization.

## References

- MemFail. <https://arxiv.org/abs/2605.26667>
- MemOps. <https://arxiv.org/abs/2607.12893>
- Mem2ActBench. <https://arxiv.org/abs/2601.19935>
- LongMemEval. <https://arxiv.org/abs/2410.10813>
"""
    if any(marker in readme for marker in ("TBD", "TODO", "PLACEHOLDER", "Results pending")):
        raise ValueError("report contains a placeholder")
    atomic_write_text(root / "PAPER.md", readme)


def validate_report_inputs(db: sqlite3.Connection, analysis: dict[str, Any],
                           config: ExperimentConfig) -> None:
    if db.execute("SELECT COUNT(*) FROM runs WHERE completed_at IS NULL").fetchone()[0]:
        raise ValueError("report generation blocked by incomplete run")
    stages = analysis.get("included_stages", [])
    db_count = db.execute(
        f"SELECT COUNT(*) FROM results WHERE repeat_index=0 AND stage IN ({','.join('?' for _ in stages)})",
        stages,
    ).fetchone()[0]
    if db_count != analysis.get("row_count"):
        raise ValueError("analysis/database row count mismatch")
    spent = db.execute("SELECT COALESCE(SUM(actual_cost_usd),0) FROM api_calls WHERE status='complete'").fetchone()[0]
    reserved = db.execute("SELECT COALESCE(SUM(estimated_cost_usd),0) FROM api_calls WHERE status='reserved'").fetchone()[0]
    if spent + reserved > config.spend_cap_usd + 1e-9:
        raise ValueError("spend cap exceeded")
    if not analysis.get("architecture_summary"):
        raise ValueError("results missing")


def hypothesis_text(rows: list[dict[str, Any]], analysis: dict[str, Any]) -> str:
    by_factor: dict[str, dict[int, float]] = {}
    for factor in ("temporal_distance", "interference", "contradictions"):
        levels = sorted({row[factor] for row in rows})
        by_factor[factor] = {level: sum(r["success"] for r in rows if r[factor] == level) /
                             sum(1 for r in rows if r[factor] == level) for level in levels}
    nonlinear = []
    for factor, values in by_factor.items():
        levels = sorted(values)
        if len(levels) == 3:
            low_mid = values[levels[0]] - values[levels[1]]
            mid_high = values[levels[1]] - values[levels[2]]
            nonlinear.append(f"{factor}: low→medium drop {pct(low_mid)}, medium→high drop {pct(mid_high)}")
    nonretrieval = analysis["non_retrieval_failure_fraction"]
    return (f"H2 (most failures are non-retrieval) was {'supported' if nonretrieval > .5 else 'not supported'} "
            f"under the preregistered >50% criterion ({pct(nonretrieval)}). Nonlinearity diagnostics were "
            + "; ".join(nonlinear) + ". H5 is quantified by the retrieval-to-execution funnel below.")


def build_manifest(root: Path) -> dict[str, str]:
    files = [root / "PLAN.md", root / "pyproject.toml", root / "Makefile"]
    files += sorted((root / "src").rglob("*.py"))
    files += sorted((root / "artifacts" / "tables").glob("*"))
    files += sorted((root / "artifacts" / "figures").glob("*.svg"))
    files += sorted((root / "artifacts" / "benchmark").glob("*"))
    files += [root / "artifacts" / "config.json", root / "artifacts" / "stage_decisions.json",
              root / "artifacts" / "raw" / "results.jsonl"]
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in files if path.is_file()}


def pct(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{100 * value:.1f}%"


def svg_start(title: str, width: int = 900, height: int = 500) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="24" y="34" font-family="sans-serif" font-size="21" font-weight="bold">{html.escape(title)}</text>']


def text(x: float, y: float, value: Any, size: int = 12, anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{size}" text-anchor="{anchor}">{html.escape(str(value))}</text>'


def finish(path: Path, parts: list[str]) -> None:
    parts.append("</svg>")
    atomic_write_text(path, "\n".join(parts) + "\n")


def create_stress_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    parts = svg_start("Memory stress curves", 1050, 430)
    architectures = sorted({r["architecture"] for r in rows})
    factors = (("temporal_distance", "Temporal distance"), ("interference", "Interference"),
               ("contradictions", "Contradictions"))
    for panel, (factor, label) in enumerate(factors):
        left, top, width, height = 55 + panel * 345, 75, 285, 270
        parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#444"/>',
                  f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#444"/>',
                  text(left + width / 2, 380, label, 14, "middle")]
        levels = sorted({r[factor] for r in rows})
        for index, architecture in enumerate(architectures):
            points = []
            for pos, level in enumerate(levels):
                group = [r for r in rows if r["architecture"] == architecture and r[factor] == level]
                acc = sum(r["success"] for r in group) / len(group)
                x = left + pos * width / max(1, len(levels)-1)
                y = top + (1-acc) * height
                points.append(f"{x:.1f},{y:.1f}")
                if index == 0: parts.append(text(x, top+height+18, level, 10, "middle"))
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[index]}" stroke-width="2"/>')
    for index, architecture in enumerate(architectures):
        x = 60 + index * 160
        parts += [f'<line x1="{x}" y1="410" x2="{x+18}" y2="410" stroke="{COLORS[index]}" stroke-width="3"/>',
                  text(x+23, 414, architecture, 10)]
    finish(path, parts)


def create_interaction_figure(path: Path, rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    left_factor, right_factor = selected["factor_a"], selected["factor_b"]
    architectures = sorted({r["architecture"] for r in rows})
    parts = svg_start(f"Selected interaction: {left_factor} × {right_factor}", 1100, 500)
    for aidx, architecture in enumerate(architectures):
        x0 = 35 + aidx * 175
        parts.append(text(x0+65, 65, architecture, 11, "middle"))
        left_levels = sorted({r[left_factor] for r in rows})
        right_levels = sorted({r[right_factor] for r in rows})
        for yi, lv in enumerate(left_levels):
            for xi, rv in enumerate(right_levels):
                group = [r for r in rows if r["architecture"] == architecture and
                         r[left_factor] == lv and r[right_factor] == rv]
                acc = sum(r["success"] for r in group) / len(group) if group else math.nan
                shade = 235 if not group else int(245 - 170 * acc)
                color = f"rgb({shade},{min(250,shade+25)},250)"
                x, y = x0 + xi*45, 90 + yi*45
                parts.append(f'<rect x="{x}" y="{y}" width="43" height="43" fill="{color}" stroke="white"/>')
                parts.append(text(x+21, y+26, "—" if not group else f"{acc:.2f}", 10, "middle"))
    parts.append(text(35, 260, "Rows: " + left_factor + "; columns: " + right_factor + ". Blank cells were not observed in executed stages.", 12))
    finish(path, parts)


def create_failure_figure(path: Path, analysis: dict[str, Any]) -> None:
    parts = svg_start("Failure modes and oracle rescue", 900, 500)
    rows = analysis["failure_modes"]
    maximum = max((r["count"] for r in rows), default=1)
    for index, row in enumerate(rows):
        y = 75 + index * 42
        width = 500 * row["count"] / maximum
        parts += [text(20, y+18, row["primary_cause"], 11),
                  f'<rect x="280" y="{y}" width="{width}" height="25" fill="#7c3aed"/>',
                  text(290+width, y+18, row["count"], 11)]
    rescue = analysis["oracle_rescue_rate"]
    parts += [text(20, 455, f"Oracle rescue rate: {pct(rescue)}", 15),
              f'<rect x="280" y="438" width="{500*rescue}" height="22" fill="#059669"/>']
    finish(path, parts)


def create_funnel_figure(path: Path, funnel: dict[str, int]) -> None:
    parts = svg_start("Retrieval-to-tool-execution funnel", 850, 400)
    values = [("Tool questions", funnel["tool_questions"]),
              ("Correct evidence retrieved", funnel["correct_evidence_retrieval"]),
              ("Exact local execution", funnel["correct_tool_execution"])]
    maximum = max(1, values[0][1])
    for index, (label, value) in enumerate(values):
        width = 650 * value / maximum
        x = 100 + (650-width)/2
        y = 80 + index*90
        parts += [f'<rect x="{x}" y="{y}" width="{width}" height="55" fill="{COLORS[index]}" rx="5"/>',
                  text(425, y+34, f"{label}: {value}", 15, "middle")]
    finish(path, parts)


def create_pareto_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    parts = svg_start("Accuracy–cost Pareto view (marker size reflects latency)", 900, 520)
    max_cost = max((r["cost_per_question_usd"] for r in rows), default=1) or 1
    max_latency = max((r["median_latency_ms"] for r in rows), default=1) or 1
    parts += ['<line x1="80" y1="430" x2="840" y2="430" stroke="#444"/>',
              '<line x1="80" y1="70" x2="80" y2="430" stroke="#444"/>',
              text(450, 480, "Operational cost per question (USD)", 14, "middle"),
              text(20, 65, "Accuracy", 13)]
    for index, row in enumerate(rows):
        x = 80 + 740 * row["cost_per_question_usd"] / max_cost
        y = 430 - 350 * row["accuracy"]
        radius = 5 + 12 * row["median_latency_ms"] / max_latency
        parts += [f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{COLORS[index%len(COLORS)]}" opacity="0.8"/>',
                  text(x+radius+3, y+4, row["architecture"], 10)]
    finish(path, parts)


def create_token_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    parts = svg_start("Token and cache breakdown by architecture", 1000, 520)
    maximum = max((r["input_tokens"] + r["output_tokens"] for r in rows), default=1)
    for index, row in enumerate(rows):
        y = 70 + index*65
        x = 180
        input_w = 700 * row["input_tokens"] / maximum
        cached_w = 700 * row["cached_tokens"] / maximum
        output_w = 700 * row["output_tokens"] / maximum
        parts += [text(15, y+18, row["architecture"], 11),
                  f'<rect x="{x}" y="{y}" width="{input_w}" height="22" fill="#93c5fd"/>',
                  f'<rect x="{x}" y="{y}" width="{cached_w}" height="22" fill="#2563eb"/>',
                  f'<rect x="{x+input_w}" y="{y}" width="{output_w}" height="22" fill="#dc2626"/>',
                  text(x+input_w+output_w+8, y+17, f"${row['operational_cost_usd']:.4f}", 10)]
    parts += [text(180, 490, "light blue=input, dark blue=cached subset, red=output; labels show operational cost", 12)]
    finish(path, parts)
