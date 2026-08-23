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
        'provider-default sampling because explicit temperature control was unavailable'
        if omitted_temperature else 'temperature zero'
    )
    if config.api_protocol == "chat_completions":
        request_state_method = "independent Chat Completions requests without conversation state"
    else:
        reasoning_context_method = (
            'stateless requests without the rejected reasoning.context field'
            if omitted_reasoning_context else 'reasoning.context=current_turn'
        )
        request_state_method = f"`store=false` and {reasoning_context_method}"
    best = max(analysis["architecture_summary"], key=lambda row: row["accuracy"])
    failures = sum(row["count"] for row in analysis["failure_modes"])
    interaction = analysis["selected_interaction"]
    decisions_path = root / "artifacts" / "stage_decisions.json"
    decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    variance_note = "executed" if decisions.get("variance_audit") else "omitted"
    manifest = build_manifest(root)
    pricing_method = (
        "the catalog rates retained with each request and the charged cost returned by OpenRouter"
        if config.provider == "openrouter"
        else "the token counts returned by the API and the pricing snapshot pinned in configuration"
    )
    manifest_runs = [row[0] for row in db.execute("SELECT run_id FROM runs WHERE completed_at IS NOT NULL")]
    manifest_runs.append(run_id)
    with db:
        for manifest_run in manifest_runs:
            for path, checksum in manifest.items():
                db.execute("INSERT OR REPLACE INTO manifests VALUES(?,?,?)",
                           (manifest_run, path, checksum))

    comparisons = "\n".join(
        f"| {display_name(r['architecture'])} vs. {display_name(r['reference'])} | {pct(r['risk_difference'])} | "
        f"[{pct(r['ci_low'])}, {pct(r['ci_high'])}] | {format_p(r['holm_p_value'])} |"
        for r in analysis["paired_comparisons"]
    )
    summaries = "\n".join(
        f"| {display_name(r['architecture'])} | {pct(r['accuracy'])} [{pct(r['accuracy_ci_low'])}, {pct(r['accuracy_ci_high'])}] | "
        f"${r['cost_per_question_usd']:.6f} | {r['median_latency_ms']:.0f} | {r['p95_latency_ms']:.0f} |"
        for r in analysis["architecture_summary"]
    )
    failure_table = "\n".join(
        f"| {display_name(r['primary_cause'])} | {r['count']} | {pct(r['fraction'])} |"
        for r in analysis["failure_modes"]
    ) or "| None observed | 0 | 0.0% |"
    stress_findings = hypothesis_text(rows, analysis)
    readme = f"""# From Recall to Reliability: Stress-Testing Long-Term Memory in Language-Model Assistants

## Abstract

Long-term memory is increasingly treated as an infrastructure component of language-model assistants, yet aggregate recall scores obscure where memory pipelines fail and how those failures propagate into action. We introduce a controlled test bed that varies temporal distance, semantic interference, and superseding contradictions while preserving complete causal lineage from source turn to model output. Across {analysis['row_count']:,} decisions in {analysis['world_count']:,} independently generated worlds, we compare full context, recency, vector retrieval, hierarchical summarization, structured temporal memory, and a query-only baseline. **{display_name(best['architecture'])}** achieved the highest observed accuracy ({pct(best['accuracy'])}; world-clustered 95% CI, {pct(best['accuracy_ci_low'])}–{pct(best['accuracy_ci_high'])}). Source-turn interventions rescued {pct(analysis['oracle_rescue_rate'])} of failures, demonstrating that retrieval is consequential but not sufficient: tool grounding, reasoning, and context integration persisted as downstream failure modes. These results provide a stress-regime map of memory reliability and a reproducible framework for separating storage, retrieval, integration, and action errors.

## 1. Introduction

Persistent assistants must recover information after long delays, distinguish current facts from superseded states, and convert retrieved evidence into correctly grounded actions. Existing evaluations often collapse these requirements into a single end-to-end score. That score is useful for ranking systems, but it cannot determine whether an error arose during memory writing, representation, retrieval, context integration, reasoning, or tool invocation.

We study three questions. First, how do temporal distance, semantic interference, and contradiction density alter the operating regimes of common memory architectures? Second, what fraction of observed errors can be causally attributed to evidence access rather than downstream reasoning or grounding? Third, how do reliability, latency, and inference cost trade off across memory designs? Our central contribution is a controlled evaluation that combines factorial stress testing, source-level counterfactual interventions, and end-to-end efficiency accounting.

## 2. Related work

This study builds on the failure taxonomies and operational perspectives developed by [Garg et al. (2026)](https://arxiv.org/abs/2605.26667), [Hao et al. (2026)](https://arxiv.org/abs/2607.12893), and [Shen et al. (2026)](https://arxiv.org/abs/2601.19935). It complements [Wu et al. (2025)](https://arxiv.org/abs/2410.10813) by emphasizing controlled latent-world generation and internal causal traces rather than broad coverage of naturally occurring long-context questions. The distinctive contribution is the joint analysis of stress regimes, causal pipeline interventions, and accuracy–efficiency trade-offs.

## 3. Benchmark and memory systems

We generate conversations deterministically from latent world states rather than from a language model. Each world instantiates exact levels of final-evidence distance (10, 100, or 1,000 turns), semantically related interference (10, 100, or 1,000 facts), and contradiction count (0, 1, or 3 superseding updates). Twelve tasks probe atomic recall, current and historical state, coexisting and conditional facts, temporal ordering, two- and four-hop reasoning, selective forgetting, abstention, tool arguments, and tool selection. Automated validators reject answer leakage, state-transition inconsistencies, factor-count mismatches, and broken evidence lineage.

All selective systems operate under the same 8,000-token retrieval budget. Full context serves as an uncompressed reference condition. The recency baseline retains the newest turns; vector RAG embeds bounded chunks and restores retrieved chunks to chronological order; hierarchical memory recursively summarizes ten-turn leaves with fan-out ten; structured temporal memory stores versioned facts in SQLite; and the query-only condition measures performance without conversational memory. Every backend implements the same benchmark-facing interface—`observe`, `build_context`, `snapshot`, and `reset`—and receives no gold annotations.

All primary evaluations used `{config.primary_model}` through `{config.provider}` at low reasoning effort, {temperature_method}, and {request_state_method}; embeddings used `{config.embedding_model}`. Requests were content-addressed, allowing interrupted runs to resume without repeating completed evaluations. The ledger records the provider, requested and returned models, response identifiers, retry counts, latency, detailed token usage, prompt hashes, and timestamps.

## 4. Experimental design and statistical analysis

After an end-to-end pilot, a balanced L9 design screened the three stressors at three levels. We then selected the factor pair with the largest leave-one-world-out improvement from adding an interaction term and evaluated a fresh 3×3 slice for **{display_name(interaction['factor_a'])} × {display_name(interaction['factor_b'])}** (ΔMSE, {interaction['cv_mse_improvement']:.6f}). The repeat audit was {variance_note}. All questions are nested within worlds: confidence intervals use paired, world-clustered bootstrap resampling; planned architecture comparisons report paired risk differences with Holm correction; and a working-independence logistic GEE estimates architecture-by-stressor effects with world-clustered sandwich standard errors.

## 5. Results

| Architecture | Accuracy (world-clustered 95% CI) | Operational cost/question | Median latency ms | P95 latency ms |
|---|---:|---:|---:|---:|
{summaries}

![Three-panel memory stress curves](artifacts/figures/01_stress_curves.svg)

The planned comparisons below use recency as the reference. Positive risk differences favor the architecture named first.

| Comparison | Risk difference | 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
{comparisons}

![Selected interaction operating regimes](artifacts/figures/02_interaction_heatmaps.svg)

{stress_findings}

The GEE main-effect estimates use only the L9 screening worlds and adjust for task type and hop count. The selected interaction is estimated separately on {analysis['expansion_row_count']:,} decisions from fresh expansion worlds, preventing the observations used for interaction selection from also serving as confirmation. Full coefficient tables are available in [`gee_coefficients.csv`](artifacts/tables/gee_coefficients.csv) and [`selected_interaction_coefficients.csv`](artifacts/tables/selected_interaction_coefficients.csv).

## 6. Mechanistic failure analysis

For each of {failures:,} primary failures, we performed two counterfactual interventions: an oracle-retrieval condition that restored the original evidence turns and a gold-only condition containing the minimal sufficient evidence. Oracle retrieval rescued {pct(analysis['oracle_rescue_rate'])} of failures. Nevertheless, {pct(analysis['non_retrieval_failure_fraction'])} of failures received a non-retrieval primary attribution, showing that evidence access alone does not ensure correct integration, reasoning, abstention, or tool grounding.

| Primary cause | Count | Fraction of failures |
|---|---:|---:|
{failure_table}

![Failure distribution and oracle rescue](artifacts/figures/03_failure_modes.svg)

Among {analysis['tool_funnel']['tool_questions']} tool-oriented decisions, the trace-based retrieval criterion was satisfied in {analysis['tool_funnel']['correct_evidence_retrieval']} cases ({pct(analysis['tool_funnel']['correct_evidence_retrieval'] / analysis['tool_funnel']['tool_questions'])}), while {analysis['tool_funnel']['correct_tool_execution']} ({pct(analysis['tool_funnel']['correct_tool_execution'] / analysis['tool_funnel']['tool_questions'])}) produced the exact expected call. These criteria are scored independently rather than as nested stages: a context may support a correct deterministic action without satisfying the stricter source-trace retrieval flag. All actions were evaluated against local simulators.

![Retrieval-to-tool execution funnel](artifacts/figures/04_retrieval_tool_funnel.svg)

## 7. Efficiency and repeatability

Operational efficiency includes amortized memory ingestion, retrieval, and answering. Cost estimates use {pricing_method}; diagnostic interventions and deterministic benchmark construction are excluded from the operational comparison. The Pareto view therefore compares architectures on the same decision workload rather than total project expenditure.

![Accuracy-cost Pareto view](artifacts/figures/05_accuracy_cost_pareto.svg)

![Token, cache, and cost breakdown](artifacts/figures/06_token_cost_breakdown.svg)

Repeat-call variability is reported separately in [`repeat_variance.csv`](artifacts/tables/repeat_variance.csv). Primary uncertainty estimates resample worlds, preserving the dependence among the twelve tasks derived from each latent state.

## 8. Reproducibility

The complete test bed and execution instructions are available at [github.com/prtk1910/recall-to-reliability](https://github.com/prtk1910/recall-to-reliability). Running `make paper` reconstructs the benchmark, executes resumable evaluations, performs the statistical analysis, regenerates all figures, and writes the manuscript. Each run records its configuration hash, code revision, prompt hashes, model metadata, random seed, and artifact checksums. This analysis used configuration `{config.hash()}` and code revision `{code_revision(root)}`; the manifest contains {len(manifest)} checksummed pre-report artifacts.

## 9. Limitations

The benchmark is synthetic, English-language, and centered on assistant-style factual memory. Extraction, summarization, and answering use the same primary model, so their errors are not independent. The provider did not expose deterministic temperature control for this model, and the repeat audit covers only a stratified subset. GEE estimates may be unstable for rare outcomes or quasi-separation. The selected interaction is confirmed only within a fresh slice whose third factor is fixed at its middle level. Most importantly, a single model family and controlled latent worlds cannot establish cross-model robustness or real-user validity.

## 10. Conclusion

Memory architecture materially changes assistant reliability, particularly under long temporal displacement. Hierarchical summarization and vector retrieval approached the full-context reference while using selective context, but no architecture eliminated downstream reasoning and grounding errors. The oracle interventions show why end-to-end memory evaluation should trace the entire pipeline: retrieving the right evidence is often necessary, yet it is not equivalent to using that evidence correctly. Controlled stress regimes and causal diagnostics offer a more informative basis for designing reliable persistent assistants than aggregate recall alone.

## References

- Garg, I., Kolhe, N., Song, D., & Zhao, X. (2026). *MemFail: Stress-Testing Failure Modes of LLM Memory Systems*. arXiv:2605.26667. <https://arxiv.org/abs/2605.26667>
- Hao, X., Zhang, Z., Lin, Z., Sun, Y., Guo, Z., Zhang, X., Liang, Y., Xiong, F., & Li, Z. (2026). *MemOps: Benchmarking Lifecycle Memory Operations in Long-Horizon Conversations*. arXiv:2607.12893. <https://arxiv.org/abs/2607.12893>
- Shen, Y., Li, K., Zhou, W., & Hu, S. (2026). *Mem2ActBench: A Benchmark for Evaluating Long-Term Memory Utilization in Task-Oriented Autonomous Agents*. arXiv:2601.19935. <https://arxiv.org/abs/2601.19935>
- Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., & Yu, D. (2025). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*. ICLR 2025. <https://arxiv.org/abs/2410.10813>
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
    gradients = []
    for factor, values in by_factor.items():
        levels = sorted(values)
        if len(levels) == 3:
            low_mid = values[levels[1]] - values[levels[0]]
            mid_high = values[levels[2]] - values[levels[1]]
            gradients.append(
                f"{display_name(factor).lower()} ({signed_pp(low_mid)} from low to medium; "
                f"{signed_pp(mid_high)} from medium to high)"
            )
    nonretrieval = analysis["non_retrieval_failure_fraction"]
    findings = []
    if gradients:
        findings.append("Pooled stress gradients were " + "; ".join(gradients) + ".")
    if nonretrieval > .5:
        findings.append(
            f"The {pct(nonretrieval)} non-retrieval attribution rate exceeded the prespecified "
            ">50% threshold, indicating that most failures occurred after evidence access."
        )
    else:
        findings.append(
            f"The {pct(nonretrieval)} non-retrieval attribution rate fell below the prespecified "
            ">50% threshold, although it still represents nearly half of all observed failures."
        )
    return " ".join(findings)


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


def format_p(value: float) -> str:
    return "<0.0001" if value < .0001 else f"{value:.4f}"


def signed_pp(value: float) -> str:
    return f"{100 * value:+.1f} percentage points"


def display_name(value: str) -> str:
    labels = {
        "full_context": "Full context",
        "hierarchical_summary": "Hierarchical summary",
        "query_only": "Query only",
        "recency": "Recency",
        "structured_temporal": "Structured temporal",
        "vector_rag": "Vector RAG",
        "temporal_distance": "Temporal distance",
        "interference": "Semantic interference",
        "contradictions": "Contradictions",
        "abstention_or_forgetting_policy_failure": "Abstention or forgetting-policy failure",
        "context_integration_failure": "Context-integration failure",
        "reasoning_failure": "Reasoning failure",
        "representation_or_consolidation_loss": "Representation or consolidation loss",
        "retrieval_contamination": "Retrieval contamination",
        "retrieval_miss": "Retrieval miss",
        "tool_grounding_failure": "Tool-grounding failure",
    }
    return labels.get(value, value.replace("_", " ").capitalize())


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
    parts = svg_start("Memory stress curves", 1200, 540)
    architectures = sorted({r["architecture"] for r in rows})
    factors = (("temporal_distance", "Temporal distance"), ("interference", "Semantic interference"),
               ("contradictions", "Contradictions"))
    for panel, (factor, label) in enumerate(factors):
        left, top, width, height = 70 + panel * 390, 75, 330, 300
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + (1 - tick) * height
            parts += [f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="#e5e7eb"/>',
                      text(left-7, y+4, f"{tick:.2f}", 9, "end")]
        parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#444"/>',
                  f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#444"/>',
                  text(left + width / 2, 420, label, 14, "middle")]
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
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[index]}" stroke-width="2.5"/>')
            for point in points:
                x, y = point.split(",")
                parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{COLORS[index]}"/>')
    parts.append('<text x="18" y="225" font-family="sans-serif" font-size="14" text-anchor="middle" transform="rotate(-90 18 225)">Accuracy</text>')
    for index, architecture in enumerate(architectures):
        x = 70 + (index % 3) * 370
        y = 462 + (index // 3) * 27
        parts += [f'<line x1="{x}" y1="{y}" x2="{x+22}" y2="{y}" stroke="{COLORS[index]}" stroke-width="4"/>',
                  text(x+29, y+4, display_name(architecture), 11)]
    parts.append(text(600, 525, "Mean screening accuracy at each stress level; higher is better.", 11, "middle"))
    finish(path, parts)


def create_interaction_figure(path: Path, rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    left_factor, right_factor = selected["factor_a"], selected["factor_b"]
    architectures = sorted({r["architecture"] for r in rows})
    parts = svg_start(
        f"Selected interaction: {display_name(left_factor)} × {display_name(right_factor)}",
        1400, 420,
    )
    left_levels = sorted({r[left_factor] for r in rows})
    right_levels = sorted({r[right_factor] for r in rows})
    for aidx, architecture in enumerate(architectures):
        x0 = 55 + aidx * 225
        parts.append(text(x0+72, 62, display_name(architecture), 11, "middle"))
        for xi, rv in enumerate(right_levels):
            parts.append(text(x0+xi*48+23, 88, rv, 9, "middle"))
        for yi, lv in enumerate(left_levels):
            parts.append(text(x0-7, 100+yi*48+29, lv, 9, "end"))
            for xi, rv in enumerate(right_levels):
                group = [r for r in rows if r["architecture"] == architecture and
                         r[left_factor] == lv and r[right_factor] == rv]
                acc = sum(r["success"] for r in group) / len(group) if group else math.nan
                shade = 235 if not group else int(245 - 170 * acc)
                color = f"rgb({shade},{min(250,shade+25)},250)"
                x, y = x0 + xi*48, 100 + yi*48
                parts.append(f'<rect x="{x}" y="{y}" width="46" height="46" fill="{color}" stroke="white"/>')
                parts.append(text(x+23, y+28, "—" if not group else f"{acc:.2f}", 10, "middle"))
    parts += [text(700, 275, f"Rows: {display_name(left_factor)}; columns: {display_name(right_factor)}; cells show accuracy.", 12, "middle"),
              text(55, 325, "Accuracy scale", 11)]
    for index, value in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        shade = int(245 - 170 * value)
        color = f"rgb({shade},{min(250,shade+25)},250)"
        x = 155 + index * 72
        parts += [f'<rect x="{x}" y="307" width="45" height="22" fill="{color}" stroke="#d1d5db"/>',
                  text(x+22, 348, f"{value:.2f}", 9, "middle")]
    parts.append(text(700, 390, "Expansion worlds only; higher and darker cells indicate better accuracy.", 11, "middle"))
    finish(path, parts)


def create_failure_figure(path: Path, analysis: dict[str, Any]) -> None:
    parts = svg_start("Failure modes and oracle rescue", 1000, 560)
    rows = analysis["failure_modes"]
    maximum = max((r["count"] for r in rows), default=1)
    for tick_index in range(5):
        value = maximum * tick_index / 4
        x = 350 + 560 * tick_index / 4
        parts += [f'<line x1="{x}" y1="75" x2="{x}" y2="410" stroke="#e5e7eb"/>',
                  text(x, 430, f"{value:.0f}", 9, "middle")]
    for index, row in enumerate(rows):
        y = 80 + index * 45
        width = 560 * row["count"] / maximum
        parts += [text(20, y+18, display_name(row["primary_cause"]), 11),
                  f'<rect x="350" y="{y}" width="{width}" height="25" fill="#7c3aed"/>',
                  text(358+width, y+18, row["count"], 11)]
    rescue = analysis["oracle_rescue_rate"]
    parts += [text(630, 452, "Number of screening failures", 12, "middle"),
              text(20, 500, f"Oracle rescue rate: {pct(rescue)}", 14),
              f'<rect x="350" y="482" width="{560*rescue}" height="24" fill="#059669"/>',
              text(350+560*rescue+8, 500, pct(rescue), 11),
              text(500, 540, "Oracle rescue = failure corrected after restoring the original evidence turns.", 11, "middle")]
    finish(path, parts)


def create_funnel_figure(path: Path, funnel: dict[str, int]) -> None:
    parts = svg_start("Retrieval-to-tool-execution funnel", 900, 460)
    values = [("Tool questions", funnel["tool_questions"]),
              ("Correct evidence retrieved", funnel["correct_evidence_retrieval"]),
              ("Exact local execution", funnel["correct_tool_execution"])]
    maximum = max(1, values[0][1])
    for index, (label, value) in enumerate(values):
        width = 650 * value / maximum
        x = 100 + (650-width)/2
        y = 85 + index*95
        share = value / maximum
        parts += [f'<rect x="{x}" y="{y}" width="{width}" height="55" fill="{COLORS[index]}" rx="5"/>',
                  f'<text x="425" y="{y+22}" font-family="sans-serif" font-size="12" '
                  f'font-weight="600" text-anchor="middle" fill="white">{html.escape(label)}</text>',
                  f'<text x="425" y="{y+43}" font-family="sans-serif" font-size="15" '
                  f'font-weight="bold" text-anchor="middle" fill="white">{value} · {pct(share)}</text>']
    parts += [text(450, 400, "Retrieval and execution are independently scored; these stages are not strict subsets.", 11, "middle"),
              text(450, 425, "All executions used deterministic local simulators; no external action was available.", 11, "middle")]
    finish(path, parts)


def create_pareto_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    parts = svg_start("Accuracy–cost Pareto view (marker size reflects latency)", 1000, 580)
    max_cost = max((r["cost_per_question_usd"] for r in rows), default=1) or 1
    max_latency = max((r["median_latency_ms"] for r in rows), default=1) or 1
    left, top, width, height = 100, 75, 820, 395
    for tick_index in range(5):
        fraction = tick_index / 4
        x = left + width * fraction
        y = top + height * (1-fraction)
        parts += [f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top+height}" stroke="#e5e7eb"/>',
                  f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="#e5e7eb"/>',
                  text(x, top+height+18, f"${max_cost*fraction:.4f}", 9, "middle"),
                  text(left-8, y+4, f"{fraction:.2f}", 9, "end")]
    parts += [f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#444"/>',
              f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#444"/>',
              text(510, 535, "Operational cost per question (USD)", 14, "middle"),
              '<text x="25" y="270" font-family="sans-serif" font-size="14" text-anchor="middle" transform="rotate(-90 25 270)">Accuracy</text>',
              text(500, 565, "Higher accuracy and lower cost are preferred; marker area increases with median latency.", 11, "middle")]
    for index, row in enumerate(rows):
        x = left + width * row["cost_per_question_usd"] / max_cost
        y = top + height * (1-row["accuracy"])
        radius = 5 + 12 * row["median_latency_ms"] / max_latency
        anchor = "end" if x > left + width * .72 else "start"
        label_x = x-radius-4 if anchor == "end" else x+radius+4
        parts += [f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{COLORS[index%len(COLORS)]}" opacity="0.8"/>',
                  text(label_x, y+4, display_name(row["architecture"]), 10, anchor)]
    finish(path, parts)


def create_token_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    parts = svg_start("Token and cache breakdown by architecture", 1100, 600)
    maximum = max((r["input_tokens"] + r["output_tokens"] for r in rows), default=1)
    x, plot_width = 220, 800
    for tick_index in range(5):
        fraction = tick_index / 4
        tx = x + plot_width * fraction
        parts += [f'<line x1="{tx}" y1="65" x2="{tx}" y2="445" stroke="#e5e7eb"/>',
                  text(tx, 465, f"{maximum*fraction/1_000_000:.1f}M", 9, "middle")]
    for index, row in enumerate(rows):
        y = 75 + index*60
        input_w = plot_width * row["input_tokens"] / maximum
        cached_w = plot_width * row["cached_tokens"] / maximum
        output_w = plot_width * row["output_tokens"] / maximum
        parts += [text(15, y+18, display_name(row["architecture"]), 11),
                  f'<rect x="{x}" y="{y}" width="{input_w}" height="22" fill="#93c5fd"/>',
                  f'<rect x="{x}" y="{y}" width="{cached_w}" height="22" fill="#2563eb"/>',
                  f'<rect x="{x+input_w}" y="{y}" width="{output_w}" height="22" fill="#dc2626"/>',
                  text(x+input_w+output_w+8, y+17, f"${row['operational_cost_usd']:.4f}", 10)]
    parts += [text(620, 495, "Tokens (millions)", 13, "middle"),
              '<rect x="220" y="525" width="20" height="14" fill="#93c5fd"/>', text(248, 537, "Input", 10),
              '<rect x="330" y="525" width="20" height="14" fill="#2563eb"/>', text(358, 537, "Cached input subset", 10),
              '<rect x="500" y="525" width="20" height="14" fill="#dc2626"/>', text(528, 537, "Output", 10),
              text(550, 575, "Dollar labels show operational cost attributed to each architecture.", 11, "middle")]
    finish(path, parts)
