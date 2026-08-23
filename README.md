# Recall to Reliability

An experimental framework for identifying when long-term memory fails in language-model assistants—and whether the failure originates in storage, retrieval, context integration, reasoning, or tool grounding.

The benchmark generates deterministic conversations from latent world states and varies three stressors independently: temporal distance, semantic interference, and superseding contradictions. Six memory conditions are evaluated under a shared interface and, for selective systems, a shared context budget:

- full context;
- recency;
- vector retrieval-augmented generation;
- hierarchical summarization;
- structured temporal memory; and
- a query-only baseline.

Every prediction retains a causal trace from the source turn through memory writing, consolidation, retrieval, prompt assembly, and final output. Failures are probed with oracle-retrieval and gold-only counterfactuals. Tool tasks execute exclusively against deterministic local simulators.

## Reproducing the study

The canonical repository is [github.com/prtk1910/recall-to-reliability](https://github.com/prtk1910/recall-to-reliability).

Requirements:

- Python 3.12 or newer
- GNU Make or a compatible `make`
- an OpenAI API key with access to the configured models

Create a local environment file:

```bash
cp .env.example .env
chmod 600 .env
```

Add the key to `.env`:

```dotenv
OPENAI_API_KEY=your-key-here
```

Then run:

```bash
make test
make smoke
make paper
```

`make paper` generates the benchmark, executes the staged evaluation, performs the statistical analysis, renders the figures, and writes the empirical manuscript. Runs are resumable: completed requests and evaluations are content-addressed and recovered from SQLite after interruption.

No external database or vector service is required. API keys, raw responses, and generated research artifacts are excluded from version control.

## Experimental design

The study proceeds through an end-to-end pilot, a balanced L9 screening design, and a fresh 3×3 expansion of the interaction selected by leave-one-world-out predictive improvement. Questions are nested within independently generated worlds. Analysis uses paired world-clustered bootstrap intervals, Holm-adjusted paired risk differences, and logistic GEE with world-clustered sandwich standard errors.

The benchmark-facing memory contract is:

```python
observe(turn) -> WriteTrace
build_context(task) -> ContextTrace
snapshot() -> MemorySnapshot
reset(world_id) -> None
```

Gold facts never enter this interface. Configuration, model metadata, prompt hashes, detailed token usage, latency, retry history, random seeds, code revision, and artifact checksums are retained for each run.

## Generated outputs

- `PAPER.md` — empirical manuscript
- `artifacts/results.sqlite` — normalized results and request ledger
- `artifacts/raw/results.jsonl` — append-only results and diagnostic records
- `artifacts/benchmark/` — deterministic worlds and manifest
- `artifacts/tables/` — derived statistical tables
- `artifacts/figures/` — publication-ready SVG figures

Generated outputs remain local by default. To create the benchmark without model calls, run `make generate`.

## License

MIT
