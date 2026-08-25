# Recall to Reliability

An experimental framework for identifying when long-term memory fails in language-model assistants—and whether the failure originates in storage, retrieval, context integration, reasoning, or downstream action.

The benchmark generates deterministic conversations from latent world states and varies three stressors independently: temporal distance, semantic interference, and superseding contradictions. Six memory conditions are evaluated under a shared interface and, for selective systems, a shared context budget:

- full context;
- recency;
- vector retrieval-augmented generation;
- hierarchical summarization;
- structured temporal memory; and
- a query-only baseline.

Every prediction retains a causal trace from the source turn through memory writing, consolidation, retrieval, prompt assembly, and final output. Failures are probed with oracle-retrieval and gold-only counterfactuals. Tool tasks execute exclusively against deterministic local simulators.

## Audited benchmark and results

The reported analysis uses **11 retained task types** across 45 independently generated screening worlds and six memory conditions, for **2,970 screening decisions**. A post hoc benchmark audit excluded the original `tool_argument_grounding` task because the required tool identifier appeared only in hidden gold metadata rather than in the model-visible query or memory; failures on that task therefore measured hidden-name guessing rather than argument grounding.

On the audited benchmark, hierarchical summarization achieved **91.7%** accuracy, full context **90.5%**, vector RAG **88.5%**, structured temporal memory **82.2%**, recency **63.6%**, and query-only **0.0%**. Restoring the original source evidence rescued **79.4%** of screening failures, with retrieval misses accounting for the largest failure category.

A fixed-context reader-model replication replayed the same stored screening prompts through Ox Alpha while holding memory contexts and retrieved evidence fixed. Across the five memory-bearing architectures, semantic accuracy was **90.18% for Luna and 91.23% for Ox** (paired difference +1.05 percentage points; world-clustered 95% CI, +0.20 to +1.98). The original exact/protocol evaluator was substantially more response-style-sensitive, motivating a deterministic task-specific semantic rescore. This experiment tests downstream reader robustness rather than a full cross-model memory-pipeline replication.

## Reproducing the study

The canonical repository is [github.com/prtk1910/recall-to-reliability](https://github.com/prtk1910/recall-to-reliability).

Requirements:

- Python 3.12 or newer
- GNU Make or a compatible `make`
- an OpenAI or OpenRouter API key

Create a local environment file:

```bash
cp .env.example .env
chmod 600 .env
```

For direct OpenAI, add the key and model identifiers to `.env`:

```dotenv
RELIABMEM_PROVIDER=openai
OPENAI_API_KEY=your-key-here
RELIABMEM_MODEL=gpt-5.6-luna
RELIABMEM_EMBEDDING_MODEL=text-embedding-3-small
```

To evaluate any text-generation model in the [OpenRouter catalog](https://openrouter.ai/models), use its catalog slug:

```dotenv
RELIABMEM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-key-here
RELIABMEM_MODEL=anthropic/claude-sonnet-4.5
RELIABMEM_EMBEDDING_MODEL=openai/text-embedding-3-small
```

The same configuration works with OpenRouter-hosted Gemini, Claude, OpenAI, Meta, and other catalog models by changing `RELIABMEM_MODEL`. Choose `RELIABMEM_EMBEDDING_MODEL` from OpenRouter's [embedding catalog](https://openrouter.ai/docs/api/api-reference/embeddings/list-embeddings-models). Model prices are resolved from the catalog; the exact charged cost and native token accounting are read from each response. Models without native JSON Schema support automatically use a schema-in-prompt fallback and remain subject to the same local schema validation and retry policy.

Then run:

```bash
make test
make smoke
make paper
```

`make paper` generates the benchmark, executes the staged evaluation, performs the statistical analysis, renders the figures, and writes the empirical manuscript. Runs are resumable: completed requests and evaluations are content-addressed and recovered from SQLite after interruption.

Provider, base URL, model identifiers, pricing, and protocol are included in the configuration hash, so results from different models cannot silently share completed requests or run identifiers. The runner refuses to combine configurations in one results database. Before changing models, archive the existing `artifacts/` directory outside the repository and run `make clean`; this keeps each model's statistical analysis isolated.

No external database or vector service is required. API keys, raw responses, and large experiment databases remain local and are excluded from version control. Lightweight paper tables, figures, and reproducibility summaries may be versioned with the manuscript.

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
