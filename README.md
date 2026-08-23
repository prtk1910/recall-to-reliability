# Recall to Reliability

A reproducible Python test bed for studying long-term memory reliability in LLM assistants. It compares full context, recency windows, vector RAG, hierarchical summaries, structured temporal memory, and a query-only baseline under controlled temporal distance, interference, and contradiction stress.

The benchmark is generated deterministically from latent world state. Tool tasks use local simulators only; the test bed cannot send messages, make purchases, create bookings, or perform other external actions.

## Requirements

- Python 3.12 or newer
- GNU Make or compatible `make`
- An OpenAI API key with access to the configured models

No external database or vector service is required. SQLite and vector operations run locally.

## Setup

```bash
cp .env.example .env
chmod 600 .env
```

Open `.env` and replace the placeholder with your API key:

```dotenv
OPENAI_API_KEY=your-key-here
```

The `.env` file and all generated artifacts are ignored by Git. Never commit a real API key.

## Run

Run the tests first:

```bash
make test
```

Verify live API compatibility with a minimal request:

```bash
make smoke
```

Generate the benchmark, run the staged experiment, analyze results, create figures, and write the empirical paper:

```bash
make paper
```

The runner is resumable. Completed requests and evaluations are checkpointed in SQLite and are reused after interruption.

## Outputs

Generated outputs are local and intentionally excluded from version control:

- `PAPER.md` — completed empirical paper
- `artifacts/results.sqlite` — normalized results and API-call ledger
- `artifacts/raw/results.jsonl` — append-only result and diagnostic records
- `artifacts/benchmark/` — deterministic benchmark worlds and manifest
- `artifacts/tables/` — derived CSV and JSON analysis tables
- `artifacts/figures/` — generated SVG figures

## Other commands

```bash
make generate  # Generate benchmark data without API calls
make clean     # Remove generated artifacts
```

## Experiment controls

- Requests are reserved against a hard configured spend cap before execution.
- API responses record requested and returned model IDs, response IDs, usage, latency, retries, and timestamps.
- Requests use `store=false` and do not rely on hidden conversation state.
- Paid calls and completed evaluations are keyed for retry-safe recovery.
- The report refuses to run when experiment stages are incomplete or inconsistent.

Model names, pricing snapshots, stress levels, seeds, context budgets, and stage sizes are defined in `src/reliabmem/config.py`.

## License

MIT
