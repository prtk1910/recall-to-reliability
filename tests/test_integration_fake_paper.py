from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

from reliabmem.analysis import analyze
from reliabmem.api import APILedger, OpenAIClient
from reliabmem.benchmark import StressProfile, generate_world
from reliabmem.config import ExperimentConfig
from reliabmem.harness import ExperimentHarness, ResultStore
from reliabmem.report import generate_report


class FakeAPI:
    def __init__(self):
        self.count = 0

    def __call__(self, url, payload, headers):
        self.count += 1
        if url.endswith("/embeddings"):
            data = [{"index": index, "embedding": [1.0, float(len(value) % 7 + 1)]}
                    for index, value in enumerate(payload["input"])]
            return {"id": f"emb_{self.count}", "model": payload["model"], "data": data,
                    "usage": {"prompt_tokens": 20}}
        name = payload["text"]["format"]["name"]
        if name == "memory_summary":
            value = {"summary": "compressed memory", "source_turn_ids": []}
        elif name == "temporal_facts":
            value = {"facts": []}
        else:
            value = {"status": "unknown", "answer": "", "tool_name": None,
                     "tool_arguments_json": None}
        return {"id": f"resp_{self.count}", "model": payload["model"],
                "output_text": json.dumps(value),
                "usage": {"input_tokens": 30, "output_tokens": 8,
                          "input_tokens_details": {"cached_tokens": 2},
                          "output_tokens_details": {"reasoning_tokens": 1}}}


class FakePaperIntegrationTests(unittest.TestCase):
    def test_database_analysis_figures_and_paper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src" / "dummy").mkdir(parents=True)
            (root / "src" / "dummy" / "x.py").write_text("x=1\n")
            (root / "PLAN.md").write_text("plan\n")
            (root / "Makefile").write_text("all:\n\t@true\n")
            (root / "pyproject.toml").write_text("[project]\nname='x'\nversion='1'\n")
            artifacts = root / "artifacts"
            config = replace(ExperimentConfig(), architectures=("query_only", "recency"))
            db_path = artifacts / "results.sqlite"
            ledger = APILedger(db_path, 50)
            self.addCleanup(ledger.db.close)
            client = OpenAIClient(config, ledger, api_key="fake", transport=FakeAPI(), sleep=lambda _: None)
            store = ResultStore(db_path)
            self.addCleanup(store.db.close)
            harness = ExperimentHarness(config, client, store, artifacts)
            last_run = ""
            for stage, seed in (("screening", 31), ("expansion", 32)):
                run_id = f"{stage}-fake"
                last_run = run_id
                world = generate_world(StressProfile(10, 10, 0), seed)
                store.start_run(run_id, stage, config, "fake-revision", config.primary_model)
                store.register_world(world, 100)
                for architecture in config.architectures:
                    harness.run_world(run_id, stage, world, architecture)
                store.complete_run(run_id)
            result = analyze(db_path, artifacts / "tables", config.seed)
            generate_report(root, db_path, result, config, last_run)
            self.assertIn("## 5. Results", (root / "PAPER.md").read_text())
            self.assertEqual(len(list((artifacts / "figures").glob("*.svg"))), 6)
            self.assertTrue((artifacts / "tables" / "analysis.json").exists())


if __name__ == "__main__":
    unittest.main()
