from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reliabmem.analysis import fit_working_independence_gee, holm_adjust, select_interaction
from reliabmem.api import APIResult, Usage
from reliabmem.benchmark import StressProfile, generate_world
from reliabmem.config import ExperimentConfig
from reliabmem.harness import ExperimentHarness, ResultStore, evaluate


class AnswerClient:
    def respond_json(self, **kwargs):
        return StaticResult()
    def embed(self, texts, purpose):
        return [[1.0] for _ in texts], StaticResult()


class StaticResult(APIResult):
    def __new__(cls):
        return super().__new__(cls)
    def __init__(self):
        super().__init__("hash", "resp", "gpt-5.6-luna", "gpt-5.6-luna",
                         '{"status":"unknown","answer":"","tool_name":null,"tool_arguments_json":null}',
                         Usage(10, 1, 0, 3, 0), 0.00001, 5, 0, {})


class HarnessAnalysisTests(unittest.TestCase):
    def test_local_tool_evaluation(self) -> None:
        task = generate_world(StressProfile(10, 10, 0), 4).tasks[-2]
        expected = task.valid_tool_call
        output = {"status": "tool", "answer": "", "tool_name": expected["name"],
                  "tool_arguments_json": __import__("json").dumps(expected["arguments"])}
        result = evaluate(task, output)
        self.assertTrue(result.success)
        self.assertTrue(result.local_tool_result["simulated"])

    def test_result_store_end_to_end(self) -> None:
        world = generate_world(StressProfile(10, 10, 0), 2)
        abstention = {world.tasks[9].task_id}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ResultStore(root / "results.sqlite")
            self.addCleanup(store.db.close)
            harness = ExperimentHarness(ExperimentConfig(), AnswerClient(), store, root)
            store.start_run("run", "pilot", ExperimentConfig(), "revision", "gpt-5.6-luna")
            store.register_world(world, 100)
            harness.run_world("run", "pilot", world, "query_only", task_ids=abstention)
            store.complete_run("run")
            row = store.db.execute("SELECT success,prompt_hash FROM results").fetchone()
            self.assertEqual(row[0], 1)
            self.assertEqual(len(row[1]), 64)

    def test_statistics_known_architecture_effect(self) -> None:
        rows = []
        for world in range(8):
            for arch in ("a", "b"):
                for level in (10, 100, 1000):
                    rows.append({"world_id": f"w{world}", "architecture": arch,
                                 "success": int(arch == "b" or level == 10),
                                 "temporal_distance": level, "interference": level,
                                 "contradictions": {10: 0, 100: 1, 1000: 3}[level],
                                 "task_type": "atomic", "hop_count": 1, "stage": "screening"})
        coefficients = {row["term"]: row for row in fit_working_independence_gee(rows)}
        self.assertGreater(coefficients["arch=b"]["coefficient"], 0)
        selected = select_interaction(rows, 3)
        self.assertIn(selected["factor_a"], ("temporal_distance", "interference", "contradictions"))
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertTrue(all(a >= b for a, b in zip(adjusted, [0.01, 0.04, 0.03])))


if __name__ == "__main__":
    unittest.main()
