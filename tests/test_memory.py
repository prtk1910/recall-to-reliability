from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

from reliabmem.benchmark import StressProfile, generate_world
from reliabmem.config import ExperimentConfig
from reliabmem.memory import BACKENDS, StructuredTemporalMemory, create_backend
from reliabmem.util import estimate_tokens


class FakeResult:
    def __init__(self, value, request_hash="fake"):
        self.value = value
        self.request_hash = request_hash
    def json(self):
        return self.value


class FakeClient:
    def respond_json(self, **kwargs):
        if kwargs["schema_name"] == "memory_summary":
            ids = [int(part) for part in __import__("re").findall(r"\[turn (\d+)\]", kwargs["input_text"])]
            return FakeResult({"summary": kwargs["input_text"][:1200], "source_turn_ids": ids}, "summary-hash")
        if kwargs["schema_name"] == "temporal_facts":
            return FakeResult({"facts": []}, "facts-hash")
        return FakeResult({"status": "unknown", "answer": "", "tool_name": None,
                           "tool_arguments_json": None})
    def embed(self, texts, purpose):
        vectors = []
        for value in texts:
            lowered = value.casefold()
            vectors.append([float(lowered.count("aster")), float(lowered.count("region")),
                            float(len(value) % 17 + 1)])
        return vectors, FakeResult({}, purpose)


class MemoryContractTests(unittest.TestCase):
    def test_all_backends_contract_and_budget(self) -> None:
        world = generate_world(StressProfile(10, 10, 1), 11)
        config = replace(ExperimentConfig(), context_budget_tokens=500)
        with tempfile.TemporaryDirectory() as directory:
            for name in BACKENDS:
                with self.subTest(name=name):
                    backend = create_backend(name, config, FakeClient(), Path(directory))
                    backend.reset(world.world_id + name)
                    writes = [backend.observe(turn) for turn in world.turns]
                    self.assertEqual(len(writes), len(world.turns))
                    trace = backend.build_context(world.tasks[0])
                    if name not in ("full_context",):
                        self.assertLessEqual(trace.token_count, config.context_budget_tokens + 5)
                    self.assertEqual(trace.architecture, name)
                    snapshot = backend.snapshot()
                    self.assertEqual(snapshot.observed_turns, len(world.turns))

    def test_temporal_version_resolution_and_forgetting(self) -> None:
        config = ExperimentConfig()
        backend = StructuredTemporalMemory(config, FakeClient())
        backend.reset("fixture")
        backend._upsert_fact("Aster", "region", "Oslo", "current", (1,))
        backend._upsert_fact("Aster", "region", "Lima", "current", (2,))
        rows = list(backend.db.execute("SELECT value,status FROM facts ORDER BY version"))
        self.assertEqual([(r["value"], r["status"]) for r in rows],
                         [("Oslo", "historical"), ("Lima", "current")])
        backend._upsert_fact("Aster", "region", None, "forgotten", (3,))
        statuses = [r[0] for r in backend.db.execute("SELECT status FROM facts")]
        self.assertTrue(all(value == "forgotten" for value in statuses))


if __name__ == "__main__":
    unittest.main()
