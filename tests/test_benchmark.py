from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reliabmem.benchmark import StressProfile, generate_world, load_worlds, save_worlds, validate_world
from reliabmem.config import TASK_TYPES
from reliabmem.util import estimate_tokens


class BenchmarkTests(unittest.TestCase):
    def test_deterministic_world_and_lineage(self) -> None:
        profile = StressProfile(100, 100, 3)
        left = generate_world(profile, 19)
        right = generate_world(profile, 19)
        self.assertEqual(left, right)
        self.assertEqual(tuple(task.task_type for task in left.tasks), TASK_TYPES)
        self.assertEqual(sum(t.kind == "interference" for t in left.turns), 100)
        final_evidence = max(t.turn_id for t in left.turns if t.kind == "evidence")
        self.assertEqual(len(left.turns) - final_evidence - 1, 100)
        self.assertEqual(len(left.transitions) - 1, 3)
        self.assertLess(estimate_tokens("\n".join(t.text for t in left.turns)), 200_000)
        validate_world(left)

    def test_all_factor_extremes_and_no_query_leak(self) -> None:
        for distance in (10, 1_000):
            for interference in (10, 1_000):
                for contradictions in (0, 3):
                    world = generate_world(StressProfile(distance, interference, contradictions), 7)
                    for task in world.tasks:
                        if task.answer not in {"UNKNOWN", "FORGOTTEN", "TOOL"}:
                            self.assertNotIn(task.answer.casefold(), task.query.casefold())

    def test_jsonl_round_trip(self) -> None:
        world = generate_world(StressProfile(10, 10, 1), 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worlds.jsonl"
            save_worlds(path, [world])
            self.assertEqual(load_worlds(path), [world])


if __name__ == "__main__":
    unittest.main()
