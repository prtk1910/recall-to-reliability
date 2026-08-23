from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import random

from .config import TASK_TYPES
from .util import estimate_tokens, sha256_json


@dataclass(frozen=True)
class StressProfile:
    temporal_distance: int
    interference: int
    contradictions: int

    @property
    def key(self) -> str:
        return f"d{self.temporal_distance}-i{self.interference}-c{self.contradictions}"


@dataclass(frozen=True)
class Turn:
    turn_id: int
    speaker: str
    text: str
    kind: str


@dataclass(frozen=True)
class Task:
    task_id: str
    task_type: str
    query: str
    answer: str
    evidence_turn_ids: tuple[int, ...]
    hop_count: int
    unanswerable: bool = False
    forgotten: bool = False
    valid_tool_call: dict[str, Any] | None = None


@dataclass(frozen=True)
class World:
    world_id: str
    seed: int
    profile: StressProfile
    turns: tuple[Turn, ...]
    tasks: tuple[Task, ...]
    gold_facts: tuple[dict[str, Any], ...]
    transitions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "World":
        return cls(
            world_id=value["world_id"], seed=value["seed"],
            profile=StressProfile(**value["profile"]),
            turns=tuple(Turn(**turn) for turn in value["turns"]),
            tasks=tuple(Task(
                **{**task, "evidence_turn_ids": tuple(task["evidence_turn_ids"])}
            ) for task in value["tasks"]),
            gold_facts=tuple(value["gold_facts"]),
            transitions=tuple(value["transitions"]),
        )


CALLSIGNS = ("Kestrel", "Juniper", "Mariner", "Saffron", "Tern", "Vela")
REGIONS = ("Oslo", "Quito", "Tallinn", "Busan", "Lima", "Riga")
PEOPLE = ("Mira", "Niko", "Asha", "Leena", "Tomas", "Inez")
DESKS = ("D-17", "F-04", "K-22", "B-09", "H-31", "C-14")
COLORS = ("cobalt", "saffron", "jade", "umber", "violet", "silver")


def generate_world(profile: StressProfile, seed: int) -> World:
    rng = random.Random(seed * 1_000_003 + profile.temporal_distance * 101
                        + profile.interference * 17 + profile.contradictions)
    values = {
        "callsign": rng.choice(CALLSIGNS),
        "region0": rng.choice(REGIONS),
        "custodian": rng.choice(PEOPLE),
        "desk": rng.choice(DESKS),
        "primary": rng.choice(COLORS),
        "secondary": rng.choice(COLORS),
        "channel": f"archive-{rng.randrange(20, 90)}",
        "event1": f"Frost-{rng.randrange(100, 999)}",
        "event2": f"Ember-{rng.randrange(100, 999)}",
        "shelf": f"S-{rng.randrange(10, 99)}",
        "crate": f"C-{rng.randrange(100, 999)}",
        "folder": f"F-{rng.randrange(1000, 9999)}",
        "token": f"TKN-{rng.randrange(10000, 99999)}",
        "forgotten": f"OLD-{rng.randrange(1000, 9999)}",
        "sku": f"SKU-{rng.randrange(10000, 99999)}",
        "warehouse": f"WH-{rng.randrange(10, 99)}",
        "target": f"node-{rng.randrange(100, 999)}",
    }
    while values["secondary"] == values["primary"]:
        values["secondary"] = rng.choice(COLORS)

    turns: list[Turn] = []
    distractor_values: set[str] = set()
    for index in range(profile.interference):
        value = f"DecoyZone-{rng.randrange(100000, 999999)}"
        while value in distractor_values or value in values.values():
            value = f"DecoyZone-{rng.randrange(100000, 999999)}"
        distractor_values.add(value)
        turns.append(Turn(len(turns), "user",
            f"Unrelated project record {index:04d}: its deployment region is {value}; "
            f"retain this only for project decoy-{seed}-{index}.", "interference"))

    evidence: dict[str, list[int]] = {name: [] for name in TASK_TYPES}
    facts: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    def add(text: str, keys: tuple[str, ...], fact: dict[str, Any] | None = None) -> int:
        turn_id = len(turns)
        turns.append(Turn(turn_id, "user", text, "evidence"))
        for key in keys:
            evidence[key].append(turn_id)
        if fact:
            facts.append({**fact, "source_turn_id": turn_id})
        return turn_id

    add(f"For project Aster, the permanent call sign is {values['callsign']}.",
        ("atomic_recall",), {"entity": "Aster", "attribute": "callsign", "value": values["callsign"]})
    region_turn = add(f"Project Aster's deployment region is {values['region0']}.",
        ("current_state", "historical_value"),
        {"entity": "Aster", "attribute": "region", "value": values["region0"], "version": 0})
    current_region = values["region0"]
    old_region = current_region
    region_choices = [region for region in REGIONS if region != current_region]
    for version in range(1, profile.contradictions + 1):
        next_region = region_choices[(version + seed) % len(region_choices)]
        turn_id = add(
            f"Superseding update {version}: replace Aster's deployment region {current_region} "
            f"with {next_region}; {next_region} is current.",
            ("current_state", "historical_value"),
            {"entity": "Aster", "attribute": "region", "value": next_region, "version": version},
        )
        transitions.append({"entity": "Aster", "attribute": "region", "from": current_region,
                            "to": next_region, "source_turn_id": turn_id})
        current_region = next_region
    add(f"Aster uses {values['primary']} as its primary label and {values['secondary']} as its secondary label.",
        ("coexisting_facts",), {"entity": "Aster", "attribute": "labels",
                                "value": [values["primary"], values["secondary"]]})
    add(f"Audit mode for Aster is enabled. When audit mode is enabled, route reports to {values['channel']}.",
        ("conditional_fact",), {"entity": "Aster", "attribute": "audit_channel", "value": values["channel"]})
    add(f"Milestone {values['event1']} completed before milestone {values['event2']}.",
        ("temporal_ordering",), {"entity": "Aster", "attribute": "milestone_order",
                                 "value": [values["event1"], values["event2"]]})
    add(f"Aster's custodian is {values['custodian']}.", ("two_hop_reasoning",),
        {"entity": "Aster", "attribute": "custodian", "value": values["custodian"]})
    add(f"{values['custodian']}'s assigned desk is {values['desk']}.", ("two_hop_reasoning",),
        {"entity": values["custodian"], "attribute": "desk", "value": values["desk"]})
    add(f"Aster's archive points to shelf {values['shelf']}.", ("four_hop_reasoning",))
    add(f"Shelf {values['shelf']} contains crate {values['crate']}.", ("four_hop_reasoning",))
    add(f"Crate {values['crate']} contains folder {values['folder']}.", ("four_hop_reasoning",))
    add(f"Folder {values['folder']} carries token {values['token']}.", ("four_hop_reasoning",),
        {"entity": "Aster", "attribute": "archive_token", "value": values["token"]})
    forgotten_turn = add(f"Aster's temporary access phrase is {values['forgotten']}.",
        ("selective_forgetting",), {"entity": "Aster", "attribute": "temporary_access",
                                    "value": values["forgotten"], "forgotten": False})
    forget_turn = add("Forget Aster's temporary access phrase. It must no longer be recalled or used.",
        ("selective_forgetting",), {"entity": "Aster", "attribute": "temporary_access",
                                    "value": None, "forgotten": True})
    transitions.append({"entity": "Aster", "attribute": "temporary_access", "from": values["forgotten"],
                        "to": None, "forgotten": True, "source_turn_id": forget_turn})
    add(f"For Aster inventory checks, use SKU {values['sku']} at warehouse {values['warehouse']}.",
        ("tool_argument_grounding",), {"entity": "Aster", "attribute": "inventory_key",
                                       "value": [values["sku"], values["warehouse"]]})
    add(f"Aster diagnostics are local-only. Use run_local_diagnostic with target {values['target']}.",
        ("tool_selection",), {"entity": "Aster", "attribute": "diagnostic_target", "value": values["target"]})

    # The final evidence precedes every query by exactly the requested distance.
    for index in range(profile.temporal_distance):
        turns.append(Turn(len(turns), "assistant",
            f"Routine log {index:04d} for unrelated system zeta-{seed}: heartbeat nominal, sequence {rng.randrange(10**8)}.",
            "distance_filler"))

    world_id = f"{profile.key}-s{seed}-{sha256_json(values)[:8]}"
    tasks = (
        Task(f"{world_id}:atomic", "atomic_recall", "What is project Aster's permanent call sign?",
             values["callsign"], tuple(evidence["atomic_recall"]), 1),
        Task(f"{world_id}:current", "current_state", "What is Aster's current deployment region?",
             current_region, tuple(evidence["current_state"]), 1),
        Task(f"{world_id}:historical", "historical_value", "What deployment region was first recorded for Aster?",
             old_region, (region_turn,), 1),
        Task(f"{world_id}:coexisting", "coexisting_facts", "List Aster's primary and secondary labels in that order.",
             f"{values['primary']} | {values['secondary']}", tuple(evidence["coexisting_facts"]), 2),
        Task(f"{world_id}:conditional", "conditional_fact", "Given Aster's recorded mode, where should reports be routed?",
             values["channel"], tuple(evidence["conditional_fact"]), 2),
        Task(f"{world_id}:ordering", "temporal_ordering", "Which of Aster's two recorded milestones completed first?",
             values["event1"], tuple(evidence["temporal_ordering"]), 1),
        Task(f"{world_id}:twohop", "two_hop_reasoning", "What desk is assigned to Aster's custodian?",
             values["desk"], tuple(evidence["two_hop_reasoning"]), 2),
        Task(f"{world_id}:fourhop", "four_hop_reasoning", "What token is reached by following Aster's archive chain?",
             values["token"], tuple(evidence["four_hop_reasoning"]), 4),
        Task(f"{world_id}:forget", "selective_forgetting", "What is Aster's temporary access phrase?",
             "FORGOTTEN", (forgotten_turn, forget_turn), 1, forgotten=True),
        Task(f"{world_id}:abstain", "abstention", "What is Aster's launch-day catering vendor?",
             "UNKNOWN", (), 1, unanswerable=True),
        Task(f"{world_id}:toolargs", "tool_argument_grounding", "Check Aster inventory using the permitted local simulator.",
             "TOOL", tuple(evidence["tool_argument_grounding"]), 1,
             valid_tool_call={"name": "lookup_inventory", "arguments": {"sku": values["sku"],
                                                                          "warehouse": values["warehouse"]}}),
        Task(f"{world_id}:toolselect", "tool_selection", "Run Aster's permitted diagnostic using the local simulator.",
             "TOOL", tuple(evidence["tool_selection"]), 1,
             valid_tool_call={"name": "run_local_diagnostic", "arguments": {"target": values["target"]}}),
    )
    world = World(world_id, seed, profile, tuple(turns), tasks, tuple(facts), tuple(transitions))
    validate_world(world)
    return world


def validate_world(world: World, max_tokens: int = 200_000) -> None:
    if tuple(task.task_type for task in world.tasks) != TASK_TYPES:
        raise ValueError("world must contain exactly one task of each required type")
    if sum(turn.kind == "interference" for turn in world.turns) != world.profile.interference:
        raise ValueError("interference count mismatch")
    if len(world.transitions) != world.profile.contradictions + 1:
        raise ValueError("contradiction/forget transition mismatch")
    final_evidence = max(turn.turn_id for turn in world.turns if turn.kind == "evidence")
    if len(world.turns) - final_evidence - 1 != world.profile.temporal_distance:
        raise ValueError("temporal distance mismatch")
    if estimate_tokens("\n".join(turn.text for turn in world.turns)) >= max_tokens:
        raise ValueError("world exceeds token ceiling")
    turn_ids = {turn.turn_id for turn in world.turns}
    if len(turn_ids) != len(world.turns):
        raise ValueError("duplicate turn IDs")
    for task in world.tasks:
        if any(turn_id not in turn_ids for turn_id in task.evidence_turn_ids):
            raise ValueError(f"bad evidence lineage for {task.task_id}")
        answer = task.answer.casefold()
        if answer not in {"unknown", "forgotten", "tool"} and answer in task.query.casefold():
            raise ValueError(f"query leaks answer for {task.task_id}")
    interference_text = "\n".join(t.text for t in world.turns if t.kind == "interference")
    for task in world.tasks:
        if task.answer not in {"UNKNOWN", "FORGOTTEN", "TOOL"} and task.answer in interference_text:
            raise ValueError("gold answer collides with distractor")


def save_worlds(path: Path, worlds: list[World]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for world in worlds:
            handle.write(json.dumps(world.to_dict(), sort_keys=True) + "\n")


def load_worlds(path: Path) -> list[World]:
    with path.open(encoding="utf-8") as handle:
        return [World.from_dict(json.loads(line)) for line in handle if line.strip()]
