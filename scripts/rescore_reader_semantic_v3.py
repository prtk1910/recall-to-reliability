from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path


INPUT_DB = Path("artifacts/ox_zen_replay/semantic_rescore.sqlite")
OUTPUT_DB = Path("artifacts/ox_zen_replay/semantic_rescore_v3.sqlite")
AUDIT_CSV = Path("artifacts/ox_zen_replay/semantic_rescued_v3.csv")


def norm(x: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (x or "").casefold()).strip()


def phrase(text: str, value: str) -> bool:
    t = f" {norm(text)} "
    v = f" {norm(value)} "
    return v in t


def regex_gold(gold: str) -> str:
    return re.escape(norm(gold))


def not_negated(text: str, gold: str) -> bool:
    tokens = norm(text).split()
    g = norm(gold).split()

    for i in range(len(tokens) - len(g) + 1):
        if tokens[i:i + len(g)] == g:
            before = tokens[max(0, i - 4):i]
            after = tokens[i + len(g):i + len(g) + 4]

            if "not" in before or "never" in before or "not" in after:
                continue
            return True

    return False


def atomic_recall(answer: str, gold: str) -> bool:
    a = norm(answer)
    g = regex_gold(gold)

    patterns = [
        rf"(?:permanent\s+)?call\s+sign(?:\s+is|\s*:)?\s+{g}(?:\s|$)",
        rf"(?:permanent\s+)?callsign(?:\s+is|\s*:)?\s+{g}(?:\s|$)",
    ]
    return any(re.search(p, a) for p in patterns)


def coexisting(answer: str, gold: str) -> bool:
    parts = [norm(x) for x in gold.split("|")]

    if len(parts) != 2:
        return False

    first, second = parts
    a = norm(answer)

    p1 = a.find(first)
    p2 = a.find(second)

    if p1 < 0 or p2 < 0 or p1 >= p2:
        return False

    first_context = a[max(0, p1 - 60):p1]
    second_context = a[max(0, p2 - 60):p2]

    return (
        "primary" in first_context
        and "secondary" in second_context
    )


def conditional_fact(answer: str, gold: str) -> bool:
    a = norm(answer)
    g = regex_gold(gold)

    patterns = [
        rf"(?:routed|route|routing)(?:\s+\w+){{0,6}}\s+(?:to\s+)?{g}(?:\s|$)",
        rf"reports(?:\s+\w+){{0,8}}\s+{g}(?:\s|$)",
    ]

    return (
        phrase(answer, gold)
        and not_negated(answer, gold)
        and any(re.search(p, a) for p in patterns)
    )


def current_state(answer: str, gold: str) -> bool:
    a = norm(answer)
    g = regex_gold(gold)

    patterns = [
        rf"current(?:\s+\w+){{0,4}}\s+region(?:\s+is|\s*:)?\s+{g}(?:\s|$)",
        rf"current\s+deployment\s+region(?:\s+is|\s*:)?\s+{g}(?:\s|$)",
        rf"deployment\s+region\s+is\s+{g}(?:\s|$)",
    ]

    if not any(re.search(p, a) for p in patterns):
        return False

    # Avoid accepting a clearly historical statement.
    historical = [
        rf"(?:first|initial|original)(?:\s+\w+){{0,5}}\s+{g}",
        rf"{g}(?:\s+\w+){{0,5}}\s+(?:first|initial|original)",
    ]

    return not any(re.search(p, a) for p in historical)


def historical_value(answer: str, gold: str) -> bool:
    a = norm(answer)
    g = regex_gold(gold)

    patterns = [
        rf"(?:first|initial|original)(?:\s+\w+){{0,8}}\s+{g}(?:\s|$)",
        rf"{g}(?:\s+\w+){{0,8}}\s+(?:first|initial|original)(?:\s|$)",
        rf"{g}(?:\s+\w+){{0,6}}\s+first\s+recorded",
    ]

    return any(re.search(p, a) for p in patterns)


def temporal_ordering(answer: str, gold: str) -> bool:
    a = norm(answer)
    g = regex_gold(gold)

    if a == norm(gold):
        return True

    patterns = [
        rf"(?:^|\s){g}(?:\s+\w+){{0,10}}\s+(?:first|earlier|before)(?:\s|$)",
        rf"(?:^|\s)(?:first|earlier)(?:\s+\w+){{0,10}}\s+{g}(?:\s|$)",
    ]

    return any(re.search(p, a) for p in patterns)


def two_hop(answer: str, gold: str) -> bool:
    a = norm(answer)
    g = regex_gold(gold)

    patterns = [
        rf"(?:assigned\s+)?desk(?:\s+is|\s*:)?\s+{g}(?:\s|$)",
        rf"assigned(?:\s+\w+){{0,4}}\s+{g}(?:\s|$)",
    ]

    return any(re.search(p, a) for p in patterns)


def four_hop(answer: str, gold: str) -> bool:
    a = norm(answer)
    g = regex_gold(gold)

    patterns = [
        rf"token(?:\s+is|\s*:)?\s+{g}(?:\s|$)",
        rf"(?:carries|carry|carrying)(?:\s+the)?\s+token\s+{g}(?:\s|$)",
    ]

    return any(re.search(p, a) for p in patterns)


def semantic_score(
    task_type: str,
    status: str,
    answer: str,
    gold: str,
    exact: int,
) -> tuple[int, str]:

    # Never remove an original exact success.
    if exact:
        return 1, "original_exact_success"

    # Policy/tool tasks remain scored by the original protocol.
    if task_type in {
        "abstention",
        "selective_forgetting",
        "tool_argument_grounding",
        "tool_selection",
    }:
        return 0, "keep_exact_protocol"

    # A normal memory question must actually be answered.
    if status != "answered":
        return 0, f"status_{status}"

    scorers = {
        "atomic_recall": atomic_recall,
        "coexisting_facts": coexisting,
        "conditional_fact": conditional_fact,
        "current_state": current_state,
        "historical_value": historical_value,
        "temporal_ordering": temporal_ordering,
        "two_hop_reasoning": two_hop,
        "four_hop_reasoning": four_hop,
    }

    scorer = scorers.get(task_type)

    if scorer is None:
        return 0, "unknown_task_keep_exact"

    ok = scorer(answer, gold)

    return int(ok), (
        f"{task_type}_semantic_match"
        if ok
        else "not_rescued"
    )


def main():
    if not INPUT_DB.exists():
        raise SystemExit(f"Missing {INPUT_DB}")

    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    src = sqlite3.connect(INPUT_DB)
    src.row_factory = sqlite3.Row

    rows = src.execute("""
        SELECT *
        FROM semantic_results
        ORDER BY world_id, task_id, architecture
    """).fetchall()

    out = sqlite3.connect(OUTPUT_DB)

    out.execute("""
        CREATE TABLE semantic_results_v3 (
            original_result_id TEXT PRIMARY KEY,
            world_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            architecture TEXT NOT NULL,
            query TEXT NOT NULL,
            gold_answer TEXT NOT NULL,

            luna_status TEXT NOT NULL,
            luna_answer TEXT NOT NULL,
            luna_exact INTEGER NOT NULL,
            luna_semantic_v3 INTEGER NOT NULL,
            luna_reason_v3 TEXT NOT NULL,

            ox_status TEXT NOT NULL,
            ox_answer TEXT NOT NULL,
            ox_exact INTEGER NOT NULL,
            ox_semantic_v3 INTEGER NOT NULL,
            ox_reason_v3 TEXT NOT NULL
        )
    """)

    rescued = []

    for r in rows:
        luna_sem, luna_reason = semantic_score(
            r["task_type"],
            r["luna_status"],
            r["luna_answer"],
            r["gold_answer"],
            int(r["luna_exact"]),
        )

        ox_sem, ox_reason = semantic_score(
            r["task_type"],
            r["ox_status"],
            r["ox_answer"],
            r["gold_answer"],
            int(r["ox_exact"]),
        )

        vals = (
            r["original_result_id"],
            r["world_id"],
            r["task_id"],
            r["task_type"],
            r["architecture"],
            r["query"],
            r["gold_answer"],

            r["luna_status"],
            r["luna_answer"],
            r["luna_exact"],
            luna_sem,
            luna_reason,

            r["ox_status"],
            r["ox_answer"],
            r["ox_exact"],
            ox_sem,
            ox_reason,
        )

        out.execute("""
            INSERT INTO semantic_results_v3 VALUES (
                ?,?,?,?,?,?,?,
                ?,?,?,?,?,
                ?,?,?,?,?
            )
        """, vals)

        if (
            (not r["luna_exact"] and luna_sem)
            or
            (not r["ox_exact"] and ox_sem)
        ):
            rescued.append(vals)

    out.commit()

    print("\nOVERALL")
    print(out.execute("""
        SELECT
            COUNT(*) AS n,
            ROUND(100.0 * AVG(luna_exact), 2),
            ROUND(100.0 * AVG(luna_semantic_v3), 2),
            ROUND(100.0 * AVG(ox_exact), 2),
            ROUND(100.0 * AVG(ox_semantic_v3), 2)
        FROM semantic_results_v3
    """).fetchone())

    print("\nBY ARCHITECTURE")
    for row in out.execute("""
        SELECT
            architecture,
            COUNT(*),
            ROUND(100.0 * AVG(luna_semantic_v3), 1),
            ROUND(100.0 * AVG(ox_semantic_v3), 1),
            ROUND(
                100.0 * (
                    AVG(ox_semantic_v3) -
                    AVG(luna_semantic_v3)
                ), 1
            )
        FROM semantic_results_v3
        GROUP BY architecture
        ORDER BY AVG(ox_semantic_v3) DESC
    """):
        print(row)

    print("\nBY TASK")
    for row in out.execute("""
        SELECT
            task_type,
            COUNT(*),
            ROUND(100.0 * AVG(luna_semantic_v3), 1),
            ROUND(100.0 * AVG(ox_semantic_v3), 1),
            ROUND(
                100.0 * (
                    AVG(ox_semantic_v3) -
                    AVG(luna_semantic_v3)
                ), 1
            )
        FROM semantic_results_v3
        GROUP BY task_type
        ORDER BY task_type
    """):
        print(row)

    print("\nPAIRED")
    for row in out.execute("""
        SELECT
            CASE
                WHEN luna_semantic_v3=1 AND ox_semantic_v3=1
                    THEN 'Both correct'
                WHEN luna_semantic_v3=1 AND ox_semantic_v3=0
                    THEN 'Luna only'
                WHEN luna_semantic_v3=0 AND ox_semantic_v3=1
                    THEN 'Ox only'
                ELSE 'Both wrong'
            END,
            COUNT(*)
        FROM semantic_results_v3
        GROUP BY 1
        ORDER BY 2 DESC
    """):
        print(row)

    headers = [d[1] for d in out.execute(
        "PRAGMA table_info(semantic_results_v3)"
    ).fetchall()]

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rescued)

    print(f"\nRescued cases: {len(rescued)}")
    print(f"Database: {OUTPUT_DB}")
    print(f"Audit CSV: {AUDIT_CSV}")

    src.close()
    out.close()


if __name__ == "__main__":
    main()
