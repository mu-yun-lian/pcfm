from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("run_bridge.py")
SPEC = importlib.util.spec_from_file_location("reality_bridge_v1", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load reality bridge harness")
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class RealityBridgeHarnessTests(unittest.TestCase):
    def test_cohort_selection_is_input_order_invariant(self) -> None:
        rows = [
            {"person_id": str(index), "party_code": party}
            for party in (100, 200)
            for index in range(party, party + 8)
        ]
        forward = bridge.select_cohort(rows, "a" * 64)
        reverse = bridge.select_cohort(tuple(reversed(rows)), "a" * 64)
        self.assertEqual(forward, reverse)
        self.assertEqual(
            [row["party_code"] for row in forward],
            [100, 100, 100, 200, 200, 200],
        )

    def test_role_allocation_is_complete_and_disjoint(self) -> None:
        rows = [
            {"congress": "119", "rollnumber": str(index)}
            for index in range(1, 501)
        ]
        roles = bridge.allocate_roles(tuple(reversed(rows)))
        self.assertEqual(
            {name: len(values) for name, values in roles.items()},
            bridge.ROLE_COUNTS,
        )
        allocated = [
            row["rollnumber"]
            for role_rows in roles.values()
            for row in role_rows
        ]
        self.assertEqual(len(allocated), len(set(allocated)))
        self.assertEqual(allocated[0], "1")
        self.assertEqual(allocated[-1], "480")

    def test_rollnumber_encoding_is_strictly_ordered_within_day(self) -> None:
        earlier = bridge.ordered_timestamp(
            {"date": "2026-01-02", "rollnumber": 18}
        )
        later = bridge.ordered_timestamp(
            {"date": "2026-01-02", "rollnumber": 19}
        )
        self.assertLess(bridge.parse_timestamp(earlier), bridge.parse_timestamp(later))


if __name__ == "__main__":
    unittest.main()
