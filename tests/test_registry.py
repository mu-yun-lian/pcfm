from __future__ import annotations

import unittest

from pcfm.interfaces import CognitiveModule
from pcfm.registry import ModuleRegistry, ModuleSlot


class DummyMemory(CognitiveModule):
    module_id = "dummy-memory"
    module_version = "0.0"

    def required_inputs(self) -> tuple[str, ...]:
        return ("person_id", "scenario")

    def diagnostics(self) -> dict[str, object]:
        return {"status": "test-only"}


class RegistryTests(unittest.TestCase):
    def test_unimplemented_slots_are_reserved(self) -> None:
        registry = ModuleRegistry()
        manifest = {item.slot: item for item in registry.manifest()}
        self.assertEqual(manifest[ModuleSlot.MEMORY].status, "reserved")
        self.assertIsNone(manifest[ModuleSlot.MEMORY].module_id)
        for slot in (
            ModuleSlot.BELIEF_MODEL,
            ModuleSlot.VALUE_MODEL,
            ModuleSlot.GOAL_MODEL,
            ModuleSlot.CONCEPT_MODEL,
            ModuleSlot.SOCIAL_MODEL,
            ModuleSlot.SELF_MODEL,
        ):
            self.assertEqual(manifest[slot].status, "reserved")

    def test_module_can_fill_reserved_slot(self) -> None:
        registry = ModuleRegistry()
        registry.register(ModuleSlot.MEMORY, DummyMemory())
        manifest = {item.slot: item for item in registry.manifest()}
        self.assertEqual(manifest[ModuleSlot.MEMORY].status, "implemented")
        self.assertEqual(manifest[ModuleSlot.MEMORY].module_id, "dummy-memory")

    def test_slot_cannot_be_overwritten_silently(self) -> None:
        registry = ModuleRegistry()
        registry.register(ModuleSlot.MEMORY, DummyMemory())
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(ModuleSlot.MEMORY, DummyMemory())

    def test_non_plugin_core_implementation_can_be_declared(self) -> None:
        registry = ModuleRegistry()
        registry.register_implementation(
            ModuleSlot.EVALUATOR,
            "behavioral-evaluator",
            "v1",
        )
        manifest = {item.slot: item for item in registry.manifest()}
        self.assertEqual(manifest[ModuleSlot.EVALUATOR].status, "implemented")
        self.assertEqual(
            manifest[ModuleSlot.EVALUATOR].module_id,
            "behavioral-evaluator",
        )
        self.assertIsNone(registry.get(ModuleSlot.EVALUATOR))


if __name__ == "__main__":
    unittest.main()
