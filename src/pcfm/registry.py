from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .interfaces import CognitiveModule


class ModuleSlot(StrEnum):
    POPULATION_PRIOR = "population_prior"
    PERSON_ENCODER = "person_encoder"
    ADAPTER_GENERATOR = "adapter_generator"
    DECISION_INTEGRATOR = "decision_integrator"
    EVALUATOR = "evaluator"
    UPDATER = "updater"
    UNCERTAINTY = "uncertainty"
    APPLICABILITY_GUARD = "applicability_guard"
    TEMPORAL_DRIFT = "temporal_drift"
    ACTIVE_EXPERIMENT = "active_experiment"
    DYNAMIC_STATE = "dynamic_state"
    MEMORY = "memory"
    CAUSAL_MODEL = "causal_model"
    MECHANISM_DISTILLER = "mechanism_distiller"
    COMPOSITE_MODEL = "composite_model"
    LANGUAGE_INTERFACE = "language_interface"
    STYLE_RENDERER = "style_renderer"
    MULTIMODAL = "multimodal"
    BELIEF_MODEL = "belief_model"
    VALUE_MODEL = "value_model"
    GOAL_MODEL = "goal_model"
    CONCEPT_MODEL = "concept_model"
    SOCIAL_MODEL = "social_model"
    SELF_MODEL = "self_model"


@dataclass(frozen=True)
class SlotManifest:
    slot: ModuleSlot
    status: str
    module_id: str | None
    module_version: str | None


@dataclass(frozen=True)
class _RegistryEntry:
    module_id: str
    module_version: str
    instance: CognitiveModule | None = None


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[ModuleSlot, _RegistryEntry] = {}

    def register(self, slot: ModuleSlot, module: CognitiveModule) -> None:
        self.register_implementation(
            slot,
            module.module_id,
            module.module_version,
            instance=module,
        )

    def register_implementation(
        self,
        slot: ModuleSlot,
        module_id: str,
        module_version: str,
        *,
        instance: CognitiveModule | None = None,
    ) -> None:
        if slot in self._modules:
            raise ValueError(f"slot {slot} is already registered")
        if not module_id or not module_version:
            raise ValueError("module_id and module_version are required")
        self._modules[slot] = _RegistryEntry(
            module_id=module_id,
            module_version=module_version,
            instance=instance,
        )

    def get(self, slot: ModuleSlot) -> CognitiveModule | None:
        entry = self._modules.get(slot)
        return entry.instance if entry else None

    def manifest(self) -> tuple[SlotManifest, ...]:
        return tuple(
            SlotManifest(
                slot=slot,
                status="implemented" if slot in self._modules else "reserved",
                module_id=(
                    self._modules[slot].module_id
                    if slot in self._modules
                    else None
                ),
                module_version=(
                    self._modules[slot].module_version
                    if slot in self._modules
                    else None
                ),
            )
            for slot in ModuleSlot
        )
