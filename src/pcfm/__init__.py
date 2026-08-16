"""PCFM 对话式人物模拟（网页产品）。研究阶段旧模块已隔离到 legacy/。"""

from .applicability import ApplicabilityProfile, PredictionRefusedError, TemporalStability
from .contracts import EvaluationReport, Observation, PersonRepresentation, Prediction, Scenario
from .core import (
    DecisionIntegrator,
    IdentityAdapterGenerator,
    MapPersonEncoder,
    ModelUpdater,
    PopulationPriorEstimator,
)
from .decision_evidence_v1 import (
    DecisionEvidenceBundle,
    DecisionEvidenceConfig,
    DecisionEvidenceRecord,
    DecisionEvidenceRefusedError,
    DecisionEvidenceSummary,
    DecisionOption,
    EvidenceCitation,
    SourceSnapshot,
    create_decision_evidence_bundle,
    load_decision_evidence_bundle,
    save_decision_evidence_bundle,
    validate_decision_evidence_bundle,
)
from .ledger import EventLedger, EventRecord, VerificationAuthority
from .storage import ModelManifest, ModelValidation, PersonModelBundle

__all__ = [
    "ApplicabilityProfile",
    "PredictionRefusedError",
    "TemporalStability",
    "EvaluationReport",
    "Observation",
    "PersonRepresentation",
    "Prediction",
    "Scenario",
    "DecisionIntegrator",
    "IdentityAdapterGenerator",
    "MapPersonEncoder",
    "ModelUpdater",
    "PopulationPriorEstimator",
    "DecisionEvidenceBundle",
    "DecisionEvidenceConfig",
    "DecisionEvidenceRecord",
    "DecisionEvidenceRefusedError",
    "DecisionEvidenceSummary",
    "DecisionOption",
    "EvidenceCitation",
    "SourceSnapshot",
    "create_decision_evidence_bundle",
    "load_decision_evidence_bundle",
    "save_decision_evidence_bundle",
    "validate_decision_evidence_bundle",
    "EventLedger",
    "EventRecord",
    "VerificationAuthority",
    "ModelManifest",
    "ModelValidation",
    "PersonModelBundle",
]
