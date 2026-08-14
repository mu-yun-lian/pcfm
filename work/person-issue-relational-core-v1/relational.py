from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from math import isfinite
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

import numpy as np


_KERNEL_PATH = Path(__file__).parents[1] / "joint-person-core-v1" / "candidate.py"
_KERNEL_SPEC = importlib.util.spec_from_file_location(
    "person_issue_shared_joint_kernel", _KERNEL_PATH
)
if _KERNEL_SPEC is None or _KERNEL_SPEC.loader is None:
    raise RuntimeError("could not load the shared joint candidate kernel")
kernel = importlib.util.module_from_spec(_KERNEL_SPEC)
sys.modules[_KERNEL_SPEC.name] = kernel
_KERNEL_SPEC.loader.exec_module(kernel)


StateConfig = kernel.StateConfig
PrequentialStateResult = kernel.PrequentialStateResult
negative_log_likelihood = kernel.negative_log_likelihood

MOTION_TYPES = ("nomination", "cloture", "proceed", "amendment", "other")
BILL_TYPES = ("nomination", "bill", "resolution", "amendment", "other")
ALLOWED_FIELDS = (
    "vote_desc",
    "vote_question",
    "dtl_desc",
    "crs_policy_area",
    "crs_subjects",
    "bill_number_prefix",
)
FORBIDDEN_FIELDS = (
    "cast_code",
    "prob",
    "yea_count",
    "nay_count",
    "vote_result",
    "nominate_mid_1",
    "nominate_mid_2",
    "nominate_spread_1",
    "nominate_spread_2",
    "nominate_log_likelihood",
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MAJORITY_PARTY_BY_CONGRESS = {116: 200, 117: 100, 118: 100, 119: 200}
PRESIDENT_PARTY_BY_CONGRESS = {116: 200, 117: 100, 118: 100, 119: 200}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _motion_type(value: object) -> str:
    text = str(value or "")
    if re.search(r"nomination", text, re.IGNORECASE):
        return "nomination"
    if re.search(r"cloture", text, re.IGNORECASE):
        return "cloture"
    if re.search(r"proceed", text, re.IGNORECASE):
        return "proceed"
    if re.search(r"amendment", text, re.IGNORECASE):
        return "amendment"
    return "other"


def _bill_prefix(value: object) -> str:
    compact = re.sub(r"[^A-Za-z]", "", str(value or "")).upper()
    return compact[:12]


def _bill_type(value: object) -> str:
    prefix = _bill_prefix(value)
    if prefix.startswith("PN"):
        return "nomination"
    if "AMD" in prefix:
        return "amendment"
    if "RES" in prefix:
        return "resolution"
    if prefix.startswith(("S", "HR", "H", "LAW")):
        return "bill"
    return "other"


def _field_tokens(prefix: str, value: object) -> list[str]:
    words = TOKEN_PATTERN.findall(str(value or "").lower())
    tokens = [f"{prefix}:u:{word}" for word in words]
    tokens.extend(
        f"{prefix}:b:{first}_{second}" for first, second in zip(words, words[1:])
    )
    return tokens


def declared_tokens(rollcall: Mapping[str, object]) -> tuple[str, ...]:
    tokens: list[str] = []
    tokens.extend(_field_tokens("desc", rollcall.get("vote_desc")))
    tokens.extend(_field_tokens("question", rollcall.get("vote_question")))
    tokens.extend(_field_tokens("detail", rollcall.get("dtl_desc")))
    tokens.extend(_field_tokens("policy", rollcall.get("crs_policy_area")))
    subjects = rollcall.get("crs_subjects") or []
    if not isinstance(subjects, (list, tuple)):
        raise ValueError("crs_subjects must be a list when supplied")
    for subject in sorted(str(value) for value in subjects):
        tokens.extend(_field_tokens("subject", subject))
    prefix = _bill_prefix(rollcall.get("bill_number"))
    if prefix:
        tokens.append(f"bill_prefix:{prefix}")
    return tuple(sorted(tokens))


def _hashed_vector(tokens: Sequence[str], dimension: int) -> np.ndarray:
    if dimension <= 0:
        raise ValueError("hash dimension must be positive")
    vector = np.zeros(dimension, dtype=np.float64)
    for token in tokens:
        digest = hashlib.sha256(str(token).encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def _canonical_rollcall_content(rollcall: Mapping[str, object]) -> dict[str, object]:
    return {
        "tokens": list(declared_tokens(rollcall)),
        "motion_type": _motion_type(rollcall.get("vote_question")),
        "bill_type": _bill_type(rollcall.get("bill_number")),
    }


@dataclass(frozen=True)
class RelationalFeatureMap:
    hash_dimension: int
    svd_rank: int
    components: tuple[tuple[float, ...], ...]
    map_version: str = "person-issue-relational-feature-map-v1"
    map_id: str = ""

    def __post_init__(self) -> None:
        matrix = np.asarray(self.components, dtype=np.float64)
        if (
            self.hash_dimension <= 0
            or self.svd_rank <= 0
            or matrix.shape != (self.svd_rank, self.hash_dimension)
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("relational feature-map dimensions are invalid")
        gram = matrix @ matrix.T
        if not np.allclose(gram, np.eye(self.svd_rank), atol=1e-8):
            raise ValueError("relational SVD components must be orthonormal")
        expected = _digest(self._unsigned_dict())
        if self.map_id and self.map_id != expected:
            raise ValueError("relational feature-map identity mismatch")
        object.__setattr__(self, "map_id", expected)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "intercept",
            *(f"motion:{value}" for value in MOTION_TYPES),
            *(f"bill_type:{value}" for value in BILL_TYPES),
            *(f"text_factor:{index}" for index in range(self.svd_rank)),
        )

    @classmethod
    def fit(
        cls,
        rollcalls: Sequence[Mapping[str, object]],
        *,
        hash_dimension: int = 128,
        svd_rank: int = 8,
    ) -> RelationalFeatureMap:
        unique: dict[str, Mapping[str, object]] = {}
        for rollcall in rollcalls:
            content = _canonical_rollcall_content(rollcall)
            unique.setdefault(_digest(content), rollcall)
        if len(unique) < svd_rank:
            raise ValueError("too few distinct roll-call descriptions for SVD rank")
        matrix = np.asarray(
            [
                _hashed_vector(declared_tokens(unique[key]), hash_dimension)
                for key in sorted(unique)
            ],
            dtype=np.float64,
        )
        _u, _singular, vh = np.linalg.svd(matrix, full_matrices=False)
        components = np.asarray(vh[:svd_rank], dtype=np.float64)
        for row in components:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0.0:
                row *= -1.0
        return cls(
            hash_dimension=hash_dimension,
            svd_rank=svd_rank,
            components=tuple(tuple(float(value) for value in row) for row in components),
        )

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "hash_dimension": self.hash_dimension,
            "svd_rank": self.svd_rank,
            "components": [list(row) for row in self.components],
            "map_version": self.map_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_dict(), "map_id": self.map_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RelationalFeatureMap:
        return cls(
            hash_dimension=int(value["hash_dimension"]),
            svd_rank=int(value["svd_rank"]),
            components=tuple(
                tuple(float(item) for item in row) for row in value["components"]
            ),
            map_version=str(value["map_version"]),
            map_id=str(value["map_id"]),
        )

    def transform(self, rollcall: Mapping[str, object]) -> np.ndarray:
        raw = _hashed_vector(declared_tokens(rollcall), self.hash_dimension)
        latent = raw @ np.asarray(self.components, dtype=np.float64).T
        motion = _motion_type(rollcall.get("vote_question"))
        bill = _bill_type(rollcall.get("bill_number"))
        return np.asarray(
            [
                1.0,
                *(float(motion == value) for value in MOTION_TYPES),
                *(float(bill == value) for value in BILL_TYPES),
                *latent,
            ],
            dtype=np.float64,
        )

    def transform_many(self, rollcalls: Sequence[Mapping[str, object]]) -> np.ndarray:
        return np.asarray([self.transform(row) for row in rollcalls], dtype=np.float64)


def _broadcast_int(value: int | Sequence[int], count: int, name: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)):
        result = np.full(count, int(value), dtype=np.int64)
    else:
        result = np.asarray(value, dtype=np.int64)
    if result.shape != (count,):
        raise ValueError(f"{name} must be a scalar or one value per row")
    return result


def environment_feature_names(feature_map: RelationalFeatureMap) -> tuple[str, ...]:
    return (
        *(f"republican_x:{name}" for name in feature_map.feature_names),
        "party_matches_majority",
        "party_matches_president",
    )


def environment_features(
    scenario: np.ndarray,
    party_codes: int | Sequence[int],
    congresses: int | Sequence[int],
    *,
    majority_party_by_congress: Mapping[int, int] = MAJORITY_PARTY_BY_CONGRESS,
    president_party_by_congress: Mapping[int, int] = PRESIDENT_PARTY_BY_CONGRESS,
) -> np.ndarray:
    matrix = np.asarray(scenario, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("scenario features must be a finite matrix")
    count = matrix.shape[0]
    parties = _broadcast_int(party_codes, count, "party_codes")
    periods = _broadcast_int(congresses, count, "congresses")
    if not np.all(np.isin(parties, (100, 200))):
        raise ValueError("only frozen Democratic and Republican party codes are supported")
    if any(int(value) not in majority_party_by_congress for value in periods):
        raise ValueError("unknown Congress in majority-party mapping")
    republican = (parties == 200).astype(np.float64)
    majority = np.asarray(
        [float(parties[index] == majority_party_by_congress[int(periods[index])]) for index in range(count)]
    )
    president = np.asarray(
        [float(parties[index] == president_party_by_congress[int(periods[index])]) for index in range(count)]
    )
    return np.concatenate(
        (matrix * republican[:, None], majority[:, None], president[:, None]), axis=1
    )


@dataclass(frozen=True)
class RelationalCoreArtifact:
    feature_map: RelationalFeatureMap
    joint_model: object
    artifact_version: str = "person-issue-relational-core-artifact-v1"
    artifact_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.joint_model, kernel.JointCoreModel):
            raise ValueError("relational artifact requires the shared joint model")
        if tuple(self.joint_model.scenario_feature_names) != self.feature_map.feature_names:
            raise ValueError("relational scenario schema does not match feature map")
        if tuple(self.joint_model.environment_feature_names) != environment_feature_names(
            self.feature_map
        ):
            raise ValueError("relational environment schema does not match feature map")
        expected = _digest(self._unsigned_dict())
        if self.artifact_id and self.artifact_id != expected:
            raise ValueError("relational artifact identity mismatch")
        object.__setattr__(self, "artifact_id", expected)

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "feature_map": self.feature_map.to_dict(),
            "joint_model": self.joint_model.to_dict(),
            "artifact_version": self.artifact_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_dict(), "artifact_id": self.artifact_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RelationalCoreArtifact:
        if value.get("artifact_version") != "person-issue-relational-core-artifact-v1":
            raise ValueError("unsupported relational artifact version")
        return cls(
            feature_map=RelationalFeatureMap.from_dict(value["feature_map"]),
            joint_model=kernel.JointCoreModel.from_dict(value["joint_model"]),
            artifact_version=str(value["artifact_version"]),
            artifact_id=str(value["artifact_id"]),
        )

    def _matrices(
        self,
        rollcalls: Sequence[Mapping[str, object]],
        party_codes: int | Sequence[int],
        congresses: int | Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        scenario = self.feature_map.transform_many(rollcalls)
        environment = environment_features(scenario, party_codes, congresses)
        return scenario, environment

    def probabilities(
        self,
        person_id: str,
        rollcalls: Sequence[Mapping[str, object]],
        party_codes: int | Sequence[int],
        congresses: int | Sequence[int],
        *,
        profile_person_id: str | None = None,
        include_person: bool = True,
    ) -> np.ndarray:
        scenario, environment = self._matrices(rollcalls, party_codes, congresses)
        return self.joint_model.probabilities(
            profile_person_id or person_id,
            scenario,
            environment,
            include_person=include_person,
        )

    def logits_and_variances(
        self,
        person_id: str,
        rollcalls: Sequence[Mapping[str, object]],
        party_codes: int | Sequence[int],
        congresses: int | Sequence[int],
        *,
        profile_person_id: str | None = None,
        include_person: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        scenario, environment = self._matrices(rollcalls, party_codes, congresses)
        return self.joint_model.logits_and_variances(
            profile_person_id or person_id,
            scenario,
            environment,
            include_person=include_person,
        )

    def run_prequential(
        self,
        person_id: str,
        rollcalls: Sequence[Mapping[str, object]],
        party_codes: int | Sequence[int],
        congresses: int | Sequence[int],
        choices: Sequence[int],
        timestamps: Sequence[str],
        config: object,
        *,
        profile_person_id: str | None = None,
        include_person: bool = True,
        initial_state_mean: float = 0.0,
        initial_state_variance: float | None = None,
        previous_timestamp: str | None = None,
    ) -> object:
        logits, variances = self.logits_and_variances(
            person_id,
            rollcalls,
            party_codes,
            congresses,
            profile_person_id=profile_person_id,
            include_person=include_person,
        )
        return kernel.run_prequential_state(
            logits,
            variances,
            choices,
            timestamps,
            config,
            initial_state_mean=initial_state_mean,
            initial_state_variance=initial_state_variance,
            previous_timestamp=previous_timestamp,
        )


def fit_relational_artifact(
    feature_map: RelationalFeatureMap,
    rollcalls: Sequence[Mapping[str, object]],
    choices: Sequence[int],
    person_ids: Sequence[str],
    party_codes: Sequence[int],
    congresses: Sequence[int],
    *,
    stable_person_precision: float,
    global_l2_precision: float = 1.0,
    coordinate_passes: int = 6,
) -> RelationalCoreArtifact:
    count = len(rollcalls)
    if not (
        len(choices) == len(person_ids) == len(party_codes) == len(congresses) == count
    ):
        raise ValueError("relational fitting inputs are misaligned")
    scenario = feature_map.transform_many(rollcalls)
    environment = environment_features(scenario, party_codes, congresses)
    model = kernel.fit_joint_core(
        scenario,
        environment,
        choices,
        person_ids,
        scenario_feature_names=feature_map.feature_names,
        environment_feature_names=environment_feature_names(feature_map),
        stable_person_precision=stable_person_precision,
        global_l2_precision=global_l2_precision,
        coordinate_passes=coordinate_passes,
    )
    return RelationalCoreArtifact(feature_map=feature_map, joint_model=model)


def refit_profiles_with_fixed_global(
    artifact: RelationalCoreArtifact,
    rollcalls: Sequence[Mapping[str, object]],
    choices: Sequence[int],
    person_ids: Sequence[str],
    party_codes: Sequence[int],
    congresses: Sequence[int],
) -> RelationalCoreArtifact:
    count = len(rollcalls)
    if not (
        len(choices) == len(person_ids) == len(party_codes) == len(congresses) == count
    ):
        raise ValueError("profile refit inputs are misaligned")
    scenario, environment = artifact._matrices(rollcalls, party_codes, congresses)
    design = np.concatenate((scenario, environment), axis=1)
    global_offsets = design @ np.asarray(artifact.joint_model.global_weights)
    y = np.asarray(choices, dtype=np.float64)
    people = tuple(str(value) for value in person_ids)
    weights: dict[str, tuple[float, ...]] = {}
    covariances: dict[str, tuple[tuple[float, ...], ...]] = {}
    for person_id in sorted(set(people)):
        mask = np.asarray([value == person_id for value in people])
        fitted, covariance = kernel._fit_ridge_logistic(
            scenario[mask],
            y[mask],
            global_offsets[mask],
            artifact.joint_model.stable_person_precision,
        )
        weights[person_id] = tuple(float(value) for value in fitted)
        covariances[person_id] = tuple(
            tuple(float(value) for value in row) for row in covariance
        )
    model = kernel.JointCoreModel(
        scenario_feature_names=artifact.joint_model.scenario_feature_names,
        environment_feature_names=artifact.joint_model.environment_feature_names,
        global_weights=artifact.joint_model.global_weights,
        global_covariance=artifact.joint_model.global_covariance,
        person_weights=weights,
        person_covariances=covariances,
        stable_person_precision=artifact.joint_model.stable_person_precision,
        model_version=artifact.joint_model.model_version,
    )
    return RelationalCoreArtifact(feature_map=artifact.feature_map, joint_model=model)


def validate_configuration(stable_person_precision: float) -> None:
    if not isfinite(stable_person_precision) or stable_person_precision <= 0.0:
        raise ValueError("stable person precision must be finite and positive")
