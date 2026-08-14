from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcfm.conversation_mvp import _extract_qa, _segments  # noqa: E402
from pcfm.simulation_v3 import (  # noqa: E402
    SimulationKernelV3,
    SimulationV3Error,
)


OUTPUT = ROOT / "artifacts" / "simulation_v3_full_chain_audit"
RAW = OUTPUT / "raw"


SOURCES = (
    {
        "source_id": "obama-2009-02-09-first-press-conference",
        "source_date": "2009-02-09",
        "role": "parameter_fitting",
        "url": "https://obamawhitehouse.archives.gov/video/EVR020909",
    },
    {
        "source_id": "obama-2010-11-03-press-conference",
        "source_date": "2010-11-03",
        "role": "parameter_fitting",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2010/11/03/press-conference-president/",
    },
    {
        "source_id": "obama-2011-02-15-press-conference",
        "source_date": "2011-02-15",
        "role": "parameter_fitting",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2011/02/15/press-conference-president/",
    },
    {
        "source_id": "obama-2011-07-15-press-conference",
        "source_date": "2011-07-15",
        "role": "parameter_fitting",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2011/07/15/press-conference-president",
    },
    {
        "source_id": "obama-2012-03-06-press-conference",
        "source_date": "2012-03-06",
        "role": "parameter_fitting",
        "url": "https://obamawhitehouse.archives.gov/realitycheck/the-press-office/2012/03/06/press-conference-president",
    },
    {
        "source_id": "obama-2013-10-08-press-conference",
        "source_date": "2013-10-08",
        "role": "parameter_fitting",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2013/10/08/press-conference-president/",
    },
    {
        "source_id": "obama-2014-08-01-press-conference",
        "source_date": "2014-08-01",
        "role": "parameter_fitting",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2014/08/01/press-conference-president",
    },
    {
        "source_id": "obama-2015-07-15-press-conference",
        "source_date": "2015-07-15",
        "role": "applicability_calibration",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2015/07/15/press-conference-president/",
    },
    {
        "source_id": "obama-2015-10-02-press-conference",
        "source_date": "2015-10-02",
        "role": "applicability_calibration",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2015/10/02/press-conference-president",
    },
    {
        "source_id": "obama-2015-12-01-press-conference",
        "source_date": "2015-12-01",
        "role": "applicability_calibration",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2015/12/01/press-conference-president-obama/",
    },
    {
        "source_id": "obama-2016-07-09-nato-press-conference",
        "source_date": "2016-07-09",
        "role": "sealed_confirmation",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2016/07/09/press-conference-president-obama-after-nato-summit/",
    },
    {
        "source_id": "obama-2016-08-04-national-security-press-conference",
        "source_date": "2016-08-04",
        "role": "sealed_confirmation",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2016/08/04/press-conference-president-after-meeting-national-security-officials",
    },
    {
        "source_id": "obama-2016-11-14-press-conference",
        "source_date": "2016-11-14",
        "role": "sealed_confirmation",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2016/11/14/press-conference-president-0",
    },
    {
        "source_id": "obama-2016-12-16-press-conference",
        "source_date": "2016-12-16",
        "role": "sealed_confirmation",
        "url": "https://obamawhitehouse.archives.gov/the-press-office/2016/12/16/press-conference-president/",
    },
)


class VisibleText(HTMLParser):
    BLOCKS = frozenset(
        {
            "article",
            "br",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "li",
            "main",
            "p",
            "section",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.blocked = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self.blocked += 1
        if not self.blocked and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.blocked:
            self.blocked -= 1
        if not self.blocked and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.blocked and data.strip():
            self.parts.extend((" ", html.unescape(data), " "))

    def text(self) -> str:
        value = "".join(self.parts).replace("\xa0", " ")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
        return "\n".join(line for line in lines if line)


LABEL = re.compile(
    r"(?m)^(?P<label>Q|THE PRESIDENT|PRESIDENT OBAMA|PRESIDENT BARACK OBAMA)\s*:\s*"
    r"|^(?P<bare_q>Q)\s+(?=\S)"
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_turns(text: str) -> list[dict[str, str]]:
    matches = list(LABEL.finditer(text))
    chunks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = match.group("label") or match.group("bare_q") or ""
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunks.append((label, text[match.end() : end].strip()))
    turns: list[dict[str, str]] = []
    pending_questions: list[str] = []
    for label, content in chunks:
        if label == "Q":
            if content:
                pending_questions.append(content)
            continue
        if label in {"THE PRESIDENT", "PRESIDENT OBAMA", "PRESIDENT BARACK OBAMA"}:
            if pending_questions and content:
                question = " ".join(pending_questions)
                turns.append(
                    {
                        "question": re.sub(r"\s+", " ", question).strip(),
                        "answer": re.sub(r"\s+", " ", content).strip(),
                        "speaker_label": label,
                    }
                )
            pending_questions = []
        else:
            pending_questions = []
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for turn in turns:
        digest = sha256(f"{turn['question']}\n{turn['answer']}".encode("utf-8"))
        if digest not in seen:
            seen.add(digest)
            flags: list[str] = []
            if len(turn["question"]) > 1500:
                flags.append("unusually_long_question_review_required")
            if len(turn["answer"]) < 20:
                flags.append("unusually_short_answer_review_required")
            if len(turn["answer"]) > 8000:
                flags.append("unusually_long_answer_review_required")
            if re.search(r"(?:^|\s)END(?:\s|$)", turn["answer"]):
                flags.append("transcript_end_marker_in_answer")
            unique.append({**turn, "content_hash": digest, "quality_flags": flags})
    return unique


def fetch(raw: dict[str, str]) -> dict[str, object]:
    request = Request(raw["url"], headers={"User-Agent": "PCFM-Full-Chain-Audit/1.0"})
    with urlopen(request, timeout=45) as response:
        body = response.read()
        status = int(response.status)
        final_url = response.geturl()
        content_type = str(response.headers.get("Content-Type", ""))
    parser = VisibleText()
    parser.feed(body.decode("utf-8", errors="replace"))
    text = parser.text()
    turns = parse_turns(text)
    return {
        **raw,
        "http_status": status,
        "final_url": final_url,
        "content_type": content_type,
        "byte_count": len(body),
        "content_hash": sha256(body),
        "normalized_text_hash": sha256(text.encode("utf-8")),
        "raw_body": body,
        "text": text,
        "oracle_turns": turns,
        "oracle_turn_count": len(turns),
    }


def source_record(
    item: dict[str, object],
    *,
    view: str,
) -> dict[str, object]:
    role = str(item["role"])
    dataset_role = {
        "parameter_fitting": "model_source",
        "applicability_calibration": "applicability_reference",
        "sealed_confirmation": "final_holdout",
    }[role]
    if view == "oracle":
        qas = [
            {
                "question": str(turn["question"]),
                "answer": str(turn["answer"]),
                "locator": f"labelled Q/A turn {index}",
            }
            for index, turn in enumerate(item["oracle_turns"], start=1)
        ]
        segments: list[dict[str, object]] = []
        speaker_scope = "candidate_span_confirmed"
    else:
        text = str(item["text"])
        qas = _extract_qa(text)
        segments = _segments(text)
        speaker_scope = (
            "mixed_speakers" if view == "honest_raw" else "single_speaker_entire_document"
        )
    return {
        "source_id": str(item["source_id"]),
        "person_id": "audit-barack-obama",
        "review_status": "confirmed",
        "dataset_role": dataset_role,
        "content_authenticity": "verbatim_transcript",
        "speaker": "Barack Obama",
        "speaker_role": "President of the United States",
        "speaker_scope": speaker_scope,
        "audience": "press and public",
        "source_date": str(item["source_date"]),
        "title": str(item["source_id"]),
        "source_context": "official archived presidential press conference",
        "source_url": str(item["final_url"]),
        "source_locator": "official archived transcript",
        "near_duplicate_of": None,
        "content_hash": str(item["content_hash"]),
        "qas": qas,
        "segments": segments,
    }


def prediction_summary(result: dict[str, object]) -> dict[str, object]:
    structured = dict(result["structured_prediction"])
    basis = dict(structured.get("response_basis", {}))
    return {
        "answer_status": result["answer_status"],
        "person_prediction_status": basis.get("person_prediction_status"),
        "path": basis.get("path"),
        "applicability": structured.get("applicability"),
        "confidence": structured.get("confidence"),
        "evidence_event_ids": structured.get("evidence_event_ids", []),
        "resolved_context_message_ids": dict(basis.get("query_frame", {})).get(
            "resolved_context_message_ids", []
        ),
        "temporal_applicability": basis.get("temporal_applicability"),
    }


def synthetic_source(
    source_id: str,
    *,
    question: str,
    answer: str,
    date: str,
    context: str,
    lineage: str | None = None,
    language: str = "en",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "person_id": "synthetic-audit-person",
        "review_status": "confirmed",
        "dataset_role": "model_source",
        "content_authenticity": "verbatim_transcript",
        "speaker": "Synthetic Audit Person",
        "speaker_role": "public official",
        "speaker_scope": "candidate_span_confirmed",
        "audience": "public",
        "source_date": date,
        "title": context,
        "source_context": context,
        "source_url": f"https://example.test/{source_id}",
        "source_locator": "qa:1",
        "near_duplicate_of": lineage,
        "original_language": language,
        "qas": [{"question": question, "answer": answer, "locator": "qa:1"}],
        "segments": [],
    }


def run_adversarial_probes(kernel: SimulationKernelV3) -> dict[str, object]:
    chinese = [
        synthetic_source(
            "zh-1",
            question="在团队研发中，安全和速度哪个更重要？",
            answer="在团队开发中，我会把安全放在速度之前，因为错误代价不可逆。",
            date="2020-01-01",
            context="产品研发访谈",
            language="zh",
        ),
        synthetic_source(
            "zh-2",
            question="医疗系统应该先保证安全还是追求速度？",
            answer="在医疗系统中，我宁可优先保障安全，也不追求更快上线。",
            date="2021-01-01",
            context="医疗技术听证会",
            language="zh",
        ),
    ]
    chinese_artifact = kernel.fit(
        person_id="synthetic-audit-person", version=1, reviewed_sources=chinese
    )

    paraphrase = [
        synthetic_source(
            "para-1",
            question="How should a hospital deploy a risky system?",
            answer="We should protect safety before speed because failures are irreversible.",
            date="2020-01-01",
            context="hospital technology hearing",
        ),
        synthetic_source(
            "para-2",
            question="How should a product team decide when to launch?",
            answer="Preventing harm matters more than moving quickly because a bad release is costly.",
            date="2021-01-01",
            context="product strategy interview",
        ),
    ]
    paraphrase_artifact = kernel.fit(
        person_id="synthetic-audit-person", version=1, reviewed_sources=paraphrase
    )

    same_lineage = [
        synthetic_source(
            "repeat-1",
            question="What matters in hospital deployment?",
            answer="We should protect safety before speed because failures are irreversible.",
            date="2020-01-01",
            context="hospital technology hearing",
            lineage="one-interview",
        ),
        synthetic_source(
            "repeat-2",
            question="What matters in another hospital deployment?",
            answer="We should protect safety before speed because failures are irreversible.",
            date="2020-01-01",
            context="hospital technology hearing duplicate",
            lineage="one-interview",
        ),
    ]
    same_lineage_artifact = kernel.fit(
        person_id="synthetic-audit-person", version=1, reviewed_sources=same_lineage
    )
    same_lineage_prediction = kernel.predict(
        same_lineage_artifact,
        text="Should hospitals prioritize safety or speed?",
        history=[],
    )

    cross_domain = [
        synthetic_source(
            "cross-1",
            question="What matters in hospital deployment?",
            answer="We should protect safety before speed because failures are irreversible.",
            date="2020-01-01",
            context="hospital technology hearing",
        ),
        synthetic_source(
            "cross-2",
            question="What matters in a product launch?",
            answer="Safety must come before speed because a harmful launch is costly.",
            date="2021-01-01",
            context="product launch interview",
        ),
    ]
    cross_artifact = kernel.fit(
        person_id="synthetic-audit-person", version=1, reviewed_sources=cross_domain
    )
    role_transfer = kernel.predict(
        cross_artifact,
        text="As a private parent, should I prioritize safety or speed?",
        history=[],
    )
    today_transfer = kernel.predict(
        cross_artifact,
        text="Today, should an AI company prioritize safety or speed?",
        history=[],
    )
    explicit_future = kernel.predict(
        cross_artifact,
        text="In 2026, should an AI company prioritize safety or speed?",
        history=[],
    )

    base = synthetic_source(
        "candidate-disconnect",
        question="Public statement",
        answer="This is a long attributable narrative statement without an explicit tradeoff.",
        date="2020-01-01",
        context="public interview",
    )
    without_candidate = kernel.fit(
        person_id="synthetic-audit-person", version=1, reviewed_sources=[base]
    )
    with_candidate_source = dict(base)
    with_candidate_source["response_events"] = [
        {
            "review_status": "confirmed_promoted",
            "actual_response": "We should protect safety before speed.",
            "protected_interest": "safety",
            "accepted_cost": "speed",
        }
    ]
    with_candidate = kernel.fit(
        person_id="synthetic-audit-person",
        version=1,
        reviewed_sources=[with_candidate_source],
    )

    demo_path = (
        ROOT
        / "artifacts"
        / "conversation_mvp_v03"
        / "local_runtime"
        / "people"
        / "demo-sally-ride"
        / "simulation_models"
        / "simulation-model-v1.json"
    )
    false_retrieval: dict[str, object]
    context_followup: dict[str, object]
    if demo_path.exists():
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        false_retrieval = prediction_summary(
            kernel.predict(
                demo,
                text="How can phone calls and private life improve a crypto product?",
                history=[],
            )
        )
        history = [
            {
                "message_id": "context-user-1",
                "role": "user",
                "text": "How did you learn that you were selected to be a candidate?",
            },
            {
                "message_id": "context-assistant-1",
                "role": "assistant",
                "text": "Historical response.",
                "context_role": "model_generated_context",
            },
        ]
        context_followup = {
            "fixed_short_reference": prediction_summary(
                kernel.predict(demo, text="Why?", history=history)
            ),
            "natural_elliptical_reference": prediction_summary(
                kernel.predict(
                    demo,
                    text="Given that, would you have reacted differently if it happened publicly?",
                    history=history,
                )
            ),
        }
    else:
        false_retrieval = {"status": "demo_artifact_missing"}
        context_followup = {"status": "demo_artifact_missing"}

    return {
        "chinese_tradeoff": {
            "frame_count": len(chinese_artifact["event_frames"]),
            "preference_atom_count": len(chinese_artifact["preference_atoms"]),
            "preference_structure_count": len(
                chinese_artifact["preference_structures"]
            ),
            "primary_domains": sorted(
                {
                    frame["event_classification"]["primary_domain"]
                    for frame in chinese_artifact["event_frames"]
                }
            ),
        },
        "semantic_paraphrase": {
            "preference_atom_count": len(paraphrase_artifact["preference_atoms"]),
            "preference_structure_count": len(
                paraphrase_artifact["preference_structures"]
            ),
            "structures": paraphrase_artifact["preference_structures"],
        },
        "same_lineage_runtime": {
            "structures": same_lineage_artifact["preference_structures"],
            "prediction": prediction_summary(same_lineage_prediction),
        },
        "role_and_time_transfer": {
            "role_transfer": prediction_summary(role_transfer),
            "today_transfer": prediction_summary(today_transfer),
            "explicit_2026_transfer": prediction_summary(explicit_future),
        },
        "confirmed_candidate_integration": {
            "semantic_digest_without_response_events": without_candidate[
                "semantic_model_digest"
            ],
            "semantic_digest_with_response_events": with_candidate[
                "semantic_model_digest"
            ],
            "response_events_changed_v3": without_candidate[
                "semantic_model_digest"
            ]
            != with_candidate["semantic_model_digest"],
        },
        "unrelated_compound_retrieval": false_retrieval,
        "dialogue_context": context_followup,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    collected: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for source in SOURCES:
        try:
            item = fetch(dict(source))
        except Exception as error:  # audit records the exact collection failure
            failures.append(
                {
                    "source_id": source["source_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        collected.append(item)
        (RAW / f"{item['source_id']}.html").write_bytes(item["raw_body"])
        (RAW / f"{item['source_id']}.txt").write_text(
            str(item["text"]), encoding="utf-8"
        )

    public_manifest = [
        {
            key: value
            for key, value in item.items()
            if key not in {"raw_body", "text", "oracle_turns"}
        }
        for item in collected
    ]
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {"sources": public_manifest, "collection_failures": failures},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "oracle_turns.json").write_text(
        json.dumps(
            [
                {
                    "source_id": item["source_id"],
                    "source_date": item["source_date"],
                    "role": item["role"],
                    "url": item["final_url"],
                    "turns": item["oracle_turns"],
                }
                for item in collected
            ],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    kernel = SimulationKernelV3()
    fitting = [item for item in collected if item["role"] == "parameter_fitting"]
    confirmation = [
        item for item in collected if item["role"] == "sealed_confirmation"
    ]
    honest_sources = [source_record(item, view="honest_raw") for item in fitting]
    unsafe_sources = [source_record(item, view="unsafe_raw") for item in fitting]
    oracle_sources = [source_record(item, view="oracle") for item in fitting]
    holdout_sources = [source_record(item, view="oracle") for item in confirmation]

    honest_result: dict[str, object]
    try:
        honest_artifact = kernel.fit(
            person_id="audit-barack-obama",
            version=1,
            reviewed_sources=honest_sources,
        )
    except SimulationV3Error as error:
        honest_result = {
            "status": "rejected",
            "error": str(error),
            "accepted_frame_count": 0,
        }
    else:
        honest_result = {
            "status": "accepted",
            "accepted_frame_count": len(honest_artifact["event_frames"]),
        }

    unsafe_artifact = kernel.fit(
        person_id="audit-barack-obama",
        version=1,
        reviewed_sources=unsafe_sources,
    )
    oracle_artifact = kernel.fit(
        person_id="audit-barack-obama",
        version=1,
        reviewed_sources=oracle_sources,
    )
    validation = kernel.evaluate(oracle_artifact, holdout_sources)
    adversarial = run_adversarial_probes(kernel)
    runtime_eligible_structures = [
        structure
        for structure in oracle_artifact["preference_structures"]
        if structure.get("status")
        in {"repeated_public_preference", "cross_domain_public_preference"}
    ]

    oracle_answers = {
        str(turn["answer"])
        for item in fitting
        for turn in item["oracle_turns"]
    }
    unsafe_frames = unsafe_artifact["event_frames"]
    unsafe_exact_answer_frames = sum(
        str(frame["observed_response"]["verbatim"]) in oracle_answers
        for frame in unsafe_frames
    )
    unsafe_question_frames = sum(
        bool(re.search(r"(?:^|\n)Q\s+", str(frame["observed_response"]["verbatim"])))
        for frame in unsafe_frames
    )

    role_counts = {
        role: sum(item["oracle_turn_count"] for item in collected if item["role"] == role)
        for role in {str(item["role"]) for item in collected}
    }
    years = sorted({str(item["source_date"])[:4] for item in collected})
    flagged_turn_count = sum(
        bool(turn.get("quality_flags"))
        for item in collected
        for turn in item["oracle_turns"]
    )
    human_verified_turn_count = 0
    corpus_ready = (
        len(collected) >= 10
        and sum(item["oracle_turn_count"] for item in collected) >= 60
        and len(years) >= 5
        and not failures
    )

    results = {
        "schema_version": "pcfm-simulation-v3-full-chain-audit-v1",
        "protocol": "SIMULATION_V3_FULL_CHAIN_TEST_PROTOCOL.md",
        "frozen_kernel": "pcfm.simulation_v3.SimulationKernelV3",
        "corpus": {
            "source_count": len(collected),
            "collection_failure_count": len(failures),
            "label_parsed_turn_candidate_count": sum(
                item["oracle_turn_count"] for item in collected
            ),
            "quality_flagged_turn_count": flagged_turn_count,
            "human_verified_turn_count": human_verified_turn_count,
            "role_turn_counts": role_counts,
            "years": years,
            "source_availability_gate_passed": corpus_ready,
            "confirmed_model_corpus_ready": False,
            "confirmed_model_corpus_blocker": (
                "label-parsed candidates require exact-span human review; the official archive itself contains occasional speaker-label anomalies"
            ),
        },
        "honest_raw_document_path": honest_result,
        "unsafe_whole_document_override": {
            "frame_count": len(unsafe_frames),
            "exact_oracle_answer_frame_count": unsafe_exact_answer_frames,
            "question_label_frame_count": unsafe_question_frames,
            "preference_atom_count": len(unsafe_artifact["preference_atoms"]),
            "preference_structure_count": len(
                unsafe_artifact["preference_structures"]
            ),
        },
        "independent_attributable_turn_oracle": {
            "fit_turn_count": sum(item["oracle_turn_count"] for item in fitting),
            "event_frame_count": len(oracle_artifact["event_frames"]),
            "preference_atom_count": len(oracle_artifact["preference_atoms"]),
            "preference_structure_count": len(
                oracle_artifact["preference_structures"]
            ),
            "runtime_eligible_preference_structure_count": len(
                runtime_eligible_structures
            ),
            "preference_relation_count": len(
                oracle_artifact["preference_relations"]
            ),
            "knowledge_claim_count": len(oracle_artifact["knowledge_claims"]),
            "primary_domains": sorted(
                {
                    frame["event_classification"]["primary_domain"]
                    for frame in oracle_artifact["event_frames"]
                }
            ),
            "preference_atoms": oracle_artifact["preference_atoms"],
            "preference_structures": oracle_artifact["preference_structures"],
        },
        "strict_later_holdout_evaluation": validation,
        "adversarial_probes": adversarial,
    }

    no_go_reasons: list[str] = []
    if not corpus_ready:
        no_go_reasons.append("corpus_readiness_gate_failed")
    if human_verified_turn_count == 0:
        no_go_reasons.append("label_parsed_corpus_requires_human_confirmation")
    if honest_result.get("accepted_frame_count") == 0:
        no_go_reasons.append("honest_raw_mixed_speaker_path_has_no_frames")
    if not adversarial["confirmed_candidate_integration"][
        "response_events_changed_v3"
    ]:
        no_go_reasons.append("confirmed_event_candidate_does_not_change_v3")
    if not runtime_eligible_structures:
        no_go_reasons.append(
            "real_person_fit_has_no_runtime_eligible_preference_structure"
        )
    if (
        validation.get("status") == "not_assessed"
        or validation.get("coverage") == 0
        or validation.get("covered_direction_accuracy") == "not_assessed"
    ):
        no_go_reasons.append("strict_later_covered_accuracy_not_assessed")
    if adversarial["chinese_tradeoff"]["preference_atom_count"] == 0:
        no_go_reasons.append("chinese_tradeoff_extraction_failed")
    if (
        adversarial["unrelated_compound_retrieval"].get("answer_status")
        == "similar_event_evidence_answer"
    ):
        no_go_reasons.append("unrelated_compound_false_retrieval_reproduced")
    if (
        adversarial["dialogue_context"]
        .get("natural_elliptical_reference", {})
        .get("resolved_context_message_ids")
        == []
    ):
        no_go_reasons.append("natural_followup_context_not_resolved")

    results["decision"] = {
        "status": "NO_GO_REBUILD_CORE" if no_go_reasons else "GO",
        "reasons": no_go_reasons,
        "software_regression_can_override": False,
    }
    (OUTPUT / "full_chain_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
