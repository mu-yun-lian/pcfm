from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .product_service import ProductError, ProductService


STATIC_DIR = Path(__file__).with_name("web_static")
APP_VERSION = "0.10.0-simulation-v5"
DEFAULT_DATA_DIR = Path.home() / "PCFM人物对话系统数据"


def create_handler(service: ProductService):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"PCFMConversationMVP/{APP_VERSION}"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, value: object, status: int = HTTPStatus.OK) -> None:
            rendered = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(rendered)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-PCFM-Version", APP_VERSION)
            self.end_headers()
            self.wfile.write(rendered)

        def _body(self) -> dict[str, object]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ProductError("请求长度无效。") from error
            if length > 25 * 1024 * 1024:
                raise ProductError("导入文件超过 25 MB。")
            try:
                value = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ProductError("提交内容不是有效 JSON。") from error
            if not isinstance(value, dict):
                raise ProductError("提交内容必须是对象。")
            return value

        def _parts(self) -> list[str]:
            return [unquote(part) for part in urlparse(self.path).path.split("/") if part]

        def _error(self, error: Exception) -> None:
            if isinstance(error, ProductError):
                self._send_json({"ok": False, "message": str(error)}, HTTPStatus.BAD_REQUEST)
            else:
                self._send_json(
                    {"ok": False, "message": "本地应用发生错误，请查看启动窗口。"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                print(f"PCFM web error: {type(error).__name__}: {error}")

        def do_GET(self) -> None:
            try:
                parts = self._parts()
                if parts == ["api", "health"]:
                    self._send_json(
                        {
                            "ok": True,
                            "product": "PCFM 对话式人物响应预测 MVP",
                            "app_version": APP_VERSION,
                            "capabilities": service.capabilities(),
                        }
                    )
                    return
                if False:
                    self._send_json({"ok": True, "product": "PCFM 对话式人物模拟 MVP v0.3"})
                    return
                if parts == ["api", "expression", "profile"]:
                    self._send_json(
                        {"ok": True, "profile": service.expression_profile()}
                    )
                    return
                if parts == ["api", "assistant", "conversation"]:
                    self._send_json(
                        {"ok": True, "conversation": service.assistant.conversation()}
                    )
                    return
                if parts == ["api", "model-services"]:
                    self._send_json(
                        {"ok": True, "model_services": service.model_service_state()}
                    )
                    return
                if len(parts) == 4 and parts[:2] == ["api", "model-services"] and parts[3] == "key":
                    self._send_json({"ok": True, "key": service.reveal_model_service_key(parts[2])})
                    return
                if parts == ["api", "expression", "renders"]:
                    self._send_json(
                        {"ok": True, "renders": service.expression_records()}
                    )
                    return
                if parts == ["api", "people"]:
                    self._send_json({"ok": True, "people": service.list_people()})
                    return
                if parts == ["api", "archived-people"]:
                    self._send_json(
                        {"ok": True, "people": service.list_archived_people()}
                    )
                    return
                if (
                    len(parts) == 4
                    and parts[:2] == ["api", "people"]
                    and parts[3] == "conversation"
                ):
                    self._send_json(
                        {
                            "ok": True,
                            "conversation": service.conversation_summary(parts[2]),
                        }
                    )
                    return
                if (
                    len(parts) == 4
                    and parts[:2] == ["api", "people"]
                    and parts[3] == "processing-progress"
                ):
                    self._send_json({"ok": True, "progress": service.processing_progress(parts[2])})
                    return
                if (
                    len(parts) == 5
                    and parts[:2] == ["api", "people"]
                    and parts[3:5] == ["conversation", "sessions"]
                ):
                    self._send_json({"ok": True, "sessions": service.list_sessions(parts[2])})
                    return
                if len(parts) == 3 and parts[:2] == ["api", "people"]:
                    self._send_json({"ok": True, "person": service.get_person(parts[2])})
                    return
                if len(parts) == 5 and parts[:2] == ["api", "people"] and parts[3:] == ["history", "template.csv"]:
                    rendered = service.csv_template(parts[2]).encode("utf-8-sig")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="pcfm-history-template.csv"')
                    self.send_header("Content-Length", str(len(rendered)))
                    self.end_headers()
                    self.wfile.write(rendered)
                    return
                if len(parts) == 4 and parts[:2] == ["api", "people"] and parts[3] == "export":
                    payload = service.export_person(parts[2])
                    rendered = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="pcfm-{parts[2]}.json"')
                    self.send_header("Content-Length", str(len(rendered)))
                    self.end_headers()
                    self.wfile.write(rendered)
                    return
                if len(parts) == 4 and parts[:2] == ["api", "people"] and parts[3] == "avatar":
                    content, mime = service.get_avatar(parts[2])
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                self._serve_static(parts)
            except Exception as error:
                self._error(error)

        def _serve_static(self, parts: list[str]) -> None:
            relative = "index.html" if not parts else "/".join(parts)
            target = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in target.parents or not target.is_file():
                target = STATIC_DIR / "index.html"
            content = target.read_bytes()
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") or mime == "application/javascript" else mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-PCFM-Version", APP_VERSION)
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:
            try:
                parts = self._parts()
                body = self._body()
                if parts == ["api", "assistant", "message"]:
                    result = service.assistant.handle(str(body.get("text", "")))
                    self._send_json(
                        {"ok": True, "assistant": result}, HTTPStatus.CREATED
                    )
                    return
                if parts == ["api", "assistant", "reset"]:
                    service.assistant.reset()
                    self._send_json({"ok": True, "assistant": {"reply": "已重置。", "state": service.assistant._load_state()}})
                    return
                if parts == ["api", "assistant", "model"]:
                    state = service.assistant.set_model(str(body.get("model_ref", "")))
                    self._send_json({"ok": True, "assistant": {"reply": "已设置助手模型。", "state": state}})
                    return
                if len(parts) == 4 and parts[:2] == ["api", "people"] and parts[3] == "avatar":
                    person = service.set_avatar(parts[2], str(body.get("avatar", "")))
                    self._send_json({"ok": True, "person": person})
                    return
                if parts == ["api", "conversation", "people"]:
                    person = service.create_conversation_person(
                        name=str(body.get("name", "")),
                        aliases=body.get("aliases", []),
                        language=str(body.get("language", "zh")),
                        description=str(body.get("description", "")),
                        time_start=str(body.get("time_start", "")),
                        time_end=str(body.get("time_end", "")),
                        source_mode=str(body.get("source_mode", "user_provided")),
                        identity_note=str(body.get("identity_note", "")),
                        focus_domain=str(body.get("focus_domain", "")),
                        avatar=str(body.get("avatar", "")),
                        notes=str(body.get("notes", "")),
                    )
                    self._send_json(
                        {"ok": True, "person": person}, HTTPStatus.CREATED
                    )
                    return
                if parts == ["api", "model-services"]:
                    saved = service.save_model_service(body)
                    self._send_json(
                        {"ok": True, "service": saved}, HTTPStatus.CREATED
                    )
                    return
                if parts == ["api", "model-roles"]:
                    roles = service.set_model_role(
                        str(body.get("role", "")), str(body.get("model_ref", ""))
                    )
                    self._send_json({"ok": True, "roles": roles})
                    return
                if (
                    len(parts) == 4
                    and parts[:2] == ["api", "model-services"]
                    and parts[3] in {"test", "refresh-models"}
                ):
                    if parts[3] == "test":
                        result = service.test_model_service(
                            parts[2], str(body.get("model_id", ""))
                        )
                        self._send_json({"ok": True, "result": result})
                    else:
                        models = service.refresh_model_service_models(parts[2])
                        self._send_json({"ok": True, "models": models})
                    return
                if parts == ["api", "expression", "render"]:
                    result = service.render_expression(
                        dict(body.get("contract", {})),
                        include_adversarial_probe=bool(
                            body.get("include_adversarial_probe", False)
                        ),
                    )
                    self._send_json({"ok": True, "render": result})
                    return
                if parts == ["api", "people"]:
                    person = service.create_person(
                        name=str(body.get("name", "")),
                        description=str(body.get("description", "")),
                        feature_names=body.get("feature_names", []),
                    )
                    self._send_json({"ok": True, "person": person}, HTTPStatus.CREATED)
                    return
                if parts == ["api", "import-product"]:
                    person = service.import_product_export(
                        dict(body.get("payload", {})),
                        replace=bool(body.get("replace", False)),
                    )
                    self._send_json({"ok": True, "person": person})
                    return
                if (
                    len(parts) == 4
                    and parts[:2] == ["api", "archived-people"]
                    and parts[3] == "restore"
                ):
                    person = service.restore_person(parts[2])
                    self._send_json({"ok": True, "person": person})
                    return
                if len(parts) >= 4 and parts[:2] == ["api", "people"]:
                    person_id = parts[2]
                    action = parts[3:]
                    if action == ["conversation", "sources", "text"]:
                        result = service.add_conversation_text_source(
                            person_id,
                            title=str(body.get("title", "")),
                            text=str(body.get("text", "")),
                            speaker=str(body.get("speaker", "")),
                            source_date=str(body.get("source_date", "")),
                            dataset_role=str(body.get("dataset_role", "model_source")),
                            content_authenticity=str(
                                body.get("content_authenticity", "unverified_material")
                            ),
                            source_locator=str(body.get("source_locator", "")),
                            source_context=str(body.get("source_context", "")),
                            source_url=str(body.get("source_url", "")),
                            original_language=str(body.get("original_language", "")),
                            translation_of=str(body.get("translation_of", "")),
                            speaker_scope=str(body.get("speaker_scope", "single_speaker_entire_document")),
                        )
                        self._send_json(
                            {"ok": True, "source": result}, HTTPStatus.CREATED
                        )
                        return
                    if action == ["conversation", "sources", "file"]:
                        result = service.add_conversation_file_source(
                            person_id,
                            filename=str(body.get("filename", "")),
                            content_base64=str(body.get("content_base64", "")),
                            speaker=str(body.get("speaker", "")),
                            source_date=str(body.get("source_date", "")),
                            dataset_role=str(body.get("dataset_role", "model_source")),
                            content_authenticity=str(body.get("content_authenticity", "unverified_material")),
                            source_locator=str(body.get("source_locator", "")),
                            source_context=str(body.get("source_context", "")),
                            speaker_scope=str(body.get("speaker_scope", "single_speaker_entire_document")),
                        )
                        self._send_json(
                            {"ok": True, "source": result}, HTTPStatus.CREATED
                        )
                        return
                    if action == ["conversation", "sources", "url"]:
                        result = service.add_conversation_url_source(
                            person_id,
                            url=str(body.get("url", "")),
                            speaker=str(body.get("speaker", "")),
                            source_date=str(body.get("source_date", "")),
                            dataset_role=str(body.get("dataset_role", "model_source")),
                            content_authenticity=str(body.get("content_authenticity", "unverified_material")),
                            source_locator=str(body.get("source_locator", "")),
                            source_context=str(body.get("source_context", "")),
                            speaker_scope=str(body.get("speaker_scope", "single_speaker_entire_document")),
                        )
                        self._send_json(
                            {"ok": True, "source": result}, HTTPStatus.CREATED
                        )
                        return
                    if action == ["conversation", "search"]:
                        result = service.collect_public_sources(person_id)
                        self._send_json({"ok": True, "collection": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "sources"]
                        and action[3] == "review"
                    ):
                        result = service.review_conversation_source(
                            person_id, action[2], str(body.get("decision", ""))
                        )
                        self._send_json({"ok": True, "source": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "sources"]
                        and action[3] == "extract-candidates"
                    ):
                        result = service.extract_conversation_response_candidates(
                            person_id, action[2]
                        )
                        self._send_json({"ok": True, "source": result})
                        return
                    if (
                        len(action) == 6
                        and action[:2] == ["conversation", "sources"]
                        and action[3] == "candidates"
                        and action[5] == "review"
                    ):
                        result = service.review_conversation_response_candidate(
                            person_id,
                            action[2],
                            action[4],
                            str(body.get("decision", "")),
                        )
                        self._send_json({"ok": True, "source": result})
                        return
                    if action == ["conversation", "messages"]:
                        result = service.send_conversation_message(
                            person_id,
                            str(body.get("text", "")),
                            reality_lookup_requested=bool(
                                body.get("reality_lookup_requested", False)
                            ),
                            dialogue_model_ref=str(
                                body.get("dialogue_model_ref", "")
                            ),
                        )
                        self._send_json(
                            {"ok": True, "message": result}, HTTPStatus.CREATED
                        )
                        return
                    if action == ["conversation", "new"]:
                        result = service.start_new_conversation(person_id)
                        self._send_json({"ok": True, "conversation": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "sessions"]
                        and action[3] == "switch"
                    ):
                        result = service.switch_session(person_id, action[2])
                        self._send_json({"ok": True, "session": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "sessions"]
                        and action[3] == "rename"
                    ):
                        result = service.rename_session(person_id, action[2], str(body.get("title", "")))
                        self._send_json({"ok": True, "session": result})
                        return
                    if action == ["conversation", "model"]:
                        result = service.select_dialogue_model(
                            person_id, str(body.get("model_ref", ""))
                        )
                        self._send_json({"ok": True, "state": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "messages"]
                        and action[3] == "reality"
                    ):
                        result = service.find_conversation_reality_answer(
                            person_id, action[2]
                        )
                        self._send_json({"ok": True, "comparison": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "messages"]
                        and action[3] == "optimization"
                    ):
                        result = service.create_optimization_candidate(
                            person_id,
                            action[2],
                            allow_retry=bool(body.get("allow_retry", False)),
                            comparison_candidate_id=str(
                                body.get("comparison_candidate_id", "")
                            ),
                        )
                        self._send_json(
                            {"ok": True, "candidate": result}, HTTPStatus.CREATED
                        )
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "messages"]
                        and action[3] == "feedback"
                    ):
                        result = service.record_conversation_feedback(
                            person_id, action[2], str(body.get("value", ""))
                        )
                        self._send_json({"ok": True, "message": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "optimization"]
                        and action[3] == "review"
                    ):
                        result = service.review_optimization_candidate(
                            person_id, action[2], str(body.get("decision", ""))
                        )
                        self._send_json({"ok": True, "candidate": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "optimization"]
                        and action[3] == "style-review"
                    ):
                        result = service.review_optimization_style_candidate(
                            person_id, action[2], str(body.get("decision", ""))
                        )
                        self._send_json({"ok": True, "candidate": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "versions"]
                        and action[3] == "rollback"
                    ):
                        result = service.rollback_conversation_version(
                            person_id, int(action[2])
                        )
                        self._send_json({"ok": True, "state": result})
                        return
                    if action == ["cognitive", "evidence"]:
                        result = service.add_cognitive_evidence(
                            person_id, dict(body.get("evidence", body))
                        )
                        self._send_json({"ok": True, "evidence": result}, HTTPStatus.CREATED)
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["cognitive", "evidence"]
                        and action[3] == "review"
                    ):
                        result = service.review_cognitive_evidence(
                            person_id,
                            action[2],
                            str(body.get("decision", "")),
                        )
                        self._send_json({"ok": True, "evidence": result})
                        return
                    if action == ["cognitive", "card", "generate"]:
                        result = service.generate_cognitive_card(person_id)
                        self._send_json({"ok": True, "card": result})
                        return
                    if action == ["cognitive", "scenarios", "draft"]:
                        result = service.draft_cognitive_scenario(
                            person_id, str(body.get("text", ""))
                        )
                        self._send_json({"ok": True, "scenario": result}, HTTPStatus.CREATED)
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["cognitive", "scenarios"]
                        and action[3] == "confirm"
                    ):
                        result = service.confirm_cognitive_scenario(
                            person_id, action[2], dict(body)
                        )
                        self._send_json({"ok": True, "scenario": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["cognitive", "scenarios"]
                        and action[3] == "predict"
                    ):
                        result = service.predict_cognitive_scenario(person_id, action[2])
                        self._send_json({"ok": True, "prediction": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["cognitive", "predictions"]
                        and action[3] == "outcome"
                    ):
                        result = service.record_cognitive_outcome(
                            person_id, action[2], dict(body)
                        )
                        self._send_json({"ok": True, "outcome": result})
                        return
                    if action == ["history", "import"]:
                        result = service.import_history(
                            person_id,
                            body.get("payload"),
                            input_format=str(body.get("format", "form")),
                        )
                        self._send_json({"ok": True, **result})
                        return
                    if action == ["advanced-evidence"]:
                        result = service.import_decision_evidence(
                            person_id,
                            dict(body.get("bundle", {})),
                            dict(body.get("verification_keys", {})),
                        )
                        self._send_json({"ok": True, **result})
                        return
                    if action == ["train"]:
                        result = service.train(person_id)
                        self._send_json({"ok": True, "model": result})
                        return
                    if action == ["predict"]:
                        result = service.predict(
                            person_id,
                            dict(body.get("scenario", {})),
                            diagnostic_override=bool(body.get("diagnostic_override", False)),
                        )
                        self._send_json({"ok": True, "prediction": result})
                        return
                    if len(action) == 3 and action[0] == "predictions" and action[2] == "outcome":
                        result = service.record_outcome(
                            person_id,
                            action[1],
                            body.get("actual_choice"),
                            str(body["observed_at"]) if body.get("observed_at") else None,
                        )
                        self._send_json({"ok": True, "prediction": result})
                        return
                raise ProductError("不支持的操作。")
            except Exception as error:
                self._error(error)

        def do_PUT(self) -> None:
            try:
                parts = self._parts()
                if len(parts) == 3 and parts[:2] == ["api", "people"]:
                    person = service.update_person(parts[2], self._body())
                    self._send_json({"ok": True, "person": person})
                    return
                raise ProductError("不支持的操作。")
            except Exception as error:
                self._error(error)

        def do_DELETE(self) -> None:
            try:
                parts = self._parts()
                if len(parts) == 3 and parts[:2] == ["api", "people"]:
                    service.delete_person(parts[2])
                    self._send_json({"ok": True})
                    return
                if (
                    len(parts) == 6
                    and parts[:2] == ["api", "people"]
                    and parts[3:5] == ["conversation", "sessions"]
                ):
                    result = service.delete_session(parts[2], parts[5])
                    self._send_json({"ok": True, **result})
                    return
                if len(parts) == 3 and parts[:2] == ["api", "model-services"]:
                    service.delete_model_service(parts[2])
                    self._send_json({"ok": True})
                    return
                if len(parts) == 3 and parts[:2] == ["api", "archived-people"]:
                    body = self._body()
                    service.permanently_delete_archived_person(
                        parts[2], expected_name=str(body.get("expected_name", ""))
                    )
                    self._send_json({"ok": True})
                    return
                raise ProductError("不支持的操作。")
            except Exception as error:
                self._error(error)

    return Handler


def build_server(
    host: str,
    port: int,
    data_dir: Path,
    *,
    seed_example: bool = True,
    seed_demos: bool = False,
) -> ThreadingHTTPServer:
    service = ProductService(data_dir, seed_example=seed_example, seed_demos=seed_demos)
    server = ThreadingHTTPServer((host, port), create_handler(service))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="PCFM 对话式人物模拟 MVP v0.3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--seed-example", action="store_true")
    parser.add_argument("--seed-demos", action="store_true")
    parser.add_argument("--no-seed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    server = build_server(
        args.host,
        args.port,
        args.data_dir,
        seed_example=args.seed_example and not args.no_seed,
        seed_demos=args.seed_demos,
    )
    address = f"http://{args.host}:{server.server_address[1]}"
    print("PCFM 对话式人物模拟 MVP v0.3 已经启动。")
    print(f"访问地址：{address}")
    print(f"本地数据：{args.data_dir.resolve()}")
    print("关闭本窗口即可停止应用。")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
