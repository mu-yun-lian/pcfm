from __future__ import annotations

import base64
import copy
import ctypes
import hashlib
import json
import os
import ssl
import uuid
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


MODEL_SERVICE_SCHEMA = "pcfm-model-services-v1"
MODEL_ROLE_SCHEMA = "pcfm-model-role-assignments-v1"
SUPPORTED_PROTOCOLS = frozenset(
    {
        "openai_native",
        "openai_compatible",
        "anthropic",
        "gemini",
        "ollama",
        "custom_compatible",
    }
)
ROLE_NAMES = frozenset({"dialogue", "material_processing", "validation"})


class ModelServiceError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return copy.deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes) -> tuple[_DataBlob, object]:
    buffer = ctypes.create_string_buffer(data)
    return (
        _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))),
        buffer,
    )


def _dpapi_encrypt(value: str) -> str:
    if os.name != "nt":
        raise ModelServiceError("当前系统没有可用的本机凭据加密存储，请改用环境变量密钥引用。")
    source, source_buffer = _blob(value.encode("utf-8"))
    output = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptProtectData(
        ctypes.byref(source),
        "PCFM model service key",
        None,
        None,
        None,
        0,
        ctypes.byref(output),
    ):
        raise ModelServiceError("Windows 本机凭据加密失败。")
    try:
        encrypted = ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)
        del source_buffer
    return base64.b64encode(encrypted).decode("ascii")


def _dpapi_decrypt(value: str) -> str:
    encrypted = base64.b64decode(value.encode("ascii"), validate=True)
    source, source_buffer = _blob(encrypted)
    output = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)
    ):
        raise ModelServiceError("Windows 本机凭据解密失败。")
    try:
        plain = ctypes.string_at(output.pbData, output.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(output.pbData)
        del source_buffer
    return plain


class SecureSecretStore:
    """Server-only DPAPI secrets. Public configuration stores references only."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            _write_json(self.path, {"schema_version": "pcfm-dpapi-secrets-v1", "items": {}})

    def _document(self) -> dict[str, object]:
        raw = _read_json(
            self.path, {"schema_version": "pcfm-dpapi-secrets-v1", "items": {}}
        )
        if not isinstance(raw, dict) or raw.get("schema_version") != "pcfm-dpapi-secrets-v1":
            raise ModelServiceError("模型密钥存储版本不受支持。")
        return dict(raw)

    def set(self, reference: str, value: str) -> None:
        document = self._document()
        items = dict(document.get("items", {}))
        clean = str(value)
        if clean:
            items[reference] = {"ciphertext": _dpapi_encrypt(clean), "updated_at": _utc_now()}
        else:
            items.pop(reference, None)
        document["items"] = items
        _write_json(self.path, document)

    def get(self, reference: str) -> str:
        document = self._document()
        item = dict(dict(document.get("items", {})).get(reference, {}))
        ciphertext = str(item.get("ciphertext", ""))
        return _dpapi_decrypt(ciphertext) if ciphertext else ""

    def delete(self, reference: str) -> None:
        self.set(reference, "")

    def has(self, reference: str) -> bool:
        document = self._document()
        return reference in dict(document.get("items", {}))


class ModelServiceManager:
    """Versioned provider configuration with no secret in its public surface."""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir).resolve() / "model_services"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.root / "services.json"
        self.roles_path = self.root / "role_assignments.json"
        self.secret_store = SecureSecretStore(self.root / "secrets.dpapi.json")
        if not self.config_path.exists():
            _write_json(self.config_path, {"schema_version": MODEL_SERVICE_SCHEMA, "services": []})
        if not self.roles_path.exists():
            _write_json(
                self.roles_path,
                {
                    "schema_version": MODEL_ROLE_SCHEMA,
                    "material_processing": "",
                    "validation": "",
                    "default_dialogue": "",
                },
            )

    def _services(self) -> list[dict[str, object]]:
        raw = _read_json(self.config_path, {})
        if not isinstance(raw, dict) or raw.get("schema_version") != MODEL_SERVICE_SCHEMA:
            raise ModelServiceError("模型服务配置版本不受支持。")
        values = raw.get("services", [])
        if not isinstance(values, list):
            raise ModelServiceError("模型服务配置损坏。")
        return [dict(item) for item in values]

    def _save(self, services: Sequence[Mapping[str, object]]) -> None:
        _write_json(
            self.config_path,
            {"schema_version": MODEL_SERVICE_SCHEMA, "services": [dict(item) for item in services]},
        )

    @staticmethod
    def _public(item: Mapping[str, object], *, secret_configured: bool) -> dict[str, object]:
        public = {
            key: copy.deepcopy(value)
            for key, value in item.items()
            if key not in {"secret_ref", "api_key", "environment_key"}
        }
        public["api_key_configured"] = bool(secret_configured or item.get("environment_key"))
        public["api_key_source"] = (
            "environment" if item.get("environment_key") else "windows_credentials" if secret_configured else "not_configured"
        )
        public["call_readiness"] = (
            "ready"
            if item.get("connection_status") == "connected" and item.get("last_probe_at")
            else "needs_test"
        )
        return public

    def public_state(self) -> dict[str, object]:
        services = self._services()
        return {
            "schema_version": MODEL_SERVICE_SCHEMA,
            "services": [
                self._public(item, secret_configured=self.secret_store.has(str(item["secret_ref"])))
                for item in services
            ],
            "roles": self.roles(),
        }

    def save_service(self, payload: Mapping[str, object]) -> dict[str, object]:
        services = self._services()
        service_id = str(payload.get("service_id", "")).strip() or f"model-service-{uuid.uuid4().hex[:12]}"
        protocol = str(payload.get("protocol", "")).strip()
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ModelServiceError("不支持的模型服务协议。")
        display_name = str(payload.get("display_name", "")).strip()
        if not display_name:
            raise ModelServiceError("请输入模型服务显示名称。")
        base_url = str(payload.get("base_url", "")).strip().rstrip("/")
        if not base_url:
            base_url = {
                "openai_native": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com",
                "gemini": "https://generativelanguage.googleapis.com/v1beta",
                "ollama": "http://127.0.0.1:11434",
            }.get(protocol, "")
        if not base_url.startswith(("http://", "https://")):
            raise ModelServiceError("Base URL 必须使用 http 或 https。")
        existing = next((item for item in services if item.get("service_id") == service_id), None)
        secret_ref = str(existing.get("secret_ref")) if existing else f"pcfm:{service_id}"
        environment_key = str(payload.get("environment_key", "")).strip()
        item = {
            "schema_version": MODEL_SERVICE_SCHEMA,
            "service_id": service_id,
            "display_name": display_name,
            "provider": str(payload.get("provider", protocol)).strip() or protocol,
            "protocol": protocol,
            "base_url": base_url,
            "enabled": bool(payload.get("enabled", True)),
            "timeout_seconds": max(2, min(int(payload.get("timeout_seconds", 30)), 300)),
            "models": sorted({str(value).strip() for value in payload.get("models", []) if str(value).strip()}),
            "enabled_models": sorted({str(value).strip() for value in payload.get("enabled_models", payload.get("models", [])) if str(value).strip()}),
            "default_model": str(payload.get("default_model", "")).strip(),
            "capabilities": {
                "structured_output": bool(dict(payload.get("capabilities", {})).get("structured_output", True)),
                "streaming": bool(dict(payload.get("capabilities", {})).get("streaming", False)),
                "reasoning": bool(dict(payload.get("capabilities", {})).get("reasoning", False)),
            },
            # Saving may change the URL, key, protocol or model IDs.  A previous
            # success is therefore no longer evidence that chat completion works.
            "connection_status": "not_tested",
            "last_tested_at": "",
            "last_probe_at": "",
            "last_probe_model": "",
            "last_error": "",
            "secret_ref": secret_ref,
            "environment_key": environment_key,
            "updated_at": _utc_now(),
        }
        api_key = str(payload.get("api_key", ""))
        if api_key:
            self.secret_store.set(secret_ref, api_key)
            item["environment_key"] = ""
        elif bool(payload.get("clear_api_key", False)):
            self.secret_store.delete(secret_ref)
        if existing:
            services[services.index(existing)] = item
        else:
            services.append(item)
        self._save(services)
        return self._public(item, secret_configured=self.secret_store.has(secret_ref))

    def delete_service(self, service_id: str) -> None:
        services = self._services()
        item = next((value for value in services if value.get("service_id") == service_id), None)
        if item is None:
            raise ModelServiceError("模型服务不存在。")
        self.secret_store.delete(str(item["secret_ref"]))
        self._save([value for value in services if value.get("service_id") != service_id])
        roles = self.roles()
        changed = False
        for key, value in roles.items():
            if key == "schema_version":
                continue
            if str(value).startswith(service_id + ":"):
                roles[key] = ""
                changed = True
        if changed:
            _write_json(self.roles_path, roles)

    def _private(self, service_id: str) -> dict[str, object]:
        item = next((value for value in self._services() if value.get("service_id") == service_id), None)
        if item is None:
            raise ModelServiceError("模型服务不存在。")
        return item

    def _api_key(self, item: Mapping[str, object]) -> str:
        env_name = str(item.get("environment_key", ""))
        if env_name:
            return os.environ.get(env_name, "")
        return self.secret_store.get(str(item["secret_ref"]))

    def reveal_api_key(self, service_id: str) -> str:
        """返回解密后的密钥（仅本地应用按需显示，页面默认隐藏）。"""
        item = self._private(service_id)
        return self._api_key(item)

    def roles(self) -> dict[str, object]:
        raw = _read_json(self.roles_path, {})
        if not isinstance(raw, dict) or raw.get("schema_version") != MODEL_ROLE_SCHEMA:
            raise ModelServiceError("模型角色配置版本不受支持。")
        return dict(raw)

    def set_role(self, role: str, model_ref: str) -> dict[str, object]:
        key = "default_dialogue" if role == "dialogue" else role
        if role not in ROLE_NAMES:
            raise ModelServiceError("模型角色无效。")
        if model_ref:
            self.resolve_model_ref(model_ref)
        roles = self.roles()
        roles[key] = str(model_ref)
        _write_json(self.roles_path, roles)
        return roles

    def resolve_model_ref(self, model_ref: str, *, require_available: bool = True) -> tuple[dict[str, object], str]:
        service_id, separator, model_id = str(model_ref).partition(":")
        if not separator or not service_id or not model_id:
            raise ModelServiceError("模型引用格式无效。")
        item = self._private(service_id)
        if require_available and not bool(item.get("enabled")):
            raise ModelServiceError("所选模型服务当前未启用；系统没有自动回退。")
        enabled_models = set(map(str, item.get("enabled_models", [])))
        if require_available and model_id not in enabled_models:
            raise ModelServiceError("所选模型当前未启用；系统没有自动回退。")
        if require_available and not (
            item.get("connection_status") == "connected" and item.get("last_probe_at")
        ):
            raise ModelServiceError(
                "所选模型尚未通过真实对话调用验证。请在模型服务中点击“验证调用”；系统没有自动回退。"
            )
        return item, model_id

    def snapshot(self, model_ref: str) -> dict[str, object]:
        item, model_id = self.resolve_model_ref(model_ref, require_available=False)
        payload = {
            "service_id": item["service_id"],
            "provider": item["provider"],
            "protocol": item["protocol"],
            "base_url_hash": _hash(item["base_url"]),
            "model_id": model_id,
            "capabilities": copy.deepcopy(item["capabilities"]),
            "configuration_updated_at": item["updated_at"],
        }
        payload["snapshot_id"] = f"model-snapshot-{_hash(payload)[:16]}"
        return payload

    def _json_request(
        self,
        item: Mapping[str, object],
        path: str,
        *,
        method: str = "GET",
        body: Mapping[str, object] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        api_key = self._api_key(item)
        protocol = str(item["protocol"])
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if protocol in {"openai_native", "openai_compatible", "custom_compatible"} and api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif protocol == "anthropic" and api_key:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        if extra_headers:
            headers.update(extra_headers)
        url = str(item["base_url"]).rstrip("/") + path
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = Request(url, data=data, method=method, headers=headers)
        try:
            with urlopen(
                request,
                timeout=float(item.get("timeout_seconds", 30)),
                context=ssl.create_default_context(),
            ) as response:
                return dict(json.loads(response.read().decode("utf-8")))
        except HTTPError as error:
            base_url = str(item["base_url"])
            protocol = str(item["protocol"])
            hint = ""
            if error.code in {404, 405}:
                hint = f" 请检查 Base URL 是否正确（当前：{base_url}）。DeepSeek 的 API 地址是 https://api.deepseek.com，不是 https://platform.deepseek.com。"
            elif error.code == 401:
                hint = " 请检查 API Key 是否正确。"
            elif error.code == 429:
                hint = " 请求频率超限，请稍后重试。"
            raise ModelServiceError(
                f"模型服务返回 HTTP {error.code}；未进行自动回退。{hint} 请到模型服务配置检查 Base URL 与 API Key。"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            base_url = str(item["base_url"])
            hint = ""
            if "platform.deepseek" in base_url:
                hint = " DeepSeek 的 API 地址是 https://api.deepseek.com（不是 platform.deepseek.com）。"
            elif protocol == "ollama":
                hint = " 请确认 Ollama 已安装并正在运行（默认端口 11434）。"
            raise ModelServiceError(
                f"模型服务连接失败：{type(error).__name__}；未进行自动回退。{hint} 请到模型服务配置检查 Base URL。"
            ) from error
        except json.JSONDecodeError as error:
            base_url = str(item["base_url"])
            hint = ""
            if "platform.deepseek" in base_url:
                hint = " DeepSeek 的 API 地址是 https://api.deepseek.com（不是 platform.deepseek.com）。"
            raise ModelServiceError(
                f"模型服务返回了非 JSON 内容（{type(error).__name__}）；未进行自动回退。{hint} 请检查 Base URL 是否指向了 API 而非控制台页面。"
            ) from error

    def refresh_models(self, service_id: str) -> list[str]:
        item = self._private(service_id)
        protocol = str(item["protocol"])
        api_key = self._api_key(item)
        if protocol == "gemini":
            suffix = f"/models?key={quote(api_key)}" if api_key else "/models"
            payload = self._json_request(item, suffix)
            models = [str(value.get("name", "")).removeprefix("models/") for value in payload.get("models", [])]
        elif protocol == "ollama":
            payload = self._json_request(item, "/api/tags")
            models = [str(value.get("name", "")) for value in payload.get("models", [])]
        else:
            path = "/v1/models" if protocol == "anthropic" and not str(item["base_url"]).endswith("/v1") else "/models"
            payload = self._json_request(item, path)
            models = [str(value.get("id", "")) for value in payload.get("data", [])]
        clean = sorted({value for value in models if value})
        services = self._services()
        stored = next(value for value in services if value["service_id"] == service_id)
        if clean:
            stored["models"] = clean
            stored["enabled_models"] = (
                sorted(set(map(str, stored.get("enabled_models", []))) & set(clean))
                or clean
            )
        # Listing models proves only that the catalogue endpoint is reachable.
        # It must not be presented as proof that chat completion works.
        stored["connection_status"] = "models_loaded"
        stored["last_tested_at"] = _utc_now()
        stored["last_probe_at"] = ""
        stored["last_probe_model"] = ""
        stored["last_error"] = ""
        self._save(services)
        return clean

    def test_connection(self, service_id: str, model_id: str = "") -> dict[str, object]:
        catalogue_error = ""
        try:
            try:
                models = self.refresh_models(service_id)
            except ModelServiceError as error:
                catalogue_error = str(error)
                item = self._private(service_id)
                models = list(map(str, item.get("enabled_models", [])))
                if not models:
                    raise
            item = self._private(service_id)
            candidates = [
                value
                for value in map(str, item.get("enabled_models", []))
                if not models or value in set(models)
            ]
            if not candidates:
                raise ModelServiceError("已读取模型列表，但没有可验证的已启用模型。")
            requested_model = str(model_id).strip()
            if requested_model and requested_model not in candidates:
                raise ModelServiceError("要验证的模型不在当前服务模型列表中。")
            model_id = requested_model or (
                str(item.get("default_model", ""))
                if str(item.get("default_model", "")) in candidates
                else candidates[0]
            )
            structured = bool(dict(item.get("capabilities", {})).get("structured_output"))
            response = self.invoke(
                service_id,
                model_id,
                [
                    {
                        "role": "system",
                        "content": (
                            'Return only {"pcfm_probe":true} as valid JSON.'
                            if structured
                            else "Reply with exactly PCFM_OK."
                        ),
                    },
                    {"role": "user", "content": "connection probe"},
                ],
                structured=structured,
                temperature=0.0,
                # Reasoning models may spend part of the completion budget in
                # reasoning_content before emitting the short visible probe.
                max_tokens=128,
                _allow_unverified=True,
            )
            text = str(response["text"]).strip()
            if structured:
                try:
                    valid = json.loads(text) == {"pcfm_probe": True}
                except json.JSONDecodeError:
                    valid = False
            else:
                valid = text == "PCFM_OK"
            if not valid:
                raise ModelServiceError(
                    "模型列表可读取，但真实对话调用没有返回约定结果。请检查模型 ID 与结构化输出设置。"
                )
            services = self._services()
            stored = next(value for value in services if value["service_id"] == service_id)
            stored["connection_status"] = "connected"
            stored["last_tested_at"] = _utc_now()
            stored["last_probe_at"] = stored["last_tested_at"]
            stored["last_probe_model"] = model_id
            stored["last_error"] = (
                "模型列表接口不可用，但指定模型的真实对话调用已验证。"
                if catalogue_error
                else ""
            )
            self._save(services)
            status, error = "connected", ""
        except ModelServiceError as exc:
            models, status, error = [], "unavailable", str(exc)
            services = self._services()
            stored = next(value for value in services if value["service_id"] == service_id)
            stored["connection_status"] = status
            stored["last_tested_at"] = _utc_now()
            stored["last_probe_at"] = ""
            stored["last_probe_model"] = ""
            stored["last_error"] = error
            self._save(services)
        return {
            "status": status,
            "model_count": len(models),
            "message": error or (
                "真实对话调用验证成功；模型列表接口不可用，保留手动模型 ID。"
                if catalogue_error
                else "真实对话调用验证成功。"
            ),
        }

    def invoke(
        self,
        service_id: str,
        model_id: str,
        messages: Sequence[Mapping[str, str]],
        *,
        structured: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1800,
        _allow_unverified: bool = False,
    ) -> dict[str, object]:
        item, resolved_model = self.resolve_model_ref(
            f"{service_id}:{model_id}", require_available=not _allow_unverified
        )
        token_limit = max(1, min(int(max_tokens), 8192))
        protocol = str(item["protocol"])
        if protocol in {"openai_native", "openai_compatible", "custom_compatible"}:
            body: dict[str, object] = {
                "model": resolved_model,
                "messages": [dict(value) for value in messages],
                "temperature": float(temperature),
                "max_tokens": token_limit,
            }
            # 禁用思考模式（reasoning），避免结构化输出时 reasoning 耗尽 token 导致 content 为空
            body["thinking"] = {"type": "disabled"}
            if structured and bool(dict(item["capabilities"]).get("structured_output")):
                body["response_format"] = {"type": "json_object"}
            payload = self._json_request(item, "/chat/completions", method="POST", body=body)
            text = str(payload["choices"][0]["message"]["content"])
        elif protocol == "ollama":
            payload = self._json_request(
                item,
                "/api/chat",
                method="POST",
                body={"model": resolved_model, "messages": [dict(value) for value in messages], "stream": False, "format": "json" if structured else "", "options": {"num_predict": token_limit}},
            )
            text = str(dict(payload.get("message", {})).get("content", ""))
        elif protocol == "anthropic":
            system = "\n".join(value["content"] for value in messages if value.get("role") == "system")
            turns = [dict(value) for value in messages if value.get("role") != "system"]
            payload = self._json_request(
                item,
                "/v1/messages" if not str(item["base_url"]).endswith("/v1") else "/messages",
                method="POST",
                body={"model": resolved_model, "max_tokens": token_limit, "temperature": float(temperature), "system": system, "messages": turns},
            )
            text = "".join(str(value.get("text", "")) for value in payload.get("content", []))
        elif protocol == "gemini":
            api_key = self._api_key(item)
            path = f"/models/{quote(resolved_model)}:generateContent" + (f"?key={quote(api_key)}" if api_key else "")
            prompt = "\n\n".join(f"{value.get('role')}: {value.get('content')}" for value in messages)
            payload = self._json_request(item, path, method="POST", body={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": float(temperature), "maxOutputTokens": token_limit, "responseMimeType": "application/json" if structured else "text/plain"}})
            text = str(payload["candidates"][0]["content"]["parts"][0]["text"])
        else:
            raise ModelServiceError("该自定义协议没有可执行调用路径。")
        if not text.strip():
            raise ModelServiceError("模型服务返回空内容；未进行自动回退。")
        snapshot = self.snapshot(f"{service_id}:{model_id}")
        return {"text": text, "model_ref": f"{service_id}:{model_id}", "snapshot": snapshot, "fallback_used": False}
