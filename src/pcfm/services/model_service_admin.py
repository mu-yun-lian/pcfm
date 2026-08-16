from __future__ import annotations

from ._shared import *  # noqa: F401, F403
from ._shared import (  # noqa: F401
    _as_choice,
    _canonical_hash,
    _parse_time,
    _read_json,
    _reason_text,
    _slug,
    _utc_now,
    _write_json,
)



class ModelServiceAdminMixin:
    def model_service_state(self) -> dict[str, object]:
        return self.model_services.public_state()

    def save_model_service(self, payload: Mapping[str, object]) -> dict[str, object]:
        try:
            return self.model_services.save_service(payload)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def delete_model_service(self, service_id: str) -> None:
        try:
            self.model_services.delete_service(service_id)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def reveal_model_service_key(self, service_id: str) -> str:
        try:
            return self.model_services.reveal_api_key(service_id)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def test_model_service(
        self, service_id: str, model_id: str = ""
    ) -> dict[str, object]:
        try:
            return self.model_services.test_connection(service_id, model_id)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def refresh_model_service_models(self, service_id: str) -> list[str]:
        try:
            return self.model_services.refresh_models(service_id)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def set_model_role(self, role: str, model_ref: str) -> dict[str, object]:
        try:
            return self.model_services.set_role(role, model_ref)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def select_dialogue_model(self, person_id: str, model_ref: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            try:
                if model_ref:
                    self.model_services.resolve_model_ref(model_ref)
                return self.conversation.select_dialogue_model(person_id, model_ref)
            except (ModelServiceError, ConversationError) as error:
                raise ProductError(str(error)) from error
