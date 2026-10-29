"""Domain error hierarchy.

The same error object is mapped by each interface:

* FastAPI  -> HTTP status code
* CLI      -> process exit code
* Qt       -> user-facing dialog

Interfaces must only show ``user_message``; ``detail`` belongs in logs.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    code = "internal_error"
    http_status = 500
    exit_code = 5

    def __init__(
        self,
        message: str,
        *,
        detail: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.context = dict(context or {})

    @property
    def user_message(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.user_message,
        }


class ValidationError(AppError):
    code = "validation_error"
    http_status = 422
    exit_code = 2


class ResourceLimitError(AppError):
    code = "resource_limit"
    http_status = 413
    exit_code = 2


class NotFoundError(AppError):
    code = "not_found"
    http_status = 404
    exit_code = 1


class ModelUnavailableError(AppError):
    code = "model_unavailable"
    http_status = 503
    exit_code = 3


class BusyError(AppError):
    code = "busy"
    http_status = 503
    exit_code = 4


class PostconditionError(AppError):
    code = "postcondition_failed"
    http_status = 500
    exit_code = 5


class InternalError(AppError):
    code = "internal_error"
    http_status = 500
    exit_code = 5
