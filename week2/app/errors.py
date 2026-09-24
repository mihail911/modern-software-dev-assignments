"""Domain exceptions and FastAPI exception handlers.

Route handlers raise small domain errors (e.g. :class:`NotFoundError`) instead
of constructing ``HTTPException`` inline. A single set of handlers converts
those, plus request-validation failures, into one consistent JSON envelope:

    {"error": {"type": "...", "message": "...", "details": ...}}
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for expected, client-facing application errors."""

    status_code: int = 500
    error_type: str = "internal_error"

    def __init__(self, message: str, *, details: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(AppError):
    """A requested resource does not exist."""

    status_code = 404
    error_type = "not_found"


class BadRequestError(AppError):
    """The request is syntactically valid but semantically unusable."""

    status_code = 400
    error_type = "bad_request"


class UpstreamServiceError(AppError):
    """A dependency (e.g. the LLM service) failed in an unexpected way."""

    status_code = 502
    error_type = "upstream_error"


def _envelope(error_type: str, message: str, details: object | None = None) -> dict:
    return {"error": {"type": error_type, "message": message, "details": details}}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the application's exception handlers to ``app``."""

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.error_type, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # ``exc.errors()`` can contain non-JSON-serialisable values (e.g.
        # ``ValueError`` instances), so coerce them to strings first.
        details = [
            {
                "loc": list(error.get("loc", [])),
                "msg": str(error.get("msg", "")),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_envelope("validation_error", "Request validation failed", details),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Never leak stack traces or internal messages to clients.
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "An unexpected error occurred"),
        )
