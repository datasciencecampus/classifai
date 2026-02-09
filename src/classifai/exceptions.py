from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class ClassifaiError(Exception):
    """Base error for the package.

    - message: what happened (human readable)
    - code: stable identifier (machine readable; optional but useful)
    - context: small debug hints (counts, ids, model name; avoid secrets / full text)
    """

    message: str
    code: str = "classifai_error"
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:
        base = self.message
        if self.code:
            base = f"{base} (code={self.code})"
        if self.context:
            # keep it readable + stable order
            try:
                ctx = json.dumps(self.context, ensure_ascii=False, sort_keys=True, default=str)
            except Exception:
                ctx = str(self.context)
            base = f"{base} | context={ctx}"
        return base

    def to_dict(self) -> dict[str, Any]:
        data = {"error": self.code, "detail": self.message}
        if self.context:
            data["context"] = self.context
        return data


# ---- Subclasses ----


@dataclass(eq=False)
class ConfigurationError(ClassifaiError):
    code: str = "configuration_error"


@dataclass(eq=False)
class DependencyError(ClassifaiError):
    code: str = "dependency_error"


@dataclass(eq=False)
class DataValidationError(ClassifaiError):
    code: str = "validation_error"


@dataclass(eq=False)
class ExternalServiceError(ClassifaiError):
    code: str = "external_service_error"


@dataclass(eq=False)
class VectorisationError(ClassifaiError):
    code: str = "vectorisation_error"


@dataclass(eq=False)
class IndexBuildError(ClassifaiError):
    code: str = "index_build_error"


@dataclass(eq=False)
class HookError(ClassifaiError):
    code: str = "hook_error"
