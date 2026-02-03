from __future__ import annotations

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
class HookValidationError(ClassifaiError):
    code: str = "hook_validation_error"
