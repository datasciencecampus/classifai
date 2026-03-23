from importlib.metadata import PackageNotFoundError, version


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required but missing."""

    pass


def _message(req: str | list[str], extra: str | None) -> str:
    singular_or_plural = "dependency" if isinstance(req, str) else "dependencies"
    if isinstance(req, list):
        req = ", ".join(req)
    return (
        f"Optional {singular_or_plural} '{req}' is required. Install with: pip install 'classifai[{extra}]'."
        if extra
        else ""
    )


def check_deps(reqs: list[str], extra: str | None = None) -> None:
    """Check if optional dependencies are installed.

    Args:
        reqs (list[str]): A list of package names to check.
        extra (str): [optional] The name of the extra installation group. Defaults to None.

    Raises:
        OptionalDependencyError: If any of the required packages are not installed.
    """
    missing = []
    for req in reqs:
        try:
            version(req)
        except PackageNotFoundError:
            missing.append(req)
    if missing:
        raise OptionalDependencyError(_message(missing, extra))
