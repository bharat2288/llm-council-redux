"""Typed exceptions for the council engine."""


class CouncilError(Exception):
    """Base for council-engine errors. Catch this in callers for friendly handling."""


class ConfigError(CouncilError):
    """Config JSON is malformed or violates a schema constraint."""


class RoutingError(CouncilError):
    """A theorist's routing path is unavailable in the current environment.

    Message should explain WHY (which env var or binary is missing) and
    suggest the smallest fix the operator can apply.
    """


class TheoristFailure(CouncilError):
    """A specific theorist run failed. Captures partial state for retry."""

    def __init__(self, theorist_name: str, message: str) -> None:
        super().__init__(f"[{theorist_name}] {message}")
        self.theorist_name = theorist_name


class SynthesisFailure(CouncilError):
    """Chairman synthesis pass failed; theorist responses may still be salvageable."""
