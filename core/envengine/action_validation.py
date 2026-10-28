"""Errors raised when a player command crosses an environment rule boundary."""


class IllegalActionError(ValueError):
    """The supplied action is not legal for its executor or information scope."""
