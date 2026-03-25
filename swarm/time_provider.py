"""Time provider abstraction for swarm execution.

Provides a pluggable time source to support:
- Normal operation using wall clock time
- Resumed runs that continue from a previous timestamp

This enables logical time continuity when resuming swarm runs,
so agents see consistent time progression rather than gaps.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class TimeProvider(ABC):
    """Abstract base class for time providers."""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current time."""
        ...


class WallClockTimeProvider(TimeProvider):
    """Default time provider using wall clock time."""

    def now(self) -> datetime:
        """Return current UTC wall clock time."""
        return datetime.utcnow()


class OffsetTimeProvider(TimeProvider):
    """
    Time provider for resumed runs.

    Continues time from where the previous run left off,
    maintaining logical time progression.

    Example:
        If previous run ended at 10:30:00 and we resume at 14:00:00,
        the first call to now() returns 10:30:00, and subsequent calls
        return 10:30:00 + (current_wall_clock - 14:00:00).
    """

    def __init__(self, last_timestamp: datetime):
        """
        Initialize with the timestamp from the previous run.

        Args:
            last_timestamp: The last timestamp from the previous run
        """
        self._last_timestamp = last_timestamp
        self._resume_start = datetime.utcnow()

    def now(self) -> datetime:
        """
        Return time offset from last_timestamp.

        Returns:
            last_timestamp + (current_wall_clock - resume_start)
        """
        elapsed = datetime.utcnow() - self._resume_start
        return self._last_timestamp + elapsed

    @property
    def last_timestamp(self) -> datetime:
        """Get the last timestamp from the previous run."""
        return self._last_timestamp

    @property
    def resume_start(self) -> datetime:
        """Get the wall clock time when the run was resumed."""
        return self._resume_start


# Module-level time provider instance
_time_provider: TimeProvider = WallClockTimeProvider()


def now() -> datetime:
    """Get the current time from the configured time provider."""
    return _time_provider.now()


def set_time_provider(provider: TimeProvider) -> None:
    """
    Set the time provider for the module.

    Args:
        provider: The time provider to use
    """
    global _time_provider
    _time_provider = provider


def reset_time_provider() -> None:
    """Reset to the default wall clock time provider."""
    global _time_provider
    _time_provider = WallClockTimeProvider()


def get_time_provider() -> TimeProvider:
    """Get the current time provider."""
    return _time_provider
