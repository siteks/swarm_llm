"""In-memory storage backend for swarm coordination."""

import asyncio
import json
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

# Async-safe context for current agent ID in storage operations
_storage_agent_context: ContextVar[Optional[str]] = ContextVar('storage_agent', default=None)


class StorageBackend(Protocol):
    """Abstract storage interface - allows swapping implementations."""

    async def read(self, key: str) -> Optional[Any]:
        """Read a value from storage."""
        ...

    async def write(self, key: str, value: Any) -> bool:
        """Write a value to storage (overwrites existing)."""
        ...

    async def append(self, key: str, value: Any) -> bool:
        """Append a value to a list in storage."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a key from storage."""
        ...

    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by prefix."""
        ...

    async def clear(self) -> None:
        """Clear all storage."""
        ...


@dataclass
class StorageAccessEvent:
    """Record of a storage access for telemetry."""
    cycle: int
    operation: str  # "read", "write", "append", "delete"
    key: str
    agent_id: Optional[str]
    success: bool
    value_size: Optional[int] = None


class InMemoryStorage:
    """
    In-memory storage backend with asyncio locks.

    Thread-safe concurrent access via asyncio.Lock.
    Uses per-key locks for append operations to allow concurrent
    appends to different keys.

    Storage values are wrapped with cycle metadata:
    - Simple values: {"value": X, "cycle": N}
    - List elements: each is {"value": X, "cycle": N}

    Agents see unwrapped values (transparent). Exports contain wrapped format.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._append_locks: Dict[str, asyncio.Lock] = {}
        self._access_log: List[StorageAccessEvent] = []
        self._current_cycle: int = 0

    @property
    def _current_agent_id(self) -> Optional[str]:
        """Get current agent ID from async-safe context."""
        return _storage_agent_context.get()

    def set_current_agent(self, agent_id: Optional[str]) -> None:
        """Set the current agent ID for access logging (async-safe)."""
        _storage_agent_context.set(agent_id)

    def set_current_cycle(self, cycle: int) -> None:
        """Set the current cycle for storage metadata."""
        self._current_cycle = cycle

    def _wrap_value(self, value: Any) -> Dict[str, Any]:
        """Wrap a value with cycle metadata."""
        return {"value": value, "cycle": self._current_cycle}

    def _unwrap_value(self, wrapped: Any) -> Any:
        """
        Unwrap a value from cycle metadata.

        Handles both wrapped format and legacy unwrapped format for compatibility.
        """
        if isinstance(wrapped, dict) and "value" in wrapped and "cycle" in wrapped:
            return wrapped["value"]
        return wrapped

    def _unwrap_list(self, wrapped_list: List[Any]) -> List[Any]:
        """Unwrap a list of wrapped values."""
        return [self._unwrap_value(item) for item in wrapped_list]

    def _log_access(self, operation: str, key: str, success: bool, value: Any = None) -> None:
        """Log a storage access event."""
        value_size = None
        if value is not None:
            try:
                value_size = len(json.dumps(value))
            except (TypeError, ValueError):
                value_size = len(str(value))

        self._access_log.append(StorageAccessEvent(
            cycle=self._current_cycle,
            operation=operation,
            key=key,
            agent_id=self._current_agent_id,
            success=success,
            value_size=value_size
        ))

    async def read(self, key: str) -> Optional[Any]:
        """Read a value from storage. Returns unwrapped value for agents."""
        async with self._lock:
            wrapped = self._data.get(key)
            self._log_access("read", key, True, wrapped)
            if wrapped is None:
                return None
            # Unwrap lists or single values
            if isinstance(wrapped, list):
                return self._unwrap_list(wrapped)
            return self._unwrap_value(wrapped)

    async def write(self, key: str, value: Any) -> bool:
        """Write a value to storage (overwrites existing). Wraps with cycle metadata."""
        async with self._lock:
            wrapped = self._wrap_value(value)
            self._data[key] = wrapped
            self._log_access("write", key, True, wrapped)
            return True

    async def append(self, key: str, value: Any) -> bool:
        """
        Append a value to a list in storage.

        Each appended element is wrapped with cycle metadata.
        If key doesn't exist, creates a new list.
        If key exists but isn't a list, converts to list first.
        """
        # Get or create per-key lock
        if key not in self._append_locks:
            self._append_locks[key] = asyncio.Lock()

        async with self._append_locks[key]:
            async with self._lock:
                existing = self._data.get(key)
                wrapped = self._wrap_value(value)

                if existing is None:
                    self._data[key] = [wrapped]
                elif isinstance(existing, list):
                    existing.append(wrapped)
                elif isinstance(existing, dict) and "value" in existing and "cycle" in existing:
                    # Handle case where key was written with write() instead of append()
                    # The existing value is a wrapped dict - extract inner value
                    inner = existing["value"]
                    if isinstance(inner, list):
                        # Inner value is a list - convert each item to wrapped format
                        # and append the new item
                        converted = [
                            {"value": item, "cycle": existing["cycle"]}
                            for item in inner
                        ]
                        converted.append(wrapped)
                        self._data[key] = converted
                    else:
                        # Inner value is not a list - wrap it and create list
                        self._data[key] = [existing, wrapped]
                else:
                    # Convert existing value to list
                    self._data[key] = [existing, wrapped]

                self._log_access("append", key, True, wrapped)
                return True

    async def delete(self, key: str) -> bool:
        """Delete a key from storage."""
        async with self._lock:
            if key in self._data:
                del self._data[key]
                self._log_access("delete", key, True)
                return True
            self._log_access("delete", key, False)
            return False

    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by prefix."""
        async with self._lock:
            if prefix:
                return [k for k in self._data.keys() if k.startswith(prefix)]
            return list(self._data.keys())

    async def clear(self) -> None:
        """Clear all storage."""
        async with self._lock:
            self._data.clear()
            self._access_log.append(StorageAccessEvent(
                cycle=self._current_cycle,
                operation="clear",
                key="*",
                agent_id=self._current_agent_id,
                success=True
            ))

    async def mark_inbox_read(self, key: str, message_ids: set) -> bool:
        """
        Mark inbox messages as read without resetting wrapper cycles.

        This preserves the original write cycle for entropy calculations.
        """
        async with self._lock:
            wrapped = self._data.get(key)
            if wrapped is None:
                return False

            if isinstance(wrapped, list):
                # List of wrapped items - update in place
                for item in wrapped:
                    if isinstance(item, dict) and 'value' in item:
                        msg = item['value']
                        if isinstance(msg, dict) and msg.get('id') in message_ids:
                            msg['read'] = True
            elif isinstance(wrapped, dict) and 'value' in wrapped:
                # Single wrapped value containing a list
                inner = wrapped['value']
                if isinstance(inner, list):
                    for msg in inner:
                        if isinstance(msg, dict) and msg.get('id') in message_ids:
                            msg['read'] = True

            self._log_access("mark_read", key, True)
            return True

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of all storage data with wrapped format (for export)."""
        return dict(self._data)

    def get_snapshot_unwrapped(self) -> Dict[str, Any]:
        """Get a snapshot of all storage data with values unwrapped (for display)."""
        result = {}
        for key, wrapped in self._data.items():
            if isinstance(wrapped, list):
                result[key] = self._unwrap_list(wrapped)
            else:
                result[key] = self._unwrap_value(wrapped)
        return result

    def get_access_log(self) -> List[StorageAccessEvent]:
        """Get the access log for telemetry."""
        return list(self._access_log)

    def clear_access_log(self) -> None:
        """Clear the access log."""
        self._access_log.clear()

    def get_access_stats(self) -> Dict[str, Any]:
        """Get access statistics for telemetry."""
        if not self._access_log:
            return {"total_accesses": 0, "by_operation": {}, "by_key": {}}

        by_operation: Dict[str, int] = {}
        by_key: Dict[str, Dict[str, int]] = {}

        for event in self._access_log:
            by_operation[event.operation] = by_operation.get(event.operation, 0) + 1

            if event.key not in by_key:
                by_key[event.key] = {"reads": 0, "writes": 0, "appends": 0}

            if event.operation == "read":
                by_key[event.key]["reads"] += 1
            elif event.operation == "write":
                by_key[event.key]["writes"] += 1
            elif event.operation == "append":
                by_key[event.key]["appends"] += 1

        return {
            "total_accesses": len(self._access_log),
            "by_operation": by_operation,
            "by_key": by_key
        }
