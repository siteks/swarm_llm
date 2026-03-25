"""Inter-agent messaging system for swarm coordination."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from .storage import InMemoryStorage


@dataclass
class Message:
    """A message between agents."""
    id: str
    from_agent: str
    to_agent: str
    content: str
    cycle: int = 0
    arrival_order: int = 0  # For sorting within a cycle
    read: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage and display."""
        return {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to_agent,
            "content": self.content,
            "cycle": self.cycle,
            "arrival_order": self.arrival_order,
            "read": self.read
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            from_agent=data["from"],
            to_agent=data["to"],
            content=data["content"],
            cycle=data.get("cycle", 0),
            arrival_order=data.get("arrival_order", 0),
            read=data.get("read", False)
        )


class MessagingSystem:
    """
    Inter-agent messaging using storage backend.

    Messages are stored in inbox keys: _inbox_{agent_id}
    Each inbox is a list of message dicts.

    Messages are immediately visible to recipients once sent.
    """

    INBOX_PREFIX = "_inbox_"

    def __init__(self, storage: InMemoryStorage):
        self.storage = storage
        self.current_cycle = 0
        self._message_log: List[Message] = []
        self._arrival_counter: int = 0

    def _inbox_key(self, agent_id: str) -> str:
        """Get the storage key for an agent's inbox."""
        return f"{self.INBOX_PREFIX}{agent_id}"

    async def send(self, from_agent: str, to_agent: str, content: str) -> Message:
        """
        Send a message from one agent to another.

        The message will be available in the recipient's inbox
        after the current cycle completes.
        """
        self._arrival_counter += 1
        message = Message(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            cycle=self.current_cycle,
            arrival_order=self._arrival_counter
        )

        inbox_key = self._inbox_key(to_agent)
        await self.storage.append(inbox_key, message.to_dict())

        self._message_log.append(message)
        return message

    async def read_inbox(
        self,
        agent_id: str,
        unread_only: bool = True,
        mark_as_read: bool = True
    ) -> List[Message]:
        """
        Read messages from an agent's inbox.

        Args:
            agent_id: The agent whose inbox to read
            unread_only: Only return unread messages
            mark_as_read: Mark returned messages as read

        Returns:
            List of messages, newest first (by arrival_order)
        """
        inbox_key = self._inbox_key(agent_id)
        raw_messages = await self.storage.read(inbox_key) or []

        messages = [Message.from_dict(m) for m in raw_messages]

        if unread_only:
            messages = [m for m in messages if not m.read]

        # Sort by arrival_order, newest first
        messages.sort(key=lambda m: m.arrival_order, reverse=True)

        # Mark as read if requested
        if mark_as_read and messages:
            read_ids = {m.id for m in messages}
            # Update items in-place to preserve wrapper cycle metadata
            await self.storage.mark_inbox_read(inbox_key, read_ids)

            # Update returned messages
            for m in messages:
                m.read = True

        return messages

    async def get_inbox_count(self, agent_id: str, unread_only: bool = True) -> int:
        """Get the count of messages in an agent's inbox."""
        inbox_key = self._inbox_key(agent_id)
        raw_messages = await self.storage.read(inbox_key) or []

        if unread_only:
            return sum(1 for m in raw_messages if not m.get("read", False))
        return len(raw_messages)

    async def clear_inbox(self, agent_id: str) -> None:
        """Clear all messages from an agent's inbox."""
        inbox_key = self._inbox_key(agent_id)
        await self.storage.delete(inbox_key)

    def set_cycle(self, cycle: int) -> None:
        """Update the current cycle number."""
        self.current_cycle = cycle

    def get_message_log(self) -> List[Message]:
        """Get the log of all messages sent (for telemetry)."""
        return list(self._message_log)

    def clear_message_log(self) -> None:
        """Clear the message log."""
        self._message_log.clear()

    def get_message_graph(self) -> Dict[str, Any]:
        """
        Build a message flow graph for visualization.

        Returns a dict with:
        - nodes: list of agent IDs
        - edges: list of (from, to, count) tuples
        """
        nodes = set()
        edge_counts: Dict[tuple, int] = {}

        for msg in self._message_log:
            nodes.add(msg.from_agent)
            nodes.add(msg.to_agent)

            edge = (msg.from_agent, msg.to_agent)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        return {
            "nodes": list(nodes),
            "edges": [
                {"from": f, "to": t, "count": c}
                for (f, t), c in edge_counts.items()
            ]
        }

    def export_graphml(self) -> str:
        """Export message graph in GraphML format for visualization tools."""
        graph = self.get_message_graph()

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="weight" for="edge" attr.name="weight" attr.type="int"/>',
            '  <graph id="G" edgedefault="directed">'
        ]

        for node in graph["nodes"]:
            lines.append(f'    <node id="{node}"/>')

        for i, edge in enumerate(graph["edges"]):
            lines.append(f'    <edge id="e{i}" source="{edge["from"]}" target="{edge["to"]}">')
            lines.append(f'      <data key="weight">{edge["count"]}</data>')
            lines.append('    </edge>')

        lines.append('  </graph>')
        lines.append('</graphml>')

        return '\n'.join(lines)
