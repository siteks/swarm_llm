"""
Swarm Runtime Engine.

Manages clock cycles where all agents act in parallel,
coordinates tool execution, and tracks swarm state.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import uuid

from . import time_provider
from .agent import Agent, AgentStatus, ToolCall, ToolResult
from .storage import InMemoryStorage
from .messaging import MessagingSystem
from .primitives import SwarmPrimitives, reset_turn_budget
from .telemetry import TelemetryCollector, EventType, TelemetryEvent, CycleSummary
from .state import SwarmState, AgentState, STATE_VERSION

from core.llm_client import LLMClient, LLMClientPool, TurnResult
from core.tools import ToolRegistry, create_swarm_registry

logger = logging.getLogger(__name__)


@dataclass
class AgentTurnResult:
    """Result of a single agent's turn within a cycle."""
    agent_id: str
    success: bool
    content: str = ""
    tool_calls: int = 0
    tokens_used: int = 0
    iterations: int = 0
    error: Optional[str] = None


@dataclass
class CycleResult:
    """Result of a complete cycle."""
    cycle: int
    duration_ms: float
    agents_active: int
    total_tool_calls: int
    total_tokens: int
    messages_sent: int
    agent_results: List[AgentTurnResult] = field(default_factory=list)


@dataclass
class SwarmResult:
    """Final result after swarm execution completes."""
    total_cycles: int
    total_tokens: int
    total_tool_calls: int
    total_messages: int
    duration_s: float
    stop_reason: str
    cycle_results: List[CycleResult] = field(default_factory=list)


class SwarmRuntime:
    """
    Main execution engine for the swarm.

    Manages:
    - Agent lifecycle (spawn, terminate)
    - Clock cycles with parallel agent execution
    - Shared storage and messaging
    - Telemetry collection
    """

    def __init__(
        self,
        llm_pool: Optional[LLMClientPool] = None,
        max_concurrent_agents: int = 10,
        max_tool_iterations: int = 10,
        char_budget_per_turn: Optional[int] = None,
        entropy_enabled: bool = True,
        entropy_tipping_point: float = 15.0,
        entropy_steepness: float = 0.5
    ):
        """
        Initialize the swarm runtime.

        Args:
            llm_pool: Pool of LLM clients (creates default if None)
            max_concurrent_agents: Max agents to call concurrently
            max_tool_iterations: Max tool iterations per agent turn
            char_budget_per_turn: Character budget for write ops per turn (None = unlimited)
            entropy_enabled: Whether to apply storage entropy (decay)
            entropy_tipping_point: Cycles before decay accelerates
            entropy_steepness: Decay curve sharpness (higher = sharper)
        """
        # Core components
        self.llm_pool = llm_pool or LLMClientPool()
        self.storage = InMemoryStorage()
        self.telemetry = TelemetryCollector()
        self.messaging = MessagingSystem(self.storage)
        self.primitives = SwarmPrimitives(self.storage, self.messaging, self.telemetry)
        self.tool_registry = create_swarm_registry(self.primitives)

        # Configuration
        self.max_concurrent_agents = max_concurrent_agents
        self.max_tool_iterations = max_tool_iterations
        self.char_budget_per_turn = char_budget_per_turn
        self._semaphore = asyncio.Semaphore(max_concurrent_agents)

        # Entropy configuration
        self._entropy_enabled = entropy_enabled
        self._entropy_tipping_point = entropy_tipping_point
        self._entropy_steepness = entropy_steepness
        self._entropy_rot_levels: Dict[str, float] = {}  # Deprecated: rot levels now computed analytically

        # State
        self._agents: Dict[str, Agent] = {}
        self._current_cycle = 0
        self._running = False
        self._stop_requested = False

        # Activity callbacks (for watch mode)
        self._on_agent_content: Optional[Callable[[str, str], None]] = None  # (agent_id, content)
        self._on_agent_thinking: Optional[Callable[[str, str], None]] = None  # (agent_id, thinking)
        self._on_agent_tool_call: Optional[Callable[[str, str, Dict], None]] = None  # (agent_id, tool, args)
        self._on_agent_tool_result: Optional[Callable[[str, str, Dict, Dict], None]] = None  # (agent_id, tool, args, result)
        self._on_agent_turn_end: Optional[Callable[[str], None]] = None  # (agent_id,)

        entropy_status = "enabled" if entropy_enabled else "disabled"
        logger.info(
            f"SwarmRuntime initialized: max_concurrent={max_concurrent_agents}, "
            f"max_tool_iterations={max_tool_iterations}, entropy={entropy_status}"
        )
        if entropy_enabled:
            logger.info(f"Entropy settings: tipping_point={entropy_tipping_point}, steepness={entropy_steepness}")

    def set_activity_callbacks(
        self,
        on_content: Optional[Callable[[str, str], None]] = None,
        on_thinking: Optional[Callable[[str, str], None]] = None,
        on_tool_call: Optional[Callable[[str, str, Dict], None]] = None,
        on_tool_result: Optional[Callable[[str, str, Dict, Dict], None]] = None,
        on_turn_end: Optional[Callable[[str], None]] = None
    ) -> None:
        """
        Set callbacks for watching agent activity.

        Args:
            on_content: Called when agent produces content (agent_id, content)
            on_thinking: Called when agent produces thinking (agent_id, thinking)
            on_tool_call: Called when agent calls a tool (agent_id, tool_name, args)
            on_tool_result: Called when tool returns result (agent_id, tool_name, args, result)
            on_turn_end: Called when agent finishes its turn (agent_id)
        """
        self._on_agent_content = on_content
        self._on_agent_thinking = on_thinking
        self._on_agent_tool_call = on_tool_call
        self._on_agent_tool_result = on_tool_result
        self._on_agent_turn_end = on_turn_end

    def spawn_agent(
        self,
        system_prompt: str,
        model: str,
        agent_id: Optional[str] = None
    ) -> Agent:
        """
        Spawn a new agent in the swarm.

        Args:
            system_prompt: The agent's system prompt
            model: The LLM model ID to use for this agent
            agent_id: Optional specific ID (auto-generated if None)

        Returns:
            The created Agent
        """
        agent = Agent.create(system_prompt, model, agent_id)
        agent.spawn_cycle = self._current_cycle

        self._agents[agent.id] = agent

        # Register in primitives for list_agents()
        self.primitives.register_agent(agent.id, {
            "status": agent.status.value,
            "cycles_active": agent.cycles_active,
            "model": model
        })

        # Record telemetry
        self.telemetry.record_event(
            EventType.AGENT_SPAWN,
            agent_id=agent.id,
            data={"system_prompt": system_prompt, "model": model}
        )

        logger.info(f"Spawned agent: {agent.id} (model={model})")
        return agent

    def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent (removes from future cycles).

        Args:
            agent_id: ID of agent to terminate

        Returns:
            True if agent was found and terminated
        """
        if agent_id in self._agents:
            self._agents[agent_id].terminate()
            self.primitives.unregister_agent(agent_id)

            self.telemetry.record_event(
                EventType.AGENT_TERMINATE,
                agent_id=agent_id
            )

            logger.info(f"Terminated agent: {agent_id}")
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[Agent]:
        """List all agents."""
        return list(self._agents.values())

    def get_active_agents(self) -> List[Agent]:
        """Get all active agents."""
        return [a for a in self._agents.values() if a.is_active()]

    async def _apply_storage_entropy(self) -> None:
        """
        Apply storage entropy at the end of each cycle.

        Uses sigmoid-based decay to gradually corrupt stored text over time.
        Preserves structural markers (whitespace, punctuation) and removes
        items that exceed the death threshold.
        """
        if not self._entropy_enabled:
            return

        from .entropy import apply_entropy_to_storage

        entropy_result = apply_entropy_to_storage(
            storage_data=self.storage._data,
            current_cycle=self._current_cycle,
            tipping_point=self._entropy_tipping_point,
            steepness=self._entropy_steepness,
        )

        # Keep rot_levels for telemetry/export (no longer needed for marginal calc)
        self._entropy_rot_levels = entropy_result.rot_levels

        # Record telemetry event if any items were processed
        if entropy_result.items_processed > 0:
            self.telemetry.record_event(
                EventType.STORAGE_ENTROPY,
                data=entropy_result.to_dict()
            )

    async def _execute_agent_turn(self, agent: Agent) -> AgentTurnResult:
        """
        Execute a single agent's turn within a cycle.

        Args:
            agent: The agent to execute

        Returns:
            AgentTurnResult with turn outcome
        """
        start_time = time_provider.now()

        # Record turn start
        self.telemetry.record_event(
            EventType.AGENT_TURN_START,
            agent_id=agent.id
        )

        try:
            # Set agent context for primitives
            self.primitives.set_current_agent(agent.id)

            # Reset character budget for this turn
            reset_turn_budget(self.char_budget_per_turn)

            # Build initial messages for this cycle
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": f"Cycle {self._current_cycle}. Continue your work."}
                # {"role": "user", "content": "."}
            ]

            # Tool executor that uses the registry
            async def tool_executor(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
                return await self.tool_registry.execute_tool(name, args)

            # Callbacks for logging and activity watching
            def on_tool_call(name: str, args: Dict[str, Any]):
                logger.debug(f"[{agent.id}] Tool call: {name}")
                # Activity callback only - telemetry recorded in on_tool_result
                if self._on_agent_tool_call:
                    self._on_agent_tool_call(agent.id, name, args)

            def on_tool_result(name: str, args: Dict[str, Any], result: Dict[str, Any]):
                logger.debug(f"[{agent.id}] Tool result: {name} -> success={result.get('success')}")
                # Record combined tool call event with args and result
                self.telemetry.record_event(
                    EventType.TOOL_CALL,
                    agent_id=agent.id,
                    data={"tool": name, "args": args, "result": result}
                )
                # Activity callback (passes args for context like send_message content)
                if self._on_agent_tool_result:
                    self._on_agent_tool_result(agent.id, name, args, result)

            # Thinking callback - records reasoning to telemetry and activity
            def on_thinking(text: str):
                if text.strip():
                    self.telemetry.record_event(
                        EventType.AGENT_THINKING,
                        agent_id=agent.id,
                        data={"content": text}
                    )
                    if self._on_agent_thinking:
                        self._on_agent_thinking(agent.id, text)

            # Content callback - records output to telemetry and activity
            def on_content(text: str):
                if text.strip():
                    self.telemetry.record_event(
                        EventType.AGENT_CONTENT,
                        agent_id=agent.id,
                        data={"content": text}
                    )
                if self._on_agent_content:
                    self._on_agent_content(agent.id, text)

            # Get the LLM client for this agent's model
            llm_client = self.llm_pool.get_client(agent.model)

            # Execute the turn with multi-step tool calling
            turn_result = await llm_client.agent_turn(
                messages=messages,
                tools=self.tool_registry.get_tool_definitions(),
                tool_executor=tool_executor,
                max_iterations=self.max_tool_iterations,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                on_thinking=on_thinking,
                on_content=on_content
            )

            # Update agent state (cycles_active already incremented at cycle start)
            agent.last_activity_cycle = self._current_cycle
            agent.record_turn_usage(
                tokens=turn_result.usage.total_tokens,
                tool_calls=len(turn_result.tool_calls)
            )

            # Record turn end
            self.telemetry.record_event(
                EventType.AGENT_TURN_END,
                agent_id=agent.id,
                data={
                    "tokens": turn_result.usage.total_tokens,
                    "tool_calls": len(turn_result.tool_calls),
                    "iterations": turn_result.iterations
                }
            )

            # Activity callback
            if self._on_agent_turn_end:
                self._on_agent_turn_end(agent.id)

            return AgentTurnResult(
                agent_id=agent.id,
                success=True,
                content=turn_result.content,
                tool_calls=len(turn_result.tool_calls),
                tokens_used=turn_result.usage.total_tokens,
                iterations=turn_result.iterations
            )

        except Exception as e:
            logger.error(f"Agent {agent.id} turn failed: {e}")

            self.telemetry.record_event(
                EventType.TOOL_ERROR,
                agent_id=agent.id,
                data={"error": str(e), "source": "swarm.runtime"}
            )

            # Record turn end even on failure to maintain telemetry symmetry
            self.telemetry.record_event(
                EventType.AGENT_TURN_END,
                agent_id=agent.id,
                data={
                    "tokens": 0,
                    "tool_calls": 0,
                    "iterations": 0,
                    "error": str(e)
                }
            )

            return AgentTurnResult(
                agent_id=agent.id,
                success=False,
                error=str(e)
            )

        finally:
            # Clear agent context
            self.primitives.set_current_agent(None)

    async def run_cycle(self) -> CycleResult:
        """
        Run a single cycle where all active agents execute their turns.

        Returns:
            CycleResult with cycle outcome
        """
        self._current_cycle += 1
        cycle_start = time_provider.now()

        active_agents = self.get_active_agents()
        logger.info(f"======================================================================================")
        logger.info(f"=== Cycle {self._current_cycle} starting with {len(active_agents)} agents")
        logger.info(f"======================================================================================")

        # Update messaging and storage cycle
        self.messaging.set_cycle(self._current_cycle)
        self.storage.set_current_cycle(self._current_cycle)

        # Record cycle start
        self.telemetry.start_cycle(self._current_cycle, len(active_agents))

        # Reset agent turn state and increment cycles_active for all agents
        # (done at cycle start so all agents see consistent values during the cycle)
        for agent in active_agents:
            agent.reset_turn_state()
            agent.cycles_active += 1
            self.primitives.register_agent(agent.id, {
                "status": agent.status.value,
                "cycles_active": agent.cycles_active
            })

        # Execute all agents in parallel with semaphore
        async def execute_with_semaphore(agent: Agent) -> AgentTurnResult:
            async with self._semaphore:
                return await self._execute_agent_turn(agent)

        results = await asyncio.gather(
            *[execute_with_semaphore(agent) for agent in active_agents],
            return_exceptions=True
        )

        # Process results
        agent_results: List[AgentTurnResult] = []
        total_tool_calls = 0
        total_tokens = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Agent execution exception: {result}")
                agent_results.append(AgentTurnResult(
                    agent_id="unknown",
                    success=False,
                    error=str(result)
                ))
            else:
                agent_results.append(result)
                total_tool_calls += result.tool_calls
                total_tokens += result.tokens_used

        # Count messages sent this cycle (from send_message tool calls)
        messages_sent = len([
            e for e in self.telemetry.get_events(
                event_type=EventType.TOOL_CALL,
                cycle=self._current_cycle
            )
            if e.data.get("tool") == "send_message"
        ])

        # Record cycle end
        duration_ms = (time_provider.now() - cycle_start).total_seconds() * 1000
        self.telemetry.end_cycle(
            tool_calls=total_tool_calls,
            messages_sent=messages_sent,
            tokens=total_tokens
        )

        # Apply storage entropy (placeholder for future decay logic)
        await self._apply_storage_entropy()

        cycle_result = CycleResult(
            cycle=self._current_cycle,
            duration_ms=duration_ms,
            agents_active=len(active_agents),
            total_tool_calls=total_tool_calls,
            total_tokens=total_tokens,
            messages_sent=messages_sent,
            agent_results=agent_results
        )

        # Log summary
        logger.info(
            f"=== Cycle {self._current_cycle} complete: "
            f"{duration_ms:.1f}ms, {total_tool_calls} tool calls, "
            f"{messages_sent} messages, {total_tokens} tokens ==="
        )

        return cycle_result

    async def run(
        self,
        max_cycles: int = 100,
        on_cycle_complete: Optional[Callable[[CycleResult], None]] = None
    ) -> SwarmResult:
        """
        Run the swarm for multiple cycles.

        Args:
            max_cycles: Maximum number of cycles to run
            on_cycle_complete: Optional callback after each cycle

        Returns:
            SwarmResult with final outcome
        """
        if self._running:
            raise RuntimeError("Swarm is already running")

        self._running = True
        self._stop_requested = False
        swarm_start = time_provider.now()

        # Record swarm start
        self.telemetry.record_event(
            EventType.SWARM_START,
            data={"max_cycles": max_cycles, "agents": len(self._agents)}
        )

        cycle_results: List[CycleResult] = []
        total_tokens = 0
        total_tool_calls = 0
        total_messages = 0
        stop_reason = "max_cycles"

        try:
            for cycle_num in range(max_cycles):
                # Check for stop request
                if self._stop_requested:
                    stop_reason = "manual_stop"
                    break

                # Check for active agents
                if not self.get_active_agents():
                    stop_reason = "no_active_agents"
                    break

                # Run cycle
                result = await self.run_cycle()
                cycle_results.append(result)

                total_tokens += result.total_tokens
                total_tool_calls += result.total_tool_calls
                total_messages += result.messages_sent

                # Callback
                if on_cycle_complete:
                    on_cycle_complete(result)

        except Exception as e:
            logger.error(f"Swarm execution failed: {e}")
            stop_reason = f"error: {str(e)}"

        finally:
            self._running = False

        duration_s = (time_provider.now() - swarm_start).total_seconds()

        # Record swarm stop
        self.telemetry.record_event(
            EventType.SWARM_STOP,
            data={
                "stop_reason": stop_reason,
                "cycles": len(cycle_results),
                "total_tokens": total_tokens
            }
        )

        logger.info(
            f"Swarm complete: {len(cycle_results)} cycles, "
            f"{duration_s:.1f}s, {total_tokens} tokens, "
            f"stop_reason={stop_reason}"
        )

        return SwarmResult(
            total_cycles=len(cycle_results),
            total_tokens=total_tokens,
            total_tool_calls=total_tool_calls,
            total_messages=total_messages,
            duration_s=duration_s,
            stop_reason=stop_reason,
            cycle_results=cycle_results
        )

    def stop(self) -> None:
        """Request the swarm to stop after the current cycle."""
        logger.info("Stop requested")
        self._stop_requested = True

    def is_running(self) -> bool:
        """Check if the swarm is currently running."""
        return self._running

    def get_telemetry(self) -> TelemetryCollector:
        """Get the telemetry collector."""
        return self.telemetry

    def export_telemetry_graphml(self, path: str) -> None:
        """Export message graph to GraphML file."""
        self.telemetry.export_graphml(path)
        logger.info(f"Message graph exported to: {path}")

    def print_cycle_summary(self, cycle: Optional[int] = None) -> None:
        """Print a summary of a specific cycle or the last cycle."""
        target_cycle = cycle if cycle is not None else self._current_cycle
        self.telemetry.print_cycle_summary(target_cycle)

    def _build_resume_state(self) -> SwarmState:
        """
        Build complete swarm state for resume capability.

        Returns:
            SwarmState containing all information needed to resume
        """
        # Get complete storage snapshot (including _inbox_* keys)
        full_storage = self.storage.get_snapshot()

        # Build agent states with full system prompts
        agent_states = []
        for agent in self._agents.values():
            agent_states.append(AgentState(
                id=agent.id,
                model=agent.model,
                system_prompt=agent.system_prompt,  # Full prompt, not truncated
                tokens_used=agent.tokens_used,
                cycles_active=agent.cycles_active,
                last_activity_cycle=agent.last_activity_cycle,
                status=agent.status.value,
                spawn_cycle=agent.spawn_cycle
            ))

        # Get telemetry data
        events = [e.to_dict() for e in self.telemetry._events]
        summaries = [s.to_dict() for s in self.telemetry.get_cycle_summaries()]

        return SwarmState(
            version=STATE_VERSION,
            current_cycle=self._current_cycle,
            messaging_cycle=self.messaging.current_cycle,
            storage=full_storage,
            agents=agent_states,
            telemetry_events=events,
            cycle_summaries=summaries,
            entropy_rot_levels=self._entropy_rot_levels.copy(),
            can_resume=True
        )

    def export_telemetry_json(self, path: str) -> None:
        """Export telemetry to JSON file, with swarm_state as primary data repository."""
        # Build complete swarm state for resume capability
        swarm_state = self._build_resume_state()

        self.telemetry.export_json(
            path,
            swarm_state=swarm_state.to_dict()
        )
        logger.info(f"Telemetry exported to: {path}")

    @classmethod
    def restore_from_file(
        cls,
        json_path: str,
        llm_pool: Optional[LLMClientPool] = None,
        max_concurrent_agents: int = 10,
        max_tool_iterations: int = 10,
        char_budget_per_turn: Optional[int] = None,
        entropy_enabled: bool = True,
        entropy_tipping_point: float = 15.0,
        entropy_steepness: float = 0.5
    ) -> "SwarmRuntime":
        """
        Restore a SwarmRuntime from a JSON export file.

        Args:
            json_path: Path to the JSON file with swarm_state
            llm_pool: Optional LLMClientPool (creates default if None)
            max_concurrent_agents: Max agents to call concurrently
            max_tool_iterations: Max tool iterations per agent turn
            char_budget_per_turn: Character budget for write ops per turn (None = unlimited)
            entropy_enabled: Whether to apply storage entropy (decay)
            entropy_tipping_point: Cycles before decay accelerates
            entropy_steepness: Decay curve sharpness (higher = sharper)

        Returns:
            SwarmRuntime restored to the saved state

        Raises:
            ValueError: If the file doesn't contain valid swarm_state
            FileNotFoundError: If the file doesn't exist
        """
        logger.info(f"Restoring swarm from: {json_path}")

        # Load and parse JSON
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # Parse swarm state
        state = SwarmState.from_json(json_data)

        # Create new runtime instance
        runtime = cls(
            llm_pool=llm_pool,
            max_concurrent_agents=max_concurrent_agents,
            max_tool_iterations=max_tool_iterations,
            char_budget_per_turn=char_budget_per_turn,
            entropy_enabled=entropy_enabled,
            entropy_tipping_point=entropy_tipping_point,
            entropy_steepness=entropy_steepness
        )

        # Restore cycle counters
        runtime._current_cycle = state.current_cycle
        runtime.messaging.current_cycle = state.messaging_cycle
        runtime.storage.set_current_cycle(state.current_cycle)

        # Restore storage (complete, including _inbox_* keys)
        runtime.storage._data = state.storage.copy()
        logger.info(f"Restored {len(state.storage)} storage keys")

        # Entropy rot levels no longer needed for marginal decay (computed analytically)
        # but keep for backward compatibility with saved state files
        runtime._entropy_rot_levels = state.entropy_rot_levels.copy()

        # Restore agents
        for agent_state in state.agents:
            # Create agent with restored state
            agent = Agent(
                id=agent_state.id,
                system_prompt=agent_state.system_prompt,
                model=agent_state.model,
                spawn_cycle=agent_state.spawn_cycle,
                tokens_used=agent_state.tokens_used,
                cycles_active=agent_state.cycles_active,
                last_activity_cycle=agent_state.last_activity_cycle,
                status=AgentStatus(agent_state.status)
            )

            runtime._agents[agent.id] = agent

            # Register in primitives if active
            if agent.is_active():
                runtime.primitives.register_agent(agent.id, {
                    "status": agent.status.value,
                    "cycles_active": agent.cycles_active,
                    "model": agent.model
                })

        logger.info(f"Restored {len(runtime._agents)} agents")

        # Restore telemetry
        runtime._restore_telemetry(state)

        # Record resume event
        runtime.telemetry.record_event(
            EventType.SWARM_RESUME,
            data={
                "restored_from": json_path,
                "previous_cycles": state.current_cycle,
                "agents_restored": len(state.agents)
            }
        )

        logger.info(
            f"Swarm restored: cycle={state.current_cycle}, "
            f"agents={len(runtime._agents)}"
        )

        return runtime

    def _restore_telemetry(self, state: SwarmState) -> None:
        """
        Restore telemetry events and cycle summaries from state.

        Args:
            state: The SwarmState containing telemetry data
        """
        # Restore events
        for event_data in state.telemetry_events:
            event = TelemetryEvent(
                event_type=EventType(event_data["event_type"]),
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                cycle=event_data["cycle"],
                agent_id=event_data.get("agent_id"),
                data=event_data.get("data", {}),
                id=event_data["id"]
            )
            self.telemetry._events.append(event)

        # Restore cycle summaries
        for summary_data in state.cycle_summaries:
            summary = CycleSummary(
                cycle=summary_data["cycle"],
                start_time=datetime.fromisoformat(summary_data["start_time"]),
                end_time=datetime.fromisoformat(summary_data["end_time"]) if summary_data.get("end_time") else None,
                duration_ms=summary_data.get("duration_ms"),
                agents_active=summary_data.get("agents_active", 0),
                total_tool_calls=summary_data.get("total_tool_calls", 0),
                total_messages_sent=summary_data.get("total_messages_sent", 0),
                total_tokens=summary_data.get("total_tokens", 0),
                storage_reads=summary_data.get("storage_reads", 0),
                storage_writes=summary_data.get("storage_writes", 0)
            )
            self.telemetry._cycle_summaries[summary.cycle] = summary

        # Set current cycle in telemetry
        self.telemetry._current_cycle = state.current_cycle

        logger.info(
            f"Restored telemetry: {len(state.telemetry_events)} events, "
            f"{len(state.cycle_summaries)} cycle summaries"
        )

    def reconcile_agents(self, requested_agents: List[tuple]) -> None:
        """
        Reconcile agents after restore with a new agent list.

        Handles the case where --agents differs from restored state:
        - Agents in restored but not requested: terminate them
        - Agents in requested but not restored: spawn fresh

        Args:
            requested_agents: List of (model, prompt_name, agent_id) tuples
        """
        requested_ids = {agent_id for _, _, agent_id in requested_agents}
        current_ids = set(self._agents.keys())

        # Terminate agents not in requested list
        to_terminate = current_ids - requested_ids
        for agent_id in to_terminate:
            logger.info(f"Terminating agent not in requested list: {agent_id}")
            self.terminate_agent(agent_id)

        # Note: Spawning new agents should be done by the caller
        # as they need access to AGENT_PROMPTS
        missing_ids = requested_ids - current_ids
        if missing_ids:
            logger.info(f"Agents to spawn (not in restored state): {missing_ids}")
