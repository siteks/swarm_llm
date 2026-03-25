"""
LLM Client for swarm agents.

Adapted from backend_reference/llm_client.py with focus on multi-step
tool calling within a single agent turn.
"""

import os
import json
import logging
import asyncio
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator

import litellm
from litellm import stream_chunk_builder
from litellm.exceptions import ContextWindowExceededError

# Suppress Pydantic serializer warnings from LiteLLM's stream_chunk_builder
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Configure LiteLLM
litellm.drop_params = True
litellm.set_verbose = False

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


@dataclass
class UsageStats:
    """Token usage statistics from LLM response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: Optional[str] = None
    thinking_tokens: int = 0


@dataclass
class ToolCallRecord:
    """Record of a tool call made during a turn."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]


@dataclass
class TurnResult:
    """
    Result of a complete agent turn.

    A turn may include multiple LLM calls and tool executions
    as the agent reasons through its actions.
    """
    content: str  # Final response content
    tool_calls: List[ToolCallRecord]  # All tool calls made
    usage: UsageStats  # Total token usage
    iterations: int  # Number of LLM calls in this turn


class LLMClient:
    """
    LLM client for swarm agents with multi-step tool calling.

    Supports the tool execution loop where an agent can:
    1. Make tool calls
    2. See results
    3. Reason and make more tool calls
    4. Repeat until done or limit reached

    Environment variables:
    - LLM_MODEL: Model identifier (default: claude-sonnet-4-5-20250929)
    - LLM_MAX_TOKENS: Maximum tokens per completion (default: 4096)
    """

    def __init__(self, model_override: Optional[str] = None):
        self.model = model_override or os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        logger.info(f"LLM Client initialized: model={self.model}, max_tokens={self.max_tokens}")

    def _get_provider(self, model: str) -> str:
        """Infer provider from model name."""
        model_lower = model.lower()
        if "claude" in model_lower:
            return "anthropic"
        elif "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "gemini" in model_lower:
            return "google"
        elif "deepseek" in model_lower:
            return "deepseek"
        return "unknown"

    def _supports_reasoning(self, model: str) -> bool:
        """Check if model supports extended thinking/reasoning."""
        try:
            return litellm.supports_reasoning(model=model)
        except Exception:
            # Fallback pattern matching
            patterns = ["claude-opus-4", "claude-sonnet-4", "gemini", "deepseek"]
            return any(p in model.lower() for p in patterns)

    def _convert_tools_to_anthropic(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Anthropic-native format.

        Avoids relying on litellm's conversion which can lose the input_schema.type field.
        """
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                parameters = func.get("parameters", {})

                # Ensure type is always present in input_schema
                input_schema = {
                    "type": parameters.get("type", "object"),
                    "properties": parameters.get("properties", {}),
                }
                if "required" in parameters:
                    input_schema["required"] = parameters["required"]

                anthropic_tools.append({
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "input_schema": input_schema
                })
            else:
                # Pass through non-function tools as-is
                anthropic_tools.append(tool)

        return anthropic_tools

    def _build_params(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Build parameters for LLM completion request."""
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        if tools:
            provider = self._get_provider(self.model)
            if provider == "anthropic":
                # Use native Anthropic format to avoid litellm conversion issues
                params["tools"] = self._convert_tools_to_anthropic(tools)
            else:
                params["tools"] = tools
            params["tool_choice"] = "auto"

        # Add reasoning parameters if supported
        if self._supports_reasoning(self.model):
            provider = self._get_provider(self.model)
            if provider == "anthropic":
                thinking_budget = min(int(self.max_tokens * 0.5), 10000)
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
                # Enable interleaved thinking with tools
                if tools:
                    params["extra_headers"] = {
                        "anthropic-beta": "interleaved-thinking-2025-05-14"
                    }
            else:
                params["reasoning_effort"] = "medium"

        return params

    async def agent_turn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_executor: callable,
        max_iterations: int = 10,
        on_content: Optional[callable] = None,
        on_thinking: Optional[callable] = None,
        on_tool_call: Optional[callable] = None,
        on_tool_result: Optional[callable] = None
    ) -> TurnResult:
        """
        Execute a complete agent turn with multi-step tool calling.

        The agent can make multiple tool calls, see results, and continue
        reasoning until it stops calling tools or reaches max_iterations.

        Args:
            messages: Initial messages (system + user)
            tools: Tool definitions in OpenAI format
            tool_executor: Async function to execute tools: (name, args) -> result
            max_iterations: Maximum number of LLM calls in this turn
            on_content: Callback for aggregated content after each iteration: (text) -> None
            on_thinking: Callback for thinking chunks: (text) -> None
            on_tool_call: Callback when tool is called: (name, args) -> None
            on_tool_result: Callback when tool completes: (name, args, result) -> None

        Returns:
            TurnResult with final content, all tool calls, and usage stats
        """
        current_messages = messages.copy()
        all_tool_calls: List[ToolCallRecord] = []
        total_usage = UsageStats(model=self.model)
        iteration = 0
        final_content = ""

        while iteration < max_iterations:
            iteration += 1
            logger.debug(f"Agent turn iteration {iteration}/{max_iterations}")

            # Build request parameters
            params = self._build_params(current_messages, tools)

            try:
                # Make streaming request
                response = await litellm.acompletion(**params)

                # Accumulate response
                accumulated_content = ""
                accumulated_thinking = ""
                accumulated_tool_calls = []
                iteration_usage = UsageStats()
                collected_chunks = []  # For stream_chunk_builder

                async for chunk in response:
                    collected_chunks.append(chunk)
                    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta

                        # Handle thinking content (accumulate, don't emit per-chunk)
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                            accumulated_thinking += delta.reasoning_content

                        # Handle regular content (accumulate, don't emit per-chunk)
                        if hasattr(delta, "content") and delta.content:
                            accumulated_content += delta.content

                        # Accumulate tool calls
                        if hasattr(delta, "tool_calls") and delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                if hasattr(tc_delta, "index"):
                                    idx = tc_delta.index
                                    while len(accumulated_tool_calls) <= idx:
                                        accumulated_tool_calls.append({
                                            "id": None,
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        })

                                    if hasattr(tc_delta, "id") and tc_delta.id:
                                        accumulated_tool_calls[idx]["id"] = tc_delta.id
                                    if hasattr(tc_delta, "function"):
                                        if hasattr(tc_delta.function, "name") and tc_delta.function.name:
                                            accumulated_tool_calls[idx]["function"]["name"] += tc_delta.function.name
                                        if hasattr(tc_delta.function, "arguments") and tc_delta.function.arguments:
                                            accumulated_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

                    # Capture usage from final chunk
                    if hasattr(chunk, "usage") and chunk.usage:
                        iteration_usage.prompt_tokens = chunk.usage.prompt_tokens or 0
                        iteration_usage.completion_tokens = chunk.usage.completion_tokens or 0
                        iteration_usage.total_tokens = chunk.usage.total_tokens or 0
                        iteration_usage.thinking_tokens = getattr(chunk.usage, "thinking_tokens", 0) or 0

                # Accumulate total usage
                total_usage.prompt_tokens += iteration_usage.prompt_tokens
                total_usage.completion_tokens += iteration_usage.completion_tokens
                total_usage.total_tokens += iteration_usage.total_tokens
                total_usage.thinking_tokens += iteration_usage.thinking_tokens

                # Filter completed tool calls
                completed_tool_calls = [
                    tc for tc in accumulated_tool_calls
                    if tc.get("id") and tc.get("function", {}).get("name")
                ]

                # Emit aggregated thinking for this iteration
                if accumulated_thinking and on_thinking:
                    on_thinking(accumulated_thinking)

                # Emit aggregated content for this iteration
                if accumulated_content and on_content:
                    on_content(accumulated_content)

                # If no tool calls, turn is complete
                if not completed_tool_calls:
                    final_content = accumulated_content
                    logger.debug(f"Agent turn complete after {iteration} iterations (no tool calls)")
                    break

                # Process tool calls
                logger.debug(f"Processing {len(completed_tool_calls)} tool calls")

                # Reconstruct complete message to get thinking_blocks with signatures
                # This is required by Anthropic when using extended thinking with tools
                assistant_message = {
                    "role": "assistant",
                    "content": accumulated_content or "",
                    "tool_calls": completed_tool_calls
                }

                try:
                    complete_response = stream_chunk_builder(
                        chunks=collected_chunks,
                        messages=current_messages
                    )

                    if complete_response and hasattr(complete_response, 'choices') and len(complete_response.choices) > 0:
                        complete_message = complete_response.choices[0].message

                        # Preserve thinking_blocks if present (includes Anthropic's signature)
                        if hasattr(complete_message, 'thinking_blocks') and complete_message.thinking_blocks:
                            assistant_message["thinking_blocks"] = complete_message.thinking_blocks
                            logger.debug(f"Preserved {len(complete_message.thinking_blocks)} thinking block(s)")

                        # Use content from complete message if available
                        if hasattr(complete_message, 'content') and complete_message.content:
                            assistant_message["content"] = complete_message.content

                except Exception as e:
                    logger.warning(f"Failed to reconstruct complete message: {e}")

                current_messages.append(assistant_message)

                # Parse and fix tool call arguments
                # For malformed args, create placeholder values so Anthropic accepts the
                # message history, but the tool will return a useful error to the LLM
                for tool_call in completed_tool_calls:
                    tool_name = tool_call["function"]["name"]
                    args_str = tool_call["function"]["arguments"]

                    try:
                        parsed_args = json.loads(args_str) if args_str else {}
                        if not isinstance(parsed_args, dict):
                            parsed_args = {}
                    except json.JSONDecodeError:
                        parsed_args = {}

                    # If args are empty/malformed, fill in required params with placeholders
                    # This makes the tool call syntactically valid (Anthropic accepts it)
                    # but semantically invalid (tool returns useful error)
                    tool_def = next(
                        (t for t in tools if t.get("function", {}).get("name") == tool_name),
                        None
                    )
                    if not parsed_args:
                        if tool_def:
                            params = tool_def.get("function", {}).get("parameters", {})
                            required = params.get("required", [])
                            properties = params.get("properties", {})
                            for param in required:
                                prop_type = properties.get(param, {}).get("type", "string")
                                if prop_type == "string":
                                    parsed_args[param] = ""
                                elif prop_type == "boolean":
                                    parsed_args[param] = False
                                elif prop_type == "number" or prop_type == "integer":
                                    parsed_args[param] = 0
                                else:
                                    parsed_args[param] = None
                            # Update the tool call's arguments so Anthropic sees valid JSON
                            tool_call["function"]["arguments"] = json.dumps(parsed_args)
                            logger.debug(f"Fixed malformed args for {tool_name}: {parsed_args}")

                    # Check for type mismatches and coerce to empty values of correct type
                    # This handles cases where LLM passes wrong type (e.g., list instead of string)
                    # The tool validation will catch empty values and return a useful error
                    if tool_def and parsed_args:
                        params = tool_def.get("function", {}).get("parameters", {})
                        properties = params.get("properties", {})
                        args_modified = False

                        for param, value in list(parsed_args.items()):
                            prop_type = properties.get(param, {}).get("type", "string")

                            # Check if value matches expected type
                            type_ok = (
                                (prop_type == "string" and isinstance(value, str)) or
                                (prop_type == "boolean" and isinstance(value, bool)) or
                                (prop_type in ("number", "integer") and isinstance(value, (int, float)) and not isinstance(value, bool)) or
                                (prop_type == "array" and isinstance(value, list)) or
                                (prop_type == "object" and isinstance(value, dict))
                            )

                            if not type_ok:
                                # Coerce to empty value of correct type - tool validation will catch it
                                logger.debug(f"Type mismatch for {tool_name}.{param}: expected {prop_type}, got {type(value).__name__}")
                                if prop_type == "string":
                                    parsed_args[param] = ""
                                elif prop_type == "boolean":
                                    parsed_args[param] = False
                                elif prop_type in ("number", "integer"):
                                    parsed_args[param] = 0
                                elif prop_type == "array":
                                    parsed_args[param] = []
                                elif prop_type == "object":
                                    parsed_args[param] = {}
                                args_modified = True

                        if args_modified:
                            # Update tool call arguments for Anthropic
                            tool_call["function"]["arguments"] = json.dumps(parsed_args)

                    tool_call["_parsed_args"] = parsed_args

                # Execute each tool and add results
                for tool_call in completed_tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["_parsed_args"]

                    if on_tool_call:
                        on_tool_call(tool_name, tool_args)

                    # Execute tool
                    tool_result = await tool_executor(tool_name, tool_args)

                    if on_tool_result:
                        on_tool_result(tool_name, tool_args, tool_result)

                    # Record tool call
                    all_tool_calls.append(ToolCallRecord(
                        tool_name=tool_name,
                        arguments=tool_args,
                        result=tool_result
                    ))

                    # Add tool result to messages
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result)
                    })

                # Continue loop for next iteration

            except ContextWindowExceededError as e:
                # Handle like max_iterations - end turn gracefully with accumulated content
                logger.warning(f"Context window exceeded in iteration {iteration}: {e}")
                final_content = accumulated_content if "accumulated_content" in dir() else ""
                break

            except Exception as e:
                logger.error(f"LLM call failed in iteration {iteration}: {e}")
                raise LLMError(f"LLM request failed: {str(e)}") from e

        # If we hit max iterations, use last accumulated content
        if iteration >= max_iterations:
            logger.warning(f"Agent turn hit max iterations ({max_iterations})")
            final_content = accumulated_content if "accumulated_content" in dir() else ""

        return TurnResult(
            content=final_content,
            tool_calls=all_tool_calls,
            usage=total_usage,
            iterations=iteration
        )

    async def simple_completion(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Tuple[str, UsageStats]:
        """
        Simple non-tool completion for testing.

        Args:
            messages: Messages to send
            system_prompt: Optional system prompt to prepend

        Returns:
            Tuple of (content, usage_stats)
        """
        full_messages = messages.copy()
        if system_prompt:
            full_messages.insert(0, {"role": "system", "content": system_prompt})

        params = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        try:
            response = await litellm.acompletion(**params)

            content = ""
            if hasattr(response, "choices") and len(response.choices) > 0:
                content = response.choices[0].message.content or ""

            usage = UsageStats(model=self.model)
            if hasattr(response, "usage") and response.usage:
                usage.prompt_tokens = response.usage.prompt_tokens or 0
                usage.completion_tokens = response.usage.completion_tokens or 0
                usage.total_tokens = response.usage.total_tokens or 0

            return content, usage

        except Exception as e:
            raise LLMError(f"LLM request failed: {str(e)}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "provider": self._get_provider(self.model),
            "supports_reasoning": self._supports_reasoning(self.model)
        }


class LLMClientPool:
    """
    Pool of LLMClients, one per model.

    Caches LLMClient instances to avoid recreating them for the same model.
    Useful when running agents with different models in the same swarm.
    """

    def __init__(self):
        self._clients: Dict[str, LLMClient] = {}

    def get_client(self, model: str) -> LLMClient:
        """Get or create an LLMClient for the specified model."""
        if model not in self._clients:
            logger.info(f"Creating LLMClient for model: {model}")
            self._clients[model] = LLMClient(model_override=model)
        return self._clients[model]

    def get_models(self) -> List[str]:
        """Get list of models currently in the pool."""
        return list(self._clients.keys())
