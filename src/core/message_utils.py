"""Message processing utilities for evaluation."""

import json
import logging
from typing import Any

from deepeval.test_case import ToolCall as DeepEvalToolCall
from deepeval.test_case import Turn
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

logger = logging.getLogger(__name__)


def build_deepeval_turns(raw_messages: list[dict[str, Any]]) -> list[Turn]:
    """Convert raw messages to DeepEval Turn objects."""
    turns = []
    last_assistant = None

    for msg in raw_messages:
        role = msg["role"]
        content = msg["content"]

        if role in ("human", "user"):
            turns.append(Turn(**msg))
            last_assistant = None

        elif role in ("ai", "assistant"):
            tool_calls = _parse_tool_calls(msg.get("tool_calls") or [])
            turn = Turn(
                role="assistant",
                content=content,
                tools_called=[
                    DeepEvalToolCall(name=parsed.name, input_parameters=parsed.args)
                    for parsed in tool_calls
                    if parsed.name
                ],
                additional_metadata={"tool_outputs": []},
            )
            turns.append(turn)
            last_assistant = turn

        elif role == "tool":
            if not last_assistant or last_assistant.additional_metadata is None:
                continue
            last_assistant.additional_metadata["tool_outputs"].append(
                {
                    "name": msg["name"],
                    "output": msg["content"],
                }
            )

    return turns


def build_ragas_messages(raw_messages: list[dict[str, Any]]) -> list[Any]:
    """Convert raw messages to Ragas message sequence."""
    messages: list[HumanMessage | AIMessage | ToolMessage] = []
    for msg in raw_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls = _parse_ragas_tool_calls(msg)
            messages.append(AIMessage(content=content, tool_calls=tool_calls))
        elif role == "tool":
            messages.append(ToolMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    return messages


def build_reference_tool_calls(expected_tools: list[dict[str, Any]]) -> list[Any]:
    """Convert expected tool references into Ragas ToolCall objects."""
    return [
        ToolCall(name=tool["name"], args=tool.get("args", {}))
        for tool in expected_tools
        if isinstance(tool, dict) and "name" in tool
    ]


def _parse_ragas_tool_calls(msg: dict[str, Any]) -> list[ToolCall]:
    """Extract Ragas ToolCall objects from a message dict."""
    return [
        ToolCall(name=parsed.name, args=parsed.args)
        for parsed in _parse_tool_calls(msg.get("tool_calls") or [])
        if parsed.name
    ]


def _parse_tool_calls(raw_tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Parse a list of raw tool call dicts into Ragas ToolCall objects."""
    return [_parse_tool_call(raw) for raw in raw_tool_calls]


def _parse_tool_call(raw_tool_call: dict[str, Any]) -> ToolCall:
    """Parse a single raw tool call dict, deserializing JSON args if needed."""
    args = raw_tool_call["args"]
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool call args: %s", args)
    return ToolCall(**{**raw_tool_call, "args": args})
