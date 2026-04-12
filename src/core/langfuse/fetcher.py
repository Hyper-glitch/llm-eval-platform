"""Langfuse fetcher — pulls production traces and converts them to eval DataFrame."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any

from langfuse.api.client import LangfuseAPI
from langfuse.api.commons.types.trace_with_full_details import TraceWithFullDetails
import pandas as pd

from settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TraceFilters:
    """Parameters for fetching traces from Langfuse."""

    from_date: datetime | None = None
    to_date: datetime | None = None
    tags: list[str] = field(default_factory=list)
    name: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    limit: int = 50


class LangfuseFetcher:
    """Fetches production traces from Langfuse and converts them to an eval DataFrame.

    The resulting DataFrame is compatible with the eval pipeline:
    columns ticket_id, trace_id, scenario, messages, expected_tools.
    """

    def __init__(self) -> None:
        self._api = LangfuseAPI(
            base_url=settings.LANGFUSE_BASE_URL,
            username=settings.LANGFUSE_PUBLIC_KEY,
            password=settings.LANGFUSE_SECRET_KEY,
        )

    def fetch(self, filters: TraceFilters) -> pd.DataFrame:
        """Fetch traces matching the filters and return as an eval-ready DataFrame."""
        traces = self._list_traces(filters)
        if not traces:
            return pd.DataFrame()

        rows = [self._trace_to_row(t) for t in traces]
        df = pd.DataFrame(rows)
        logger.info("Fetched %d traces from Langfuse", len(df))
        return df

    def _list_traces(self, filters: TraceFilters) -> list[TraceWithFullDetails]:
        """Fetch traces page by page up to filters.limit."""
        results = []
        page = 1
        page_size = min(filters.limit, 100)

        while len(results) < filters.limit:
            resp = self._api.trace.list(
                page=page,
                limit=page_size,
                from_timestamp=filters.from_date,
                to_timestamp=filters.to_date,
                tags=filters.tags or None,
                name=filters.name or None,
                user_id=filters.user_id or None,
                session_id=filters.session_id or None,
            )
            batch = resp.data if hasattr(resp, "data") else []
            if not batch:
                break

            for trace in batch:
                full = self._api.trace.get(trace.id)
                results.append(full)
                if len(results) >= filters.limit:
                    break

            if len(batch) < page_size:
                break
            page += 1

        return results

    def _trace_to_row(self, trace: TraceWithFullDetails) -> dict[str, Any]:
        return {
            "ticket_id": trace.id,
            "trace_id": trace.id,
            "scenario": trace.name or "",
            "messages": self._extract_messages(trace),
            "expected_tools": [],
        }

    def _extract_messages(self, trace: TraceWithFullDetails) -> list[dict[str, Any]]:
        """Reconstruct the conversation from trace observations.

        Strategy: find the generation observation with the most messages in its
        input (i.e. the last LLM call), which captures the full conversation
        history up to that point, then append the final output.
        """
        generations = sorted(
            [o for o in trace.observations if o.type == "GENERATION"],
            key=lambda o: o.start_time,
        )

        if generations:
            messages = self._messages_from_generation(generations[-1])
            if messages:
                return messages

        return self._messages_from_trace_io(trace)

    @staticmethod
    def _messages_from_generation(obs: Any) -> list[dict[str, Any]]:
        """Extract messages from a generation observation's input/output."""
        raw_input = obs.input
        messages: list[dict[str, Any]] = []

        if isinstance(raw_input, list):
            for msg in raw_input:
                if isinstance(msg, dict) and "role" in msg:
                    messages.append(msg)

        raw_output = obs.output
        if isinstance(raw_output, dict) and raw_output.get("role") == "assistant":
            messages.append(raw_output)
        elif isinstance(raw_output, str) and raw_output:
            messages.append({"role": "assistant", "content": raw_output})

        return messages

    @staticmethod
    def _messages_from_trace_io(trace: TraceWithFullDetails) -> list[dict[str, Any]]:
        """Fallback: build a minimal 2-message conversation from trace input/output."""
        messages: list[dict[str, Any]] = []

        inp = trace.input
        if isinstance(inp, str) and inp:
            messages.append({"role": "user", "content": inp})
        elif isinstance(inp, dict) and inp.get("content"):
            messages.append({"role": "user", "content": str(inp["content"])})

        out = trace.output
        if isinstance(out, str) and out:
            messages.append({"role": "assistant", "content": out})
        elif isinstance(out, dict) and out.get("content"):
            messages.append({"role": "assistant", "content": str(out["content"])})

        return messages
