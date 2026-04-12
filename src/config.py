"""Central evaluation configuration passed through the pipeline."""

from dataclasses import dataclass, field

from core.criteria import GEVAL_CRITERIA
from core.prompts import CUSTOMER_CHATBOT_ROLE, EXPECTED_OUTCOME, SCENARIO, USER_DESCRIPTION


@dataclass
class EvalConfig:
    """All runtime-configurable settings for a single evaluation run."""

    chatbot_role: str = CUSTOMER_CHATBOT_ROLE
    scenario: str = SCENARIO
    user_description: str = USER_DESCRIPTION
    expected_outcome: str = EXPECTED_OUTCOME

    run_role_adherence: bool = True
    run_conversation_completeness: bool = True

    geval_criteria: dict[str, str] = field(default_factory=lambda: dict(GEVAL_CRITERIA))

    run_tool_call_accuracy: bool = True
    run_agent_goal_accuracy: bool = True

    tone_criteria: list[dict[str, str]] = field(default_factory=list)
