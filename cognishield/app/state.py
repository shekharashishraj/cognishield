from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognishield.app.schemas import CogniShieldState, TurnContext


def init_state(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
    learner_profile: Optional[Dict[str, Any]] = None,
    rubric_constraints: Optional[Dict[str, Any]] = None,
    task_context: Optional[Dict[str, Any]] = None,
) -> CogniShieldState:
    return CogniShieldState(
        context=TurnContext(
            user_query=user_query,
            history=history or [],
            learner_profile=learner_profile or {},
            rubric_constraints=rubric_constraints or {},
            task_context=task_context or {},
        )
    )
