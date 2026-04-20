from __future__ import annotations

import sys

import tyro

from cognishield.app.logging_setup import configure_logging, get_logger
from cognishield.app.orchestrator import run_turn
from cognishield.app.settings import Settings
from cognishield.app.state import init_state
from cognishield.app.trace import JsonlTracer

_LOG = get_logger("cognishield.cli")


def main() -> None:
    settings = tyro.cli(Settings)
    configure_logging(settings)

    query = (settings.query or "").strip()
    if not query:
        _LOG.error("No query provided. Set COGNISHIELD_QUERY or pass --query.")
        sys.exit(2)

    tracer: JsonlTracer | None = None
    if settings.trace_path or settings.trace_stdout:
        tracer = JsonlTracer(settings)
    try:
        state = init_state(
            user_query=query,
            history=[],
            learner_profile={"level": "undergrad"},
            rubric_constraints={"graded": True},
            task_context={"assignment_type": "homework"},
        )
        response = run_turn(state, settings, tracer=tracer)
    finally:
        if tracer:
            tracer.close()

    # Final answer on stdout for easy piping when trace uses stderr or file
    sys.stderr.write("\n=== FINAL RESPONSE ===\n\n")
    sys.stderr.flush()
    sys.stdout.write(response + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
