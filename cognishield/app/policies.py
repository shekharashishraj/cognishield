INTERVENTIONS = {"scaffold", "hint", "redirect", "defer", "refuse"}

DISALLOWED_PATTERNS = [
    "give me the final answer",
    "just solve it for me",
    "write this essay for me",
    "just give me the code",
    "i am in a hurry just answer",
    "copy-paste answer",
]

DEFAULT_POLICY = {
    "allowed_support": [
        "conceptual_hint",
        "guided_question",
        "partial_scaffold",
        "error_feedback",
    ],
    "forbidden_support": [
        "full_solution",
        "copyable_final_answer",
        "complete_essay_ghostwriting",
    ],
}
