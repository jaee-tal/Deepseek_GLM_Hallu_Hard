"""Map factual outcomes to organic/inorganic truth and hallucination labels."""
from __future__ import annotations

STATES = ("O0", "O1", "I0", "I1")
DOMAIN_OF = {"O0": "organic", "O1": "organic", "I0": "inorganic", "I1": "inorganic"}
LABEL_OF = {"O0": 0, "O1": 1, "I0": 0, "I1": 1}

def state_for(condition: str, verdict: str, framing: str | None = None) -> tuple[str | None, int | None, str | None]:
    """Returns (state, label, exclude_reason)."""
    if verdict == "unknown":
        return None, None, "judge_unknown"
    if condition == "organic":
        if verdict == "correct":
            return "O0", 0, None
        if verdict == "incorrect":
            return "O1", 1, None
        return None, None, "organic_abstain"
    if condition == "truthful":
        if verdict == "correct":
            return "I0", 0, None
        return None, None, "truthful_failed_" + verdict
    if condition == "incorrect":
        if verdict == "incorrect":
            return "I1", 1, None
        return None, None, "incorrect_failed_" + verdict
    raise ValueError(condition)
