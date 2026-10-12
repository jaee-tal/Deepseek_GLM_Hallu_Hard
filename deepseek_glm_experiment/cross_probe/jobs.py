"""Generation jobs.

Organic rounds cover every question. Inorganic rounds (truthful + one round per error register) are
restricted to the questions the model demonstrably knows, so I0 and I1 come from one population
instead of the near-disjoint ones the unrestricted design produced.

Conversation ids are deterministic. The standard frame uses compact condition names;
the strong frame and non-lie prompts have their own namespaces.
"""
from __future__ import annotations

from typing import Any

from .prompts import (DEFAULT_ERROR_FRAMING, ERROR_FRAMINGS, INSTRUCTION_TYPE, TRUTHFUL_FOR, assign_templates,
                      build_messages, frame_for)
from .vendor.generation import render_prompt

CONDITION_SETS = ("organic", "inorganic", "all")


def condition_token(condition: str, framing: str | None) -> str:
    """The condition component of a stable conversation id."""
    if condition == "organic":
        return "organic"
    if condition == "truthful":
        return "truthful" if (framing or "v1") == "v1" else f"truthful_{framing}"
    return "incorrect" if (framing or DEFAULT_ERROR_FRAMING) == DEFAULT_ERROR_FRAMING else f"incorrect_{framing}"


def conversation_id(qid: str, condition: str, template_id: int | None, framing: str | None, sample: int) -> str:
    t = "na" if template_id is None else str(template_id)
    return f"{qid}__{condition_token(condition, framing)}__t{t}__s{sample}"


def error_samples_for(n_known: int, target_rows: int = 8000, lo: int = 2, hi: int = 10) -> int:
    """Samples per known question per register, so each register gets ~`target_rows` instructed rows.
    The obey rate then decides how many of those become usable I1 rows."""
    if n_known <= 0:
        return lo
    return int(min(hi, max(lo, round(target_rows / n_known))))


def build_jobs(questions: list[dict[str, Any]], split_lookup: dict[str, str], gen_cfg: dict[str, Any], tokenizer, family,
               style: str = "short_answer", framings: list[str] | None = None, conditions: str = "all",
               known_questions: set[str] | None = None, n_organic: int | None = None, n_truthful: int | None = None,
               n_error: int | None = None) -> list[dict[str, Any]]:
    if conditions not in CONDITION_SETS:
        raise SystemExit(f"unknown condition set {conditions!r}; known: {list(CONDITION_SETS)}")
    framings = list(framings if framings is not None else gen_cfg.get("error_framings") or ERROR_FRAMINGS)
    unknown = [f for f in framings if f not in ERROR_FRAMINGS]
    if unknown:
        raise SystemExit(f"unknown error framing(s) {unknown}; known: {list(ERROR_FRAMINGS)}")
    n_o = n_organic if n_organic is not None else int(gen_cfg["organic_samples_per_question"])
    n_t = n_truthful if n_truthful is not None else int(gen_cfg["truthful_samples_per_question"])
    n_e = n_error if n_error is not None else int(gen_cfg["inorganic_samples_per_question"])
    # the truthful variants actually needed: one per distinct negative class across the registers
    tvars = sorted({TRUTHFUL_FOR[f] for f in framings})
    assign = assign_templates([q["question_id"] for q in questions], split_lookup,
                              bool(gen_cfg.get("heldout_template_on_test", True)))
    jobs: list[dict[str, Any]] = []
    for q in questions:
        qid = q["question_id"]
        plan: list[tuple[str, int | None, str | None, int]] = []
        if conditions in ("organic", "all"):
            plan.append(("organic", None, None, n_o))
        if conditions in ("inorganic", "all") and (known_questions is None or qid in known_questions):
            for tid in assign[qid]["template_ids"]:
                for tv in tvars:
                    plan.append(("truthful", tid, tv, n_t))
                for fr in framings:
                    plan.append(("incorrect", tid, fr, n_e))
        for cond, tid, fr, n in plan:
            msgs = build_messages(q["question"], cond, tid, style, fr)
            text, ids = render_prompt(tokenizer, family, msgs)
            for s in range(n):
                jobs.append({"question_id": qid, "sample_id": s,
                             "conversation_id": conversation_id(qid, cond, tid, fr, s),
                             "condition": cond, "template_id": tid, "framing": fr or "none",
                             "frame": frame_for(cond, fr), "instruction_type": INSTRUCTION_TYPE[cond],
                             "held_out_template": bool(tid is not None and tid not in (0, 1, 2, 3)),
                             "prompt_style": style, "split": split_lookup[qid], "messages": msgs,
                             "prompt_text": text, "prompt_token_ids": ids})
    return jobs
