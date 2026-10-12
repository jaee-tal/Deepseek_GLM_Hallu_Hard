"""Grading. Two modes, selected per dataset:

`alias_short` (PopQA, TriviaQA, AA-Omniscience) — deterministic normalised alias matching first, OpenRouter LLM
    arbiter otherwise (same rule for every condition). A fixed random 10 % of rule-matched rows is *also* sent to
    the LLM (audit) to measure rule/LLM agreement; the audit never changes the verdict.
`reference_longform` (HalluHard) — no rule stage. Research/medical answers are graded against their source
    document, legal answers against legal knowledge, and coding answers with HalluHard's software-API rubric.

verdict ∈ {correct, incorrect, abstain, unknown};  source ∈ {rule_empty, rule_match, llm};  raw LLM output is always saved.
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .vendor.io_utils import append_jsonl, iter_jsonl, write_json

log = logging.getLogger(__name__)

GRADING_RULES_VERSION = "v2"
MODES = ("alias_short", "reference_longform")

# Unchanged since the PopQA/Qwen3 run; every short-answer dataset is graded with exactly this prompt so the
# datasets stay comparable and that run's frozen judge_system_prompt hash is preserved.
JUDGE_SYSTEM = (
    "You grade a model's short answer to an entity-centric factual question. You are given the question, the canonical "
    "gold answer, a list of accepted aliases of the gold answer, and the model's response.\n"
    "Decide:\n"
    "- \"correct\": the response's answer refers to the same entity or value as the gold answer (allow different surface "
    "forms, abbreviations, translations, partial names that unambiguously identify the gold entity, or a more specific "
    "answer that entails the gold answer).\n"
    "- \"incorrect\": the response commits to an answer that is a different entity or value than the gold answer.\n"
    "- \"abstain\": the response gives no answer, says it does not know, refuses, or hedges between several candidates "
    "without committing.\n"
    "Judge only the answer content. Ignore any stated confidence or commentary about the answer being wrong or right.\n"
    "Respond with JSON only: {\"label\": \"correct\" | \"incorrect\" | \"abstain\"}"
)

JUDGE_SYSTEM_REFERENCE = (
    "You grade a model's paragraph-length answer to a question, against a source document that the question was "
    "written from. You are given the question, the source document, and the model's response.\n"
    "Decide:\n"
    "- \"incorrect\": any factual claim in the response contradicts the source document, or is clearly false.\n"
    "- \"correct\": every factual claim in the response is supported by, or consistent with, the source document. "
    "Claims that go beyond the source are judged against general knowledge, conservatively: only call the response "
    "incorrect when a claim is clearly wrong, not merely unsupported.\n"
    "- \"abstain\": the response declines to answer, says it does not know, or gives no substantive answer.\n"
    "Judge only the content of the claims. Ignore style, length, and any commentary about the answer being wrong or right.\n"
    "Respond with JSON only: {\"label\": \"correct\" | \"incorrect\" | \"abstain\"}"
)

JUDGE_SYSTEM_KNOWLEDGE = (
    "You grade a model's paragraph-length answer to a factual question. No gold answer or source document is "
    "available, so use your factual knowledge conservatively.\n"
    "Decide:\n"
    "- \"incorrect\": the response contains at least one clearly false factual claim or gives the wrong answer.\n"
    "- \"correct\": the response answers the question and contains no factual claim you can identify as false.\n"
    "- \"abstain\": the response declines to answer, says it does not know, or gives no substantive answer.\n"
    "For legal questions, check the stated rule, holding, case, statute, and jurisdiction. Do not mark an answer "
    "incorrect merely because it omits detail or citations. Judge only answer content and ignore any commentary "
    "about the answer being wrong or right.\n"
    "Respond with JSON only: {\"label\": \"correct\" | \"incorrect\" | \"abstain\"}"
)

JUDGE_SYSTEM_CODING = (
    "You grade a response to a coding question for hallucinated software facts.\n"
    "Mark \"incorrect\" if the response uses a nonexistent package, module, function, method, parameter, command, "
    "or installation instruction, or makes a clearly false claim about an API's behavior. Focus on fabricated "
    "software interfaces rather than ordinary algorithmic bugs.\n"
    "Mark \"correct\" if it gives a substantive answer and you cannot identify such a hallucinated software fact.\n"
    "Mark \"abstain\" if it declines, says it does not know, or gives no substantive answer.\n"
    "Be conservative: only mark incorrect when the software claim is clearly nonexistent or false. Ignore any "
    "commentary about the answer being wrong or right.\n"
    "Respond with JSON only: {\"label\": \"correct\" | \"incorrect\" | \"abstain\"}"
)

SYSTEM_FOR_MODE = {"alias_short": JUDGE_SYSTEM, "reference_longform": JUDGE_SYSTEM_REFERENCE}

FATAL_STATUS = {401: "the OPENROUTER_API_KEY is invalid or revoked", 402: "the OpenRouter account is out of credit",
                403: "the OPENROUTER_API_KEY is not permitted to use this model"}


class JudgeUnavailable(SystemExit):
    """A non-transient judge failure (auth / payment / permission). Retrying cannot help, and continuing would mark
    every row 'unknown', so the stage stops instead."""


def check_credit(min_balance: float = 1.0) -> dict[str, Any]:
    """Preflight: refuse to start a judging run that cannot pay for itself (it would burn GPU time upstream).

    OpenRouter returns 402 as soon as a single request costs more than the remaining balance, so the floor is a
    run-sized amount (a full dataset costs roughly $1-7 at gpt-mini prices), not zero."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise JudgeUnavailable("OPENROUTER_API_KEY missing (.env)")
    try:
        r = requests.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        d = r.json()["data"]
    except Exception as e:  # noqa: BLE001 - a preflight failure must not block a run that would otherwise work
        log.warning("could not check OpenRouter credit (%s); continuing", str(e)[:120])
        return {"checked": False, "error": str(e)[:200]}
    remaining = float(d.get("total_credits", 0)) - float(d.get("total_usage", 0))
    info = {"checked": True, "total_credits": d.get("total_credits"), "total_usage": d.get("total_usage"), "remaining": round(remaining, 4)}
    log.info("OpenRouter credit: %.2f remaining of %.2f", remaining, float(d.get("total_credits", 0)))
    if remaining < min_balance:
        raise JudgeUnavailable(f"OpenRouter has ${remaining:.2f} credit remaining (< ${min_balance:.2f} needed for a run) — "
                               f"top up before judging. Generation and replay artifacts are already on disk, so judging "
                               f"resumes from where it stopped without redoing any GPU work.")
    return info


RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "grade", "strict": True, "schema": {
    "type": "object", "properties": {"label": {"type": "string", "enum": ["correct", "incorrect", "abstain"]}},
    "required": ["label"], "additionalProperties": False}}}


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def rule_match(response: str, aliases: list[str]) -> bool:
    r = _normalize(response)
    if not r:
        return False
    return any(a and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", r) for a in (_normalize(x) for x in aliases))


def is_empty(response: str) -> bool:
    return _normalize(response) == ""


def truncate_words(text: str, max_words: int) -> str:
    w = str(text).split()
    return " ".join(w) if len(w) <= max_words else " ".join(w[:max_words]) + " […]"


def user_message(question: str, gold: str, aliases: list[str], response: str) -> str:
    return (f"Question: {question}\nCanonical gold answer: {gold}\nAccepted aliases: {json.dumps(aliases[:25], ensure_ascii=False)}\n"
            f"Model response:\n\"\"\"\n{response[:2000]}\n\"\"\"")


def user_message_reference(question: str, reference: str, response: str, max_words: int = 1500) -> str:
    return (f"Question: {question}\n\nSource document:\n\"\"\"\n{truncate_words(reference, max_words)}\n\"\"\"\n\n"
            f"Model response:\n\"\"\"\n{response[:6000]}\n\"\"\"")


def user_message_knowledge(question: str, response: str) -> str:
    return f"Question: {question}\n\nModel response:\n\"\"\"\n{response[:6000]}\n\"\"\""


def halluhard_judge_variant(question: dict[str, Any]) -> str:
    """Choose the explicit per-row HalluHard rubric, including compatibility with old inputs."""
    meta = question.get("meta") or {}
    variant = meta.get("judge_variant")
    if variant in ("reference", "knowledge", "coding"):
        return variant
    if meta.get("domain") == "coding":
        return "coding"
    return "reference" if meta.get("reference") else "knowledge"


def openrouter_grade(jcfg: dict[str, Any], system: str, user: str, retries: int = 6) -> dict[str, Any]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing (.env)")
    body: dict[str, Any] = {"model": jcfg["model"], "temperature": float(jcfg.get("temperature", 0)),
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if jcfg.get("reasoning_effort"):
        body["reasoning"] = {"effort": jcfg["reasoning_effort"]}
    use_schema = bool(jcfg.get("structured_output", True))
    last_err = None
    for t in range(retries):
        b = dict(body)
        if use_schema:
            b["response_format"] = RESPONSE_FORMAT
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=b, timeout=120,
                              headers={"Authorization": f"Bearer {key}", "HTTP-Referer": "https://localhost/cross_probe", "X-Title": "cross_probe"})
            if r.status_code == 400 and use_schema and "response_format" in r.text:
                use_schema = False          # provider rejected the schema: fall back to prompt-instructed JSON
                continue
            if r.status_code in FATAL_STATUS:
                raise JudgeUnavailable(f"OpenRouter returned {r.status_code}: {FATAL_STATUS[r.status_code]}. "
                                       f"Nothing was judged; generation and replay artifacts are untouched and judging "
                                       f"resumes from the cache once this is fixed. ({r.text[:150]})")
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"status {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            j = r.json()
            if "error" in j and not j.get("choices"):
                raise RuntimeError(f"api error: {json.dumps(j['error'])[:200]}")
            content = j["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", content, flags=re.S)
            label = json.loads(m.group(0)).get("label") if m else None
            label = str(label).strip().lower() if label else None
            if label not in ("correct", "incorrect", "abstain"):
                raise ValueError(f"bad label in judge output: {content[:200]!r}")
            return {"label": label, "resolved_model": j.get("model"), "usage": j.get("usage", {}), "structured": use_schema,
                    "raw_body": j, "attempts": t + 1}
        except JudgeUnavailable:
            raise                                   # non-transient: do not retry, do not swallow
        except Exception as e:  # noqa: BLE001
            last_err = e
            if t == retries - 1:
                break
            time.sleep(min(30, 2 ** t))
    return {"label": None, "error": str(last_err)[:400], "attempts": retries}


def _audit_ids(rule_matched_ids: list[str], fraction: float, seed: int) -> set[str]:
    ids = sorted(rule_matched_ids)
    n = int(round(fraction * len(ids)))
    if n == 0:
        return set()
    perm = np.random.RandomState(seed).permutation(len(ids))
    return {ids[i] for i in perm[:n]}


def judge_all(records: list[dict[str, Any]], questions: dict[str, dict[str, Any]], jcfg: dict[str, Any], judge_dir: Path,
              limit_llm: int | None = None, mode: str = "alias_short") -> list[dict[str, Any]]:
    """records: generation records (need conversation_id, question_id, response_text, condition, template_id).
    Writes judge/verdicts.jsonl (resumable), judge/raw.jsonl (every LLM call), judge/stats.json."""
    if mode not in MODES:
        raise SystemExit(f"unknown judge mode {mode!r}; known: {list(MODES)}")
    judge_dir.mkdir(parents=True, exist_ok=True)
    ref_words = int(jcfg.get("reference_max_words", 1500))
    v_path, raw_path = judge_dir / "verdicts.jsonl", judge_dir / "raw.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if v_path.exists():
        for d in iter_jsonl(v_path):
            if d.get("verdict") != "unknown":
                done[d["conversation_id"]] = d
    # deterministic pre-pass: empties / rule matches / audit set
    pre: dict[str, dict[str, Any]] = {}
    rule_ids = []
    for r in records:
        q = questions[r["question_id"]]
        resp = r["response_text"]
        e = is_empty(resp)
        if mode == "alias_short":
            aliases = q["meta"]["aliases"] or [q["meta"]["answer_value"]]
            m = (not e) and rule_match(resp, aliases)
        else:
            m = False                       # reference-grounded: every non-empty response is adjudicated by the LLM
        pre[r["conversation_id"]] = {"empty": e, "rule_match": m}
        if m:
            rule_ids.append(r["conversation_id"])
    audit = _audit_ids(rule_ids, float(jcfg.get("audit_fraction", 0.0)), int(jcfg.get("audit_seed", 42)))
    todo = [r for r in records if r["conversation_id"] not in done]
    n_llm_needed = sum(1 for r in todo if (not pre[r["conversation_id"]]["empty"]) and (not pre[r["conversation_id"]]["rule_match"] or r["conversation_id"] in audit))
    log.info("judge[%s]: %d records, %d cached, %d to do (%d need LLM calls)", mode, len(records), len(done), len(todo), n_llm_needed)
    if limit_llm is not None:
        todo = todo[:limit_llm]

    def one(r: dict[str, Any]) -> dict[str, Any]:
        cid, q = r["conversation_id"], questions[r["question_id"]]
        p = pre[cid]
        out: dict[str, Any] = {"conversation_id": cid, "question_id": r["question_id"], "condition": r.get("condition"),
                               "template_id": r.get("template_id"), "empty": p["empty"], "rule_match": p["rule_match"],
                               "audit": cid in audit, "llm_label": None, "llm_model": None, "error": None,
                               "judge_mode": mode, "grading_rules_version": GRADING_RULES_VERSION}
        need_llm = (not p["empty"]) and ((not p["rule_match"]) or cid in audit)
        if need_llm:
            if mode == "alias_short":
                variant = "alias"
                system = JUDGE_SYSTEM
                aliases = q["meta"]["aliases"] or [q["meta"]["answer_value"]]
                user = user_message(q["question"], q["meta"]["answer_value"], aliases, r["response_text"])
            else:
                variant = halluhard_judge_variant(q)
                if variant == "reference":
                    system = JUDGE_SYSTEM_REFERENCE
                    user = user_message_reference(q["question"], q["meta"]["reference"], r["response_text"], ref_words)
                elif variant == "coding":
                    system = JUDGE_SYSTEM_CODING
                    user = user_message_knowledge(q["question"], r["response_text"])
                else:
                    system = JUDGE_SYSTEM_KNOWLEDGE
                    user = user_message_knowledge(q["question"], r["response_text"])
            out["judge_variant"] = variant
            g = openrouter_grade(jcfg, system, user)
            append_jsonl(raw_path, [{"conversation_id": cid, "audit": cid in audit, "request_model": jcfg["model"],
                                     "judge_mode": mode, "judge_variant": variant, **g}])
            out["llm_label"], out["llm_model"], out["error"] = g.get("label"), g.get("resolved_model"), g.get("error")
            out["usage"] = g.get("usage")
        if p["empty"]:
            out["verdict"], out["source"] = "abstain", "rule_empty"
        elif p["rule_match"]:
            out["verdict"], out["source"] = "correct", "rule_match"
        else:
            out["verdict"], out["source"] = (out["llm_label"] or "unknown"), "llm"
        return out

    with ThreadPoolExecutor(max_workers=int(jcfg.get("workers", 8))) as ex:
        for i, res in enumerate(ex.map(one, todo)):
            done[res["conversation_id"]] = res
            append_jsonl(v_path, [res])
            if (i + 1) % 200 == 0:
                log.info("judge %d/%d", i + 1, len(todo))
    verdicts = [done[r["conversation_id"]] for r in records if r["conversation_id"] in done]
    write_json(judge_dir / "stats.json", judge_stats(verdicts, mode))
    return verdicts


def judge_stats(verdicts: list[dict[str, Any]], mode: str = "alias_short") -> dict[str, Any]:
    from collections import Counter

    n = len(verdicts)
    by_cond: dict[str, Counter] = {}
    for v in verdicts:
        by_cond.setdefault(str(v.get("condition")), Counter())[v["verdict"]] += 1
    llm = [v for v in verdicts if v["source"] == "llm"]
    audit = [v for v in verdicts if v.get("audit") and v.get("llm_label")]
    agree = sum(1 for v in audit if v["llm_label"] == "correct")
    usage_in = sum((v.get("usage") or {}).get("prompt_tokens", 0) for v in verdicts if v.get("usage"))
    usage_out = sum((v.get("usage") or {}).get("completion_tokens", 0) for v in verdicts if v.get("usage"))
    models = Counter(v["llm_model"] for v in verdicts if v.get("llm_model"))
    variants = Counter(v.get("judge_variant", "alias" if mode == "alias_short" else "reference") for v in verdicts)
    return {"n_responses": n, "judge_mode": mode, "n_llm_adjudicated": len(llm), "adjudication_rate": (len(llm) / n) if n else None,
            "n_rule_empty": sum(1 for v in verdicts if v["source"] == "rule_empty"),
            "n_rule_match": sum(1 for v in verdicts if v["source"] == "rule_match"),
            "n_unknown": sum(1 for v in verdicts if v["verdict"] == "unknown"),
            "audit": {"n": len(audit), "llm_agrees_correct": agree, "agreement_rate": (agree / len(audit)) if audit else None},
            "verdicts_by_condition": {k: dict(c) for k, c in by_cond.items()},
            "llm_models_resolved": dict(models), "judge_variants": dict(variants),
            "llm_prompt_tokens": usage_in, "llm_completion_tokens": usage_out,
            "grading_rules_version": GRADING_RULES_VERSION}
