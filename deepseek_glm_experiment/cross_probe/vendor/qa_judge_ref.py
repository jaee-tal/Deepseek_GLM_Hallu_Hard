"""Labels for short-answer QA domains: gold-alias matching plus a cheap LLM check (no web search).

Output schema mirrors the HalluHard path (labels.jsonl / claims.jsonl) so every downstream stage is shared:
one "claim" per response = the answer; hallucination = "Yes" when the answer is wrong.
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

log = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You grade answers to trivia questions. You are given the question, the list of accepted gold answers, "
    "and a model's response. Decide whether the response's final answer is correct (matches any gold answer in "
    "meaning, allowing different surface forms, e.g. 'the 60s' vs '1960s', 'JFK' vs 'John F. Kennedy'). "
    "If the response gives no answer, says it does not know, or hedges between several candidates without "
    "committing, grade it as abstain. Respond with JSON only: {\"verdict\": \"correct\"|\"incorrect\"|\"abstain\", "
    "\"answer_given\": \"<the response's final answer or null>\", \"reason\": \"<one sentence>\"}"
)


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def rule_match(response: str, aliases: list[str]) -> bool:
    r = _normalize(response)
    return any(a and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", r) for a in (_normalize(x) for x in aliases))


def _openrouter_chat(model: str, messages: list[dict[str, str]], reasoning_effort: str | None, retries: int = 6) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing in .env")
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    for t in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {key}", "HTTP-Referer": "https://localhost/hallu_probe"}, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"status {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            j = r.json()
            content = j["choices"][0]["message"]["content"]
            usage = j.get("usage", {})
            return json.dumps({"content": content, "usage": usage})
        except Exception as e:  # noqa: BLE001
            if t == retries - 1:
                raise
            time.sleep(min(30, 2 ** t))
    raise RuntimeError("unreachable")


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    return json.loads(m.group(0))


def judge_one(rec: dict[str, Any], q: dict[str, Any], model: str, effort: str | None) -> dict[str, Any]:
    aliases = q["meta"]["aliases"] or [q["meta"]["answer_value"]]
    matched = rule_match(rec["response_text"], aliases)
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {q['question']}\nGold answers: {json.dumps(aliases[:25], ensure_ascii=False)}\n"
                                        f"Model response:\n\"\"\"\n{rec['response_text'][:4000]}\n\"\"\""}]
    out: dict[str, Any] = {"conversation_id": rec["conversation_id"], "rule_match": matched}
    try:
        raw = json.loads(_openrouter_chat(model, msgs, effort))
        j = _extract_json(raw["content"])
        v = str(j.get("verdict", "")).strip().lower()
        if v not in ("correct", "incorrect", "abstain"):
            raise ValueError(f"bad verdict {v!r}")
        out.update({"verdict": v, "answer_given": j.get("answer_given"), "reason": j.get("reason", ""),
                    "verification_error": "No", "usage": raw.get("usage", {})})
    except Exception as e:  # noqa: BLE001
        out.update({"verdict": "unknown", "answer_given": None, "reason": "", "verification_error": "Yes", "error": str(e)[:300]})
    return out


def judge_all(records: list[dict[str, Any]], questions: dict[str, dict[str, Any]], model: str, effort: str | None,
              workers: int, cache_path) -> list[dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                if d.get("verification_error") != "Yes":
                    done[d["conversation_id"]] = d
    todo = [r for r in records if r["conversation_id"] not in done]
    log.info("qa judge: %d responses, %d cached, %d to judge", len(records), len(done), len(todo))
    with ThreadPoolExecutor(max_workers=workers) as ex, open(cache_path, "a", encoding="utf-8") as f:
        for i, res in enumerate(ex.map(lambda r: judge_one(r, questions[r["question_id"]], model, effort), todo)):
            done[res["conversation_id"]] = res
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                log.info("qa judge %d/%d", i + 1, len(todo))
    return [done[r["conversation_id"]] for r in records]


def to_label_rows(records: list[dict[str, Any]], verdicts: list[dict[str, Any]], label_source: str) -> tuple[list, list]:
    claims, labels = [], []
    for r, v in zip(records, verdicts):
        cid, qid, sid = r["conversation_id"], r["question_id"], r["sample_id"]
        h = {"correct": "No", "incorrect": "Yes", "abstain": "No", "unknown": "Unknown"}[v["verdict"]]
        abst = "Yes" if v["verdict"] == "abstain" else "No"
        claims.append({"conversation_id": cid, "question_id": qid, "sample_id": sid, "claim_id": f"{cid}__answer", "turn_number": 1,
                       "claim": {"data": {"claimed_content": v.get("answer_given") or r["response_text"][:300]}},
                       "hallucination": h, "abstention": abst, "verification_error": v.get("verification_error", "No"),
                       "reference_grounding": "N/A", "content_grounding": v["verdict"], "reason": v.get("reason", ""),
                       "rule_match": v.get("rule_match"), **({"error": v["error"]} if v.get("error") else {})})
        if v["verdict"] == "incorrect":
            label, reason = 1, "answer_incorrect"
        elif v["verdict"] == "correct":
            label, reason = 0, "answer_correct"
        elif v["verdict"] == "abstain":
            label, reason = None, "zero_claims"
        else:
            label, reason = None, "unverifiable"
        labels.append({"conversation_id": cid, "question_id": qid, "sample_id": sid, "label_source": label_source,
                       "score": None, "total_claims": 1, "hallucinations": int(label == 1),
                       "label": label, "label_reason": reason, "n_claims_total": 1, "n_yes": int(label == 1), "n_no": int(label == 0),
                       "n_unknown": int(v["verdict"] == "unknown"), "n_skipped": 0, "n_verr": int(v.get("verification_error") == "Yes"),
                       "n_abstain": int(v["verdict"] == "abstain"), "n_checked": int(label in (0, 1)), "n_usable": int(label in (0, 1)),
                       "rule_match": v.get("rule_match")})
    return claims, labels
