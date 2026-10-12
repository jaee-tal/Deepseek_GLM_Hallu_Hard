"""Per-model-family knowledge: chat-template kwargs, EOS/pad ids, answer-span detection.

Layer/position conventions used everywhere:
  * `prompt_token_ids`     : ids of the rendered chat prompt (incl. template-injected tokens such as
                             Qwen3's empty <think></think> block), tokenised with add_special_tokens=False.
  * `generated_token_ids`  : sampled ids, cut *before* the first EOS/pad.
  * `answer_start/answer_end` : half-open span *within generated_token_ids* holding the final answer.
  * `response_token_start/end`: the same span as absolute offsets in prompt+generated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SpanResult:
    start: int
    end: int
    exclude_reason: Optional[str] = None
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class Family:
    name: str
    template_kwargs: dict[str, Any]
    eos_ids: list[int]
    pad_id: int
    prompt_tail: str                       # rendered prompt must end with this
    reasoning_marker_ids: list[int]        # must not appear inside the answer span
    tokenizer: Any

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        """Render one chat prompt. Families with non-Jinja protocols override this."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )

    def answer_span(self, generated: list[int]) -> SpanResult:
        raise NotImplementedError


class Qwen3Family(Family):
    def answer_span(self, generated: list[int]) -> SpanResult:
        think_open, think_close = self.reasoning_marker_ids
        if think_open not in generated and think_close not in generated:
            return SpanResult(0, len(generated))
        # Unexpected: the model re-emitted a think block despite enable_thinking=False.
        flags = {"has_think_tokens": True}
        if think_close in generated:
            last = len(generated) - 1 - generated[::-1].index(think_close)
            start = last + 1
            # skip whitespace-only tokens following </think>
            while start < len(generated) and self.tokenizer.decode([generated[start]]).strip() == "":
                start += 1
            return SpanResult(start, len(generated), exclude_reason="unexpected_think", flags=flags)
        return SpanResult(0, len(generated), exclude_reason="unexpected_think", flags=flags)


class Llama31Family(Family):
    def answer_span(self, generated: list[int]) -> SpanResult:
        return SpanResult(0, len(generated))


class GptOssFamily(Family):
    """Harmony format: ... <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>ANSWER<|return|>"""

    def _init_ids(self):
        t = self.tokenizer
        self.channel_id = t.convert_tokens_to_ids("<|channel|>")
        self.message_id = t.convert_tokens_to_ids("<|message|>")
        self.end_id = t.convert_tokens_to_ids("<|end|>")
        self.return_id = t.convert_tokens_to_ids("<|return|>")
        self.call_id = t.convert_tokens_to_ids("<|call|>")
        final_ids = t.encode("final", add_special_tokens=False)
        assert len(final_ids) == 1, final_ids
        self.final_id = final_ids[0]

    def answer_span(self, generated: list[int]) -> SpanResult:
        if not hasattr(self, "final_id"):
            self._init_ids()
        marker = [self.channel_id, self.final_id, self.message_id]
        starts = [i for i in range(len(generated) - 2) if generated[i:i + 3] == marker]
        if not starts:
            return SpanResult(0, 0, exclude_reason="no_final_channel", flags={"n_final_markers": 0})
        s = starts[-1] + 3
        e = len(generated)
        for j in range(s, len(generated)):
            if generated[j] in (self.return_id, self.end_id, self.call_id):
                e = j
                break
        flags = {"n_final_markers": len(starts), "n_reasoning_tokens": starts[-1]}
        if e <= s:
            return SpanResult(s, e, exclude_reason="empty_final_answer", flags=flags)
        return SpanResult(s, e, flags=flags)


class ThinkingFinalFamily(Family):
    """A chat format whose generated answer follows the final ``</think>`` token."""

    def answer_span(self, generated: list[int]) -> SpanResult:
        close = getattr(self, "think_close_ids", [self.reasoning_marker_ids[-1]])
        starts = [i for i in range(len(generated) - len(close) + 1)
                  if generated[i:i + len(close)] == close]
        if not starts:
            return SpanResult(0, 0, exclude_reason="no_final_answer_marker",
                              flags={"has_think_close": False, "n_reasoning_tokens": len(generated)})
        last = starts[-1]
        start = last + len(close)
        while start < len(generated) and self.tokenizer.decode([generated[start]]).strip() == "":
            start += 1
        if start >= len(generated):
            return SpanResult(start, start, exclude_reason="empty_final_answer",
                              flags={"has_think_close": True, "n_reasoning_tokens": start})
        return SpanResult(start, len(generated),
                          flags={"has_think_close": True, "n_reasoning_tokens": start})


class DeepSeekV41Family(ThinkingFinalFamily):
    """DeepSeek V4.1 uses the checkpoint's official Python encoder, not Jinja."""

    encoder: Any = None

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        return self.encoder.encode_messages(
            messages,
            thinking_mode=str(self.template_kwargs.get("thinking_mode", "thinking")),
            reasoning_effort=self.template_kwargs.get("reasoning_effort", 1),
        )


class Glm53Family(ThinkingFinalFamily):
    def render_messages(self, messages: list[dict[str, str]]) -> str:
        processor = getattr(self.tokenizer, "_cross_probe_processor", None)
        renderer = processor if processor is not None and hasattr(processor, "apply_chat_template") else self.tokenizer
        return renderer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )


class Gemma4Family(Family):
    """Gemma 4 final-answer spans with optional thinking-channel support.

    With ``enable_thinking=false`` the canonical template puts an empty thought
    channel in the prompt, so generated tokens are final-answer tokens.  Keep
    the parser tolerant of thinking-enabled generations so a config change
    cannot silently pool reasoning tokens into the probe representation.
    """

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        processor = getattr(self.tokenizer, "_cross_probe_processor", None)
        renderer = processor if processor is not None and hasattr(processor, "apply_chat_template") else self.tokenizer
        return renderer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )

    def answer_span(self, generated: list[int]) -> SpanResult:
        close = getattr(self, "think_close_ids", [])
        opened = getattr(self, "think_open_ids", [])
        starts = ([i for i in range(len(generated) - len(close) + 1)
                   if generated[i:i + len(close)] == close] if close else [])
        if starts:
            start = starts[-1] + len(close)
            while start < len(generated) and self.tokenizer.decode([generated[start]]).strip() == "":
                start += 1
            if start >= len(generated):
                return SpanResult(start, start, exclude_reason="empty_final_answer",
                                  flags={"has_think_channel": True, "n_reasoning_tokens": start})
            return SpanResult(start, len(generated),
                              flags={"has_think_channel": True, "n_reasoning_tokens": start})
        has_open = bool(opened) and any(
            generated[i:i + len(opened)] == opened
            for i in range(len(generated) - len(opened) + 1)
        )
        if has_open:
            return SpanResult(0, 0, exclude_reason="unclosed_think_channel",
                              flags={"has_think_channel": True, "n_reasoning_tokens": len(generated)})
        return SpanResult(0, len(generated), flags={"has_think_channel": False, "n_reasoning_tokens": 0})


def _special_id(tokenizer, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or value == getattr(tokenizer, "unk_token_id", None):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"expected {token!r} to be one special token; got {ids}")
        value = ids[0]
    return int(value)


def _token_ids(tokenizer, token: str) -> list[int]:
    return list(map(int, tokenizer.encode(token, add_special_tokens=False)))


def _load_deepseek_encoder(checkpoint: str, revision: str | None):
    """Load the official MIT-licensed encoder from the pinned model revision."""
    import importlib.util
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(checkpoint, "encoding/encoding.py", revision=revision)
    spec = importlib.util.spec_from_file_location("cross_probe_deepseek_v41_encoding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import DeepSeek encoder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_family(name: str, tokenizer, template_kwargs: dict[str, Any], model_cfg: dict[str, Any] | None = None) -> Family:
    if name == "qwen3":
        eos = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        pad = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        markers = [tokenizer.convert_tokens_to_ids("<think>"), tokenizer.convert_tokens_to_ids("</think>")]
        tail = "<|im_start|>assistant\n<think>\n\n</think>\n\n" if not template_kwargs.get("enable_thinking", True) \
            else "<|im_start|>assistant\n"
        return Qwen3Family("qwen3", template_kwargs, eos, pad, tail, markers, tokenizer)
    if name == "llama31":
        eos = [tokenizer.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>")]
        pad = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tail = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return Llama31Family("llama31", template_kwargs, eos, pad, tail, [], tokenizer)
    if name == "gpt_oss":
        eos = [tokenizer.convert_tokens_to_ids(t) for t in ("<|return|>", "<|call|>")]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.convert_tokens_to_ids("<|endoftext|>")
        markers = [tokenizer.convert_tokens_to_ids("<|channel|>")]
        fam = GptOssFamily("gpt_oss", template_kwargs, eos, pad, "<|start|>assistant", markers, tokenizer)
        fam._init_ids()
        return fam
    if name == "deepseek_v41":
        mcfg = model_cfg or {}
        open_ids, close_ids = _token_ids(tokenizer, "<think>"), _token_ids(tokenizer, "</think>")
        markers = list(dict.fromkeys(open_ids + close_ids))
        eos = [_special_id(tokenizer, "<｜end▁of▁sentence｜>")]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
        fam = DeepSeekV41Family(name, template_kwargs, eos, int(pad), "<｜Assistant｜><think>", markers, tokenizer)
        fam.think_open_ids, fam.think_close_ids = open_ids, close_ids
        fam.encoder = _load_deepseek_encoder(mcfg.get("checkpoint", "deepseek-ai/DeepSeek-V4.1-Flash"),
                                             mcfg.get("revision"))
        return fam
    if name == "glm53":
        open_ids, close_ids = _token_ids(tokenizer, "<think>"), _token_ids(tokenizer, "</think>")
        markers = list(dict.fromkeys(open_ids + close_ids))
        eos = [int(tokenizer.eos_token_id)]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
        fam = Glm53Family(name, template_kwargs, eos, int(pad), "<|assistant|><think>", markers, tokenizer)
        fam.think_open_ids, fam.think_close_ids = open_ids, close_ids
        return fam
    if name == "gemma4":
        # Gemma 4 encodes thought channels as
        # <|channel>thought\n ... <channel|>. Both delimiters may be more than
        # one tokenizer id, so span matching is sequence based.
        open_ids = _token_ids(tokenizer, "<|channel>thought\n")
        close_ids = _token_ids(tokenizer, "<channel|>")
        # Only the channel boundary tokens are forbidden inside the answer.
        # ``open_ids`` also contains the ordinary token for "thought" and a
        # newline, which are valid final-answer tokens on their own.
        markers = [_special_id(tokenizer, "<|channel>"), _special_id(tokenizer, "<channel|>")]
        eos = [_special_id(tokenizer, token) for token in ("<eos>", "<turn|>", "<|tool_response>")]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else _special_id(tokenizer, "<pad>")
        tail = ("<|turn>model\n" if template_kwargs.get("enable_thinking", False)
                else "<|turn>model\n<|channel>thought\n<channel|>")
        fam = Gemma4Family(name, template_kwargs, eos, int(pad), tail, markers, tokenizer)
        fam.think_open_ids, fam.think_close_ids = open_ids, close_ids
        return fam
    raise ValueError(f"unknown family {name}")
