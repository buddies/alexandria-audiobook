"""Read the final answer out of OpenAI-compatible chat responses that may also
carry server-side reasoning ("thinking") output.

Which shape you get depends on the backend (vLLM, SGLang, llama.cpp, Ollama,
LM Studio, OpenRouter, ...) and on how it was started:

* a sibling field on the assistant message, e.g. ``reasoning_content``
  (vLLM ``--reasoning-parser``, DeepSeek-R1, GLM-4.x, Qwen3 thinking),
  ``reasoning`` (OpenRouter-style gateways) or ``thinking`` (llama.cpp);
* inline tags inside ``content``, e.g. `` thinking...<｜end▁of▁thinking｜>`` (DeepSeek-R1) or
  ``<thinking>...</thinking>`` / ``<reflection>...</reflection>`` (GLM, Qwen);
* raw tokenizer special tokens leaked into ``content``, e.g.
  ``<|begin_of_thought|>...<|end_of_thought|>`` / ``<|begin_of_solution|>`` (QwQ,
  R1 distills) or `` Thinking...`` with SentencePiece ``▁`` separators;
* inline channel markup inside ``content``, e.g. gpt-oss/Harmony
  ``<|channel|>analysis<|message|>...<|end|><|channel|>final<|message|>...``.

:func:`analyze_response` normalises all of these into an :class:`LLMOutput`, so the
JSON parsers downstream only ever see the answer body, while the thinking text stays
available for logging and diagnostics.

Two guarantees matter for callers:

* **The body is never truncated by thinking markup.** Only *paired* delimiters are
  removed, and an unterminated thinking prefix is dropped only when it starts the
  message (where the templates always put it). Anything else is left in place,
  because guessing where thinking ends would silently delete real content.
* **Thinking never eats the answer's budget.** :func:`next_max_tokens` grows the
  output budget after a truncated response, so a retry can return the complete body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Inline thinking markup
# ---------------------------------------------------------------------------

# Tag names used by popular models to delimit chain of thought inside `content`.
THINKING_TAGS = (
    "think",           # DeepSeek-R1, Qwen3 thinking
    "thinking",
    "thought",
    "thought_process",
    "reflection",
    "reasoning",
    "scratchpad",
)

_TAGS = "|".join(THINKING_TAGS)
# <think ...>...</think> — attributes tolerated, case-insensitive, multi-line.
_PAIRED_TAG_SRC = rf"<\s*(?P<tag>{_TAGS})\b[^>]*>.*?<\s*/\s*(?P=tag)\s*>"
# Standalone variants (no back-reference, so they can be used on their own).
# Unclosed opener: everything from the opener to the end of the string is thinking.
_OPEN_TAG_RE = re.compile(rf"<\s*(?:{_TAGS})\b[^>]*>", re.IGNORECASE)
# Closing tag left behind by a truncated/split stream.
_CLOSE_TAG_RE = re.compile(rf"<\s*/\s*(?:{_TAGS})\s*>", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Raw tokenizer special tokens
# ---------------------------------------------------------------------------

# Some servers leak the chat template's control tokens into `content`, using
# half-width (<|...|>) or full-width (<｜...｜>, SentencePiece ▁) pipes:
#   <|begin_of_thought|>...<|end_of_thought|>      QwQ / R1 distills
#    Thinking...                              DeepSeek-R1, Qwen3
#   <|begin_of_solution|>answer<|end_of_solution|>  QwQ
_PIPE = r"[|\uff5c]"          # half-width and full-width pipe
_NOT_PIPE = r"[^|\uff5c>]"    # flat class (no nesting), matches ▁ too

_SPECIAL_PAIR_SRC = rf"<{_PIPE}begin(?P<spec>{_NOT_PIPE}+){_PIPE}>.*?<{_PIPE}end(?P=spec){_PIPE}>"
_SPECIAL_OPEN_RE = re.compile(rf"<{_PIPE}begin{_NOT_PIPE}+{_PIPE}>", re.IGNORECASE)
# A "solution"/"answer" section *is* the answer, so it ends the thinking part.
# A token that also mentions thinking/thought is a thinking marker, not a solution.
_SOLUTION_RE = re.compile(
    rf"<{_PIPE}(?![^|\uff5c>]*(?:thinking|thought|reasoning))[^|\uff5c>]*(?:solution|answer)[^|\uff5c>]*{_PIPE}>",
    re.IGNORECASE,
)
# Any leftover control token: <|end|>, <|message|>, <｜end▁of▁sentence｜>, ...
_CONTROL_TOKEN_RE = re.compile(rf"<{_PIPE}[A-Za-z_\u2581]+{_PIPE}>?", re.IGNORECASE)

# ---------------------------------------------------------------------------
# gpt-oss / Harmony channel markup
# ---------------------------------------------------------------------------

_CHANNEL_RE = re.compile(
    rf"<{_PIPE}channel{_PIPE}>\s*(?P<channel>[A-Za-z_]+)\s*<{_PIPE}message{_PIPE}>",
    re.IGNORECASE,
)
_FINAL_CHANNELS = frozenset({"final", "answer", "output"})

# ---------------------------------------------------------------------------
# Message fields that may carry the chain of thought
# ---------------------------------------------------------------------------

# Checked in order; the first non-empty value wins.
REASONING_FIELDS = (
    "reasoning_content",   # vLLM --reasoning-parser, DeepSeek-R1, GLM-4.x, Qwen3
    "reasoning",           # OpenRouter and assorted gateways
    "reasoning_text",
    "thinking",            # llama.cpp server, LM Studio style
    "thought",
    "thoughts",
    "analysis",            # gpt-oss style field
)

_CONTENT_FIELDS = ("content",)

# Upper bound for the output budget used when retrying a truncated response.
DEFAULT_MAX_TOKENS_CAP = 32768


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LLMOutput:
    """The assistant turn split into answer body and reasoning.

    ``body`` is what callers should parse (JSON, prose, ...). ``reasoning`` is the
    server-side thinking, if the server returned any; it is never fed to parsers but
    is useful in logs when an answer comes back empty or malformed.
    """

    body: str = ""
    reasoning: str = ""
    finish_reason: Optional[str] = None
    usage: Any = None
    raw_message: Any = None

    @property
    def has_reasoning(self) -> bool:
        return bool(self.reasoning)

    @property
    def only_thinking(self) -> bool:
        """True when the server spent the whole budget thinking and never answered."""
        return not self.body and bool(self.reasoning)


# ---------------------------------------------------------------------------
# Low-level value access (objects, openai SDK models and plain dicts)
# ---------------------------------------------------------------------------

def _get_field(source: Any, name: str) -> Any:
    """Read ``name`` off a dict, a pydantic model or an arbitrary object.

    The OpenAI SDK models are configured with ``extra="allow"``, so unknown server
    fields such as ``reasoning_content`` are still reachable as attributes; the
    ``model_extra`` lookup covers SDK versions that keep them in a dict instead.
    """
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    value = getattr(source, name, None)
    if value is None:
        extra = getattr(source, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(name)
    return value


def _first_field(source: Any, names: Tuple[str, ...]) -> str:
    """First non-blank field among ``names``."""
    for name in names:
        text = _as_text(_get_field(source, name))
        if text.strip():
            return text
    return ""


def _as_text(value: Any) -> str:
    """Flatten the shapes a message field can take into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (_as_text(v) for v in value) if part)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return _as_text(value[key])
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Markup stripping
# ---------------------------------------------------------------------------

def strip_control_tokens(text: str) -> str:
    """Drop leftover special tokens such as ``<|end|>`` / ``<|message|>``."""
    if not text:
        return ""
    return _CONTROL_TOKEN_RE.sub("", text)


def split_channel_markup(text: str) -> Tuple[str, str]:
    """Split gpt-oss/Harmony style channel markup into (body, thinking).

    Returns ``(text, "")`` unchanged when no channel markup is present. When only a
    non-final channel (``analysis``, ``commentary``, ...) exists, the body is empty.
    """
    if not text:
        return "", ""

    matches = list(_CHANNEL_RE.finditer(text))
    if not matches:
        return text, ""

    body_parts = []
    thinking_parts = []

    # Text before the first channel marker is usually template residue
    # ("<|start|>assistant"), but keep it when it looks like real content so a
    # preamble answer is never thrown away.
    prefix = strip_control_tokens(text[:matches[0].start()]).strip()
    if prefix and ("[" in prefix or "{" in prefix):
        body_parts.append(prefix)

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = strip_control_tokens(text[start:end]).strip()
        if not chunk:
            continue
        if match.group("channel").lower() in _FINAL_CHANNELS:
            body_parts.append(chunk)
        else:
            thinking_parts.append(chunk)

    return "\n".join(body_parts).strip(), "\n".join(thinking_parts).strip()


def split_inline_reasoning(text: str) -> Tuple[str, str]:
    """Split inline thinking out of ``text``, returning ``(body, thinking)``.

    Handles channel markup, explicit solution/answer markers, raw special tokens
    (a missing/hanging delimiter is treated as thinking running to the end) and the
    common ``<think>``-style tags.
    """
    if not text:
        return "", ""

    body, channel_thinking = split_channel_markup(text)
    blocks = [channel_thinking]

    # An explicit solution/answer marker: everything before it is thinking.
    marker = _SOLUTION_RE.search(body)
    if marker:
        blocks.append(body[:marker.start()])
        body = body[marker.end():]

    # Raw special tokens first, then the <think>-style tags.
    body, found = _take_delimited(body, _SPECIAL_PAIR_SRC, _SPECIAL_OPEN_RE, None)
    blocks.extend(found)
    body, found = _take_delimited(body, _PAIRED_TAG_SRC, _OPEN_TAG_RE, _CLOSE_TAG_RE)
    blocks.extend(found)

    thinking = "\n".join(
        cleaned for cleaned in (_clean_thinking(block) for block in blocks) if cleaned
    )
    return strip_control_tokens(body).strip(), thinking.strip()


def _take_delimited(text: str, pair_source: str, unclosed_open_re, stray_close_re):
    """Move everything inside open/close delimiters into the thinking blocks.

    Returns ``(body, blocks)`` where each block still carries its delimiters.
    """
    pair_re = re.compile(pair_source, re.IGNORECASE | re.DOTALL)

    found = []

    def _capture(match: "re.Match[str]") -> str:
        found.append(match.group(0))
        return ""

    body = pair_re.sub(_capture, text)

    # Opener with no closing delimiter: the model was still thinking when it stopped.
    # Only the *first* opener can be a thinking prefix, and only when nothing but
    # whitespace precedes it (that is where the chat templates put it). A marker in
    # the middle of the text is not a reliable delimiter — truncating there would
    # delete real body content, so it is left alone.
    for opener in unclosed_open_re.finditer(body):
        if body[:opener.start()].strip():
            continue
        found.append(body[opener.start():])
        body = body[:opener.start()]
        break

    if stray_close_re is not None:
        body = stray_close_re.sub("", body)

    return body, found


def _clean_thinking(block: str) -> str:
    """Strip delimiters/control tokens from a thinking block for readable logs."""
    if not block:
        return ""
    block = _OPEN_TAG_RE.sub("", block)
    block = _CLOSE_TAG_RE.sub("", block)
    return strip_control_tokens(block).strip()


def strip_thinking_markup(text: str) -> str:
    """Return ``text`` with every detected thinking block/markup removed."""
    return split_inline_reasoning(text)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_message(message: Any) -> LLMOutput:
    """Analyze a single assistant message object."""
    if message is None:
        return LLMOutput()

    content = _first_field(message, _CONTENT_FIELDS)
    explicit_reasoning = _first_field(message, REASONING_FIELDS)

    body, inline_thinking = split_inline_reasoning(content)

    reasoning_parts = []
    if explicit_reasoning.strip():
        reasoning_parts.append(strip_control_tokens(explicit_reasoning).strip())
    if inline_thinking:
        reasoning_parts.append(inline_thinking)

    return LLMOutput(
        body=body,
        reasoning="\n".join(part for part in reasoning_parts if part).strip(),
        raw_message=message,
    )


def analyze_response(response: Any) -> LLMOutput:
    """Analyze a chat completion response and return its :class:`LLMOutput`.

    Never raises on missing fields: an empty body simply means "no usable answer",
    which lets callers keep their existing retry logic instead of crashing on
    ``None.strip()``.
    """
    choices = _get_field(response, "choices")
    choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None

    output = analyze_message(_get_field(choice, "message"))
    output.finish_reason = _as_text(_get_field(choice, "finish_reason")) or None
    output.usage = _get_field(response, "usage")
    return output


# Convenience wrappers -------------------------------------------------------

def response_body(response: Any) -> str:
    """The answer text only, with any thinking output removed."""
    return analyze_response(response).body


def response_reasoning(response: Any) -> str:
    """The server-side thinking text, or ``''`` when the server returned none."""
    return analyze_response(response).reasoning


# Diagnostics ----------------------------------------------------------------

def reasoning_warning(output: LLMOutput) -> str:
    """A short console warning for the ways thinking output breaks a run.

    Returns ``''`` when there is nothing worth warning about.
    """
    if not output.has_reasoning:
        return ""

    if output.only_thinking:
        return (
            "server returned reasoning only — the final answer is empty "
            f"({len(output.reasoning)} chars of thinking). Thinking mode consumed the "
            "whole output budget."
        )

    if (output.finish_reason or "").lower() == "length":
        return (
            f"the answer was cut off while thinking ({len(output.reasoning)} chars of "
            "reasoning before it ran out of tokens)."
        )

    return ""


def reasoning_log_section(output: LLMOutput) -> str:
    """A labelled block for the ``logs/*.log`` files, or ``''`` when there is none."""
    if not output.has_reasoning:
        return ""
    bar = "-" * 80
    return f"THINKING ({len(output.reasoning)} chars)\n{bar}\n{output.reasoning}\n{bar}\n"


def next_max_tokens(current: int, base: int, cap: int = DEFAULT_MAX_TOKENS_CAP) -> int:
    """A larger output budget to request after a truncated response.

    Thinking output is charged against ``max_tokens``, so a reasoning model can end
    up returning half an answer (or none) while ``finish_reason == "length"``.
    Doubling the budget gives the answer room on the retry; the result never drops
    below ``base`` and never exceeds ``cap``, keeping the retry bounded.
    """
    base = max(1, int(base))
    grown = max(int(current) * 2, base * 2)
    return max(base, min(grown, int(cap)))
