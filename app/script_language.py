"""Keep the generated script in the language the TTS is configured for.

The LLM writes three different things into `annotated_script.json`, and they do
not share one language rule:

* `speaker` — this becomes the voice name on the Voices tab and the key of
  `voice_config.json`,
* `instruct` — the per-line emotion/style direction, shown in the editor as
  "Emotion / Style",
* `text` — the book itself, which must never be translated or paraphrased.

The shipped prompts ask for UPPERCASE English speaker labels and English
directions, which is wrong for a Chinese book read by a Chinese TTS voice: roles
come back romanized ("CHEN JIFENG") and every direction is in English. This
module appends one overrides-everything rule block to whatever system prompt is
in play — shipped or hand-written — so both follow `tts.language`.

It also owns the narrator label. Narration used to be recognised by the literal
"NARRATOR", so localising it (`旁白`) has to be paired with a shared predicate
that downstream code (persona context, narrator merging, character styles) uses
instead of string equality.
"""

# The label the prompts use for narration, per language. Kept to one word per
# language on purpose: the model must reuse the exact same label on every
# narration line, because each distinct label would become its own voice.
NARRATOR_LABEL_BY_LANGUAGE = {
    "chinese": "旁白",
    "japanese": "ナレーター",
    "korean": "나레이터",
    "russian": "Рассказчик",
    "french": "Narrateur",
    "german": "Erzähler",
    "italian": "Narratore",
    "portuguese": "Narrador",
    "spanish": "Narrador",
    "english": "NARRATOR",
}

DEFAULT_INSTRUCT_BY_LANGUAGE = {
    "chinese": "中性叙述，语速平稳。",
    "japanese": "中立的なナレーション。",
    "korean": "중립적인 내레이션.",
    "russian": "Нейтральное повествование.",
    "french": "Narration neutre.",
    "german": "Neutrale Erzählung.",
    "italian": "Narrazione neutra.",
    "portuguese": "Narração neutra.",
    "spanish": "Narración neutra.",
    "english": "Neutral narration.",
}

# Every label that counts as narration, in any language anyone may have used
# before (or set by hand) — recognition must never depend on one spelling.
_NARRATOR_LABELS = frozenset(
    {label.upper() for label in NARRATOR_LABEL_BY_LANGUAGE.values()}
    | {"NARRATOR", "NARRATION", "NARRATIVE", "VOICE OVER", "VOICEOVER",
       "旁白者", "叙述", "叙述者", "解说", "ナレーション", "내레이션"}
)

_AUTO_VALUES = frozenset({"", "auto", "auto (detect)", "automatic", "none"})


def normalize_language(value):
    """Return the configured language name, or "" when it means "let the text decide"."""
    text = str(value or "").strip()
    return "" if text.lower() in _AUTO_VALUES else text


def narrator_label_for(language, override=""):
    """The single label narration must use, honouring an explicit override."""
    custom = str(override or "").strip()
    if custom:
        return custom
    lang = normalize_language(language)
    return NARRATOR_LABEL_BY_LANGUAGE.get(lang.lower(), "NARRATOR")


def default_instruct_for(language):
    """A sensible single-speaker default direction in the configured language."""
    lang = normalize_language(language)
    return DEFAULT_INSTRUCT_BY_LANGUAGE.get(lang.lower(), "Neutral narration.")


def is_narrator_label(label):
    """True when `label` is narration, in any language this module knows."""
    return str(label or "").strip().upper() in _NARRATOR_LABELS


def _build_directive(language, narrator_label):
    if not language:
        return ""
    return (
        "OUTPUT LANGUAGE — THIS OVERRIDES ANY EARLIER WORDING ABOVE\n"
        f"- Write the \"speaker\" label and the \"instruct\" direction in {language}.\n"
        f"- Use the single label {narrator_label} on every narration / non-dialogue entry, "
        f"spelled exactly the same way each time. It is the narrator's voice name.\n"
        f"- For characters, use the name as it appears in the source text, in {language} — "
        f"never romanize it, never translate it, never add an honorific that is not there.\n"
        "- NEVER translate, paraphrase or re-spell the \"text\" field: it must stay exactly "
        "as the source text has it.\n"
        "- Field names, JSON syntax and this instruction block stay in English."
    )


def _build_description_directive(language):
    if not language:
        return ""
    return (
        "OUTPUT LANGUAGE — THIS OVERRIDES ANY EARLIER WORDING ABOVE\n"
        f"- Write the \"description\" field in {language}.\n"
        "- NEVER translate or re-spell the \"ref_text\" field: it must be copied verbatim "
        "from the character's own lines, in the book's language.\n"
        "- Field names, JSON syntax and this instruction block stay in English."
    )


def language_context(language, narrator_label=""):
    """Everything the script prompts need to render in the configured language.

    Keys: ``language`` ("" when auto-detecting), ``display_language`` (what a
    ``{language}`` placeholder should say), ``narrator_label`` and ``directive``
    (empty when the language is auto, because there is nothing to enforce).
    """
    lang = normalize_language(language)
    label = narrator_label_for(lang, narrator_label)
    return {
        "language": lang,
        "display_language": lang or "the same language as the source text",
        "narrator_label": label,
        "directive": _build_directive(lang, label),
    }


def description_language_context(language):
    """Same shape, but for the persona pass, whose output fields differ."""
    lang = normalize_language(language)
    return {
        "language": lang,
        "display_language": lang or "the same language as the source text",
        "narrator_label": narrator_label_for(lang),
        "directive": _build_description_directive(lang),
    }


def character_style_language_context(language):
    """Same shape, for the per-voice Character Style anchor pass."""
    lang = normalize_language(language)
    directive = ""
    if lang:
        directive = (
            "OUTPUT LANGUAGE — THIS OVERRIDES ANY EARLIER WORDING ABOVE\n"
            f"- Write every \"character_style\" anchor in {lang}.\n"
            "- Keep the speaker labels you were given exactly as they are; do not "
            "translate or re-spell them.\n"
            "- Field names, JSON syntax and this instruction block stay in English."
        )
    return {
        "language": lang,
        "display_language": lang or "the same language as the source text",
        "narrator_label": narrator_label_for(lang),
        "directive": directive,
    }


def apply_language(system_prompt, ctx):
    """Append the language rule to a system prompt (idempotent)."""
    directive = (ctx or {}).get("directive", "")
    prompt = system_prompt or ""
    if not directive or directive in prompt:
        return prompt
    return f"{prompt.rstrip()}\n\n{directive}"


def safe_format(template, ctx, **kwargs):
    """Fill a prompt template, tolerating braces we do not own.

    `str.format` raises on a literal `{...}` that the author did not mean as a
    placeholder — and these templates are user-editable and often quote JSON
    examples. On failure the known placeholders are replaced textually instead.
    """
    ctx = ctx or {}
    values = dict(kwargs)
    values.setdefault("language", ctx.get("display_language", ""))
    values.setdefault("narrator_label", ctx.get("narrator_label", "NARRATOR"))
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        out = template
        for key, value in values.items():
            out = out.replace("{" + key + "}", str(value))
        return out
