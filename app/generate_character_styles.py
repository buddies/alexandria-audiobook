"""Generate one constant `character_style` anchor per speaker from the script.

Why this exists
---------------
On the CustomVoice path the TTS engine regenerates the voice from the entire
`instructions` string, and the seed only fixes the sampling noise - it does NOT
pin the timbre. Measured on this project (voice=Sohee, seed=1, one fixed text,
6 real per-line instructions): pitch register moved 206-238 Hz (about 2.5
semitones) and duration swung 13.8-19.8 s. Hallucinating a fresh emotion
sentence per line is therefore the same as re-rolling the voice every line.

A constant anchor cannot cancel that per-line drift on its own (measured: the
best wording cuts the spectral spread by roughly 30% and leaves the register
moving), but it is the piece the pipeline has always been missing: it fixes the
part of the instruction that describes who the character *is*, so switching the
per-line emotion later - or turning `tts.instruct_style` down to `voice_style` -
lands on a deliberate, documented voice instead of an accident.

What it writes
--------------
`voice_config.json`, one `character_style` string per speaker. Existing anchors
are preserved unless `--overwrite` is passed; `default_style` counts as an
existing anchor because the engine prefers `character_style` but falls back to
it, so filling one would silently change that voice.
"""

import argparse
import json
import os
import re
import sys
import time

from openai import OpenAI

from character_style_prompts import load_character_style_prompts
from llm_utils import analyze_response, next_max_tokens, reasoning_warning
from script_language import apply_language, character_style_language_context, is_narrator_label, safe_format
from utils import atomic_json_write

MAX_STYLE_CHARS = 240          # a 25-word anchor is ~150 chars; hard cap for safety
MAX_SAMPLE_LINES = 6
MAX_CONTEXT_LINES = 3
DEFAULT_BATCH_SIZE = 12

# Anchors must describe identity, not delivery. These words are legitimate when
# used as a permanent trait ("steady measured pace") which is why they only
# produce a warning, not a rejection.
_EMOTION_WORDS = (
    "angry", "sad", "excited", "nervous", "tense", "joyful", "grieving",
    "fearful", "whisper", "shout", "yelling", "crying", "laughing",
)

# Default entry written for a speaker that has no voice_config entry yet. The
# fields and their order match what the Voices tab writes for a new card.
DEFAULT_ENTRY = {
    "type": "custom",
    "voice": "Ryan",
    "character_style": "",
    "default_style": "",
    "seed": "-1",
    "ref_audio": None,
    "ref_text": None,
    "adapter_id": None,
    "adapter_path": None,
    "description": "",
}


# ── script / voice_config helpers ────────────────────────────────────────────

def _entry_speaker(entry):
    return str(entry.get("speaker") or entry.get("type") or "").strip()


def _entry_text(entry):
    return str(entry.get("text") or "").strip()


def _is_narrator(label):
    return is_narrator_label(label)


def _pick_lines(lines):
    """Spread the samples across the whole book instead of taking the first N.

    A character who starts out angry and ends up calm must not be described from
    one emotional extreme, and the anchor is supposed to ignore emotion anyway.
    """
    lines = [ln for ln in lines if ln]
    if len(lines) <= MAX_SAMPLE_LINES:
        return lines
    step = len(lines) / float(MAX_SAMPLE_LINES)
    picked = [lines[int(i * step)] for i in range(MAX_SAMPLE_LINES)]
    out = []
    for line in picked:
        if line not in out:
            out.append(line)
    return out


def collect_speaker_context(script, window=4):
    """Return {speaker: {"count": n, "lines": [...], "context": [...]}}."""
    appearances = {}
    texts = {}
    for i, entry in enumerate(script):
        speaker = _entry_speaker(entry)
        if not speaker:
            continue
        appearances.setdefault(speaker, []).append(i)
        text = _entry_text(entry)
        if text:
            texts.setdefault(speaker, []).append(text)

    info = {}
    for speaker, indices in appearances.items():
        context = []
        seen = set()
        for idx in indices:
            for j in range(max(0, idx - window), min(len(script), idx + window + 1)):
                if j == idx:
                    continue
                entry = script[j]
                if not _is_narrator(_entry_speaker(entry)):
                    continue
                line = _entry_text(entry)
                if line and line not in seen:
                    seen.add(line)
                    context.append(line)
                    if len(context) >= MAX_CONTEXT_LINES:
                        break
            if len(context) >= MAX_CONTEXT_LINES:
                break

        lines = texts.get(speaker, [])
        info[speaker] = {
            "count": len(indices),
            "lines": _pick_lines(lines),
            "context": [] if _is_narrator(speaker) else context,
        }
    return info


def _existing_anchor(entry):
    """The anchor the engine would use today (`character_style`, else `default_style`)."""
    entry = entry or {}
    return str(entry.get("character_style") or entry.get("default_style") or "").strip()


# ── LLM plumbing ─────────────────────────────────────────────────────────────

def extract_json_object(text):
    """Return the first balanced {...} object in `text` as a dict, or None."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    end = None
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return None
    try:
        parsed = json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


_PAIR_RE = re.compile(r'"([^"\\]+)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _salvage_pairs(text):
    """Last-resort parse: pull "key": "value" pairs out of a malformed object.

    A truncated or over-quoted reply still gets most characters an anchor;
    dropping the whole batch would leave them without one.
    """
    out = {}
    for key, value in _PAIR_RE.findall(text or ""):
        try:
            out.setdefault(key, json.loads(f'"{value}"'))
        except (json.JSONDecodeError, ValueError):
            out.setdefault(key, value)
    return out


def _call_llm(client, model_name, system_prompt, user_prompt, max_tokens=1200):
    """One chat call, retried once with a larger budget if the reply is truncated.

    Thinking output is charged against `max_tokens`, so a reasoning model can
    return nothing usable while `finish_reason == "length"`. A partial-but-
    non-empty body is kept: it usually still holds several complete anchors.
    """
    budget = max(1, int(max_tokens))
    output = None
    for _ in range(2):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=budget,
        )
        output = analyze_response(response)
        truncated = (output.finish_reason or "").lower() == "length"
        if not truncated or output.body.strip():
            return output
        issue = reasoning_warning(output)
        print(f"  Warning: {issue or 'empty truncated reply'}; retrying with a larger budget")
        budget = next_max_tokens(budget, max_tokens)
    return output


# ── text hygiene ─────────────────────────────────────────────────────────────

def normalize_style(value):
    """Clean one anchor string, or return "" when it is unusable."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = re.sub(r"\s+", " ", value.replace("\u3000", " ")).strip()
    text = text.strip("\"'`“”‘’ ").strip()
    # Models like to prefix the label they were given ("NARRATOR: ...").
    text = re.sub(r"^[A-Z][A-Z0-9 _.\-]{1,30}:\s*", "", text).strip()
    if not text:
        return ""
    if len(text) > MAX_STYLE_CHARS:
        cut = text[:MAX_STYLE_CHARS]
        if " " in cut:
            cut = cut[:cut.rfind(" ")]
        text = cut.rstrip(" ,;")
    return text


def _warn_if_emotional(speaker, style):
    lowered = style.lower()
    hits = [w for w in _EMOTION_WORDS if w in lowered]
    if hits:
        print(f"  Warning: {speaker} anchor mentions delivery/emotion words {hits} — "
              f"those belong to the per-line instruct, not the constant anchor.")


def _match_speaker(raw_key, speakers):
    """Map an LLM-returned key back onto the real speaker label."""
    if raw_key in speakers:
        return raw_key
    norm = {s.strip().lower(): s for s in speakers}
    key = str(raw_key or "").strip().lower()
    if key in norm:
        return norm[key]
    for lowered, original in norm.items():
        if lowered and (lowered in key or key in lowered):
            return original
    return None


# ── core pass ────────────────────────────────────────────────────────────────

def _format_characters(speakers, info):
    blocks = []
    for speaker in speakers:
        data = info.get(speaker, {})
        lines = data.get("lines") or []
        context = data.get("context") or []
        parts = [f"=== SPEAKER: {speaker} ===", f"Lines spoken: {data.get('count', 0)}"]
        if context:
            parts.append("Surrounding narration:")
            parts.extend(f"  - {line}" for line in context)
        if lines:
            parts.append("Sample lines spoken by this speaker:")
            parts.extend(f"  - {line}" for line in lines)
        else:
            parts.append("(no dialogue text available)")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def _generate_batch(client, model_name, system_prompt, user_prompt, speakers, info, lang_ctx):
    """Ask the LLM for the anchors of one batch. Returns {speaker: style}."""
    prompt = safe_format(user_prompt, lang_ctx, characters=_format_characters(speakers, info))
    output = _call_llm(client, model_name, apply_language(system_prompt, lang_ctx), prompt)
    text = output.body or ""
    if not text.strip():
        issue = reasoning_warning(output)
        print(f"  Warning: empty reply from the LLM ({issue or 'no visible content'})")
        return {}

    parsed = extract_json_object(text)
    if parsed is None:
        print("  Warning: reply was not a JSON object, trying to salvage key/value pairs")
        parsed = _salvage_pairs(text)

    styles = {}
    for raw_key, raw_value in (parsed or {}).items():
        speaker = _match_speaker(raw_key, speakers)
        if not speaker:
            print(f"  Warning: ignoring unknown speaker '{raw_key}' in the reply")
            continue
        style = normalize_style(raw_value)
        if not style:
            print(f"  Warning: empty anchor returned for {speaker}")
            continue
        styles[speaker] = style
    return styles


def generate(script, voice_config, *, client, model_name, language, system_prompt,
             user_prompt, batch_size=DEFAULT_BATCH_SIZE, overwrite=False,
             default_preset=None, dry_run=False):
    """Fill in missing anchors. Returns a summary dict."""
    lang_ctx = character_style_language_context(language)
    info = collect_speaker_context(script)
    all_speakers = list(info.keys())

    pending, skipped, aliased = [], [], []
    for speaker in all_speakers:
        entry = voice_config.get(speaker, {})
        if entry.get("alias_of"):
            aliased.append(speaker)
            continue
        existing = _existing_anchor(entry)
        if existing and not overwrite:
            skipped.append(speaker)
        else:
            pending.append(speaker)

    summary = {"updated": {}, "skipped": skipped, "aliased": aliased, "failed": []}

    if not pending:
        print("Every speaker already has a character style; nothing to generate.")
        return summary

    print(f"Requesting character styles for {len(pending)} speaker(s)"
          f"{' (overwriting existing anchors)' if overwrite else ' (only empty anchors)'}...")

    styles = {}
    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start:start + max(1, batch_size)]
        print(f"  Batch {start // max(1, batch_size) + 1}: {', '.join(batch)}")
        try:
            styles.update(
                _generate_batch(client, model_name, system_prompt, user_prompt,
                                batch, info, lang_ctx)
            )
        except Exception as e:
            print(f"  Error: LLM call failed for this batch: {e}")
            summary["failed"].extend(batch)
        time.sleep(0.3)

    for speaker in pending:
        style = styles.get(speaker)
        if not style:
            if speaker not in summary["failed"]:
                summary["failed"].append(speaker)
            continue
        _warn_if_emotional(speaker, style)
        print(f"  {speaker} -> {style}")
        if dry_run:
            summary["updated"][speaker] = style
            continue
        entry = voice_config.get(speaker)
        if not isinstance(entry, dict):
            entry = dict(DEFAULT_ENTRY)
            if default_preset:
                entry["voice"] = default_preset
            print(f"  (created a new voice_config entry for {speaker}, preset "
                  f"'{entry['voice']}' — change it in the Voices tab if it is wrong)")
        entry["character_style"] = style
        voice_config[speaker] = entry
        summary["updated"][speaker] = style

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate a constant character_style anchor for each speaker in the script")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace anchors that are already set (default: only fill empty ones)")
    parser.add_argument("--language", default="",
                        help="Language for the anchor text (default: tts.language from app/config.json)")
    parser.add_argument("--speakers", default="",
                        help="Optional comma-separated speaker allowlist")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Speakers per LLM call (default {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--default-preset", default="",
                        help="Preset voice to use when creating a missing voice_config entry (default: Ryan)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the generated anchors without writing voice_config.json")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(root, "annotated_script.json")
    voice_config_path = os.path.join(root, "voice_config.json")
    app_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found. Generate a script first.")
        return 1

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            script = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: could not read {os.path.basename(script_path)}: {e}")
        return 1

    if not isinstance(script, list) or not script:
        print("Error: the annotated script is empty.")
        return 1

    config = {}
    if os.path.exists(app_config_path):
        try:
            with open(app_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: could not read app/config.json: {e}")

    llm_cfg = config.get("llm", {})
    model_name = llm_cfg.get("model_name", "")
    language = args.language.strip() or str(config.get("tts", {}).get("language") or "English")
    client = OpenAI(base_url=llm_cfg.get("base_url", "http://localhost:11434/v1"),
                    api_key=llm_cfg.get("api_key", "local"))

    prompts_cfg = config.get("prompts", {})
    default_system, default_user = load_character_style_prompts()
    system_prompt = prompts_cfg.get("character_style_system_prompt") or default_system
    user_prompt = prompts_cfg.get("character_style_user_prompt") or default_user

    voice_config = {}
    if os.path.exists(voice_config_path):
        try:
            with open(voice_config_path, "r", encoding="utf-8") as f:
                voice_config = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: could not read voice_config.json ({e}); starting from an empty one")
            voice_config = {}

    if args.speakers.strip():
        allow = {s.strip() for s in args.speakers.split(",") if s.strip()}
        script = [e for e in script if _entry_speaker(e) in allow]
        if not script:
            print("Error: none of the requested speakers appear in the script.")
            return 1

    print(f"Model: {model_name}  Language for anchors: {language}")

    try:
        summary = generate(
            script, voice_config,
            client=client, model_name=model_name, language=language,
            system_prompt=system_prompt, user_prompt=user_prompt,
            batch_size=max(1, args.batch_size), overwrite=args.overwrite,
            default_preset=args.default_preset.strip(), dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"Error: character style generation failed: {e}")
        return 1

    if args.dry_run:
        print(f"Dry run: would have written {len(summary['updated'])} anchor(s); "
              f"voice_config.json left untouched.")
        return 0

    if summary["updated"]:
        try:
            atomic_json_write(voice_config, voice_config_path)
            print(f"Updated voice_config.json with {len(summary['updated'])} character style(s).")
        except Exception as e:
            print(f"Error: failed to save voice_config.json: {e}")
            return 1

    if summary["skipped"]:
        print(f"Kept existing anchors for: {', '.join(summary['skipped'])}")
    if summary["aliased"]:
        print(f"Skipped alias speakers: {', '.join(summary['aliased'])}")
    if summary["failed"]:
        print(f"No anchor produced for: {', '.join(summary['failed'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
