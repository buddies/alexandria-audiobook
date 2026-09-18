"""Load the character_style prompt pair.

`character_style_prompts.txt` in the project root is the editable copy; the
constants below are the built-in fallback. Unlike the other prompt loaders in
this app, a missing or malformed file must NOT be fatal: this pass runs
automatically right after script generation, and failing there would throw away
a perfectly good script. So the loader degrades to the built-ins instead of
raising, and scripts can always fall back to the shipped wording.
"""

import os

_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "..", "character_style_prompts.txt")

BUILTIN_SYSTEM_PROMPT = """You write the constant voice-identity anchor ("character_style") for every speaker in an audiobook TTS script.

The pipeline appends that anchor to each line's per-line emotion direction and sends the result to the TTS engine as its `instructions` field. In this engine the voice is regenerated from the entire instruction string, so any wording that changes between lines re-rolls the speaker's timbre, pitch register and tempo. The anchor therefore has to describe the ONE thing that must never change about a character's voice.

RULES
1. One sentence, at most 25 words. No newlines, no quotes, no brackets, no parentheses.
2. Describe acoustics only, in this order:
   - voice type and apparent age or gender when the text makes it inferable (e.g. "male voice in his fifties")
   - pitch register (e.g. "low baritone register", "medium-high register")
   - timbre and texture (e.g. "dry, slightly gravelly timbre", "bright, clear timbre")
   - accent or dialect only when the text clearly implies one
   - the permanent baseline pace and energy (e.g. "deliberate measured pace")
3. NEVER include: emotion or mood words (angry, sad, tense, calm, warm, nervous...), delivery directions (whispering, shouting, pausing), sentence-specific actions, references to the current scene, or anything that would only be true for a single line. Those belong to the per-line direction.
4. Never put the character's own name or the word "voice" followed by a name in the anchor.
5. The narrator anchor describes the book's storytelling voice, not a character inside the story.
6. Write every anchor in the requested output language.
7. Output ONLY one JSON object mapping each speaker label you were given to its anchor string, using the labels byte-for-byte. No markdown, no explanation, no code fences.

EXAMPLE OUTPUT
{"NARRATOR": "Male voice in his forties, low baritone register, dry slightly gravelly timbre, deliberate measured pace.", "ELENA": "Female voice in her thirties, medium register, bright clear timbre, precise even pace."}"""

BUILTIN_USER_PROMPT = """Write the constant voice-identity anchor for each speaker listed below.

Write the anchor text in {language}.

{characters}

Respond with only the JSON object: every speaker label above as a key, in the same order, and nothing else."""


def load_character_style_prompts():
    """Return (system_prompt, user_prompt).

    Re-reads on every call so edits to the .txt are picked up without a restart.
    Falls back to the built-in wording when the file is missing or malformed.
    """
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return BUILTIN_SYSTEM_PROMPT, BUILTIN_USER_PROMPT

    parts = raw.split("---SEPARATOR---")
    if len(parts) != 2:
        return BUILTIN_SYSTEM_PROMPT, BUILTIN_USER_PROMPT

    system_prompt, user_prompt = parts[0].strip(), parts[1].strip()
    if not system_prompt or not user_prompt:
        return BUILTIN_SYSTEM_PROMPT, BUILTIN_USER_PROMPT
    return system_prompt, user_prompt


CHARACTER_STYLE_SYSTEM_PROMPT, CHARACTER_STYLE_USER_PROMPT = load_character_style_prompts()
