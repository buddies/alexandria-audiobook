"""Deterministic clean-up and auditing for the per-line Emotion / Style instruct.

Why this exists
---------------
For CustomVoice the instruct is tokenized as a **user turn** while the spoken text
becomes an **assistant turn** (qwen_tts/inference/qwen3_tts_model.py:269-276), so
the engine literally reads it as a request addressed to the voice. Two
consequences drive the rules below:

* Only *delivery* language steers the performance. Measured on this project
  (voice=Serena, seed=42, anchor fixed, 3 different texts): an explicit Section III
  rate instruction ("语速极慢，拖长音，语气沉重迟缓。") made every take 10-28 %
  longer (+17.9 % on average), while a scene/directorial note with no acoustic
  target ("强调时间的紧迫和事件的诡异。") moved nothing consistently (+4.7 %,
  -0.9 % on one text). A note the voice cannot act on is wasted tokens.
* Anything *acoustic* in that turn (timbre, register, age, gender) duplicates the
  speaker preset and the Character Style, and every change to the string re-rolls
  the timbre - see VOICE_REFERENCE.md, whose working rule 3 is "never mix acoustic
  and behavioural descriptors".

So: `instruct` carries Section II/III vocabulary (emotion, delivery, pacing) in one
short clause; Section I vocabulary (texture, timbre, register) belongs in
Character Style; actions, gestures and scene meaning belong nowhere.

The lexicons below are a curated mirror of VOICE_REFERENCE.md Sections I-III.
`normalize_instruct` only cleans - it never invents or removes meaning - while
`audit_instruct` reports what a human (or the LLM prompt) still has to fix.
"""

import re

# ── limits ───────────────────────────────────────────────────────────────────
MAX_CHARS = 60            # hard cap after which the instruct is truncated at a clause
WARN_CHARS_EN = 90
WARN_WORDS_EN = 12
WARN_CHARS_ZH = 24        # ~16 字 of content plus punctuation
# The reviewer's own examples ("Cold fury, barely contained, voice tight") use three
# comma parts, so three is conformant and four or more reads as over-specified.
MAX_CLAUSES = 3

# ── lexicons (VOICE_REFERENCE.md mirrors) ────────────────────────────────────
# Section I - texture / timbre / register / anatomy. Belongs in Character Style.
TIMBRE_TERMS = [
    # English (Section I headline terms)
    "timbre", "gravelly", "raspy", "husky", "scratchy", "smoky", "guttural", "coarse",
    "hoarse", "throaty", "gruff", "silky", "velvety", "honeyed", "creamy", "mellow",
    "buttery", "booming", "chesty", "sonorous", "rumbling", "hollow", "cavernous",
    "bassy", "resonant", "breathy", "airy", "feathery", "reedy", "tinny", "shrill",
    "nasal", "twangy", "whiny", "brittle", "metallic", "falsetto", "vocal fry",
    "sibilant", "tremulous", "register", "baritone", "tenor", "bass", "alto",
    "soprano", "mezzo", "pitch range", "high-pitched", "low-pitched", "vocal cords",
    # Chinese
    "音色", "音区", "音质", "嗓音", "声线", "共鸣", "共振", "磁性", "沙哑", "嘶哑",
    "浑厚", "厚重", "厚实", "低沉", "明亮", "清脆", "清亮", "圆润", "尖细", "细弱",
    "鼻音", "气声", "干涩", "粗糙", "苍老", "低音", "中音", "高音", "男中音", "男高音",
    "女中音", "男声", "女声", "少女", "儿童", "声带", "喉咙", "喉音",
    # identity words that belong in the anchor, not per line
    "年轻", "年迈", "中年", "十几岁", "少年", "老妇", "老者", "岁",
]

# Section II - emotion / attitude. Wanted in the instruct.
EMOTION_TERMS = [
    "angry", "sad", "happy", "afraid", "anxious", "nervous", "tense", "calm",
    "warm", "cold", "weary", "tired", "excited", "amused", "wry", "bitter",
    "resigned", "smug", "urgent", "gentle", "stern", "tender", "defeated",
    "grieving", "startled", "sarcastic", "desperate", "curious", "guarded",
    # the sanctioned neutral-narration vocabulary must not read as "no direction"
    "neutral", "even", "somber", "sombre", "quiet", "quietly", "composed",
    "steady", "matter-of-fact", "deadpan", "wryly", "light", "grave",
    "疲惫", "愤怒", "生气", "紧张", "平静", "冷淡", "犹豫", "惊讶", "恐惧", "悲伤",
    "喜悦", "无奈", "嘲讽", "轻蔑", "轻慢", "坚定", "恭敬", "急切", "克制", "冷静",
    "焦虑", "压抑", "警惕", "温柔", "严厉", "兴奋", "茫然", "绝望", "心虚", "坦然",
    "凝重", "沉重", "戏谑", "慵懒", "震惊", "困惑", "笃定", "诚恳", "威严", "深意",
    "中性", "客观", "平稳", "平和", "沉静", "从容", "凝重", "专注", "生硬",
    "炫耀", "掌控", "试探", "示弱", "尴尬", "如释重负", "悬疑", "锐利", "深澱",
    "坚定", "郑重", "轻快", "从容不迫", "好奇", "体谅", "亲昵", "怀疑",
]

# Section III - delivery / pacing / articulation. Wanted in the instruct.
DELIVERY_TERMS = [
    "pace", "slowly", "quickly", "rapidly", "measured", "halting", "staccato",
    "legato", "drawl", "monotone", "flat", "clipped", "whisper", "murmur", "mutter",
    "shout", "softly", "loudly", "emphasis", "pause", "breathless", "articulate",
    "narration", "narrating", "deadpan", "hushed", "brisk", "clipped", "staccato",
    "语速", "缓慢", "快速", "急促", "平稳", "停顿", "拖长", "断断续续", "喃喃",
    "低语", "耳语", "低声", "轻声", "小声", "喊", "吼", "大声", "高喊", "咬字",
    "重音", "放慢", "加快", "平铺直叙", "拖腔", "一字一顿", "叹息", "喘息", "颤抖",
    "语气", "口吻", "调子", "念", "叙述", "播报", "朗读", "平缓", "语调", "声调",
    "干脆", "强调", "咬字清晰", "含蓄", "简洁", "利落", "停顿感",
]

# Actions / gestures / scene meaning / meta - belong nowhere in a voice direction.
ACTION_TERMS = [
    "nod", "shake his head", "shake her head", "smile", "smiles", "laugh", "sigh",
    "glance", "stare", "frown", "shrug", "gesture", "turn away", "reaches",
    "笑了笑", "笑", "点头", "摇头", "转身", "拿起", "放下", "看着", "盯着", "皱眉",
    "叹气", "耸肩", "挥手", "站起来", "坐下", "走向", "停下", "动作", "表情", "眼神",
    "手势", "沉默",
]
SCENE_TERMS = [
    "scene", "describes", "description", "narrating the", "the room", "setting",
    "场景", "描述", "画面", "周围", "环境", "背景", "氛围", "体现场", "突显", "渲染",
    "体现", "强调事件", "事件", "情节", "弦外之音", "暗示",
]

_BRACKETS = "「」『』“”\"'‘’（）()【】〔〕[]《》〈〉"
_CONTROL_TOKEN = re.compile(r"<\|[^|]*\|>")
_SPEAKER_PREFIX = re.compile(r"^[A-Z][A-Za-z0-9 _.\-]{0,30}[:：]\s*")
_CLAUSE_SPLIT = re.compile(r"[,，;；。.!！?？、]")


def _has_cjk(text):
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))


def _hits(text, terms):
    low = text.lower()
    found = []
    for term in terms:
        if term.isascii():
            if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", low):
                found.append(term)
        elif term in text:
            found.append(term)
    return found


def count_clauses(text):
    return len([p for p in _CLAUSE_SPLIT.split(text or "") if p.strip()])


def normalize_instruct(text, speaker=""):
    """Clean an instruct without changing what it asks for.

    Removes wrapping quotes/brackets, a "SPEAKER:" or "旁白：" prefix, leaked special
    tokens, newlines and repeated punctuation, then caps the length at a clause
    boundary. Returns "" only when there is nothing left.
    """
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = _CONTROL_TOKEN.sub(" ", cleaned)
    cleaned = cleaned.replace("\u3000", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(_BRACKETS).strip()
    cleaned = _SPEAKER_PREFIX.sub("", cleaned).strip()
    # A localised label ("旁白：…") is only stripped when we know the label, so a
    # legitimate "注意：…"-style direction is never truncated.
    label = str(speaker or "").strip()
    if label:
        cleaned = re.sub(rf"^{re.escape(label)}\s*[:：]\s*", "", cleaned).strip()
    # stray closing quote left over from an unbalanced pair
    cleaned = cleaned.strip(_BRACKETS).strip()
    cleaned = re.sub(r"([，,；;、])\1+", r"\1", cleaned)
    cleaned = re.sub(r"([！!？?。.~～…])\1+", r"\1", cleaned)
    cleaned = cleaned.strip(" ,;，；、:：")
    if len(cleaned) > MAX_CHARS:
        clauses = [p.strip() for p in _CLAUSE_SPLIT.split(cleaned) if p.strip()]
        kept = ""
        for clause in clauses:
            candidate = f"{kept}，{clause}" if kept else clause
            if len(candidate) > MAX_CHARS:
                break
            kept = candidate
        cleaned = (kept or cleaned[:MAX_CHARS].rsplit(" ", 1)[0]).rstrip(" ,;，；、")
    return cleaned.strip()


def audit_instruct(text, speaker=""):
    """Return a list of findings: {"code", "detail"}. Empty list == conformant."""
    instruction = (text or "").strip()
    findings = []
    if not instruction:
        return [{"code": "empty", "detail": "no direction at all"}]

    zh = _has_cjk(instruction)
    for code, terms, why in (
        ("timbre", TIMBRE_TERMS, "acoustic identity — belongs in Character Style"),
        ("action", ACTION_TERMS, "an action or gesture, not a voice direction"),
        ("scene", SCENE_TERMS, "scene/meta wording the voice cannot act on"),
    ):
        hits = _hits(instruction, terms)
        if hits:
            findings.append({"code": code, "detail": f"{', '.join(hits[:4])} ({why})"})

    if not (_hits(instruction, EMOTION_TERMS) or _hits(instruction, DELIVERY_TERMS)):
        findings.append({"code": "no_delivery",
                         "detail": "no emotion/delivery/pacing word — nothing for the voice to act on"})

    clauses = count_clauses(instruction)
    if clauses > MAX_CLAUSES:
        findings.append({"code": "over_specified",
                         "detail": f"{clauses} clauses (max {MAX_CLAUSES})"})
    words = len(instruction.split())
    if zh and len(instruction) > WARN_CHARS_ZH:
        findings.append({"code": "long", "detail": f"{len(instruction)} chars (aim <= {WARN_CHARS_ZH})"})
    elif not zh and (words > WARN_WORDS_EN or len(instruction) > WARN_CHARS_EN):
        findings.append({"code": "long", "detail": f"{words} words / {len(instruction)} chars"})

    if re.search(r"<\|", instruction):
        findings.append({"code": "control_token", "detail": "leaked model control token"})
    if any(ch in instruction for ch in _BRACKETS):
        findings.append({"code": "punctuation", "detail": "quotes/brackets (stage-direction style)"})
    if re.search(r"[!！?？~～…]{2,}", instruction) or "..." in instruction:
        findings.append({"code": "punctuation", "detail": "repeated/ellipsis punctuation"})
    if speaker and speaker.strip() and speaker.strip() in instruction:
        findings.append({"code": "speaker_name", "detail": f"mentions '{speaker.strip()}'"})
    return findings


def normalize_entries(entries):
    """Clean the instruct of every entry in place. Returns (entries, changed_count)."""
    changed = 0
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        before = entry.get("instruct") or ""
        after = normalize_instruct(before, entry.get("speaker") or "")
        if after != (before or "").strip():
            changed += 1
        entry["instruct"] = after
    return entries, changed


def audit_entries(entries, label=""):
    """Count findings across entries. Returns {"total", "flagged", "counts", "examples"}."""
    counts = {}
    examples = []
    evaluated = 0
    flagged = 0
    for i, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            continue
        evaluated += 1
        findings = audit_instruct(entry.get("instruct") or "", entry.get("speaker") or "")
        if not findings:
            continue
        flagged += 1
        for finding in findings:
            counts[finding["code"]] = counts.get(finding["code"], 0) + 1
        if len(examples) < 8:
            examples.append({
                "index": i,
                "speaker": entry.get("speaker") or "",
                "instruct": (entry.get("instruct") or "")[:60],
                "codes": sorted({f["code"] for f in findings}),
            })
    return {
        "label": label,
        "total": evaluated,
        "flagged": flagged,
        "counts": counts,
        "examples": examples,
    }


def format_audit(report, examples=True):
    """One-line summary (+ optional examples) for the generation logs."""
    head = (f"[instruct audit{(':' + report['label']) if report.get('label') else ''}] "
            f"{report['flagged']}/{report['total']} line(s) need attention")
    if not report["counts"]:
        return head + " — all conformant"
    detail = ", ".join(f"{code}={n}" for code, n in sorted(report["counts"].items(),
                                                           key=lambda kv: -kv[1]))
    lines = [f"{head}: {detail}"]
    if examples:
        for example in report["examples"][:5]:
            lines.append(
                f"    line {example['index']} [{example['speaker']}] {example['instruct']!r} "
                f"-> {', '.join(example['codes'])}"
            )
    return "\n".join(lines)


# ── the rule block that is appended to whatever system prompt is in use ──────
# Same idea as the output-language block: the shipped prompt can be replaced by
# the user, but these constraints are structural (they follow from how the engine
# tokenizes the field), so they must hold either way.
INSTRUCT_RULES = """EMOTION / STYLE FIELD — THIS OVERRIDES ANY EARLIER WORDING ABOVE
- The "instruct" value is sent to the TTS engine as a one-line instruction to the speaker, so write what the VOICE should DO, not what the scene means.
- Allowed, in this order: emotional tone, delivery, pacing or articulation. Example: "Alert, quiet and measured."
- FORBIDDEN in "instruct": timbre, register, age, gender or accent (that is the character's permanent voice identity and is applied separately), physical actions or gestures, descriptions of the scene or of what the line "shows", bracketed stage directions, quotes, ellipses, and the speaker's own name.
- ONE short clause, at most two: <= 10 English words, <= 16 Chinese characters, and never more than three comma-separated parts. Do not stack every quality you can think of — over-specifying makes the performance less predictable, not more precise.
- Name pace explicitly when the line needs it ("speaking slowly", "语速极慢"): concrete delivery wording measurably steers the take, vague energy wording does not.
- Every line carries its own complete direction: never "same as before", "still ...", or a reference to the previous line.
- Write it in the same language as the text unless the instructions above say otherwise."""


REVIEW_INSTRUCT_ADDENDUM = """HOW TO APPLY THE FIELD RULES ABOVE WHEN REVIEWING
- A non-conformant "instruct" MUST be rewritten, even when it reads well as prose. This is required work, not an edit of the author's text: the "text" of every entry, its "speaker", and every punctuation mark inside "text" stay exactly as they are.
- Rewrite only what is actually wrong with the entry. If the entry already names a tone or a delivery ("沉稳，清晰，语速平缓。"), leave it exactly as it is — never append a generic pace note to an entry that is already conformant, and never turn two lines into the same wording.
- Scene meaning, or a note about what the line shows -> replace that part with the delivery that would produce the effect. Examples:
    "叙述节奏放缓，营造一种压抑且充满悬念的氛围。" -> "语速放缓，压低声音，留出停顿。"
    "极力强调事情的真实性和荒谬性。" -> "重音落在关键处，语气激动。"
    "平淡，随意，像是在闲聊以掩饰探究。" -> "平淡，随意，声音放松。"
- Timbre, register, age, gender or accent -> delete that part, keep the performance direction:
    "年轻，疲惫，慵懒且略带戏谑的声音。" -> "疲惫，慵懒，略带戏谑。"
- Physical actions or gestures -> delete them; when they carry tone, keep only the vocal part:
    "自言自语，摇头，语气中带着不可思议。" -> "自言自语般的低语，语气中带着不可思议。"
- Four or more comma-separated qualities -> keep the two or three that actually change the delivery.
- Keep the language of the rest of the script, and return every entry (changed or not) as valid JSON only."""


def apply_instruct_rules(system_prompt, review=False):
    """Append the Emotion / Style rule block to a system prompt (idempotent).

    `review=True` adds the imperative how-to-fix addendum: a review prompt that
    says "do not rephrase anything" would otherwise win and leave the bad
    instructs untouched (measured: only 2 of 17 non-conformant lines got fixed).
    """
    prompt = system_prompt or ""
    if INSTRUCT_RULES not in prompt:
        prompt = f"{prompt.rstrip()}\n\n{INSTRUCT_RULES}"
    if review and REVIEW_INSTRUCT_ADDENDUM not in prompt:
        prompt = f"{prompt.rstrip()}\n\n{REVIEW_INSTRUCT_ADDENDUM}"
    return prompt
