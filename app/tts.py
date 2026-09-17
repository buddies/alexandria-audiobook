import os
import re
import json
import threading
import shutil
import zlib
import numpy as np
import soundfile as sf
from pydub import AudioSegment

DEFAULT_PAUSE_MS = 500  # Pause between different speakers
SAME_SPEAKER_PAUSE_MS = 250  # Shorter pause for same speaker continuing

# ── Join smoothing ───────────────────────────────────────────────
# Every chunk is synthesized by an independent TTS request, so the raw WAVs
# differ in lead/tail silence and in average level. Splicing them naively is
# what makes a single voice sound like it changes person mid-scene.
DEFAULT_JOIN_CONFIG = {
    "trim_silence": True,        # remove the TTS lead/tail silence
    "silence_threshold_db": -45.0,
    "keep_head_ms": 40,          # padding kept so speech onsets keep their attack
    "keep_tail_ms": 80,
    "match_loudness": True,      # gain-match every segment to one common level
    "target_dbfs": -20.0,
    "max_gain_db": 12.0,         # never boost/cut more than this
    "fade_ms": 25,               # de-click fade at every join
}


def resolve_join_config(raw):
    """Merge a user-supplied join config over DEFAULT_JOIN_CONFIG.

    Unknown keys are ignored and values are coerced to the type of the default,
    so a partially-filled or hand-edited config.json never breaks a merge.
    """
    cfg = dict(DEFAULT_JOIN_CONFIG)
    if not isinstance(raw, dict):
        return cfg
    for key, default in DEFAULT_JOIN_CONFIG.items():
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        try:
            if isinstance(default, bool):
                if isinstance(value, str):
                    cfg[key] = value.strip().lower() not in ("", "0", "false", "no", "off")
                else:
                    cfg[key] = bool(value)
            elif isinstance(default, int) and not isinstance(default, bool):
                cfg[key] = int(float(value))
            else:
                cfg[key] = float(value)
        except (TypeError, ValueError):
            continue
    return cfg


def trim_segment_silence(segment, threshold_db=-45.0, keep_head_ms=40, keep_tail_ms=80):
    """Trim lead/tail silence from one TTS segment, keeping a short padding.

    TTS adds 100-400 ms of silence to most segments; stacking that on top of the
    configured pause produces an uneven, chopped-up reading.

    Implemented with numpy instead of pydub's detect_nonsilent(), which does not
    exist in older pydub releases (0.23.x and below).
    """
    if segment is None or len(segment) == 0:
        return segment
    try:
        samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
        if segment.channels and segment.channels > 1:
            samples = samples.reshape(-1, segment.channels).mean(axis=1)
        if samples.size == 0:
            return segment

        window = max(1, int(segment.frame_rate * 0.01))  # 10 ms envelope windows
        n_windows = samples.size // window
        if n_windows == 0:
            return segment

        envelope = np.abs(samples[:n_windows * window]).reshape(n_windows, window).max(axis=1)
        peak = float(envelope.max())
        if peak <= 0:
            return segment  # all silence, nothing to keep

        full_scale = float(1 << (8 * segment.sample_width - 1))
        # Absolute dBFS threshold, but never stricter than 10% of this segment's
        # own peak (a quiet take would otherwise look "all silence").
        threshold = min(full_scale * (10 ** (float(threshold_db) / 20.0)), peak * 0.1)
        loud = np.nonzero(envelope > threshold)[0]
        if loud.size == 0:
            return segment

        start_ms = int(loud[0] * window / segment.frame_rate * 1000) - int(keep_head_ms)
        end_ms = int((loud[-1] + 1) * window / segment.frame_rate * 1000) + int(keep_tail_ms)
        start_ms = max(0, start_ms)
        end_ms = min(len(segment), end_ms)
        if end_ms <= start_ms:
            return segment
        return segment[start_ms:end_ms]
    except Exception as e:
        print(f"Warning: silence trim failed ({e}); keeping segment as-is")
        return segment


def match_segment_level(segment, target_dbfs=-20.0, max_gain_db=12.0):
    """Gain-match one segment towards a common average level.

    Keeps the whole book at a consistent perceived volume so consecutive takes
    of the same voice no longer jump in loudness at the join.
    """
    if segment is None or len(segment) == 0:
        return segment
    try:
        current_db = segment.dBFS
        peak_db = segment.max_dBFS
    except Exception:
        return segment
    if current_db == float("-inf") or peak_db == float("-inf"):
        return segment  # pure silence, nothing to match
    gain = float(target_dbfs) - current_db
    gain = max(-abs(float(max_gain_db)), min(abs(float(max_gain_db)), gain))
    # Never push the peak above -1 dBFS (avoids clipping on loud takes).
    gain = min(gain, -1.0 - peak_db)
    if abs(gain) < 0.1:
        return segment
    return segment.apply_gain(gain)


def prepare_segment_for_join(segment, join_config=None):
    """Normalize one chunk's audio before it is spliced into the timeline."""
    cfg = resolve_join_config(join_config)
    if cfg["trim_silence"]:
        segment = trim_segment_silence(
            segment,
            threshold_db=cfg["silence_threshold_db"],
            keep_head_ms=cfg["keep_head_ms"],
            keep_tail_ms=cfg["keep_tail_ms"],
        )
    if cfg["match_loudness"]:
        segment = match_segment_level(
            segment,
            target_dbfs=cfg["target_dbfs"],
            max_gain_db=cfg["max_gain_db"],
        )
    return segment


def stable_seed_for_speaker(speaker):
    """Deterministic seed derived from a speaker name.

    Two takes of the same voice then share the same initial noise sequence, so
    timbre, pitch and pacing stay close across independently generated segments
    instead of being re-rolled randomly on every request.
    """
    key = f"alexandria-tts::{speaker or ''}"
    return zlib.crc32(key.encode("utf-8")) % (2 ** 31 - 1)


def sanitize_filename(name):
    """Make a string safe for use in filenames"""
    name = re.sub(r'[^\w\-]', '_', name)
    return name.lower()


def combine_audio_with_pauses(audio_segments, speakers, pause_ms=DEFAULT_PAUSE_MS,
                              same_speaker_pause_ms=SAME_SPEAKER_PAUSE_MS,
                              pause_overrides=None, join_config=None):
    """Combine audio segments with pauses between them.

    Args:
        pause_overrides: Optional list aligned with audio_segments. Each entry is
            the pause (ms) to insert *after* that segment, or None to use the
            default speaker-change logic. The last entry is ignored.
        join_config: Optional dict (see DEFAULT_JOIN_CONFIG). Only the fade_ms
            key is used here; silence trimming / level matching happen earlier
            in ``prepare_segment_for_join`` so the timeline stays accurate.
    """
    if not audio_segments:
        return None

    segments = list(audio_segments)
    fade_ms = int(resolve_join_config(join_config)["fade_ms"])
    if fade_ms > 0 and len(segments) > 1:
        last_index = len(segments) - 1
        faded = []
        for i, segment in enumerate(segments):
            if i > 0:
                segment = segment.fade_in(fade_ms)
            if i < last_index:
                segment = segment.fade_out(fade_ms)
            faded.append(segment)
        segments = faded

    combined = segments[0]
    prev_speaker = speakers[0]

    for i, (segment, speaker) in enumerate(zip(segments[1:], speakers[1:])):
        override = pause_overrides[i] if pause_overrides else None
        if override is not None:
            gap = AudioSegment.silent(duration=override)
        elif speaker == prev_speaker:
            gap = AudioSegment.silent(duration=same_speaker_pause_ms)
        else:
            gap = AudioSegment.silent(duration=pause_ms)
        combined += gap + segment
        prev_speaker = speaker

    return combined


def compute_timeline(chunks_with_audio, pause_ms=DEFAULT_PAUSE_MS,
                     same_speaker_pause_ms=SAME_SPEAKER_PAUSE_MS):
    """Compute a timeline of (chunk, segment, abs_start_ms) tuples.

    Args:
        chunks_with_audio: list of (chunk_dict, AudioSegment) tuples.
            Each chunk_dict may have an optional 'pause_after' key (int ms)
            that overrides the default pause inserted after that chunk.
        pause_ms: Default pause between different speakers.
        same_speaker_pause_ms: Default pause when same speaker continues.

    Returns:
        list of (chunk_dict, AudioSegment, abs_start_ms) tuples.
    """
    timeline = []
    cursor_ms = 0
    prev_speaker = None
    prev_chunk = None

    for chunk, segment in chunks_with_audio:
        if prev_speaker is not None:
            override = prev_chunk.get("pause_after")
            if override is not None:
                gap = int(override)
            elif chunk["speaker"] == prev_speaker:
                gap = same_speaker_pause_ms
            else:
                gap = pause_ms
            cursor_ms += gap

        timeline.append((chunk, segment, cursor_ms))
        cursor_ms += len(segment)
        prev_speaker = chunk["speaker"]
        prev_chunk = chunk

    return timeline


class TTSEngine:
    """TTS engine supporting local (qwen-tts) and external (Gradio) backends.

    Mode is determined by config["tts"]["mode"]:
      - "local": Loads Qwen3TTSModel directly. No external server needed.
      - "external": Connects via Gradio client to a running TTS server.

    Models and clients are lazily initialized on first use.
    """

    def __init__(self, config):
        tts_config = config.get("tts", {})
        self._mode = tts_config.get("mode", "external")
        self._url = tts_config.get("url", "http://127.0.0.1:7860")
        self._device = tts_config.get("device", "auto")
        self._compile_codec_enabled = tts_config.get("compile_codec", False)

        # External backend protocol:
        #   "openai" - OpenAI-compatible POST /v1/audio/speech (vLLM-Omni, etc.)
        #   "gradio" - legacy gradio_client path
        self._external_api = str(tts_config.get("api", "openai")).lower()
        self._speech_url = self._build_speech_url(self._url)
        self._http_timeout = float(tts_config.get("http_timeout", 900))
        self._http_session = None

        # Language setting (passed to Qwen3-TTS)
        self._language = tts_config.get("language", "English")

        # Cross-segment consistency: reuse one seed per speaker so repeated takes
        # of the same voice do not drift, and smooth the splice points.
        self._deterministic_seed = tts_config.get("deterministic_seed", True) is not False
        self._join_config = resolve_join_config(tts_config.get("join"))

        # How much of the per-line emotion direction is forwarded to the engine.
        # A long, differently-worded instruct per line is the strongest driver of
        # timbre drift, so shorter/constant styles trade expressiveness for a
        # voice that stays recognizably the same person.
        #   "full"         - per-line instruct as written
        #   "first_clause" - only the first clause of it (recommended)
        #   "voice_style"  - ignore it, use character_style only
        self._instruct_style = str(tts_config.get("instruct_style", "full")).strip().lower()
        if self._instruct_style not in ("full", "first_clause", "voice_style"):
            self._instruct_style = "full"

        # Sub-batching config
        self._sub_batch_enabled = tts_config.get("sub_batch_enabled", True)
        self._sub_batch_min_size = max(1, tts_config.get("sub_batch_min_size", 4))
        self._sub_batch_ratio = max(1.0, float(tts_config.get("sub_batch_ratio", 5)))
        self._sub_batch_max_items = int(tts_config.get("sub_batch_max_items", 0))  # 0 = auto

        # Lazy-loaded backends (guarded by _model_lock to prevent concurrent loads)
        self._model_lock = threading.Lock()
        self._local_custom_model = None
        self._local_clone_model = None
        self._local_design_model = None
        self._local_lora_model = None
        self._warmup_needed = True  # cleared after first batch warmup
        self._lora_adapter_path = None  # track which adapter is currently loaded
        self._gradio_client = None

        # Clone prompt cache: speaker_name -> (ref_audio_path, reusable voice_clone_prompt)
        self._clone_prompt_cache = {}
        # LoRA clone prompt cache: adapter_path -> reusable voice_clone_prompt
        self._lora_prompt_cache = {}

    @property
    def mode(self):
        return self._mode

    @property
    def join_config(self):
        """Join-smoothing settings (trim / level match / fade) for this engine."""
        return dict(self._join_config)

    @staticmethod
    def _explicit_seed(voice_data):
        """Return the seed the user pinned for this voice, or None.

        ``-1``, an empty string or a missing key all mean "not pinned" and fall
        through to the session batch seed / the stable per-speaker seed.
        """
        voice_data = voice_data or {}
        try:
            seed = int(str(voice_data.get("seed", -1)).strip() or -1)
        except (TypeError, ValueError):
            return None
        return seed if seed >= 0 else None

    def _resolve_seed(self, voice_data, speaker, batch_seed=None):
        """Pick the random seed for one generation request.

        Precedence: explicit per-voice seed > session batch seed > a stable
        per-speaker seed (default) > -1 (fully random).

        Without this, every chunk of the same speaker was sampled from a fresh
        random seed, which is a major cause of one voice sounding like a
        different person from one chunk to the next.
        """
        explicit = self._explicit_seed(voice_data)
        if explicit is not None:
            return explicit
        if batch_seed is not None:
            try:
                session_seed = int(batch_seed)
            except (TypeError, ValueError):
                session_seed = -1
            if session_seed >= 0:
                return session_seed
        if self._deterministic_seed:
            return stable_seed_for_speaker(speaker)
        return -1

    @staticmethod
    def _style_with_anchor(instruct_text, voice_data):
        """Append the speaker's character style to a per-line instruct.

        ``character_style`` is the constant part of the voice identity (timbre,
        age, accent). Repeating it in *every* request anchors the identity while
        the per-line instruct only carries the emotion — this is what the local
        batch path already did; the external path silently dropped it.
        """
        instruct = (instruct_text or "").strip()
        parts = [instruct, (
            (voice_data.get("character_style") or voice_data.get("default_style") or "").strip()
        )]
        out = ""
        for part in parts:
            if part and part not in out:
                out = f"{out} {part}".strip()
        return out

    def _build_instruct(self, instruct_text, voice_data, fallback=""):
        """Build the instruct string sent for one request.

        Applies the configured ``tts.instruct_style`` policy on top of the
        per-line direction, then anchors the speaker's constant character style.
        """
        line = (instruct_text or "").strip()
        if self._instruct_style == "voice_style":
            line = ""
        elif self._instruct_style == "first_clause" and line:
            # "Respectful but urgent, professional yet anxious." -> "Respectful but urgent"
            line = re.split(r"[,，;；]", line, maxsplit=1)[0].strip()
        instruct = self._style_with_anchor(line, voice_data)
        return instruct or (fallback or "").strip()

    @property
    def downloads_enabled(self):
        """Whether this engine is allowed to download model weights.

        Only local mode needs the Qwen3-TTS checkpoints (~3.5 GB each), so only
        local mode may pull them. In external mode the engine talks to a remote
        Gradio server instead and downloads are refused outright; an
        already-cached snapshot is still reused, it is just never fetched.
        """
        return self._mode == "local"

    def _download_blocked_error(self, model_id, reason):
        """Build the error raised whenever a download is attempted in external mode."""
        return RuntimeError(
            f"Refusing to download '{model_id}' ({reason}): TTS mode is "
            f"'{self._mode}', and models may only be downloaded in 'local' mode. "
            f"Switch TTS mode to 'local' in the Setup tab, or pre-populate the "
            f"HuggingFace cache (HF_HOME / ~/.cache/huggingface)."
        )

    @staticmethod
    def _concat_audio(wav):
        """Concatenate audio array(s) into a single numpy array."""
        if isinstance(wav, list):
            return np.concatenate(wav) if len(wav) > 1 else wav[0]
        return wav

    @staticmethod
    def _clear_gpu_cache():
        """Free GPU memory: garbage-collect Python objects, then clear CUDA cache.

        Tolerates a missing torch so the external-mode batch paths can call this
        unconditionally without requiring a local GPU stack.
        """
        import gc
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        torch.cuda.empty_cache()

    @staticmethod
    def _reset_compile_cache():
        """Reset torch.compile dynamo state to prevent guard accumulation.

        torch.compile(dynamic=True) accumulates shape guards across calls.
        With varying batch sizes and sequence lengths, the guard list grows
        and CPU-side guard evaluation becomes a bottleneck, causing
        progressive throughput degradation.  Resetting clears all in-memory
        guards; the next call pays a one-time recompilation cost (fast due
        to inductor disk cache) but prevents the slowdown from compounding.

        Only applied on ROCm (AMD GPUs). On NVIDIA, max-autotune mode
        re-benchmarks all kernel variants after each reset, and the
        benchmarking cost scales with tensor size — causing worse slowdown
        than the guard accumulation it prevents.
        """
        import torch
        if not (hasattr(torch.version, "hip") and torch.version.hip):
            return  # skip on NVIDIA/CPU — recompilation cost outweighs benefit
        torch._dynamo.reset()

    def _estimate_max_batch_size(self, model, clone_prompt_tokens=0,
                                ref_text_chars=0, max_text_chars=0,
                                max_new_tokens=2048):
        """Estimate how many sequences fit in free VRAM based on KV cache math.

        Uses the talker's architecture (num_layers, num_kv_heads, head_dim) to
        calculate KV cache bytes per token, then estimates total tokens per
        sequence from clone prompt size + text length + max generation length.

        Returns max batch size (>= 1).  Falls back to a large default on CPU
        or if the model config is inaccessible.
        """
        import torch
        if not torch.cuda.is_available():
            return 9999

        try:
            config = model.model.talker.config
            num_layers = config.num_hidden_layers
            num_kv_heads = config.num_key_value_heads
            head_dim = config.hidden_size // config.num_attention_heads
        except AttributeError:
            return 9999  # can't read config, skip estimation

        dtype_bytes = 2  # bf16
        kv_per_token = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes

        # Total tokens per sequence (worst case: padded to longest + full generation)
        overhead = 10  # role tokens + prefix + special tokens
        ref_text_tokens = ref_text_chars // 3 if ref_text_chars else 0
        text_tokens = max_text_chars // 3 if max_text_chars else 0
        total_tokens = overhead + clone_prompt_tokens + ref_text_tokens + text_tokens + max_new_tokens

        # Overhead factor covers prefill activations, codec, allocator fragmentation
        OVERHEAD_FACTOR = 2.0
        mem_per_seq = total_tokens * kv_per_token * OVERHEAD_FACTOR

        # Available = driver-level free + PyTorch reserved-but-unallocated
        free_driver, _ = torch.cuda.mem_get_info()
        reserved_unused = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        free_total = free_driver + reserved_unused

        budget = int(free_total * 0.8)
        max_batch = max(1, budget // mem_per_seq)

        print(f"VRAM estimate: {free_total / 1e9:.1f}GB free, "
              f"{total_tokens} tok/seq ({clone_prompt_tokens} prompt + "
              f"{ref_text_tokens + text_tokens} text + {max_new_tokens} gen), "
              f"{mem_per_seq / 1e6:.0f}MB/seq -> max_batch={max_batch}")

        return max_batch

    def _build_sub_batches(self, texts, max_items=None):
        """Split sorted-by-length texts into sub-batches.

        Splits on three criteria (checked in order):
        1. VRAM item limit: when max_items is set (from _estimate_max_batch_size)
        2. Length ratio: when longest/shortest > sub_batch_ratio
        3. Minimum size: ratio splits only happen after sub_batch_min_size items

        Returns list of (start, end) index tuples.
        """
        if not self._sub_batch_enabled or len(texts) <= 1:
            return [(0, len(texts))]

        # Manual cap overrides VRAM estimate when set (take the stricter of the two)
        if self._sub_batch_max_items > 0:
            max_items = min(max_items, self._sub_batch_max_items) if max_items else self._sub_batch_max_items

        sub_batches = []
        batch_start = 0

        for i in range(1, len(texts)):
            shortest = max(len(texts[batch_start]), 1)
            should_split = False

            # VRAM-estimated item limit (highest priority — based on actual
            # free GPU memory and per-sequence KV cache cost)
            if max_items is not None and (i - batch_start) >= max_items:
                should_split = True
            # Ratio split: large length disparity wastes padding —
            # only split after min_size items to preserve parallelism
            elif (i - batch_start) >= self._sub_batch_min_size:
                if len(texts[i]) > self._sub_batch_ratio * shortest:
                    should_split = True

            if should_split:
                sub_batches.append((batch_start, i))
                batch_start = i

        sub_batches.append((batch_start, len(texts)))
        return sub_batches

    # ── Lazy initialization ──────────────────────────────────────

    def _warmup_model(self, model):
        """Run a short warmup generation to pre-tune MIOpen/GPU solvers.

        First generation after model load is ~2x slower due to MIOpen autotuning.
        This warmup pays that cost upfront so real generations run at full speed.
        """
        import time
        t0 = time.time()
        try:
            model.generate_custom_voice(
                text="The ancient library stood at the crossroads of two forgotten paths, its weathered stone walls covered in ivy that had been growing for centuries.",
                language=self._language,
                speaker="serena",
                instruct="neutral",
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            print(f"Warmup done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"Warmup failed (non-fatal): {e}")

    def _resolve_device(self):
        """Resolve 'auto' device to the best available."""
        if self._device != "auto":
            return self._device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


    def _enable_rocm_optimizations(self):
        """Apply ROCm-specific optimizations. No-op on NVIDIA/CPU.

        1. FLASH_ATTENTION_TRITON_AMD_ENABLE: Lets qwen_tts whisper encoder
           use native flash attention via Triton AMD backend.
        2. MIOPEN_FIND_MODE=2: Forces MIOpen to use fast-find instead of
           exhaustive search, avoiding workspace allocation failures that
           cause fallback to slow GEMM algorithms.
        3. MIOPEN_LOG_LEVEL=4: Suppress noisy MIOpen workspace warnings.
        4. triton_key shim: Bridges pytorch-triton-rocm's get_cache_key()
           to the triton_key() that PyTorch's inductor expects.
        """
        try:
            import torch
            if not (hasattr(torch.version, "hip") and torch.version.hip):
                return  # not ROCm
        except ImportError:
            return

        # MIOpen: use fast-find to avoid workspace allocation failures
        os.environ.setdefault("MIOPEN_FIND_MODE", "2")
        # Suppress MIOpen workspace warnings
        os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")

        # Flash attention via Triton AMD backend
        os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

        # Fix triton_key compatibility for torch.compile on ROCm
        try:
            from triton.compiler import compiler as triton_compiler
            if not hasattr(triton_compiler, "triton_key"):
                import triton
                triton_compiler.triton_key = lambda: f"pytorch-triton-rocm-{triton.__version__}"
        except ImportError:
            pass

        # Correct under-reported GPU properties on consumer RDNA2/3.
        # ROCm reports half the CU count and warp size 32 instead of 64,
        # causing PyTorch to under-schedule work on RX 6000/7000 GPUs.
        self._patch_rdna_device_properties(torch)


    @staticmethod
    def _patch_rdna_device_properties(torch):
        """Monkey-patch torch.cuda.get_device_properties to report correct
        CU count and wavefront size for consumer RDNA2/3 GPUs.

        ROCm exposes these GPUs with half CU count and warp_size=32
        (matching the CDNA/MI convention). The actual hardware has the
        full CU count and native wavefront64. Under-reporting causes
        PyTorch to generate smaller kernel launches.

        Based on AMD-GPU-BOOST (github.com/Painter3000/AMD-GPU-BOOST).
        """
        if hasattr(torch.cuda, '_rdna_props_patched'):
            return

        # Known RDNA GPU corrections: {name_substring: (true_CUs, true_warp)}
        _rdna_corrections = {
            "7900 XTX": (96, 64),
            "7900 XT":  (84, 64),
            "7900 GRE": (80, 64),
            "7800 XT":  (60, 64),
            "7700 XT":  (54, 64),
            "7600":     (32, 64),
            "6950 XT":  (80, 64),
            "6900 XT":  (80, 64),
            "6800 XT":  (72, 64),
            "6800":     (60, 64),
            "6750 XT":  (40, 64),
            "6700 XT":  (40, 64),
            "6700":     (36, 64),
            "6650 XT":  (32, 64),
            "6600 XT":  (32, 64),
            "6600":     (28, 64),
        }

        original_fn = torch.cuda.get_device_properties
        _cache = {}

        def _patched_get_device_properties(device=None):
            if device is None:
                device = torch.cuda.current_device()
            key = int(device) if not isinstance(device, int) else device

            if key in _cache:
                return _cache[key]

            props = original_fn(device)

            # Find matching correction
            correction = None
            for substr, vals in _rdna_corrections.items():
                if substr in props.name:
                    correction = vals
                    break

            if correction:
                from types import SimpleNamespace
                true_cus, true_warp = correction
                patched = SimpleNamespace()
                for attr in dir(props):
                    if not attr.startswith('_'):
                        try:
                            setattr(patched, attr, getattr(props, attr))
                        except (AttributeError, RuntimeError):
                            pass
                patched.multi_processor_count = true_cus
                patched.warp_size = true_warp
                old_threads = props.multi_processor_count * props.warp_size
                new_threads = true_cus * true_warp
                print(f"  [RDNA fix] {props.name}: CUs {props.multi_processor_count}->{true_cus}, "
                      f"warp {props.warp_size}->{true_warp}, "
                      f"threads {old_threads}->{new_threads}")
                _cache[key] = patched
                return patched

            _cache[key] = props
            return props

        torch.cuda.get_device_properties = _patched_get_device_properties
        torch.cuda._rdna_props_patched = True

    def _compile_codec(self, model):
        """Apply torch.compile to the audio codec for faster decoding.

        The codec decoder has 136 attention modules and many small ops that
        benefit enormously from compilation.  Profiling shows the codec is
        47% of single-gen time and 85% of batch time uncompiled.  With
        torch.compile (dynamic=True, max-autotune), batch throughput
        improves from ~1.3x to ~4.3x real-time and single generation
        drops from ~14s to ~9s.

        max-autotune mode benchmarks GPU kernels to pick the fastest and
        handles varying batch sizes gracefully (unlike reduce-overhead
        which uses CUDA graphs that break on shape changes).
        """
        import torch
        try:
            codec = model.model.speech_tokenizer.model
            model.model.speech_tokenizer.model = torch.compile(
                codec, mode="max-autotune", dynamic=True,
            )
            print("Codec compiled with torch.compile (dynamic=True).")
        except Exception as e:
            print(f"Codec compilation skipped (non-fatal): {e}")

    @staticmethod
    def _resolve_local_model_path(model_id):
        """Check if a HuggingFace model is cached locally and return its snapshot path.

        Uses try_to_load_from_cache to find the local snapshot directory.
        Returns the local path string if cached, or None if not cached.
        """
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_id, "config.json")
        if isinstance(result, str):
            # result is the full path to config.json inside the snapshot dir
            return os.path.dirname(result)
        return None

    def _load_model(self, model_cls, model_id, load_kwargs):
        """Load a model, preferring local cache to avoid network issues.

        Checks if the model snapshot exists in the HF cache and loads from
        the local directory path directly, bypassing all HF Hub network calls.
        Falls back to normal download on first install when cache is empty.
        If loading from local cache fails (e.g. incomplete snapshot), retries
        with the model ID so HF Hub can download any missing files.

        When downloads are disabled (external TTS mode), every path that would
        reach the HuggingFace Hub raises instead. A complete cached snapshot is
        still reused — it just is never fetched or topped up.
        """
        local_path = self._resolve_local_model_path(model_id)
        if local_path:
            print(f"  Loading from local cache: {local_path}")
            try:
                return model_cls.from_pretrained(local_path, **load_kwargs)
            except Exception as e:
                import traceback
                print(f"  Warning: Failed to load from local cache: {e}")
                traceback.print_exc()
                if not self.downloads_enabled:
                    # The retry below passes the repo id, which lets HF Hub pull
                    # down whatever is missing from the snapshot.
                    raise self._download_blocked_error(
                        model_id, "incomplete local snapshot"
                    ) from e
                print(f"  Retrying with model ID (may download missing files)...")
                return model_cls.from_pretrained(model_id, **load_kwargs)

        if not self.downloads_enabled:
            raise self._download_blocked_error(model_id, "not cached locally")

        print(f"  Model not cached locally, downloading {model_id}...")
        return model_cls.from_pretrained(model_id, **load_kwargs)

    def _init_local_custom(self):
        """Load Qwen3-TTS CustomVoice model on demand."""
        if self._local_custom_model is not None:
            return self._local_custom_model

        with self._model_lock:
            if self._local_custom_model is not None:
                return self._local_custom_model

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS CustomVoice model on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_custom_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_custom_model)
            print("CustomVoice model loaded.")
            return self._local_custom_model

    def _init_local_clone(self):
        """Load Qwen3-TTS Base model (for voice cloning) on demand."""
        if self._local_clone_model is not None:
            return self._local_clone_model

        with self._model_lock:
            if self._local_clone_model is not None:
                return self._local_clone_model

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS Base model (voice cloning) on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_clone_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_clone_model)
            print("Base model (voice cloning) loaded.")
            return self._local_clone_model

    def _init_local_design(self):
        """Load Qwen3-TTS VoiceDesign model on demand."""
        if self._local_design_model is not None:
            return self._local_design_model

        with self._model_lock:
            if self._local_design_model is not None:
                return self._local_design_model

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS VoiceDesign model on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_design_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_design_model)
            print("VoiceDesign model loaded.")
            return self._local_design_model

    def _init_local_lora(self, adapter_path):
        """Load Qwen3-TTS Base model with a LoRA adapter on demand.

        Caches the model; if a different adapter is requested the old one
        is unloaded first to free VRAM.
        """
        if self._local_lora_model is not None and self._lora_adapter_path == adapter_path:
            return self._local_lora_model

        with self._model_lock:
            if self._local_lora_model is not None and self._lora_adapter_path == adapter_path:
                return self._local_lora_model

            # Unload previous adapter if switching
            if self._local_lora_model is not None:
                print(f"Unloading previous LoRA adapter ({self._lora_adapter_path})...")
                del self._local_lora_model
                self._local_lora_model = None
                self._lora_adapter_path = None
                self._lora_prompt_cache.clear()
                self._clear_gpu_cache()

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel
            from peft import PeftModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS Base model + LoRA adapter on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device

            model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", load_kwargs,
            )

            # Wrap the talker with the LoRA adapter
            model.model.talker = PeftModel.from_pretrained(
                model.model.talker,
                adapter_path,
            )
            model.model.talker.eval()

            if self._compile_codec_enabled:
                self._compile_codec(model)

            self._local_lora_model = model
            self._lora_adapter_path = adapter_path
            print(f"LoRA adapter loaded from {adapter_path}")
            return model

    def unload_models(self):
        """Free all cached local TTS models and clear GPU cache.

        Called after a full conversion finishes (or via /api/unload) so
        Qwen3-TTS base/clone/design/LoRA weights don't sit in VRAM
        indefinitely. Next TTS call re-loads on demand.
        """
        with self._model_lock:
            unloaded = []
            for attr in ("_local_custom_model", "_local_clone_model",
                         "_local_design_model", "_local_lora_model"):
                if getattr(self, attr) is not None:
                    setattr(self, attr, None)
                    unloaded.append(attr)
            self._lora_adapter_path = None
            self._lora_prompt_cache.clear()
            self._clone_prompt_cache.clear()
            self._warmup_needed = True
            if unloaded:
                print(f"Unloaded TTS models: {', '.join(unloaded)}")
                self._clear_gpu_cache()
            return unloaded

    @staticmethod
    def _build_speech_url(url):
        """Resolve the OpenAI-compatible speech endpoint from a base URL.

        Accepts http://host:port, http://host:port/v1, or
        http://host:port/v1/audio/speech.
        """
        base = (url or "").strip().rstrip("/")
        if base.endswith("/audio/speech"):
            return base
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/audio/speech"

    @staticmethod
    def _audio_data_url(path):
        """Encode a local audio file as a base64 data URL.

        The reference clip lives on the client machine, so it has to travel
        inside the request body - a bare filesystem path is only meaningful when
        the TTS server runs on the same host.
        """
        import base64
        import mimetypes

        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        with open(path, "rb") as f:
            payload = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    def _init_external(self):
        """Create the external backend on demand.

        Returns an HTTP session for the OpenAI-compatible protocol, or a
        gradio_client.Client for the legacy protocol.
        """
        if self._external_api == "gradio":
            if self._gradio_client is not None:
                return self._gradio_client

            from gradio_client import Client

            print(f"Connecting to Gradio TTS server at {self._url}...")
            self._gradio_client = Client(self._url)
            print("Connected to external TTS server.")
            return self._gradio_client

        if self._http_session is None:
            import requests

            print(f"Using OpenAI-compatible TTS server at {self._speech_url}")
            self._http_session = requests.Session()

        return self._http_session

    def _openai_speech(self, payload, output_path):
        """POST an OpenAI-compatible speech request and write the audio response."""
        client = self._init_external()
        resp = client.post(self._speech_url, json=payload, timeout=self._http_timeout)
        if resp.status_code != 200:
            detail = resp.text[:500].replace("\n", " ")
            raise RuntimeError(f"TTS server returned HTTP {resp.status_code}: {detail}")
        if not resp.content:
            raise RuntimeError("TTS server returned an empty audio response")
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True

    # ── Clone prompt cache (local mode) ──────────────────────────

    def _get_clone_prompt(self, speaker, voice_config):
        """Get or create a cached voice clone prompt for a speaker."""
        voice_data = voice_config.get(speaker, {})
        ref_audio_path = voice_data.get("ref_audio")
        ref_text = voice_data.get("ref_text")

        if not ref_audio_path or not ref_text:
            raise ValueError(f"Clone voice for '{speaker}' missing ref_audio or ref_text")
        # Resolve relative paths against project root (parent of app/)
        if not os.path.isabs(ref_audio_path):
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ref_audio_path = os.path.join(root_dir, ref_audio_path)
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found for '{speaker}': {ref_audio_path}")

        # Check cache — invalidate if ref_audio changed
        if speaker in self._clone_prompt_cache:
            cached_path, cached_prompt = self._clone_prompt_cache[speaker]
            if cached_path == ref_audio_path:
                return cached_prompt
            print(f"Voice changed for '{speaker}', rebuilding clone prompt...")

        model = self._init_local_clone()

        # Load reference audio as numpy array
        audio_array, sample_rate = sf.read(ref_audio_path)
        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        print(f"Creating clone prompt for '{speaker}'...")
        prompt = model.create_voice_clone_prompt(
            ref_audio=(audio_array, sample_rate),
            ref_text=ref_text,
        )
        self._clone_prompt_cache[speaker] = (ref_audio_path, prompt)
        print(f"Clone prompt cached for '{speaker}'.")
        return prompt

    # ── Core generation methods ──────────────────────────────────

    def generate_custom_voice(self, text, instruct_text, speaker, voice_config, output_path,
                              batch_seed=None):
        """Generate audio using CustomVoice model. Returns True on success."""
        if self._mode == "local":
            return self._local_generate_custom(text, instruct_text, speaker, voice_config,
                                               output_path, batch_seed=batch_seed)
        else:
            return self._external_generate_custom(text, instruct_text, speaker, voice_config,
                                                  output_path, batch_seed=batch_seed)

    def generate_clone_voice(self, text, speaker, voice_config, output_path, batch_seed=None):
        """Generate audio using voice cloning. Returns True on success."""
        if self._mode == "local":
            return self._local_generate_clone(text, speaker, voice_config, output_path,
                                              batch_seed=batch_seed)
        else:
            return self._external_generate_clone(text, speaker, voice_config, output_path,
                                                 batch_seed=batch_seed)

    def generate_voice(self, text, instruct_text, speaker, voice_config, output_path,
                       batch_seed=None):
        """Generate audio using the appropriate method based on voice type config."""
        voice_data = voice_config.get(speaker)
        if not voice_data:
            print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
            return False

        voice_type = voice_data.get("type", "custom")

        if voice_type == "clone":
            return self.generate_clone_voice(text, speaker, voice_config, output_path,
                                             batch_seed=batch_seed)
        elif voice_type in ("lora", "builtin_lora"):
            return self.generate_lora_voice(text, instruct_text, voice_data, output_path,
                                            batch_seed=batch_seed)
        elif voice_type == "design":
            return self.generate_design_voice(text, instruct_text, voice_data, output_path)
        else:
            return self.generate_custom_voice(text, instruct_text, speaker, voice_config, output_path,
                                              batch_seed=batch_seed)

    # ── Voice design generation ──────────────────────────────────

    def generate_voice_design(self, description, sample_text, language=None, seed=-1):
        """Generate a voice from a text description using the VoiceDesign model.

        Args:
            description: Natural language description of the desired voice
            sample_text: Text to synthesize with the designed voice
            language: Language code (defaults to engine's configured language)
            seed: Random seed (-1 for random, >= 0 for reproducible)

        Returns:
            (wav_path, sample_rate) on success

        Raises:
            RuntimeError: If generation fails
        """
        import time
        import tempfile

        lang = language or self._language
        print(f"VoiceDesign: generating preview for description='{description[:80]}...'"
              f"{f', seed={seed}' if seed >= 0 else ''}")

        if self._mode != "local":
            return self._external_voice_design(description, sample_text, lang, seed)

        import torch

        model = self._init_local_design()

        if seed >= 0:
            torch.manual_seed(seed)

        t_start = time.time()
        wavs, sr = model.generate_voice_design(
            text=sample_text,
            instruct=description,
            language=lang,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        gen_time = time.time() - t_start

        if wavs is None or len(wavs) == 0:
            raise RuntimeError("VoiceDesign model returned no audio")

        audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
        duration = len(audio) / sr
        print(f"VoiceDesign: done in {gen_time:.1f}s -> {duration:.1f}s audio")

        # Save to previews directory
        previews_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "designed_voices", "previews")
        os.makedirs(previews_dir, exist_ok=True)

        filename = f"preview_{int(time.time() * 1000)}.wav"
        wav_path = os.path.join(previews_dir, filename)
        self._save_wav(audio, sr, wav_path)

        return wav_path, sr

    def generate_design_voice(self, text, instruct_text, voice_data, output_path):
        """Generate audio using VoiceDesign model with combined description + instruct.

        The voice_data 'description' field provides the base voice identity,
        and the per-line instruct_text is appended for delivery/emotion direction.
        """
        import shutil

        base_desc = (voice_data.get("description") or "").strip()
        instruct = (instruct_text or "").strip()

        if base_desc and instruct:
            description = f"{base_desc}, {instruct}"
        elif base_desc:
            description = base_desc
        elif instruct:
            description = instruct
        else:
            print("Warning: Design voice has no description or instruct. Using generic.")
            description = "A clear, natural speaking voice"

        # A seed pinned on the voice card applies here too; -1 keeps the previous
        # "let the engine pick" behaviour (0 is a valid seed, so test for None).
        pinned = self._explicit_seed(voice_data)
        wav_path, sr = self.generate_voice_design(
            description=description,
            sample_text=text,
            seed=pinned if pinned is not None else -1,
        )
        shutil.copy2(wav_path, output_path)
        return True

    # ── LoRA voice generation ────────────────────────────────────

    def generate_lora_voice(self, text, instruct_text, voice_data, output_path, batch_seed=None):
        """Generate audio using a LoRA-finetuned Base model.

        The adapter directory must contain:
          - PEFT adapter weights (adapter_model.safetensors / adapter_config.json)
          - ref_sample.wav (reference audio for voice cloning prompt)
          - training_meta.json (with ref_sample_text)

        The LoRA weights refine voice identity beyond what the reference alone provides.
        """
        if self._mode != "local":
            return self._external_generate_adapter_voice(text, voice_data, output_path,
                                                         batch_seed=batch_seed)

        try:
            import torch
            import time

            adapter_path = voice_data.get("adapter_path")
            if not adapter_path:
                print(f"Error: No adapter_path in voice_data")
                return False

            # Resolve relative paths against project root
            if not os.path.isabs(adapter_path):
                root_dir = os.path.dirname(os.path.dirname(__file__))
                adapter_path = os.path.join(root_dir, adapter_path)

            if not os.path.isdir(adapter_path):
                # Auto-download built-in adapters from HF
                adapter_id = os.path.basename(adapter_path)
                if adapter_id.startswith("builtin_"):
                    if not self.downloads_enabled:
                        print(f"Error: '{adapter_id}' is not downloaded, and downloading "
                              f"is disabled because TTS mode is '{self._mode}'. Download it "
                              f"from the Training tab while in 'local' mode first.")
                        return False
                    print(f"Adapter {adapter_id} not downloaded, attempting auto-download...")
                    try:
                        from hf_utils import download_builtin_adapter
                        builtin_dir = os.path.dirname(adapter_path)
                        download_builtin_adapter(adapter_id, builtin_dir,
                                                 allow_download=self.downloads_enabled)
                    except Exception as e:
                        print(f"Error: Auto-download failed for {adapter_id}: {e}")
                        return False
                else:
                    print(f"Error: LoRA adapter path not found: {adapter_path}")
                    return False

            # Load reference audio and text from adapter directory
            ref_wav_path = os.path.join(adapter_path, "ref_sample.wav")
            meta_path = os.path.join(adapter_path, "training_meta.json")

            if not os.path.exists(ref_wav_path):
                print(f"Error: ref_sample.wav not found in {adapter_path}")
                return False
            if not os.path.exists(meta_path):
                print(f"Error: training_meta.json not found in {adapter_path}")
                return False

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ref_text = meta.get("ref_sample_text", "")
            if not ref_text:
                print(f"Error: ref_sample_text missing from training_meta.json")
                return False

            print(f"TTS [local lora] generating for adapter={os.path.basename(adapter_path)}, "
                  f"text='{text[:50]}...'")

            model = self._init_local_lora(adapter_path)

            # Reuse one seed per adapter so repeated takes keep the same identity
            seed = self._resolve_seed(voice_data, os.path.basename(adapter_path), batch_seed)
            if seed >= 0:
                torch.manual_seed(seed)

            # Build or reuse voice clone prompt for this adapter
            if adapter_path not in self._lora_prompt_cache:
                audio_array, sample_rate = sf.read(ref_wav_path)
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                print(f"Creating clone prompt for LoRA adapter...")
                prompt = model.create_voice_clone_prompt(
                    ref_audio=(audio_array, sample_rate),
                    ref_text=ref_text,
                    x_vector_only_mode=True,
                )
                self._lora_prompt_cache[adapter_path] = prompt
                print(f"Clone prompt cached for LoRA adapter.")

            prompt = self._lora_prompt_cache[adapter_path]

            # Build instruct_ids so the Base model can follow style prompts
            gen_extra = {}
            instruct = self._build_instruct(instruct_text, voice_data)
            if instruct:
                instruct_formatted = f"<|im_start|>user\n{instruct}<|im_end|>\n"
                gen_extra["instruct_ids"] = model._tokenize_texts([instruct_formatted])

            t_start = time.time()
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                max_new_tokens=2048,
                **gen_extra,
            )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local lora] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating LoRA voice: {e}")
            traceback.print_exc()
            return False

    # ── Batch generation ─────────────────────────────────────────

    def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Generate multiple audio files.

        Local mode: uses native list-based batch API for custom voices.
        External mode: sequential individual calls.

        Args:
            chunks: List of dicts with 'text', 'instruct', 'speaker', 'index' keys
            voice_config: Voice configuration dict
            output_dir: Directory to save output files
            batch_seed: Single seed for all generations (-1 for random)

        Returns:
            dict with 'completed' (list of indices) and 'failed' (list of (index, error) tuples)
        """
        results = {"completed": [], "failed": []}

        if not chunks:
            return results

        # Reset torch.compile state to prevent progressive slowdown
        # from dynamo guard accumulation across batches
        if self._compile_codec_enabled:
            self._reset_compile_cache()

        # Separate chunks by voice type
        custom_chunks = []
        clone_chunks = []
        lora_chunks = []
        design_chunks = []

        for chunk in chunks:
            speaker = chunk.get("speaker")
            voice_data = voice_config.get(speaker, {})
            voice_type = voice_data.get("type", "custom")

            if voice_type == "clone":
                clone_chunks.append(chunk)
            elif voice_type in ("lora", "builtin_lora"):
                lora_chunks.append(chunk)
            elif voice_type == "design":
                design_chunks.append(chunk)
            else:
                custom_chunks.append(chunk)

        # Process custom voice chunks
        if custom_chunks:
            if self._mode == "local":
                batch_results = self._local_batch_custom(custom_chunks, voice_config, output_dir, batch_seed)
            else:
                batch_results = self._sequential_custom(custom_chunks, voice_config, output_dir, batch_seed)
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process clone voice chunks (batched by speaker in local mode)
        if clone_chunks:
            if self._mode == "local":
                batch_results = self._local_batch_clone(clone_chunks, voice_config, output_dir, batch_seed)
            else:
                batch_results = {"completed": [], "failed": []}
                for chunk in clone_chunks:
                    idx = chunk["index"]
                    output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                    try:
                        success = self.generate_clone_voice(
                            chunk["text"], chunk["speaker"], voice_config, output_path,
                            batch_seed=batch_seed,
                        )
                        if success:
                            batch_results["completed"].append(idx)
                        else:
                            batch_results["failed"].append((idx, "Clone voice generation failed"))
                    except Exception as e:
                        batch_results["failed"].append((idx, str(e)))
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process LoRA voice chunks (batched by adapter in local mode)
        if lora_chunks:
            if self._mode == "local":
                batch_results = self._local_batch_lora(lora_chunks, voice_config, output_dir, batch_seed)
            else:
                batch_results = {"completed": [], "failed": []}
                for chunk in lora_chunks:
                    idx = chunk["index"]
                    output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                    speaker = chunk.get("speaker")
                    voice_data = voice_config.get(speaker, {})
                    try:
                        success = self.generate_lora_voice(
                            text=chunk["text"],
                            instruct_text=chunk.get("instruct", ""),
                            voice_data=voice_data,
                            output_path=output_path,
                            batch_seed=batch_seed,
                        )
                        if success:
                            batch_results["completed"].append(idx)
                        else:
                            batch_results["failed"].append((idx, "LoRA voice generation failed"))
                    except Exception as e:
                        batch_results["failed"].append((idx, str(e)))
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process design voice chunks (sequential — each line has unique description)
        if design_chunks:
            for chunk in design_chunks:
                idx = chunk["index"]
                output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                speaker = chunk.get("speaker")
                voice_data = voice_config.get(speaker, {})
                try:
                    success = self.generate_design_voice(
                        text=chunk["text"],
                        instruct_text=chunk.get("instruct", ""),
                        voice_data=voice_data,
                        output_path=output_path,
                    )
                    if success:
                        results["completed"].append(idx)
                    else:
                        results["failed"].append((idx, "Design voice generation failed"))
                except Exception as e:
                    results["failed"].append((idx, str(e)))

        return results

    # ── Connection test ──────────────────────────────────────────

    # ── Local backend methods ────────────────────────────────────

    def _local_generate_custom(self, text, instruct_text, speaker, voice_config, output_path,
                               batch_seed=None):
        """Generate custom voice audio using local Qwen3-TTS model."""
        try:
            import torch

            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            voice = voice_data.get("voice", "Ryan")
            default_style = voice_data.get("default_style", "")
            seed = self._resolve_seed(voice_data, speaker, batch_seed)

            instruct = self._build_instruct(instruct_text, voice_data,
                                            fallback=default_style or "neutral")

            import time

            print(f"TTS [local] generating (seed={seed}) with instruct='{instruct}' for text='{text[:50]}...'")

            model = self._init_local_custom()

            if seed >= 0:
                torch.manual_seed(seed)

            t_start = time.time()
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=self._language,
                speaker=voice,
                instruct=instruct,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            # wavs is a list of numpy arrays; concatenate them
            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating custom voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _local_generate_clone(self, text, speaker, voice_config, output_path, batch_seed=None):
        """Generate voice-cloned audio using local Qwen3-TTS Base model."""
        try:
            import torch

            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            seed = self._resolve_seed(voice_data, speaker, batch_seed)

            import time

            print(f"TTS [local clone] generating (seed={seed}) for speaker='{speaker}', text='{text[:50]}...'")

            prompt = self._get_clone_prompt(speaker, voice_config)
            model = self._init_local_clone()

            if seed >= 0:
                torch.manual_seed(seed)

            t_start = time.time()
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local clone] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating clone voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _split_by_pinned_seed(self, chunks, voice_config):
        """Decide how pinned per-voice seeds split a batch run.

        The native batch API accepts one seed per call, so a pinned seed can only
        be honoured when the chunks needing it are generated on their own. Chunks
        whose speaker does not pin a seed stay together in a ``None`` group and
        keep the previous single-batch behaviour.

        Returns ``(groups, single_seed)``:
          - ``groups`` is a list of ``(seed_or_None, chunks)`` when the run has to
            be split, otherwise None;
          - ``single_seed`` is the one seed the whole run shares when no split is
            needed (None when nothing is pinned).
        """
        pinned = {}
        unpinned = []
        for chunk in chunks:
            voice_data = voice_config.get(chunk.get("speaker", ""), {})
            seed = self._explicit_seed(voice_data)
            if seed is None:
                unpinned.append(chunk)
            else:
                pinned.setdefault(seed, []).append(chunk)

        # The unpinned chunks form one implicit group of their own.
        if len(pinned) + (1 if unpinned else 0) <= 1:
            # At most one seed is in play for the whole run, so nothing to split;
            # the caller just has to apply that seed to the single batch.
            return None, (next(iter(pinned)) if pinned else None)

        groups = [(seed, group) for seed, group in pinned.items()]
        if unpinned:
            groups.append((None, unpinned))
        return groups, None

    def _local_batch_custom(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Batch generate custom voice using native list API with sub-batching.

        Autoregressive batch generation runs for as long as the longest sequence.
        Shorter sequences waste compute on padding. To minimize this, chunks are
        sorted by text length and split into sub-batches when the length ratio
        exceeds the configured threshold. Sub-batching can be disabled entirely
        via config, in which case everything runs as one batch.

        The native batch call also takes a single seed for the whole call, so
        chunks whose speaker pins an explicit seed are generated in their own
        group - otherwise a pinned voice would be re-rolled by the batch seed.
        """
        import torch
        import time

        grouped, single_seed = self._split_by_pinned_seed(chunks, voice_config)
        if grouped is not None:
            print(f"Batch [local]: splitting into {len(grouped)} seed group(s) "
                  f"to honour pinned voice seeds")
            # A pinned group re-seeds the global RNG, so snapshot the state here
            # and restore it before every unpinned group: those voices must keep
            # the same entropy they would have had in an unsplit run instead of
            # inheriting whatever the last pinned group left behind.
            rng_state = torch.get_rng_state()
            results = {"completed": [], "failed": []}
            for group_seed, group_chunks in grouped:
                if group_seed is None:
                    torch.set_rng_state(rng_state)
                sub = self._local_batch_custom(
                    group_chunks, voice_config, output_dir,
                    batch_seed=group_seed if group_seed is not None else batch_seed,
                )
                results["completed"].extend(sub["completed"])
                results["failed"].extend(sub["failed"])
            return results
        if single_seed is not None:
            # Every chunk shares one pinned seed; apply it to the whole batch.
            batch_seed = single_seed

        results = {"completed": [], "failed": []}

        texts = []
        speakers = []
        instructs = []
        indices = []

        for chunk in chunks:
            idx = chunk["index"]
            text = chunk.get("text", "")
            instruct_text = chunk.get("instruct", "")
            speaker_name = chunk.get("speaker", "")

            voice_data = voice_config.get(speaker_name, {})
            voice = voice_data.get("voice", "Ryan")

            instruct = self._build_instruct(instruct_text, voice_data, fallback="neutral")

            texts.append(text)
            speakers.append(voice)
            instructs.append(instruct)
            indices.append(idx)

        total_text_chars = sum(len(t) for t in texts)

        # Sort by text length to group similar-length chunks together.
        # This reduces wasted padding during autoregressive generation
        # (the LLM runs until ALL sequences finish, so short chunks
        # waste compute waiting for long ones).
        sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        texts = [texts[i] for i in sort_order]
        speakers = [speakers[i] for i in sort_order]
        instructs = [instructs[i] for i in sort_order]
        indices = [indices[i] for i in sort_order]

        model = self._init_local_custom()

        # Warmup on first batch to pre-tune MIOpen/GPU solvers
        if self._warmup_needed:
            print("Running batch warmup generation...")
            self._warmup_model(model)
            self._warmup_needed = False

        # Clear stale GPU cache from any prior generation to avoid
        # fragmented VRAM blocking large batch allocations (ROCm especially).
        self._clear_gpu_cache()


        max_items = self._estimate_max_batch_size(
            model, max_text_chars=len(texts[-1]),
        )
        sub_batches = self._build_sub_batches(texts, max_items=max_items)

        print(f"Batch [local]: generating {len(texts)} chunks ({total_text_chars} chars) "
              f"in {len(sub_batches)} sub-batch(es)...")

        t_total_start = time.time()
        total_audio_duration = 0.0

        for sb_idx, (start, end) in enumerate(sub_batches):
            sb_texts = texts[start:end]
            sb_speakers = speakers[start:end]
            sb_instructs = instructs[start:end]
            sb_indices = indices[start:end]
            sb_chars = sum(len(t) for t in sb_texts)

            print(f"  Sub-batch {sb_idx+1}/{len(sub_batches)}: {len(sb_texts)} chunks "
                  f"({sb_chars} chars, {len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk)")

            try:
                if batch_seed >= 0:
                    torch.manual_seed(batch_seed)

                t_start = time.time()
                wavs_list, sr = model.generate_custom_voice(
                    text=sb_texts,
                    language=[self._language] * len(sb_texts),
                    speaker=sb_speakers,
                    instruct=sb_instructs,
                    non_streaming_mode=True,
                    max_new_tokens=2048,
                )
                gen_time = time.time() - t_start

                if wavs_list is None:
                    for idx in sb_indices:
                        results["failed"].append((idx, "Batch returned None"))
                    continue

                sb_audio_duration = 0.0
                for i, (wav, idx) in enumerate(zip(wavs_list, sb_indices)):
                    try:
                        output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                        audio = self._concat_audio(wav)
                        self._save_wav(audio, sr, output_path)
                        results["completed"].append(idx)
                        duration = len(audio) / sr
                        sb_audio_duration += duration
                        print(f"    Chunk {idx} saved: {os.path.getsize(output_path)} bytes ({duration:.1f}s audio)")
                    except Exception as e:
                        print(f"    Error saving chunk {idx}: {e}")
                        results["failed"].append((idx, str(e)))

                total_audio_duration += sb_audio_duration
                sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

            except Exception as e:
                print(f"  Sub-batch {sb_idx+1} failed: {e}")
                for idx in sb_indices:
                    results["failed"].append((idx, f"Batch error: {e}"))

            # Free GPU memory between sub-batches to prevent VRAM exhaustion
            self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")



        return results

    def _local_batch_clone(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Batch generate clone voices, grouped by speaker.

        Chunks sharing the same speaker (same reference audio) are batched
        together through generate_voice_clone(text=[list], ...).
        Sub-batching by text length is applied within each speaker group.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}

        # Group chunks by speaker
        speaker_groups = {}
        for chunk in chunks:
            speaker = chunk.get("speaker", "")
            speaker_groups.setdefault(speaker, []).append(chunk)

        model = self._init_local_clone()

        # Warmup on first batch to pre-tune MIOpen/GPU solvers
        # Uses CustomVoice model (not Base) since warmup just needs to
        # exercise MIOpen/GPU solvers and wake the GPU from deep sleep.
        if self._warmup_needed:
            warmup_model = self._init_local_custom()
            print("Running batch warmup generation...")
            self._warmup_model(warmup_model)
            self._warmup_needed = False

        self._clear_gpu_cache()


        t_total_start = time.time()
        total_audio_duration = 0.0

        for speaker, group in speaker_groups.items():
            try:
                prompt = self._get_clone_prompt(speaker, voice_config)
            except Exception as e:
                print(f"  Error building clone prompt for '{speaker}': {e}")
                for chunk in group:
                    results["failed"].append((chunk["index"], str(e)))
                continue

            texts = [c["text"] for c in group]
            indices = [c["index"] for c in group]

            # Sort by text length for sub-batching efficiency
            sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            texts = [texts[i] for i in sort_order]
            indices = [indices[i] for i in sort_order]

            # Estimate max batch size from VRAM + clone prompt overhead
            clone_tokens = prompt[0].ref_code.shape[0] if prompt[0].ref_code is not None else 0
            ref_text_chars = len(prompt[0].ref_text) if prompt[0].ref_text else 0
            max_items = self._estimate_max_batch_size(
                model, clone_tokens, ref_text_chars, len(texts[-1]),
            )
            sub_batches = self._build_sub_batches(texts, max_items=max_items)

            print(f"Batch [clone] speaker='{speaker}': {len(texts)} chunks "
                  f"in {len(sub_batches)} sub-batch(es)")

            # Same precedence as the single-chunk clone path: a seed pinned on
            # the voice card wins, then the session batch seed, then one stable
            # seed per speaker so the identity holds across sub-batches.
            seed = self._resolve_seed(voice_config.get(speaker, {}), speaker, batch_seed)
            if seed >= 0:
                torch.manual_seed(seed)

            for sb_idx, (start, end) in enumerate(sub_batches):
                sb_texts = texts[start:end]
                sb_indices = indices[start:end]

                print(f"  Sub-batch {sb_idx+1}/{len(sub_batches)}: {len(sb_texts)} chunks "
                      f"({len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk)")

                try:
                    t_start = time.time()
                    wavs_list, sr = model.generate_voice_clone(
                        text=sb_texts,
                        voice_clone_prompt=prompt,
                        non_streaming_mode=True,
                        max_new_tokens=2048,
                    )
                    gen_time = time.time() - t_start

                    if wavs_list is None:
                        for idx in sb_indices:
                            results["failed"].append((idx, "Batch returned None"))
                        continue

                    sb_audio_duration = 0.0
                    for wav, idx in zip(wavs_list, sb_indices):
                        try:
                            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                            audio = self._concat_audio(wav)
                            self._save_wav(audio, sr, output_path)
                            results["completed"].append(idx)
                            duration = len(audio) / sr
                            sb_audio_duration += duration
                        except Exception as e:
                            print(f"    Error saving chunk {idx}: {e}")
                            results["failed"].append((idx, str(e)))

                    total_audio_duration += sb_audio_duration
                    sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                    print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

                except Exception as e:
                    print(f"  Sub-batch {sb_idx+1} failed: {e}")
                    for idx in sb_indices:
                        results["failed"].append((idx, f"Batch error: {e}"))

                self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch [clone] total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")



        return results

    def _local_batch_lora(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Batch generate LoRA voices, grouped by adapter.

        Chunks sharing the same adapter are batched together through
        generate_voice_clone(text=[list], instruct_ids=[list], ...).
        Sub-batching by text length is applied within each adapter group.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}
        root_dir = os.path.dirname(os.path.dirname(__file__))

        # Group chunks by adapter_path (resolved to absolute)
        adapter_groups = {}  # adapter_path -> (voice_data, [chunks])
        for chunk in chunks:
            speaker = chunk.get("speaker", "")
            voice_data = voice_config.get(speaker, {})
            adapter_path = voice_data.get("adapter_path", "")

            if not adapter_path:
                results["failed"].append((chunk["index"], "No adapter_path"))
                continue

            if not os.path.isabs(adapter_path):
                adapter_path = os.path.join(root_dir, adapter_path)

            if adapter_path not in adapter_groups:
                adapter_groups[adapter_path] = (voice_data, [])
            adapter_groups[adapter_path][1].append(chunk)

        self._clear_gpu_cache()


        # Warmup on first batch to pre-tune MIOpen/GPU solvers
        # Uses CustomVoice model (not Base) since warmup just needs to
        # exercise MIOpen/GPU solvers and wake the GPU from deep sleep.
        if self._warmup_needed:
            warmup_model = self._init_local_custom()
            print("Running batch warmup generation...")
            self._warmup_model(warmup_model)
            self._warmup_needed = False

        t_total_start = time.time()
        total_audio_duration = 0.0

        for adapter_path, (voice_data, group) in adapter_groups.items():
            if not os.path.isdir(adapter_path):
                print(f"  Error: adapter path not found: {adapter_path}")
                for chunk in group:
                    results["failed"].append((chunk["index"], f"Adapter not found: {adapter_path}"))
                continue

            # Same precedence as the single-chunk LoRA path: a pinned per-voice
            # seed wins, then the session batch seed, then one stable seed per
            # adapter so the identity holds across sub-batches.
            seed = self._resolve_seed(
                voice_data, os.path.basename(adapter_path), batch_seed
            )
            if seed >= 0:
                torch.manual_seed(seed)

            # Load adapter and build/get clone prompt
            try:
                ref_wav_path = os.path.join(adapter_path, "ref_sample.wav")
                meta_path = os.path.join(adapter_path, "training_meta.json")
                if not os.path.exists(ref_wav_path) or not os.path.exists(meta_path):
                    raise FileNotFoundError(f"Missing ref_sample.wav or training_meta.json in {adapter_path}")

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ref_text = meta.get("ref_sample_text", "")
                if not ref_text:
                    raise ValueError("ref_sample_text missing from training_meta.json")

                model = self._init_local_lora(adapter_path)

                if adapter_path not in self._lora_prompt_cache:
                    audio_array, sample_rate = sf.read(ref_wav_path)
                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)
                    print(f"Creating clone prompt for LoRA adapter...")
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=(audio_array, sample_rate),
                        ref_text=ref_text,
                        x_vector_only_mode=True,
                    )
                    self._lora_prompt_cache[adapter_path] = prompt
                    print(f"Clone prompt cached for LoRA adapter.")

                prompt = self._lora_prompt_cache[adapter_path]
            except Exception as e:
                print(f"  Error loading LoRA adapter {os.path.basename(adapter_path)}: {e}")
                for chunk in group:
                    results["failed"].append((chunk["index"], str(e)))
                continue

            texts = [c["text"] for c in group]
            instructs_raw = [c.get("instruct", "") for c in group]
            indices = [c["index"] for c in group]

            # Sort by text length
            sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            texts = [texts[i] for i in sort_order]
            instructs_raw = [instructs_raw[i] for i in sort_order]
            indices = [indices[i] for i in sort_order]

            # Estimate max batch size from VRAM + clone prompt overhead
            clone_tokens = prompt[0].ref_code.shape[0] if prompt[0].ref_code is not None else 0
            ref_text_chars = len(prompt[0].ref_text) if prompt[0].ref_text else 0
            max_items = self._estimate_max_batch_size(
                model, clone_tokens, ref_text_chars, len(texts[-1]),
            )
            sub_batches = self._build_sub_batches(texts, max_items=max_items)

            print(f"Batch [lora] adapter='{os.path.basename(adapter_path)}': {len(texts)} chunks "
                  f"in {len(sub_batches)} sub-batch(es)")

            for sb_idx, (start, end) in enumerate(sub_batches):
                sb_texts = texts[start:end]
                sb_instructs = instructs_raw[start:end]
                sb_indices = indices[start:end]

                print(f"  Sub-batch {sb_idx+1}/{len(sub_batches)}: {len(sb_texts)} chunks "
                      f"({len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk)")

                try:
                    # Build instruct_ids list for this sub-batch
                    instruct_ids = []
                    for inst in sb_instructs:
                        instruct = self._build_instruct(inst, voice_data)
                        if instruct:
                            instruct_formatted = f"<|im_start|>user\n{instruct}<|im_end|>\n"
                            instruct_ids.append(model._tokenize_texts([instruct_formatted])[0])
                        else:
                            instruct_ids.append(None)

                    gen_extra = {}
                    if any(iid is not None for iid in instruct_ids):
                        gen_extra["instruct_ids"] = instruct_ids

                    t_start = time.time()
                    wavs_list, sr = model.generate_voice_clone(
                        text=sb_texts,
                        voice_clone_prompt=prompt,
                        non_streaming_mode=True,
                        max_new_tokens=2048,
                        **gen_extra,
                    )
                    gen_time = time.time() - t_start

                    if wavs_list is None:
                        for idx in sb_indices:
                            results["failed"].append((idx, "Batch returned None"))
                        continue

                    sb_audio_duration = 0.0
                    for wav, idx in zip(wavs_list, sb_indices):
                        try:
                            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                            audio = self._concat_audio(wav)
                            self._save_wav(audio, sr, output_path)
                            results["completed"].append(idx)
                            duration = len(audio) / sr
                            sb_audio_duration += duration
                        except Exception as e:
                            print(f"    Error saving chunk {idx}: {e}")
                            results["failed"].append((idx, str(e)))

                    total_audio_duration += sb_audio_duration
                    sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                    print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

                except Exception as e:
                    print(f"  Sub-batch {sb_idx+1} failed: {e}")
                    for idx in sb_indices:
                        results["failed"].append((idx, f"Batch error: {e}"))

                self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch [lora] total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")



        return results

    # ── External backend methods ─────────────────────────────────

    def _external_voice_design(self, description, sample_text, language, seed=-1):
        """Generate a design-voice preview via an OpenAI-compatible TTS server.

        Requires a VoiceDesign checkpoint on the server side; a CustomVoice or
        Base server will reject the request.
        """
        import time

        print(f"VoiceDesign [external/openai] description='{description[:80]}...'")

        previews_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "designed_voices", "previews")
        os.makedirs(previews_dir, exist_ok=True)
        wav_path = os.path.join(previews_dir, f"preview_{int(time.time() * 1000)}.wav")

        payload = {
            "input": sample_text,
            "instructions": description,
            "language": language,
            "response_format": "wav",
            "task_type": "VoiceDesign",
        }
        if seed >= 0:
            payload["seed"] = seed

        self._openai_speech(payload, wav_path)
        return wav_path, sf.info(wav_path).samplerate

    def _external_generate_custom(self, text, instruct_text, speaker, voice_config, output_path,
                                  batch_seed=None):
        """Generate custom voice audio via the external TTS server."""
        try:
            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            voice = voice_data.get("voice", "Ryan")
            default_style = voice_data.get("default_style", "")
            seed = self._resolve_seed(voice_data, speaker, batch_seed)

            if self._external_api != "gradio":
                # Instruct = per-line emotion + the speaker's constant style anchor.
                instruct = self._build_instruct(instruct_text, voice_data, fallback=default_style)
                print(f"TTS [external/openai] voice='{voice}' seed={seed} instruct='{instruct}' "
                      f"text='{text[:50]}...'")
                payload = {
                    "input": text,
                    "voice": voice,
                    "language": self._language,
                    "response_format": "wav",
                }
                if instruct:
                    payload["instructions"] = instruct
                if seed >= 0:
                    payload["seed"] = seed
                return self._openai_speech(payload, output_path)

            instruct = self._build_instruct(instruct_text, voice_data, fallback=default_style or "neutral")

            print(f"TTS [external] generating (seed={seed}) with instruct='{instruct}' for text='{text[:50]}...'")

            client = self._init_external()

            result = client.predict(
                text=text,
                language=self._language,
                speaker=voice,
                instruct=instruct,
                model_size="1.7B",
                seed=seed,
                api_name="/generate_custom_voice"
            )

            generated_audio_filepath = result[0]
            if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
                print(f"Error: No audio file generated for: '{text[:50]}...'")
                return False

            if os.path.getsize(generated_audio_filepath) == 0:
                print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
                return False

            shutil.copy(generated_audio_filepath, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating custom voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _external_clone_request(self, text, ref_audio_path, ref_text, seed, output_path):
        """Send a voice-cloning request to an OpenAI-compatible TTS server."""
        print(f"TTS [external/openai] clone ref='{os.path.basename(ref_audio_path)}' "
              f"text='{text[:50]}...'")
        payload = {
            "input": text,
            "ref_audio": self._audio_data_url(ref_audio_path),
            "ref_text": ref_text,
            "language": self._language,
            "response_format": "wav",
            "task_type": "Base",
        }
        if seed >= 0:
            payload["seed"] = seed
        return self._openai_speech(payload, output_path)

    def _external_generate_adapter_voice(self, text, voice_data, output_path, batch_seed=None):
        """Approximate a LoRA voice on an external server.

        The adapter is a local training artifact the remote server knows nothing
        about, so fall back to cloning the adapter's own reference sample.
        """
        adapter_path = voice_data.get("adapter_path")
        if not adapter_path:
            print("Error: No adapter_path in voice_data")
            return False

        if not os.path.isabs(adapter_path):
            root_dir = os.path.dirname(os.path.dirname(__file__))
            adapter_path = os.path.join(root_dir, adapter_path)

        ref_wav_path = os.path.join(adapter_path, "ref_sample.wav")
        meta_path = os.path.join(adapter_path, "training_meta.json")
        if not os.path.exists(ref_wav_path) or not os.path.exists(meta_path):
            print(f"Error: LoRA adapter assets not found in {adapter_path} "
                  f"(need ref_sample.wav and training_meta.json)")
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                ref_text = json.load(f).get("ref_sample_text", "")
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error: Could not read training_meta.json: {e}")
            return False

        if not ref_text:
            print("Error: ref_sample_text missing from training_meta.json")
            return False

        try:
            seed = self._resolve_seed(voice_data, voice_data.get("name", "") or adapter_path, batch_seed)
        except (TypeError, ValueError):
            seed = -1

        print(f"TTS [external/openai] LoRA voice '{os.path.basename(adapter_path)}' "
              f"-> falling back to reference-sample cloning")
        try:
            return self._external_clone_request(text, ref_wav_path, ref_text, seed, output_path)
        except Exception as e:
            import traceback
            print(f"Error generating LoRA voice externally: {e}")
            traceback.print_exc()
            return False

    def _external_generate_clone(self, text, speaker, voice_config, output_path, batch_seed=None):
        """Generate voice-cloned audio via the external TTS server.

        Requires a Base checkpoint on the server side.
        """
        try:
            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            ref_audio = voice_data.get("ref_audio")
            ref_text = voice_data.get("ref_text")
            seed = self._resolve_seed(voice_data, speaker, batch_seed)

            if not ref_audio or not ref_text:
                print(f"Warning: Clone voice for '{speaker}' missing ref_audio or ref_text. Skipping.")
                return False

            # Resolve relative paths against project root
            if not os.path.isabs(ref_audio):
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ref_audio = os.path.join(root_dir, ref_audio)

            if not os.path.exists(ref_audio):
                print(f"Warning: Reference audio not found for '{speaker}': {ref_audio}")
                return False

            if self._external_api != "gradio":
                return self._external_clone_request(text, ref_audio, ref_text, seed, output_path)

            from gradio_client import handle_file

            client = self._init_external()

            result = client.predict(
                handle_file(ref_audio),
                ref_text,
                text,
                self._language,
                False,       # use_xvector_only
                "1.7B",
                200,         # max_chunk_chars
                0,           # chunk_gap
                seed,
                api_name="/generate_voice_clone"
            )

            generated_audio_filepath = result[0]
            if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
                print(f"Error: No audio file generated for: '{text[:50]}...'")
                return False

            if os.path.getsize(generated_audio_filepath) == 0:
                print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
                return False

            shutil.copy(generated_audio_filepath, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating clone voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _sequential_custom(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Sequential custom voice generation for external mode (no native batch)."""
        results = {"completed": [], "failed": []}

        for chunk in chunks:
            idx = chunk["index"]
            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            try:
                success = self.generate_custom_voice(
                    chunk.get("text", ""),
                    chunk.get("instruct", ""),
                    chunk.get("speaker", ""),
                    voice_config,
                    output_path,
                    batch_seed=batch_seed,
                )
                if success:
                    results["completed"].append(idx)
                    print(f"Batch chunk {idx} saved: {os.path.getsize(output_path)} bytes")
                else:
                    results["failed"].append((idx, "Custom voice generation failed"))
            except Exception as e:
                results["failed"].append((idx, str(e)))

        return results

    # ── Utility ──────────────────────────────────────────────────

    @staticmethod
    def _save_wav(audio_array, sample_rate, output_path):
        """Save a numpy audio array as a WAV file."""
        # Ensure numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        # Flatten if needed
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()
        sf.write(output_path, audio_array, sample_rate)
