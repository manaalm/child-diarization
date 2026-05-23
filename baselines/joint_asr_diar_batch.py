"""Batch inference for the USC-SAIL joint ASR + diarization model
(`AlexXu811/child-adult-joint-asr-diarization`) on a wav directory.

Output format from upstream inference.py:
    <|0.70|><adult>...<|2.20|><|2.90|><child>where<|3.00|><|3.40|><child>...

We parse the `<role>` tokens and surrounding `<|t|>` timestamps to extract
child segments, write per-file RTTMs into <results-dir>/per_file_predictions/,
matching the layout used by pyannote/unified_rttm.py for downstream
frame_localization_gt.py and onset_tolerance_f1.py to ingest.

Limitations:
- Repo's max_len=300 truncation is destructive when generation doesn't
  naturally terminate. We raise max_len to 600 here; if still truncated,
  log and skip parsing.
- Model is trained on Playlogue → evaluation on Playlogue is circular;
  intended targets here are synth_holdout and Providence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import librosa  # noqa: F401  (loaded by transcribe_audio via inference)

# Repo modules
HERE = os.path.dirname(os.path.abspath(__file__))
JOINT_DIR = os.path.join(HERE, "joint_asr_diar")
sys.path.insert(0, JOINT_DIR)

from transformers import WhisperProcessor, LogitsProcessorList  # noqa: E402
from model import WhisperWithDiarization  # noqa: E402
from inference import (  # noqa: E402
    get_vad_outputs, decode_with_timestamps,
)
from structured_logits_processor import StructuredOutputLogitsProcessor  # noqa: E402
from silence_masking_processor import SilenceMaskingProcessor  # noqa: E402


def transcribe_audio_kept(model, processor, audio_path, device='cuda',
                          enable_silence_masking=True,
                          enable_logits_processors=True, max_len=300,
                          audio_array=None, sample_rate=16000):
    """Variant of inference.transcribe_audio that:
    - clamps max_len to 300 (>300 triggers GPU IndexError on Whisper-small;
      see notes in CLAUDE.md / megadoc §8c),
    - does NOT discard tokens when generation reaches max_len; instead keeps
      whatever was generated and returns it (suffixed with `<|TRUNCATED|>`),
    - accepts a pre-loaded `audio_array` (1-D numpy float at 16 kHz) so the
      caller can chunk a long file and avoid re-loading on every call.
    """
    if audio_array is None:
        import librosa as _librosa
        audio, _ = _librosa.load(audio_path, sr=16000)
    else:
        audio = audio_array
    model = model.to(device)
    model.eval()
    input_features = processor(audio, sampling_rate=16000,
                               return_tensors="pt").input_features.to(device)

    silence_segments = []
    if enable_silence_masking:
        silence_segments = get_vad_outputs(model, input_features, device,
                                           silence_threshold=0.7)
        silence_segments = [s for s in silence_segments
                            if (s["end"] - s["start"]) >= 1.0]

    start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    notimestamps_token_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    prompt = "<|startoftranscript|><|en|><|transcribe|>"
    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    logits_processors = []
    if enable_logits_processors:
        logits_processors.append(StructuredOutputLogitsProcessor(
            tokenizer=processor.tokenizer,
            speaker_tokens=("<adult>", "<child>"),
            max_len=max_len,
            device=str(device),
        ))
        if silence_segments:
            logits_processors.append(SilenceMaskingProcessor(
                tokenizer=processor.tokenizer,
                silence_segments=silence_segments,
                buffer_s=0.2,
            ))
    logits_proc = LogitsProcessorList(logits_processors)

    import torch
    with torch.no_grad():
        generated = model.generate(
            input_features,
            logits_processor=logits_proc,
            decoder_input_ids=decoder_input_ids,
            max_length=max_len,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            suppress_tokens=[notimestamps_token_id, start_token_id]
            if notimestamps_token_id is not None else [],
            early_stopping=False,
            repetition_penalty=1.1,
            return_timestamps=False,
        )
    token_ids = generated[0].cpu().numpy()
    transcript = decode_with_timestamps(processor, token_ids)
    if len(token_ids) >= max_len:
        transcript = transcript + "<|TRUNCATED|>"
    return transcript


# Match `<|0.70|><adult>...<|2.20|>` style segments in the model output.
SEGMENT_RE = re.compile(
    r"<\|(?P<start>\d+(?:\.\d+)?)\|>"
    r"<(?P<role>adult|child)>"
    r"(?P<text>.*?)"
    r"<\|(?P<end>\d+(?:\.\d+)?)\|>",
    re.DOTALL,
)


def parse_prediction(prediction: str) -> list[dict]:
    """Return list of {'start','end','role','text'} segments."""
    segs = []
    for m in SEGMENT_RE.finditer(prediction):
        try:
            start = float(m.group("start"))
            end = float(m.group("end"))
        except ValueError:
            continue
        if end <= start:
            continue
        segs.append({
            "start": start,
            "end": end,
            "role": m.group("role"),
            "text": m.group("text").strip(),
        })
    return segs


def write_rttm(segs: list[dict], file_id: str, out_path: str,
               write_role: str = "child") -> None:
    """Write segments matching `write_role` to RTTM, label=CHI."""
    with open(out_path, "w") as f:
        for s in segs:
            if s["role"] != write_role:
                continue
            dur = s["end"] - s["start"]
            f.write(
                f"SPEAKER {file_id} 1 {s['start']:.3f} {dur:.3f} "
                f"<NA> <NA> CHI <NA> <NA>\n"
            )


def transcribe_long(model, processor, audio_path, device='cuda',
                    chunk_sec=30.0, sample_rate=16000, max_len=300,
                    enable_silence_masking=True,
                    enable_logits_processors=True):
    """Chunk a long audio into <chunk_sec> windows, transcribe each, and
    concatenate predicted segments with corrected absolute timestamps.

    Returns (combined_prediction_str, n_chunks, n_truncated_chunks).
    The returned string is the per-chunk predictions joined by " "; each
    chunk's `<|t|>` timestamps are rewritten to absolute file time.
    """
    import re as _re
    import librosa as _librosa
    audio, _ = _librosa.load(audio_path, sr=sample_rate)
    n_samples = len(audio)
    chunk_samples = int(chunk_sec * sample_rate)
    n_chunks = max(1, (n_samples + chunk_samples - 1) // chunk_samples)
    parts = []
    n_trunc = 0
    for i in range(n_chunks):
        s = i * chunk_samples
        e = min(n_samples, s + chunk_samples)
        chunk = audio[s:e]
        if len(chunk) < int(0.5 * sample_rate):
            continue  # skip <0.5 s tail
        # Whisper expects 30 s mels; pad short chunks with zeros
        if len(chunk) < chunk_samples:
            import numpy as _np
            chunk = _np.pad(chunk, (0, chunk_samples - len(chunk)))
        try:
            pred_chunk = transcribe_audio_kept(
                model, processor, audio_path, device=device,
                enable_silence_masking=enable_silence_masking,
                enable_logits_processors=enable_logits_processors,
                max_len=max_len,
                audio_array=chunk, sample_rate=sample_rate,
            )
        except Exception as e:
            parts.append(f"<|CHUNK_ERROR_{i}|>{type(e).__name__}: {e}")
            continue
        if pred_chunk.endswith("<|TRUNCATED|>"):
            n_trunc += 1
            pred_chunk = pred_chunk[: -len("<|TRUNCATED|>")].rstrip()
        # Strip prompt prefix
        pred_chunk = pred_chunk.replace(
            "<|startoftranscript|><|en|><|transcribe|>", "").strip()
        # Rewrite timestamps from chunk-relative to absolute file time
        offset = i * chunk_sec

        def _shift(m):
            return f"<|{float(m.group(1)) + offset:.2f}|>"
        pred_chunk = _re.sub(r"<\|(\d+(?:\.\d+)?)\|>", _shift, pred_chunk)
        parts.append(pred_chunk)
    combined = " ".join(parts)
    return combined, n_chunks, n_trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=300,
                    help="Max generation length. Whisper-small's decoder has "
                         "positional embeddings of size 448 but values >300 "
                         "trigger GPU IndexError during generation. Upstream "
                         "uses 300 but discards all output on max-length hit; "
                         "this wrapper keeps whatever was generated.")
    ap.add_argument("--chunk-sec", type=float, default=30.0,
                    help="Chunk audio longer than this (sec) and concatenate "
                         "per-chunk predictions. Default 30 (Whisper limit).")
    ap.add_argument("--enable-silence-masking", action="store_true", default=True)
    ap.add_argument("--enable-logits-processors", action="store_true", default=True)
    ap.add_argument("--file-list", default="",
                    help="Optional: path to a text file with one WAV path per "
                         "line. If set, overrides --wav-dir scanning.")
    args = ap.parse_args()

    pred_dir = os.path.join(args.results_dir, "per_file_predictions")
    raw_dir = os.path.join(args.results_dir, "raw_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print("Loading processor + model from "
          "AlexXu811/child-adult-joint-asr-diarization ...", flush=True)
    processor = WhisperProcessor.from_pretrained(
        "AlexXu811/child-adult-joint-asr-diarization"
    )
    model = WhisperWithDiarization.from_pretrained(
        "AlexXu811/child-adult-joint-asr-diarization"
    )

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}", flush=True)

    if args.file_list:
        with open(args.file_list) as f:
            wavs = [Path(ln.strip()) for ln in f if ln.strip()]
        print(f"Loaded {len(wavs)} files from {args.file_list}", flush=True)
    else:
        wavs = sorted(p for p in Path(args.wav_dir).iterdir()
                      if p.suffix.lower() in {".wav", ".flac", ".mp3"})
        print(f"Found {len(wavs)} audio files in {args.wav_dir}", flush=True)

    n_truncated = 0
    n_no_segs = 0
    t0 = time.time()
    for i, wav_path in enumerate(wavs):
        stem = wav_path.stem
        out_rttm = os.path.join(pred_dir, f"{stem}_pred.rttm")
        out_raw = os.path.join(raw_dir, f"{stem}.txt")
        if os.path.exists(out_rttm) and os.path.exists(out_raw):
            continue
        try:
            # Determine if file needs chunking
            import librosa as _librosa
            dur = _librosa.get_duration(path=str(wav_path))
            if dur > args.chunk_sec + 0.5:
                pred, _nc, n_chunk_trunc = transcribe_long(
                    model, processor, str(wav_path), device=device,
                    chunk_sec=args.chunk_sec, max_len=args.max_len,
                    enable_silence_masking=args.enable_silence_masking,
                    enable_logits_processors=args.enable_logits_processors,
                )
                if n_chunk_trunc:
                    n_truncated += n_chunk_trunc
            else:
                pred = transcribe_audio_kept(
                    model, processor, str(wav_path), device=device,
                    enable_silence_masking=args.enable_silence_masking,
                    enable_logits_processors=args.enable_logits_processors,
                    max_len=args.max_len,
                )
        except Exception as e:
            print(f"[{i+1}/{len(wavs)}] {stem} ERROR: {e}", flush=True)
            with open(out_raw, "w") as f:
                f.write(f"ERROR: {e}\n")
            # Still write empty RTTM so eval scripts don't choke
            with open(out_rttm, "w") as f:
                pass
            continue

        prediction = pred.replace("<|startoftranscript|><|en|><|transcribe|>", "").strip()
        with open(out_raw, "w") as f:
            f.write(prediction + "\n")

        if prediction.endswith("<|TRUNCATED|>"):
            n_truncated += 1
            prediction = prediction[: -len("<|TRUNCATED|>")].rstrip()

        segs = parse_prediction(prediction)
        if not segs:
            n_no_segs += 1
        write_rttm(segs, stem, out_rttm, write_role="child")

        if (i + 1) % 10 == 0 or i + 1 == len(wavs):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(wavs) - i - 1) / rate if rate else 0
            print(f"[{i+1}/{len(wavs)}] {stem}  "
                  f"({rate:.2f} files/s, ETA {eta/60:.1f} min)  "
                  f"truncated_so_far={n_truncated}  no_segs={n_no_segs}",
                  flush=True)

    print(f"\nDONE. n_files={len(wavs)}  truncated={n_truncated}  no_segs={n_no_segs}", flush=True)
    summary = {
        "wav_dir": args.wav_dir,
        "n_files": len(wavs),
        "n_truncated": n_truncated,
        "n_no_segs": n_no_segs,
        "max_len": args.max_len,
        "device": device,
        "model": "AlexXu811/child-adult-joint-asr-diarization",
    }
    with open(os.path.join(args.results_dir, "joint_asr_diar_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
