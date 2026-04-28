"""
run_espnet_diar.py — ESPnet2 EEND-EDA batch inference script.

Called as a subprocess by EENDEDAFrontend in nemo_diar.py.

Usage:
    python run_espnet_diar.py \\
        --audio-list /tmp/audio_paths.txt \\
        --output-dir  /path/to/rttm_output/ \\
        [--model-tag  espnet/horiguchi_INTERSPEECH2022_EEND-EDA-online_6spk] \\
        [--num-spks   0] \\
        [--device     cuda]

Output:
    One RTTM per input audio, named <stem>.rttm in <output-dir>.
    Empty RTTM is written for any file that fails (no crash).

Setup:
    pip install espnet espnet_model_zoo soundfile

Finding a pre-trained EEND-EDA model:
    from espnet_model_zoo.downloader import ModelDownloader
    d = ModelDownloader()
    # List diarization models:
    for r in d.query("diar"):
        print(r["name"])
    # Download and unpack:
    cfg = d.download_and_unpack("espnet/<model_name>")

The --model-tag argument accepts either:
  - An ESPnet Model Zoo tag, e.g. "espnet/horiguchi_INTERSPEECH2022_EEND-EDA-online_6spk"
  - A local directory containing train_config.yaml and a .pth checkpoint file.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# RTTM conversion helpers
# ---------------------------------------------------------------------------

def _frame_activity_to_rttm(
    activity: np.ndarray,
    stem: str,
    hop_sec: float = 0.01,
    min_dur: float = 0.1,
    threshold: float = 0.5,
) -> List[str]:
    """Convert (T, n_spks) per-frame activity to RTTM lines.

    activity values are probabilities [0,1] or binary {0,1}.
    Consecutive active frames are merged into RTTM segments.
    Gaps < 0.05 s between consecutive active frames of the same speaker
    are filled to avoid over-fragmentation.
    """
    if activity.ndim == 1:
        activity = activity[:, None]
    n_frames, n_spks = activity.shape
    lines = []

    for spk_idx in range(n_spks):
        label = f"spk{spk_idx + 1}"
        in_seg = False
        seg_start = 0.0
        prev_active = False

        for i in range(n_frames):
            active = activity[i, spk_idx] > threshold
            t = i * hop_sec

            if active and not in_seg:
                in_seg = True
                seg_start = t
            elif not active and in_seg:
                # Look-ahead: fill short gaps (≤5 frames = 50ms)
                gap_end = min(n_frames, i + 6)
                fill = any(activity[j, spk_idx] > threshold for j in range(i, gap_end))
                if fill:
                    continue
                dur = t - seg_start
                if dur >= min_dur:
                    lines.append(
                        f"SPEAKER {stem} 1 {seg_start:.3f} {dur:.3f} "
                        f"<NA> <NA> {label} <NA> <NA>"
                    )
                in_seg = False

        if in_seg:
            dur = n_frames * hop_sec - seg_start
            if dur >= min_dur:
                lines.append(
                    f"SPEAKER {stem} 1 {seg_start:.3f} {dur:.3f} "
                    f"<NA> <NA> {label} <NA> <NA>"
                )

    return lines


# ---------------------------------------------------------------------------
# ESPnet2 EEND-EDA inference
# ---------------------------------------------------------------------------

def _patch_espnet_model_compat():
    """Drop unknown kwargs from ESPnetDiarizationModel.__init__ for older checkpoints.

    Models trained with espnet <202511 may include 'context_size' in their
    config.yaml which was removed in 202511. Patching __init__ to accept **kwargs
    lets those checkpoints load without error.
    """
    try:
        from espnet2.diar.espnet_model import ESPnetDiarizationModel
        orig = ESPnetDiarizationModel.__init__
        import functools

        @functools.wraps(orig)
        def _compat_init(self, *args, **kwargs):
            known = {"frontend", "specaug", "normalize", "label_aggregator",
                     "encoder", "decoder", "attractor", "diar_weight", "attractor_weight"}
            kwargs = {k: v for k, v in kwargs.items() if k in known}
            orig(self, *args, **kwargs)

        ESPnetDiarizationModel.__init__ = _compat_init
    except Exception:
        pass  # non-fatal; model load will show the real error if it fails


def _patch_context_stacking(diar) -> None:
    """Re-inject frame stacking removed from ESPnetDiarizationModel in espnet 202511.

    Older checkpoints were trained with a context_size parameter that stacked
    (2*context_size+1) consecutive mel frames before the encoder, giving
    input_dim = n_mels * (2*context_size+1).  The current ESPnet silently drops
    context_size, causing a matrix-dimension mismatch.  This patches
    diar_model.encode on the loaded instance to restore the stacking step.
    """
    try:
        import types
        import torch
        import torch.nn.functional as F

        train_args = getattr(diar, "diar_train_args", None)

        # context_size lives under model_conf in ESPnet YAML, not at top level
        context_size = 0
        if train_args:
            # Try top-level first (some configs), then model_conf dict/namespace
            context_size = getattr(train_args, "context_size", 0) or 0
            if not context_size:
                mc = getattr(train_args, "model_conf", None)
                if isinstance(mc, dict):
                    context_size = mc.get("context_size", 0) or 0
                elif mc is not None:
                    context_size = getattr(mc, "context_size", 0) or 0

        # Auto-detect from encoder weight dimension as fallback:
        # If encoder first linear layer expects K*n_mels input (K>1), infer context_size.
        if not context_size:
            try:
                model_tmp = diar.diar_model
                # Walk common encoder paths to find a weight matrix
                enc = model_tmp.encoder
                # Try transformer encoder layers
                w = None
                for attr in ("encoders", "embed"):
                    sub = getattr(enc, attr, None)
                    if sub is not None:
                        # Could be ModuleList or sequential; grab first element
                        try:
                            first = sub[0] if hasattr(sub, "__getitem__") else sub
                            for lname in ("feed_forward", "ff1", "linear"):
                                lyr = getattr(first, lname, None)
                                if lyr is not None:
                                    for pname in ("w_1", "linear", "0"):
                                        try:
                                            w = getattr(lyr, pname).weight
                                            break
                                        except Exception:
                                            pass
                                if w is not None:
                                    break
                        except Exception:
                            pass
                    if w is not None:
                        break
                if w is not None and w.dim() >= 2:
                    enc_in = w.shape[1]
                    # Infer n_mels from frontend_conf, default 23
                    n_mels = 23
                    fc = getattr(train_args, "frontend_conf", {})
                    if isinstance(fc, dict):
                        n_mels = fc.get("n_mels", 23) or 23
                    elif fc is not None:
                        n_mels = getattr(fc, "n_mels", 23) or 23
                    if enc_in > n_mels and enc_in % n_mels == 0:
                        k = enc_in // n_mels
                        if k % 2 == 1:  # must be 2*c+1
                            context_size = (k - 1) // 2
                            print(f"  Auto-detected context_size={context_size} "
                                  f"from encoder input dim {enc_in} / n_mels {n_mels}")
            except Exception as _e:
                print(f"  context_size auto-detect failed: {_e}", file=sys.stderr)

        if context_size <= 0:
            return  # model doesn't need stacking

        model = diar.diar_model

        def _encode_with_stack(self, speech, speech_lengths,
                               bottleneck_feats=None, bottleneck_feats_lengths=None):
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # Frame stacking: (B, T, D) → (B, T, D*(2*context_size+1))
            B, T, D = feats.shape
            pad = feats.new_zeros(B, context_size, D)
            feats_padded = torch.cat([pad, feats, pad], dim=1)
            feats = torch.cat(
                [feats_padded[:, i:i + T, :] for i in range(2 * context_size + 1)],
                dim=-1,
            )

            if bottleneck_feats is None:
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
            elif self.frontend is None:
                encoder_out, encoder_out_lens, _ = self.encoder(
                    bottleneck_feats, bottleneck_feats_lengths
                )
            else:
                feats_interp = F.interpolate(
                    feats.transpose(1, 2), size=bottleneck_feats.shape[1]
                ).transpose(1, 2)
                encoder_out, encoder_out_lens, _ = self.encoder(
                    torch.cat((bottleneck_feats, feats_interp), 2),
                    bottleneck_feats_lengths,
                )
            return encoder_out, encoder_out_lens

        model.encode = types.MethodType(_encode_with_stack, model)
        print(f"  Applied context_size={context_size} frame-stacking patch to diar_model.encode")
    except Exception as e:
        print(f"  WARNING: context_size patch failed ({e}); inference may error", file=sys.stderr)


def _load_diar_model(model_tag: str, num_spks: int, device: str):
    """Load DiarSpeech from ESPnet Model Zoo tag or local directory."""
    _patch_espnet_model_compat()
    try:
        from espnet2.bin.diar_inference import DiarSpeech
    except ImportError:
        from espnet2.bin.diar_inference import DiarizeSpeech as DiarSpeech

    if os.path.isdir(model_tag):
        import glob

        yaml_files = glob.glob(os.path.join(model_tag, "**", "*.yaml"), recursive=True)
        pth_files = glob.glob(os.path.join(model_tag, "**", "*.pth"), recursive=True)
        if not yaml_files:
            raise FileNotFoundError(f"No .yaml config found under {model_tag}")
        if not pth_files:
            raise FileNotFoundError(f"No .pth checkpoint found under {model_tag}")
        train_config = yaml_files[0]
        model_file = pth_files[0]
        print(f"  Local model: config={train_config}, checkpoint={model_file}")
        kwargs = dict(
            train_config=train_config,
            model_file=model_file,
            device=device,
        )
        if num_spks > 0:
            kwargs["num_spks"] = num_spks
        diar = DiarSpeech(**kwargs)
        _patch_context_stacking(diar)
        return diar
    else:
        from espnet_model_zoo.downloader import ModelDownloader
        import glob as _glob

        d = ModelDownloader()
        print(f"  Downloading/unpacking model: {model_tag}")
        try:
            cfg = d.download_and_unpack(model_tag)
        except FileNotFoundError:
            # espnet_model_zoo expects a meta.yaml that some HF repos don't provide.
            # Fall back: locate the snapshot directory and pick the spk4 (or first)
            # config.yaml + checkpoint directly.
            snap_base = os.path.join(
                os.path.dirname(ModelDownloader.__module__.replace(".", "/")),
                "models--" + model_tag.replace("/", "--"),
            )
            # Use the package-relative cache that ModelDownloader uses
            import espnet_model_zoo as _emz
            emz_dir = os.path.dirname(_emz.__file__)
            snap_glob = os.path.join(emz_dir, f"models--{model_tag.replace('/', '--')}", "snapshots", "*")
            snaps = _glob.glob(snap_glob)
            if not snaps:
                raise
            snap_dir = snaps[0]
            # Prefer spk4 (4-speaker model) for naturalistic recordings
            yaml_files = sorted(_glob.glob(os.path.join(snap_dir, "**", "config.yaml"), recursive=True))
            pth_files = sorted(_glob.glob(os.path.join(snap_dir, "**", "*.pth"), recursive=True))
            spk4_yamls = [y for y in yaml_files if "spk4" in y]
            spk4_pths = [p for p in pth_files if "spk4" in p]
            train_config = (spk4_yamls or yaml_files)[0]
            model_file = (spk4_pths or pth_files)[0]
            print(f"  (meta.yaml missing — loading directly) config={train_config}")
            cfg = dict(train_config=train_config, model_file=model_file,
                       _snap_dir=snap_dir)

        if num_spks > 0:
            cfg["num_spks"] = num_spks
        cfg["device"] = device
        # The config references stats files via relative paths; chdir to snapshot.
        snap_dir = cfg.pop("_snap_dir", None)
        orig_dir = os.getcwd()
        if snap_dir:
            os.chdir(snap_dir)
        try:
            diar = DiarSpeech(**cfg)
            _patch_context_stacking(diar)
            return diar
        finally:
            os.chdir(orig_dir)


def _infer_one(diar, audio_path: str) -> np.ndarray:
    """Run EEND-EDA on a single audio file; return (T, n_spks) activity array."""
    import soundfile as sf
    import torch

    speech, rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    # Resample to model's expected rate if necessary (e.g. 8kHz AMI models).
    model_fs = getattr(getattr(diar, "diar_train_args", None), "frontend_conf", {})
    if isinstance(model_fs, dict):
        model_fs = model_fs.get("fs", None)
    if model_fs is None:
        try:
            model_fs = diar.diar_train_args.frontend_conf.get("fs", None)
        except Exception:
            model_fs = None
    if model_fs is not None:
        target_rate = int(str(model_fs).replace("k", "000")) if "k" in str(model_fs) else int(model_fs)
        if target_rate != rate:
            import torchaudio
            speech_t = torch.from_numpy(speech).unsqueeze(0)
            speech_t = torchaudio.functional.resample(speech_t, rate, target_rate)
            speech = speech_t.squeeze(0).numpy()
            rate = target_rate

    # DiarSpeech.__call__ requires a 2D input (batch, T) in current ESPnet2.
    # Pass as torch.Tensor (1, T) to satisfy the internal assertion.
    speech_tensor = torch.from_numpy(speech).unsqueeze(0)  # (1, T)

    out = diar(speech_tensor, fs=rate)

    # DiarSpeech returns different formats across ESPnet versions.
    # diar_ami_eend_eda: out = (None, ndarray(1, T, n_spks))
    #   out[0] = None (binary hard labels when num_spks fixed; None for EDA auto mode)
    #   out[1] = float32 posterior probabilities (1, T, n_spks)
    # Some versions: out = (ndarray(T, n_spks),) or dict
    activity = None
    if isinstance(out, dict):
        activity = out.get("spk_labels", out.get("diarization", None))
    elif isinstance(out, (tuple, list)):
        # Pick the first non-None array; prefer float arrays over None/binary
        for candidate in out:
            if candidate is None:
                continue
            arr = candidate.numpy() if hasattr(candidate, "numpy") else np.array(candidate)
            if arr.dtype.kind == "f" or activity is None:
                activity = arr
                if arr.dtype.kind == "f":
                    break  # prefer float probabilities over binary labels
    elif isinstance(out, np.ndarray):
        activity = out
    elif hasattr(out, "numpy"):
        activity = out.numpy()
    else:
        try:
            activity = np.array(out)
        except Exception:
            pass

    if activity is None:
        return np.zeros((1, 1))

    if hasattr(activity, "numpy"):
        activity = activity.numpy()
    activity = np.array(activity, dtype=np.float32)

    # Strip batch dimension if present: (1, T, n_spks) → (T, n_spks)
    if activity.ndim == 3 and activity.shape[0] == 1:
        activity = activity[0]
    if activity.ndim == 1:
        activity = activity[:, None]
    return activity


def run_eend_eda(audio_paths: List[str], output_dir: str, model_tag: str,
                 num_spks: int, device: str) -> None:
    print(f"Loading EEND-EDA model: {model_tag} (device={device})")
    diar = _load_diar_model(model_tag, num_spks, device)
    print(f"Model loaded. Processing {len(audio_paths)} file(s)...")

    os.makedirs(output_dir, exist_ok=True)

    for i, audio_path in enumerate(audio_paths, 1):
        stem = Path(audio_path).stem
        out_rttm = os.path.join(output_dir, f"{stem}.rttm")
        if os.path.exists(out_rttm):
            continue
        try:
            activity = _infer_one(diar, audio_path)
            lines = _frame_activity_to_rttm(activity, stem)
            with open(out_rttm, "w") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")
            print(f"  [{i}/{len(audio_paths)}] {stem}: {len(lines)} segments")
        except Exception as exc:
            print(f"  WARNING [{i}]: {audio_path}: {exc}", file=sys.stderr)
            open(out_rttm, "w").close()
        finally:
            # Free GPU memory between files to reduce fragmentation on long recordings
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ESPnet2 EEND-EDA batch diarization — outputs one RTTM per audio file."
    )
    parser.add_argument(
        "--audio-list", required=True,
        help="Text file with one audio path per line.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-tag",
        default="espnet/diar_ami_eend_eda",
        help=(
            "ESPnet Model Zoo tag (e.g. 'espnet/horiguchi_INTERSPEECH2022_EEND-EDA-online_6spk') "
            "or local directory containing train_config.yaml + *.pth."
        ),
    )
    parser.add_argument(
        "--num-spks", type=int, default=0,
        help="Number of speakers (0 = let EDA attractor mechanism determine automatically).",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.audio_list) as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    if not paths:
        print("No audio files in list — nothing to do.")
        return

    run_eend_eda(paths, args.output_dir, args.model_tag, args.num_spks, args.device)
    print("EEND-EDA inference complete.")


if __name__ == "__main__":
    main()
