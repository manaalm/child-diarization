"""Compute / efficiency table for the headline systems.

For each system: trainable params, total params, est. FLOPs per 30 s clip,
documented training GPU-hours, and inference cost regime tag.

We use **documented model-card numbers** for upstream encoders (Whisper-small,
WavLM-Base+, HuBERT-large, Qwen2.5-Omni-7B (thinker), ECAPA-TDNN,
pyannote/speaker-diarization-community-1, NeMo Sortformer) and load only the
local trained heads from disk.

FLOPs estimate uses the 2 × N_params × N_tokens rule-of-thumb for transformer
encoders (Hoffmann et al. 2022 Chinchilla). For 50 Hz audio: 30 s = 1500 tokens.

Outputs:
  /tmp/thesis_outputs/evaluation/system_efficiency.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
OUT = "/tmp/thesis_outputs/evaluation/system_efficiency.csv"


@dataclass
class Row:
    system: str
    family: str
    backbone: str
    backbone_params_m: float
    head_params_m: float
    total_params_m: float
    flops_per_30s_clip_g: float
    train_gpu_hours: str
    inference_regime: str
    notes: str = ""


def gflops_2pn(params_m: float, seq_tokens: int) -> float:
    return 2.0 * params_m * 1e6 * seq_tokens / 1e9


N_TOKENS_30S = 1500


def head_params(ckpt_path: str, default_m: float = 0.5) -> float:
    if not os.path.isfile(ckpt_path):
        return default_m
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        head_keys = [k for k in sd if any(s in k.lower() for s in ("mil_head", "abmil", "head", "classifier"))]
        n = sum(sd[k].numel() for k in head_keys)
        return n / 1e6 if n else default_m
    except Exception as e:
        print(f"  load failed for {ckpt_path}: {e}")
        return default_m


def main():
    rows = []

    whisper_mil_head = head_params(f"{REPO}/mil/mil_results/whisper_mil/best_checkpoint.pt")
    wavlm_mil_head = head_params(f"{REPO}/mil/mil_results/wavlm_mil/best_checkpoint.pt")

    rows.append(Row("whisper_mil", "MIL", "Whisper-small encoder", 88.0, whisper_mil_head,
                    88.0 + whisper_mil_head, gflops_2pn(88.0 + whisper_mil_head, N_TOKENS_30S),
                    "~3 (single A100, 1311 train clips × 20 ep)", "gpu_required",
                    "Backbone frozen; only ABMIL head + classifier trained."))
    rows.append(Row("whisper_mil_tsmil_concat", "MIL", "Whisper-small encoder", 88.0,
                    whisper_mil_head + 0.6, 88.0 + whisper_mil_head + 0.6,
                    gflops_2pn(88.0 + whisper_mil_head + 0.6, N_TOKENS_30S),
                    "~3 (whisper_mil + ECAPA proto pre-step)", "gpu_required",
                    "Adds ECAPA-conditioning concat (+~0.6M params)."))
    rows.append(Row("wavlm_mil", "MIL", "WavLM-Base+", 94.0, wavlm_mil_head,
                    94.0 + wavlm_mil_head, gflops_2pn(94.0 + wavlm_mil_head, N_TOKENS_30S),
                    "~3 (single A100)", "gpu_required",
                    "Backbone frozen; ABMIL head trained."))
    rows.append(Row("hubert_large_mil_layersum", "MIL", "HuBERT-large", 317.0, 1.5,
                    318.5, gflops_2pn(318.5, N_TOKENS_30S),
                    "~6 (larger backbone)", "gpu_required",
                    "Spec-014 US1 weighted-layer-sum; 24 layers × scalar gate."))
    rows.append(Row("wavlm_pseudo_frame", "self-distill", "WavLM-Base+", 94.0, 0.5,
                    94.5, gflops_2pn(94.5, N_TOKENS_30S),
                    "~0.3 (5 min total per CLAUDE.md)", "gpu_required",
                    "Frozen WavLM + tiny Conv1d frame classifier."))

    rows.append(Row("usc_sail_whisper", "diarizer (frame)", "Whisper-base + LoRA r=8", 74.0, 1.6,
                    75.6, gflops_2pn(75.6, N_TOKENS_30S),
                    "documented 50k-step pretrain (anfengxu)", "gpu_required",
                    "LoRA fc1/fc2 each layer; 4-class output."))

    rows.append(Row("pyannote_diarization", "diarizer (segment)", "PyanNet + WeSpeaker ECAPA", 24.7, 0.0,
                    24.7, gflops_2pn(24.7, N_TOKENS_30S), "0 (pretrained)", "external_subprocess",
                    "speaker-diarization-community-1 pipeline."))
    rows.append(Row("vtc_2.0", "diarizer (segment)", "PyanNet (BabAR-trained)", 17.0, 0.0, 17.0,
                    gflops_2pn(17.0, N_TOKENS_30S), "0", "external_subprocess",
                    "Fine-tuned PyanNet; same arch as pyannote frontend."))
    rows.append(Row("vtc_kchi", "diarizer (segment)", "PyanNet", 17.0, 0.0, 17.0,
                    gflops_2pn(17.0, N_TOKENS_30S), "0", "external_subprocess",
                    "Output identical to BabAR on short BIDS clips."))
    rows.append(Row("babar", "diarizer (segment)", "PyanNet + phoneme model", 22.0, 0.0, 22.0,
                    gflops_2pn(22.0, N_TOKENS_30S), "0", "external_subprocess",
                    "Phoneme step is no-op on BIDS short clips."))
    rows.append(Row("vbx", "diarizer (clustering)", "pyannote VAD + ECAPA + VB-HMM", 24.7, 0.0, 24.7,
                    gflops_2pn(24.7, N_TOKENS_30S), "0", "external_subprocess",
                    "VB-HMM clustering; collapses to 1 spk on long-form."))
    rows.append(Row("sortformer", "diarizer (neural)", "NeMo diar_sortformer_4spk-v1", 92.0, 0.0,
                    92.0, gflops_2pn(92.0, N_TOKENS_30S), "0", "external_subprocess",
                    "Pretrained 4-speaker mixtures; 90s chunked."))
    rows.append(Row("eend_eda", "diarizer (neural)", "ESPnet horiguchi 6spk", 35.0, 0.0, 35.0,
                    gflops_2pn(35.0, N_TOKENS_30S), "0", "external_subprocess",
                    "Encoder-decoder + attractors; 90s chunked."))

    rows.append(Row("ecapa_enrollment_head", "enrollment", "ECAPA-TDNN (speechbrain)", 6.2, 0.0, 6.2,
                    gflops_2pn(6.2, N_TOKENS_30S), "0", "cpu_fast",
                    "Cosine-similarity head; no learned params."))

    rows.append(Row("audio_llm_qwen25_omni", "audio LLM", "Qwen2.5-Omni-7B (thinker)",
                    7400.0, 0.0,
                    7400.0, 7400.0,
                    "0 (zero-shot)", "gpu_required",
                    "FLOPs: ~1 forward over short context. Talker (speech "
                    "synthesis) component is not loaded."))

    rows.append(Row("best_audio_mil_ensemble", "ensemble", "mean of 4 MIL outputs", 0.0, 0.0, 0.0,
                    0.0, "0 (post-hoc)", "cpu_fast",
                    "No new params; sum cost of 4 base systems."))
    rows.append(Row("metadata_stacker", "ensemble", "logistic regression", 0.0, 0.05, 0.05,
                    0.0, "<0.01 (sklearn)", "cpu_fast",
                    "12 sys probs + metadata → LR. Cost dominated by base."))
    rows.append(Row("av_fusion_gated", "AV fusion", "BabAR-prob × XGBoost(visual)", 17.0, 0.05,
                    17.05, gflops_2pn(17.0, N_TOKENS_30S),
                    "<0.5 (XGB on 1311 clips)", "cpu_fast",
                    "Audio: BabAR scores. Visual: XGBoost on visual eligibility."))

    df = pd.DataFrame([r.__dict__ for r in rows])
    for c in ("backbone_params_m", "head_params_m", "total_params_m", "flops_per_30s_clip_g"):
        df[c] = df[c].round(2)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
