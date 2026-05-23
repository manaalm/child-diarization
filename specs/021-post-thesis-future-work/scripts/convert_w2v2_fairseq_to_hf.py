"""Fairseq-free wav2vec2 fairseq -> HF converter for spec-021 US2 (T031).

This is a slim variant of the upstream
    transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py
that operates directly on the state_dict (which torch.load can read without
fairseq). The HF Wav2Vec2Model skeleton is built from facebook/wav2vec2-base
config (matches LL_4300's wav2vec2-base architecture: 768 hidden, 12 layers,
12 heads, 7-conv feature extractor).

Usage:
    python convert_w2v2_fairseq_to_hf.py \
        --fairseq-ckpt models/wav2vec2_naturalistic_LL_4300_hf/staging/LL_4300/checkpoint_best.pt \
        --hf-base facebook/wav2vec2-base \
        --output-dir models/wav2vec2_naturalistic_LL_4300_hf
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model


# Direct copy of MAPPING from upstream conversion script.
MAPPING = {
    "post_extract_proj":   "feature_projection.projection",
    "encoder.pos_conv.0":  "encoder.pos_conv_embed.conv",
    "self_attn.k_proj":    "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj":    "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj":    "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj":  "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm":"encoder.layers.*.layer_norm",
    "fc1":                 "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2":                 "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm":    "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm":  "encoder.layer_norm",
    "layer_norm":          "feature_projection.layer_norm",  # top-level layer_norm
    "mask_emb":            "masked_spec_embed",
}
TOP_LEVEL_KEYS = {"masked_spec_embed", "feature_projection.layer_norm"}


def set_attr_chain(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def assign_weight(hf_model, mapped_key: str, weight_type: str | None, value: torch.Tensor):
    pointer = set_attr_chain(hf_model, mapped_key)
    if weight_type == "weight_g" or weight_type == "weight_v":
        # PyTorch parametrizations rename: weight_g -> parametrizations.weight.original0
        target_name = "original0" if weight_type == "weight_g" else "original1"
        if hasattr(pointer, weight_type):
            getattr(pointer, weight_type).data = value
        else:
            getattr(pointer.parametrizations.weight, target_name).data = value
        return
    if weight_type == "weight":
        pointer.weight.data = value
    elif weight_type == "bias":
        pointer.bias.data = value
    elif weight_type is None:
        # Top-level (e.g. masked_spec_embed)
        pointer.data = value
    else:
        raise ValueError(f"unknown weight_type {weight_type}")


def convert(fairseq_ckpt_path: Path, hf_base: str, output_dir: Path,
            verbose: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    sd = torch.load(fairseq_ckpt_path, map_location="cpu", weights_only=False)
    fairseq_state = sd["model"] if "model" in sd else sd

    config = Wav2Vec2Config.from_pretrained(hf_base)
    # Preserve mask_time_prob at the upstream default; turn off at inference
    # via Wav2Vec2Model.eval() / explicit mask_time_indices=None instead.
    use_group_norm = config.feat_extract_norm == "group"

    hf_model = Wav2Vec2Model(config)
    hf_model.eval()

    unused: list[str] = []
    feature_extractor = hf_model.feature_extractor

    for name, value in fairseq_state.items():
        if "conv_layers" in name:
            # Feature extractor conv layers
            after = name.split("conv_layers.")[-1]
            parts = after.split(".")
            if len(parts) < 3:
                unused.append(name); continue
            layer_id, type_id = int(parts[0]), int(parts[1])
            if type_id == 0:
                if "bias" in name:
                    feature_extractor.conv_layers[layer_id].conv.bias.data = value
                else:
                    feature_extractor.conv_layers[layer_id].conv.weight.data = value
            elif type_id == 2 and (not use_group_norm or layer_id == 0):
                if "bias" in name:
                    feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
                else:
                    feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            else:
                unused.append(name)
            continue

        # Try mapping table.
        is_used = False
        for src, dst in MAPPING.items():
            # Match on substring or top-level alias
            if src in name or src.split("w2v_model.")[-1] == name.split(".")[0]:
                # Resolve layer index for *-keys
                if "*" in dst:
                    layer_index = name.split(src)[0].split(".")[-2]
                    if not layer_index.isdigit():
                        continue
                    mapped = dst.replace("*", layer_index)
                else:
                    mapped = dst

                # Determine weight_type from the suffix.
                if name.endswith("weight_g"):
                    wt = "weight_g"
                elif name.endswith("weight_v"):
                    wt = "weight_v"
                elif name.endswith(".bias"):
                    wt = "bias"
                elif name.endswith(".weight"):
                    wt = "weight"
                else:
                    # Top-level tensor like masked_spec_embed (no .weight suffix)
                    wt = None

                # Skip top-level layer_norm if it's a sub-layer's norm (final_layer_norm
                # already handled above as encoder.layers.*.final_layer_norm).
                if src == "layer_norm" and not name.startswith("layer_norm."):
                    continue
                # Skip encoder.layer_norm if name matches a per-layer final layer norm.
                if src == "encoder.layer_norm" and "layers." in name:
                    continue

                try:
                    assign_weight(hf_model, mapped, wt, value)
                    is_used = True
                    if verbose:
                        print(f"OK  {name:60s} -> {mapped} ({wt})")
                except Exception as e:
                    if verbose:
                        print(f"WARN  {name:60s} -> {mapped} ({wt}) failed: {e}")
                break
        if not is_used:
            unused.append(name)

    # Save HF artefact.
    hf_model.save_pretrained(output_dir)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(hf_base)
    fe.save_pretrained(output_dir)

    summary = {
        "source_ckpt": str(fairseq_ckpt_path),
        "hf_base_config": hf_base,
        "n_state_keys": len(fairseq_state),
        "n_unused": len(unused),
        "unused_keys_sample": unused[:20],
        "output_dir": str(output_dir),
    }
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fairseq-ckpt", required=True, type=Path)
    ap.add_argument("--hf-base", default="facebook/wav2vec2-base")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    s = convert(args.fairseq_ckpt, args.hf_base, args.output_dir, verbose=args.verbose)
    print(json.dumps(s, indent=2))
