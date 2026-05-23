"""Bark zero-shot smoke test for spec-019.

Generates ~10 candidates per child-targeted prompt and saves to wav/.
Goal: decide whether Bark out-of-the-box can produce plausibly child-like
vocalizations at 14-18mo, before committing to a fine-tune.
"""
import argparse
import os

import numpy as np
import scipy.io.wavfile as wav
import torch

from transformers import AutoProcessor, BarkModel

PROMPTS = {
    "babble_brackets": "[baby babbling]",
    "infant_voc_brackets": "[infant vocalization]",
    "mama_dada": "mama mama da da",
    "ah_ba_ba": "ah ba ba ba ba",
    "laughs_babble": "[laughs] ba ba ba ba",
    "ooh_aah": "ooh aah ooh",
    "single_word_dog": "doggie",
    "uhoh": "uh oh",
}


def main(out_dir: str, n_per_prompt: int, model_name: str, voice_preset: str | None):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = BarkModel.from_pretrained(model_name).to(device)
    model.eval()
    sr = model.generation_config.sample_rate

    torch.manual_seed(42)
    for prompt_name, text in PROMPTS.items():
        for i in range(n_per_prompt):
            kwargs = {}
            if voice_preset:
                kwargs["voice_preset"] = voice_preset
            inputs = processor(text=text, return_tensors="pt", **kwargs).to(device)
            with torch.no_grad():
                audio = model.generate(**inputs, do_sample=True)
            audio_np = audio.cpu().numpy().squeeze()
            audio_np = np.clip(audio_np, -1.0, 1.0)
            int16 = (audio_np * 32767).astype(np.int16)
            fname = f"{prompt_name}_{i:02d}.wav"
            wav.write(os.path.join(out_dir, fname), sr, int16)
            dur = len(int16) / sr
            print(f"  {fname}: {dur:.2f}s @ {sr}Hz", flush=True)
    print(f"\nDone. Wrote to {out_dir}.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="synth/synth_voice/spec019_bark_smoke")
    p.add_argument("--n-per-prompt", type=int, default=2)
    p.add_argument("--model", default="suno/bark-small")
    p.add_argument("--voice-preset", default=None,
                   help="Bark voice preset (e.g., v2/en_speaker_6) or None for unconditioned")
    args = p.parse_args()
    main(args.out_dir, args.n_per_prompt, args.model, args.voice_preset)
