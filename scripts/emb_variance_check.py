"""Quick diagnostic: compare embedding variance for baseline vs child-adapted WavLM."""
import torch, torchaudio
import pandas as pd
from transformers import WavLMModel

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
val_csv = f"{REPO}/whisper-modeling/seen_child_splits/val.csv"

df = pd.read_csv(val_csv)
clips = df['audio_path'].tolist()[:8]

def embed_clips(model_path, clips):
    model = WavLMModel.from_pretrained(model_path)
    model.eval()
    embs = []
    for path in clips:
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(0, keepdim=True)[:, :16000*5]
        with torch.no_grad():
            out = model(wav).last_hidden_state.mean(1)
        embs.append(out.squeeze(0))
    embs = torch.stack(embs)
    return {
        "feature_std": embs.std(dim=0).mean().item(),
        "clip_diversity": embs.std(dim=1).mean().item(),
        "mean_norm": embs.norm(dim=1).mean().item(),
    }

print("=== Baseline: microsoft/wavlm-base-plus ===")
r = embed_clips("microsoft/wavlm-base-plus", clips)
for k, v in r.items(): print(f"  {k}: {v:.4f}")

print("=== Child-adapted: step_50000 ===")
r2 = embed_clips(f"{REPO}/synth_results/child_wavlm_checkpoint/step_50000", clips)
for k, v in r2.items(): print(f"  {k}: {v:.4f}")
