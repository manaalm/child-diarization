"""One-time pre-download of whisper-tiny/base/medium models + processors."""
import os
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
# Strip rotated/expired HF token so anonymous download works for public models
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
os.environ.pop("HF_HUB_TOKEN", None)
from huggingface_hub import HfFolder
HfFolder.delete_token()
from transformers import WhisperModel, WhisperProcessor, AutoFeatureExtractor

for name in ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-medium"]:
    print(f"--- {name} ---", flush=True)
    try:
        m = WhisperModel.from_pretrained(name)
        print(f"  model OK  ({sum(p.numel() for p in m.parameters())/1e6:.1f}M params)", flush=True)
    except Exception as e:
        print(f"  model FAIL: {e}", flush=True)
    try:
        WhisperProcessor.from_pretrained(name)
        print("  processor OK", flush=True)
    except Exception as e:
        print(f"  processor FAIL: {e}", flush=True)
    try:
        AutoFeatureExtractor.from_pretrained(name)
        print("  feature_extractor OK", flush=True)
    except Exception as e:
        print(f"  feature_extractor FAIL: {e}", flush=True)
