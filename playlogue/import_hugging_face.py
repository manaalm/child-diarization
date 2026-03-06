from huggingface_hub import snapshot_download
# requires: pip install -U huggingface_hub
# and: export HF_TOKEN=... (or run `huggingface-cli login`)
snapshot_download(
    repo_id="playlogue/playlogue-v1",
    repo_type="dataset",
    local_dir="playlogue_hf",
    allow_patterns=["data/speaker_diarization/*"]
)