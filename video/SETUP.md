# Video ASD Environment Setup

## 1. Install Python environment

```bash
cd video/
uv sync
```

## 2. Clone model repositories

```bash
# TalkNet-ASD (base model + S3FD face detector)
git clone https://github.com/TaoRuijie/TalkNet-ASD video/TalkNet-ASD

# TS-TalkNet (target-speaker variant with enrollment)
git clone https://github.com/Jiang-Yidi/TS-TalkNet video/TS-TalkNet
```

## 3. Download pretrained checkpoints

Place all checkpoints in `video/pretrain/` (this directory is .gitignore'd — do NOT commit).

### S3FD face detector (~87 MB)

Download `sfd_face.pth` from the TalkNet-ASD README (Google Drive link in
`TalkNet-ASD/README.md` under "Pretrained Model"). Save to:
```
video/pretrain/sfd_face.pth
```

### TalkNet-ASD model

Download `pretrain_TalkNet-ASD_CVPR2021.model` from the TalkNet-ASD README.
Save to:
```
video/pretrain/talknet_asd.model
```

### TS-TalkNet model

Download the TS-TalkNet checkpoint from the TS-TalkNet README.
Save to:
```
video/pretrain/ts_talknet.model
```

## 4. Verify setup

```bash
cd video/
uv run python -c "import torch, cv2, torchaudio; print('torch:', torch.__version__); print('cv2:', cv2.__version__)"
```

## 5. Known limitations

- **Audio-only datasets**: Providence and Playlogue have no `.mp4` files.
  Running `run_asd.py` on these will raise `FileNotFoundError` with a clear message.
  The `pyannote/video_asd.py` frontends catch this and return `[]` automatically.
- **Child-face identification (TalkNet-ASD)**: Without speaker enrollment, the child
  is assumed to be the face with the smallest mean bounding box area — a heuristic
  that can fail in group scenes or when the child is off-screen.
- **Domain shift**: TalkNet/LoCoNet are pretrained on AVA-ActiveSpeaker (TV/film).
  Expect lower raw mAP on SAILS home videos vs. published AVA numbers.
