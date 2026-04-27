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

# TS-TalkNet (target-speaker variant with enrollment) — optional, see §4 below
git clone https://github.com/Jiang-Yidi/TS-TalkNet video/TS-TalkNet
```

## 3. Pretrained checkpoints

### TalkNet-ASD (auto-download, no manual action needed)

Both the S3FD face detector and TalkNet-ASD model checkpoint are downloaded
automatically on the first run of `run_asd.py`:

- **S3FD** (`video/TalkNet-ASD/model/faceDetector/s3fd/sfd_face.pth`):
  auto-downloaded by S3FD's `__init__.py` via gdown.
- **TalkNet-ASD model** (`video/pretrain/talknet_asd.model`):
  auto-downloaded via gdown (GDrive ID `1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea`).

No manual download is required for TalkNet-ASD. Both downloads run on first
inference; expect ~1 min of download time the first time.

### TS-TalkNet (not publicly released — frontend skipped)

TS-TalkNet requires two checkpoints that are **not publicly released**:

- `video/pretrain/ts_talknet.model` — trained TS-TalkNet checkpoint
- `video/TS-TalkNet/exps/pretrain.model` — ECAPA speaker encoder weights

These must be obtained directly from the TS-TalkNet authors
(Yidi Jiang, `jiang_yidi@outlook.com`). If they are not present,
`TSTalkNetFrontend` returns `[]` for all clips (graceful skip) and
`enrollment_video_asd.sh` prints a warning and skips the enrollment run.

The TalkNet-ASD frontend is fully functional without TS-TalkNet.

## 4. Verify setup

```bash
cd video/
uv run python -c "import torch, cv2, torchaudio; print('torch:', torch.__version__); print('cv2:', cv2.__version__)"
```

## 5. Known limitations

- **Audio-only datasets**: Providence and Playlogue have no `.mp4` files.
  Running `run_asd.py` on these will raise `FileNotFoundError` with a clear message.
  The `pyannote/video_asd.py` frontends catch this and return `[]` automatically.
- **TS-TalkNet**: Checkpoints are not publicly released. This frontend is skipped
  when checkpoints are absent; only TalkNet-ASD is used in enrollment runs.
- **Child-face identification (TalkNet-ASD)**: Without speaker enrollment, the child
  is assumed to be the face with the smallest mean bounding box area — a heuristic
  that can fail in group scenes or when the child is off-screen.
- **Domain shift**: TalkNet/LoCoNet are pretrained on AVA-ActiveSpeaker (TV/film).
  Expect lower raw mAP on SAILS home videos vs. published AVA numbers.
