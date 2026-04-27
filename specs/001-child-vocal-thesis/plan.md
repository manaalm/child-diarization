# Implementation Plan: Video-Audio ASD Models for SAILS Child Diarization

**Branch**: `003-video-audio-models` | **Date**: 2026-04-20 | **Spec**: `specs/001-child-vocal-thesis/spec.md`
**Input**: "Use video-audio models (TS-TalkNet / EG4D / LocoNet / EASEE / AS-Net etc.) to analyze SAILS videos. Set this up."

## Summary

Add video-audio active speaker detection (ASD) as a new diarization frontend category in `pyannote/unified.py`. Implement TalkNet-ASD and TS-TalkNet as `DiarizationFrontend` subclasses that derive the SAILS video path from the audio path (`_audio.wav` → `_desc-processed_beh.mp4`), run face detection + ASD inference in a new isolated `video/` uv environment, and feed resulting child vocalization segments into the existing ECAPA enrollment evaluation pipeline unchanged.

---

## Technical Context

**Language/Version**: Python 3.10 (new `video/` env); Python 3.11 (existing envs)
**Primary Dependencies**: PyTorch ≥ 1.12 + CUDA, opencv-python, speechbrain (isolated), TalkNet-ASD (cloned), TS-TalkNet (cloned)
**Storage**: NFS at `/orcd/scratch/bcs/001/sensein/sails/BIDS_data/`; caches under `pyannote/video_asd_rttm_cache/`, `pyannote/video_face_cache/`
**Testing**: Manual smoke-test on 2–3 SAILS clips; enrollment pipeline metrics (F1/AUROC/AUPRC) on seen-child split
**Target Platform**: Linux HPC cluster (SLURM), CUDA GPU node
**Project Type**: Research pipeline extension (new diarization frontend + evaluation run)
**Performance Goals**: Full seen-child split (2183 clips) inference completable in one SLURM job array (< 8h wall time); face detection cache prevents redundant compute
**Constraints**: Must not install cross-subsystem; `video/` env must be fully isolated; Providence/Playlogue have no video — frontend must fail gracefully

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Reproducibility | PASS | uv env pinned; seed=42; model checkpoint paths documented; cache invalidation rule added to CLAUDE.md |
| II. Data Integrity | PASS | Uses existing `seen_child_splits/`; no new split creation; no test-set leakage in face detection |
| III. Baseline-first | PASS | Video ASD results compared to existing audio-only baselines (USC-SAIL, VTC, VBx) on same split+protocol |
| IV. Metrics | PASS | Same enrollment pipeline; F1/Precision/Recall/AUROC/AUPRC all reported; per-timepoint breakdown included |
| V. Ablations | PASS | Two model variants (TalkNet vs TS-TalkNet) compared; face-identification strategy documented as ablation dimension |
| VI. Thesis sync | PASS | Results committed to `video_asd_ecapa_enrollment_runs/{model}/`; config.json alongside; no manual transcription |
| VII. Documentation | PASS | `video_asd.py` requires docstrings; CLAUDE.md updated; known limitation (Providence/Playlogue audio-only) documented |

**No violations requiring Complexity Tracking.**

---

## Project Structure

### Documentation (this feature)

```text
specs/001-child-vocal-thesis/
├── plan.md              # This file
├── research.md          # Updated with section 9 (video ASD research)
├── data-model.md        # Updated with VideoRecording, FaceTrack, ASDPrediction entities
├── quickstart.md        # Existing (unchanged)
├── contracts/           # Existing (unchanged)
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code

```text
video/                          # NEW: isolated uv environment for video ASD
├── pyproject.toml              # uv project file; Python 3.10
├── uv.lock                     # pinned lockfile
├── pretrain/                   # model checkpoints (not committed; .gitignore)
│   ├── sfd_face.pth            # S3FD face detector (~90 MB)
│   ├── talknet_asd.model       # TalkNet-ASD checkpoint
│   └── ts_talknet.model        # TS-TalkNet checkpoint
├── TalkNet-ASD/                # cloned repo (submodule or manual clone)
├── TS-TalkNet/                 # cloned repo (submodule or manual clone)
└── run_asd.py                  # thin wrapper script called by subprocess from unified.py

pyannote/
├── video_asd.py                # NEW: TalkNetASDFrontend, TSTalkNetFrontend implementing DiarizationFrontend
├── video_asd_rttm_cache/       # NEW: per-model RTTM cache dirs
│   ├── talknet_asd/
│   └── ts_talknet/
├── video_face_cache/           # NEW: S3FD face track JSON cache (shared across models)
├── unified.py                  # MODIFIED: import video_asd; add 'talknet_asd' and 'ts_talknet' to --diarizer choices
└── [existing files unchanged]

video_asd_ecapa_enrollment_runs/   # NEW: results root (top-level, parallel to existing runs dirs)
├── talknet_asd/
│   ├── config.json
│   ├── child_prototype_stats.csv
│   ├── role_only_*.json
│   └── enroll_*.json / test_*.json
└── ts_talknet/
    └── [same structure]
```

---

## Implementation Phases

### Phase 1: Environment + Face Detection Skeleton

1. Create `video/pyproject.toml` with Python 3.10 and core deps (opencv-python, torch, torchaudio, speechbrain, scipy, scikit-learn, tqdm).
2. Clone TalkNet-ASD and TS-TalkNet repos into `video/`.
3. Write `video/run_asd.py`: reads `--audio_path`, derives video path, runs S3FD face detection, runs ASD model, writes RTTM to `--out_rttm`.
4. Write `pyannote/video_asd.py` with `TalkNetASDFrontend` stub that subprocess-calls `video/run_asd.py` and parses the output RTTM into `List[{"start", "end"}]`.
5. Smoke-test on 2–3 SAILS clips manually.

### Phase 2: TS-TalkNet Frontend + Enrollment Test

1. Extend `video/run_asd.py` to support `--model ts_talknet` with `--ref_audio` argument (reference clip from train split).
2. Implement `TSTalkNetFrontend` in `video_asd.py` that locates one reference clip for the target child from the train split before calling the subprocess.
3. Add `talknet_asd` and `ts_talknet` to `--diarizer` choices in `unified.py`; wire to `video_asd.py`.
4. Run full enrollment evaluation on seen-child split:
   ```bash
   cd pyannote
   python unified.py --diarizer talknet_asd
   python unified.py --diarizer ts_talknet
   ```
5. Commit results to `video_asd_ecapa_enrollment_runs/`.

### Phase 3: CLAUDE.md + Documentation Update

1. Add `video_asd.py`, new cache dirs, and results folders to CLAUDE.md.
2. Document checkpoint download instructions in `video/README.md`.
3. Add video ASD enrollment metrics to the results table in CLAUDE.md.

---

## Complexity Tracking

*(No violations — no extra uv environment beyond what the architecture requires for dependency isolation.)*

---

## Known Limitations (to document in thesis)

- **Audio-only datasets**: Providence and Playlogue have no video files. Video ASD frontends raise `FileNotFoundError` and are not evaluated on those datasets.
- **Child-face identification (TalkNet-ASD)**: Without speaker enrollment, the child is identified as the smallest face in frame — a heuristic that fails in group scenes and when the child is occluded or off-screen.
- **TS-TalkNet enrollment reference**: Uses one training-split audio clip as the reference speaker; performance may vary with reference clip quality.
- **Domain shift**: ASD models (TalkNet, LoCoNet) are pretrained on AVA-ActiveSpeaker (TV/film data). Home video child speech is a different domain; expect lower raw mAP than published AVA numbers.
