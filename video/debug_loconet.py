"""Debug LocoNet inference: print actual scores and diagnose why all segments are empty.

Usage:
    cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
    video/.venv/bin/python video/debug_loconet.py
"""
import sys, os, json
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VIDEO_DIR = os.path.join(_REPO, "video")
_LOCONET_DIR = os.path.join(_VIDEO_DIR, "LoCoNet_ASD")

sys.path.insert(0, _LOCONET_DIR)
sys.path.insert(0, _VIDEO_DIR)

# ── paths ──────────────────────────────────────────────────────────────────
AUDIO = ("/orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/"
         "derivatives/preprocessed/sub-B1L0B3F6F1/ses-02/beh/"
         "sub-B1L0B3F6F1_ses-02_task-generalsocialcommunicationinteraction_run-04_audio.wav")
VIDEO = AUDIO.replace("_audio.wav", "_desc-processed_beh.mp4").replace(
    "derivatives/preprocessed", "derivatives/preprocessed")

CHECKPOINT = os.path.join(_LOCONET_DIR, "pytorch_model.bin")
FACE_CACHE = ("/orcd/scratch/orcd/008/manaal/child-adult-diarization/pyannote/"
              "video_face_cache")

# ── find face cache for this clip — face cache is keyed by VIDEO path ──────
import hashlib
from pathlib import Path
VIDEO = ("/orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/"
         "derivatives/preprocessed/sub-B1L0B3F6F1/ses-02/beh/"
         "sub-B1L0B3F6F1_ses-02_task-generalsocialcommunicationinteraction"
         "_run-04_desc-processed_beh.mp4")

key_vid = hashlib.md5(VIDEO.encode()).hexdigest()
key_aud = hashlib.md5(AUDIO.encode()).hexdigest()
face_json_vid = os.path.join(FACE_CACHE, f"{key_vid}.json")
face_json_aud = os.path.join(FACE_CACHE, f"{key_aud}.json")
face_json = face_json_vid if os.path.exists(face_json_vid) else face_json_aud
print(f"Face cache (video key): {face_json_vid} (exists={os.path.exists(face_json_vid)})")
print(f"Face cache (audio key): {face_json_aud} (exists={os.path.exists(face_json_aud)})")

if os.path.exists(face_json):
    with open(face_json) as f:
        tracks = json.load(f)
    print(f"N face tracks: {len(tracks)}")
    for i, t in enumerate(tracks[:3]):
        n_frames = len(t.get("frames", []))
        print(f"  track {i}: {n_frames} frames, mean_area={t.get('mean_area', '?'):.0f}")

# ── check video ────────────────────────────────────────────────────────────
import cv2
print(f"Video: {VIDEO} (exists={os.path.exists(VIDEO)})")
if os.path.exists(VIDEO):
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {nf} frames @ {fps:.1f} fps = {nf/fps:.1f}s")

# ── load model ────────────────────────────────────────────────────────────
print("\n--- Loading LocoNet ---")
from config_loconet import LoCoNetConfig
from modeling_loconet import loconet as LoconetModel
from loss_multi import lossAV

cfg = LoCoNetConfig.from_pretrained(_LOCONET_DIR)
print(f"num_speakers={cfg.num_speakers}, adjust_attention={cfg.adjust_attention}, av_layers={cfg.av_layers}")

model = LoconetModel(cfg)
state = torch.load(CHECKPOINT, map_location="cpu")
keys = list(state.keys())
print(f"Checkpoint keys (first 10): {keys[:10]}")
print(f"lossAV keys: {[k for k in keys if 'lossAV' in k or 'loss' in k.lower()][:10]}")

# Check if model prefix needs adjusting
has_model_prefix = any(k.startswith("model.") for k in keys)
has_loconet_prefix = any(k.startswith("lossAV") or k.startswith("lossA") or k.startswith("lossV") for k in keys)
print(f"has_model_prefix={has_model_prefix}, has_loconet_prefix={has_loconet_prefix}")

missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Missing keys ({len(missing)}): {missing[:10]}")
print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")

# Check if lossAV FC is initialized or loaded
print(f"\nlossAV.FC.weight norm: {model.lossAV.FC.weight.norm().item():.4f}")
print(f"lossAV.FC.bias: {model.lossAV.FC.bias.detach().cpu().numpy()}")

# ── run inference on one window ───────────────────────────────────────────
print("\n--- Audio features ---")
from torchvggish import vggish_input as _vggish_input
from run_asd import load_audio_16k, _track_to_bboxes, _scores_to_segments

audio_raw, sr = load_audio_16k(AUDIO)
print(f"audio shape: {audio_raw.shape}, sr: {sr}")

cap = cv2.VideoCapture(VIDEO)
fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
n_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
duration_sec = n_vid_frames / fps_vid
print(f"n_vid_frames={n_vid_frames}, fps={fps_vid:.1f}, dur={duration_sec:.1f}s")

mel_all = _vggish_input.waveform_to_examples(audio_raw, sr, n_vid_frames, fps_vid, return_tensor=False)
print(f"mel_all shape: {mel_all.shape}  (expected ~({n_vid_frames*4}, 64))")

# ── load face crops for one track ────────────────────────────────────────
# Fall back: use the cached loconet tracks JSON which already has bboxes embedded
LOCONET_TRACKS = ("/orcd/scratch/orcd/008/manaal/child-adult-diarization/pyannote/"
                  "video_asd_rttm_cache/loconet_ecapa_tracks/"
                  "sub-B1L0B3F6F1_ses-02_task-generalsocialcommunicationinteraction"
                  "_run-04_audio__bbd1e34fcc00a1dae1543a84e1c4e8ec.json")

if os.path.exists(face_json):
    with open(face_json) as f:
        face_tracks = json.load(f)
    print(f"Using face cache: {len(face_tracks)} tracks")
elif os.path.exists(LOCONET_TRACKS):
    with open(LOCONET_TRACKS) as f:
        loconet_track_data = json.load(f)
    print(f"No face cache; using cached LocoNet tracks JSON: {len(loconet_track_data)} tracks")
    # LocoNet tracks have track_id, mean_area, segments — but no bboxes
    # We need bboxes. Let's check the face cache dir for any file with this video
    import glob
    all_face = glob.glob(os.path.join(FACE_CACHE, "*.json"))
    print(f"Total face cache files: {len(all_face)}")
    # For now, skip and note the issue
    print("ERROR: No face cache for this clip and no bboxes in LocoNet tracks JSON.")
    print("Need to run face detection first. Exiting.")
    sys.exit(1)
else:
    print("ERROR: No face cache and no LocoNet tracks JSON. Exiting.")
    sys.exit(1)

track = face_tracks[0]  # first (largest) track
bboxes = _track_to_bboxes(track)
print(f"\nTrack 0: {len(bboxes)} bbox slots, {sum(1 for b in bboxes if b)} non-None")

cap = cv2.VideoCapture(VIDEO)
face_frames = []
fi = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if fi < len(bboxes) and bboxes[fi]:
        x1, y1, x2, y2 = [int(v) for v in bboxes[fi]]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            face_frames.append(cv2.resize(gray, (112, 112)))
        else:
            face_frames.append(np.zeros((112, 112), dtype=np.uint8))
    else:
        face_frames.append(np.zeros((112, 112), dtype=np.uint8))
    fi += 1
cap.release()
print(f"face_frames: {len(face_frames)}, non-zero: {sum(1 for f in face_frames if f.any())}")

# ── run one inference window ──────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

window_sec = min(4.0, duration_sec)
v_s, v_e = 0, min(int(window_sec * fps_vid), len(face_frames))
a_s, a_e = 0, min(v_e * 4, len(mel_all))

print(f"\n--- Window 0: v=[{v_s},{v_e}), a=[{a_s},{a_e}) ---")

audio_feat = torch.FloatTensor(mel_all[a_s:a_e]).unsqueeze(0).unsqueeze(0).to(device)
vis_chunk = np.stack(face_frames[v_s:v_e], axis=0)
visual_feat = torch.FloatTensor(vis_chunk).unsqueeze(0).to(device)

print(f"audio_feat shape: {audio_feat.shape}")
print(f"visual_feat shape: {visual_feat.shape}")

enc = model.model
with torch.no_grad():
    a_emb = enc.forward_audio_frontend(audio_feat)
    print(f"a_emb shape: {a_emb.shape}")

    v_emb = enc.forward_visual_frontend(visual_feat)
    print(f"v_emb shape: {v_emb.shape}")

    T = min(a_emb.shape[1], v_emb.shape[1])
    print(f"T (aligned): {T}")
    a_emb = a_emb[:, :T, :]
    v_emb = v_emb[:, :T, :]

    n_spk = cfg.num_speakers
    a_rep = a_emb.repeat(n_spk, 1, 1)
    v_rep = v_emb.repeat(n_spk, 1, 1)
    print(f"a_rep shape (after repeat): {a_rep.shape}")

    a_rep, v_rep = enc.forward_cross_attention(a_rep, v_rep)
    outsAV = enc.forward_audio_visual_backend(a_rep, v_rep, b=1, s=n_spk)
    print(f"outsAV shape: {outsAV.shape}")

    scores_np = model.lossAV(outsAV)
    print(f"scores_np shape: {scores_np.shape}")
    print(f"scores_np (first T={T}): min={scores_np[:T].min():.4f}, max={scores_np[:T].max():.4f}, mean={scores_np[:T].mean():.4f}")
    print(f"scores above 0.5: {(scores_np[:T] > 0.5).sum()}/{T}")
    print(f"First 20 scores: {scores_np[:20]}")

print("\n=== Done ===")
