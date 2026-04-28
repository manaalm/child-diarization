import argparse
from pathlib import Path

def norm_label(lbl: str) -> str:
    # Playlogue uses ADT (adult) and CHI (child)
    if lbl == "ADULT":
        return "ADT"
    return lbl

def load_rttm(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            rec_id = parts[1]
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            rows.append((rec_id, start, dur, spk, parts))
    return rows

def write_rttm(rows, out_path: Path, min_dur: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        # Sort by start time for cleanliness
        rows = sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3]))
        for rec_id, start, dur, spk, parts in rows:
            if dur < min_dur:
                continue
            parts[7] = norm_label(spk)
            # keep numeric formatting sane
            parts[3] = f"{start:.3f}"
            parts[4] = f"{dur:.3f}"
            f.write(" ".join(parts) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_dur", type=float, default=0.20)  # drop 60–120ms junk
    args = ap.parse_args()

    sys_dir = Path(args.sys_dir)
    out_dir = Path(args.out_dir)

    for p in sorted(sys_dir.glob("*.rttm")):
        rows = load_rttm(p)
        write_rttm(rows, out_dir / p.name, min_dur=args.min_dur)

    print(f"Normalized RTTMs -> {out_dir}")

if __name__ == "__main__":
    main()