import requests
import json
import time
import sys
from pathlib import Path

BASE = "https://api.databrary.org"
OUT_DIR = Path("/home/manaal/orcd/scratch/child-adult-diarization/seedlings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzc2MTcyNTE1LCJpYXQiOjE3NzYxNDI1MTUsImp0aSI6IjNkOTlhNDNmMzA1NTQ1Y2M4YjFiZWRmZWZhYTlhNzYxIiwidXNlcl9pZCI6IjI1NTMyIn0.u0gGiB5vu3Ji917ICogjImtCBZARemKf1SrfIpOK_po"
REFRESH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTc3NjcwMTc0NCwiaWF0IjoxNzc1ODM3NzQ0LCJqdGkiOiIxZWQ0NmEzMWY2NWM0MjM4ODNlOGEzMDhkMDkwMDM1NSIsInVzZXJfaWQiOiIyNTUzMiJ9.3TNUEeJASkq0zqU895zrtHkMnr-bQG8V2RJuPELeyIk"

def make_session(token):
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Origin": "https://nyu.databrary.org",
        "Referer": "https://nyu.databrary.org/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "sec-fetch-site": "same-site",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
    })
    return s

def refresh_token(s, refresh_tok):
    r = requests.post(
        "https://nyu.databrary.org/api/token/refresh/",
        json={"refresh": refresh_tok}
    )
    r.raise_for_status()
    new_token = r.json()["access"]
    s.headers.update({"Authorization": f"Bearer {new_token}"})
    return new_token

def get_all_pages(s, url):
    results, page = [], 1
    while True:
        r = s.get(url, params={"page": page, "page_size": 500})
        r.raise_for_status()
        data = r.json()
        results.extend(data.get("results", []))
        if not data.get("next"):
            break
        page += 1
    return results

def download_session(vol_id, vol_label, sess):
    SESSION = make_session(ACCESS_TOKEN)
    sid = sess["id"]
    sess_name = sess.get("name", f"session_{sid}")
    downloaded, skipped, failed = 0, 0, 0

    files = get_all_pages(SESSION, f"{BASE}/volumes/{vol_id}/sessions/{sid}/files/")

    for f in files:
        fid = f["id"]
        link_url = f"{BASE}/volumes/{vol_id}/sessions/{sid}/files/{fid}/download-link/"

        link_r = SESSION.get(link_url)
        if link_r.status_code == 401:
            refresh_token(SESSION, REFRESH_TOKEN)
            link_r = SESSION.get(link_url)
        if link_r.status_code != 200:
            print(f"  FAILED link fid={fid}: {link_r.status_code}")
            failed += 1
            continue

        link_data = link_r.json()
        download_url = link_data["downloadUrl"]
        fname = link_data.get("fileName", f"file_{fid}")
        fsize = link_data.get("fileSize", 0)

        dest = OUT_DIR / vol_label / sess_name / fname
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and dest.stat().st_size == fsize:
            skipped += 1
            continue

        print(f"  [{vol_label}] {sess_name}/{fname} ({fsize/1e6:.1f} MB)")
        dl = SESSION.get(download_url, stream=True)
        if dl.status_code != 200:
            print(f"  FAILED download {fname}: {dl.status_code}")
            failed += 1
            continue

        with open(dest, "wb") as fh:
            for chunk in dl.iter_content(chunk_size=65536):
                fh.write(chunk)
        downloaded += 1
        time.sleep(0.05)

    print(f"Session {sid}: {downloaded} done, {skipped} skipped, {failed} failed")

if __name__ == "__main__":
    # Args: vol_id vol_label session_json
    vol_id = int(sys.argv[1])
    vol_label = sys.argv[2]
    sess = json.loads(sys.argv[3])
    download_session(vol_id, vol_label, sess)