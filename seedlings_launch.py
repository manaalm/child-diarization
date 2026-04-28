import requests
import json
import subprocess
from pathlib import Path

BASE = "https://api.databrary.org"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzc2MTcyNTE1LCJpYXQiOjE3NzYxNDI1MTUsImp0aSI6IjNkOTlhNDNmMzA1NTQ1Y2M4YjFiZWRmZWZhYTlhNzYxIiwidXNlcl9pZCI6IjI1NTMyIn0.u0gGiB5vu3Ji917ICogjImtCBZARemKf1SrfIpOK_po"
REFRESH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTc3NjcwMTc0NCwiaWF0IjoxNzc1ODM3NzQ0LCJqdGkiOiIxZWQ0NmEzMWY2NWM0MjM4ODNlOGEzMDhkMDkwMDM1NSIsInVzZXJfaWQiOiIyNTUzMiJ9.3TNUEeJASkq0zqU895zrtHkMnr-bQG8V2RJuPELeyIk"

VOLUMES = {
    670: "14_month",
    694: "15_month",
    695: "16_month",
    696: "17_month",
}

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

SESSION = make_session(ACCESS_TOKEN)
tasks = []

for vol_id, vol_label in VOLUMES.items():
    page = 1
    while True:
        r = SESSION.get(
            f"{BASE}/volumes/{vol_id}/sessions/",
            params={"page": page, "page_size": 500}
        )
        data = r.json()
        sessions = data.get("results", [])
        for sess in sessions:
            if sess.get("accessibleFileCount", 0) > 0:
                tasks.append((vol_id, vol_label, sess))
        if not data.get("next"):
            break
        page += 1
    print(f"  Volume {vol_id} ({vol_label}): {len([t for t in tasks if t[0]==vol_id])} sessions with files")

with open("seedlings_tasks.json", "w") as f:
    json.dump(tasks, f)

print(f"\nTotal: {len(tasks)} tasks")
print(f"Submit with: sbatch --array=0-{len(tasks)-1}%5 seedlings_array.sh")