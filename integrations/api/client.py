"""
integrations/api/client.py
===========================
Command-line client for testing the gesture recognition REST API.

Usage
-----
    python integrations/api/client.py --health
    python integrations/api/client.py --gestures
    python integrations/api/client.py --image path/to/frame.jpg
    python integrations/api/client.py --landmarks path/to/landmarks.json
    python integrations/api/client.py --batch path/to/folder/
    python integrations/api/client.py --host 192.168.1.100 --port 5000 --health
"""

import sys
import json
import base64
import argparse
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


class GestureClient:
    def __init__(self, host: str = "localhost", port: int = 5000):
        self.base = f"http://{host}:{port}"

    def _get(self, endpoint: str) -> dict:
        r = requests.get(f"{self.base}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, endpoint: str, data: dict) -> dict:
        r = requests.post(f"{self.base}{endpoint}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        return self._get("/health")

    def gestures(self) -> dict:
        return self._get("/gestures")

    def predict_image(self, image_path: Path) -> dict:
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        return self._post("/predict", {"image": b64})

    def predict_landmarks(self, lm_path: Path) -> dict:
        with open(lm_path) as f:
            data = json.load(f)
        lms = data["landmarks"]
        vec = [coord for lm in lms for coord in [lm["x"], lm["y"], lm["z"]]]
        return self._post("/predict_landmarks", {"landmarks": vec})

    def batch_predict(self, folder: Path) -> dict:
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        images = sorted(images)[:20]          # limit to 20
        b64s   = [base64.b64encode(p.read_bytes()).decode() for p in images]
        return self._post("/batch_predict", {"images": b64s})

    def gesture_action(self, gesture_id: int) -> dict:
        return self._get(f"/gesture_action/{gesture_id}")


def _print_result(result: dict):
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    gid   = result.get("gesture_id", "?")
    gname = result.get("gesture_name", "?")
    conf  = result.get("confidence", 0)
    action = result.get("action", "")
    alert  = result.get("alert", False)
    iot    = result.get("iot_command", None)
    print(f"\n  Gesture:    [{gid:02d}] {gname}")
    print(f"  Confidence: {conf:.4f}  ({conf*100:.1f}%)")
    print(f"  Action:     {action}")
    print(f"  Alert:      {alert}  |  IoT: {iot}")
    if result.get("top5"):
        print("  Top-5:")
        for t in result["top5"]:
            print(f"    [{t['gesture_id']:02d}] {t['gesture_name']:22s}  {t['confidence']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Gesture API test client")
    ap.add_argument("--host",       default="localhost")
    ap.add_argument("--port",       type=int, default=5000)
    ap.add_argument("--health",     action="store_true")
    ap.add_argument("--gestures",   action="store_true")
    ap.add_argument("--image",      type=Path, default=None)
    ap.add_argument("--landmarks",  type=Path, default=None)
    ap.add_argument("--batch",      type=Path, default=None,
                    help="Folder of images for batch predict")
    ap.add_argument("--action",     type=int, default=None,
                    help="Get clinical action for gesture ID")
    args = ap.parse_args()

    client = GestureClient(args.host, args.port)

    if args.health:
        r = client.health()
        print(f"[Health] {json.dumps(r, indent=2)}")

    if args.gestures:
        r = client.gestures()
        print("\n[Gestures] All 25:")
        for gid, info in r.items():
            print(f"  [{gid:>2}] {info['name']:22s}  alert={info['alert']}  iot={info['iot']}")

    if args.image:
        print(f"\n[Predict] Image: {args.image}")
        _print_result(client.predict_image(args.image))

    if args.landmarks:
        print(f"\n[Predict] Landmarks: {args.landmarks}")
        _print_result(client.predict_landmarks(args.landmarks))

    if args.batch:
        print(f"\n[Batch] Folder: {args.batch}")
        r = client.batch_predict(args.batch)
        for item in r.get("results", []):
            gname = item.get("gesture_name", "?")
            conf  = item.get("confidence", 0)
            print(f"  [{item['index']:3d}] {gname:22s}  {conf:.3f}")

    if args.action is not None:
        r = client.gesture_action(args.action)
        print(f"\n[Action] {json.dumps(r, indent=2)}")

    if not any([args.health, args.gestures, args.image,
                args.landmarks, args.batch, args.action is not None]):
        ap.print_help()


if __name__ == "__main__":
    main()
