"""Smoke-test end-to-end cho service shell (Task 12).

Chay: python -m ai_engine.service.test_service
Tu khoi dong server (subprocess), goi /health -> /edit -> /qc, luu ket qua
that vao outputs/service_samples/, roi TU KILL server (khong de gi lang nghe
lai tren 8123).
"""
import glob
import json
import os
import subprocess
import sys
import time

import cv2
import requests

cv2.setNumThreads(2)

BASE_URL = "http://127.0.0.1:8123"
OUT_DIR = "outputs/service_samples"
SAMPLE_IMG = "data/pairs/before/_ML_1605.jpg"
COMMAND = "tăng sáng nhẹ, ấm hơn"


def wait_health(timeout=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    raise TimeoutError("Server khong len sau khi doi /health")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    img_path = SAMPLE_IMG
    if not os.path.exists(img_path):
        cands = sorted(glob.glob("data/pairs/before/*.jpg"))
        if not cands:
            print("[!] Khong co anh nao trong data/pairs/before")
            return False
        img_path = cands[0]

    in_u8 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    in_shape = in_u8.shape[:2]
    print(f"[*] Anh test: {img_path} ({in_shape[1]}x{in_shape[0]})")

    print("[*] Khoi dong server subprocess (python -m ai_engine.service.run_dev) ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "ai_engine.service.run_dev"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    all_ok = True
    try:
        try:
            health = wait_health()
        except TimeoutError as exc:
            out, _ = proc.communicate(timeout=5)
            print(f"[!] {exc}\n--- server log ---\n{out}")
            return False
        print(f"[HEALTH] {health}")
        all_ok = all_ok and health.get("status") == "ok" and health.get("ops", 0) > 0

        # /ops
        r_ops = requests.get(f"{BASE_URL}/ops", timeout=5.0)
        ops_list = r_ops.json()
        print(f"[OPS] {len(ops_list)} op dang ky: {[o['op'] for o in ops_list]}")

        # GET / phai tra ve trang HTML demo (Task 24)
        r_root = requests.get(f"{BASE_URL}/", timeout=5.0)
        is_html = "text/html" in r_root.headers.get("content-type", "")
        has_edit_ref = "/edit" in r_root.text
        print(f"[ROOT] status={r_root.status_code} is_html={is_html} references_edit={has_edit_ref}")
        all_ok = all_ok and r_root.status_code == 200 and is_html and has_edit_ref

        # /edit voi command tieng Viet, planner co the fallback rule-based (khong co ANTICODE_API_KEY)
        with open(img_path, "rb") as f:
            files = {"image": (os.path.basename(img_path), f, "image/jpeg")}
            data = {"command": COMMAND}
            r_edit = requests.post(f"{BASE_URL}/edit", files=files, data=data, timeout=60.0)

        print(f"[EDIT] status={r_edit.status_code}")
        if r_edit.status_code != 200:
            print(f"[!] /edit that bai: {r_edit.text}")
            return False

        plan_applied = r_edit.headers.get("X-Plan-Applied", "")
        qc_overall = r_edit.headers.get("X-QC-Overall", "")
        print(f"[EDIT] X-Plan-Applied={plan_applied}")
        print(f"[EDIT] X-QC-Overall={qc_overall}")

        out_path = os.path.join(OUT_DIR, "edited.jpg")
        with open(out_path, "wb") as f:
            f.write(r_edit.content)

        out_u8 = cv2.imread(out_path, cv2.IMREAD_COLOR)
        out_shape = out_u8.shape[:2] if out_u8 is not None else None
        size_ok = (out_shape == in_shape)
        print(f"[EDIT] out_shape={out_shape} in_shape={in_shape} size_ok={size_ok}")
        all_ok = all_ok and size_ok and bool(plan_applied)

        # /qc tren chinh anh da edit
        with open(out_path, "rb") as f:
            files_qc = {"image": ("edited.jpg", f, "image/jpeg")}
            r_qc = requests.post(f"{BASE_URL}/qc", files=files_qc, timeout=30.0)
        print(f"[QC] status={r_qc.status_code}")
        qc_json = r_qc.json()
        score_keys = ["blur_score", "exposure_score", "tilt_score", "color_cast_score",
                      "noise_score", "washout_score"]
        has_6_scores = all(k in qc_json for k in score_keys)
        print(f"[QC] overall={qc_json.get('overall')} flags={qc_json.get('flags')} "
              f"needs_human={qc_json.get('needs_human')} has_6_scores={has_6_scores}")
        all_ok = all_ok and r_qc.status_code == 200 and has_6_scores

        # test loi: anh rac -> phai 400, khong 500
        bad_files = {"image": ("bad.jpg", b"not-an-image-at-all", "image/jpeg")}
        r_bad = requests.post(f"{BASE_URL}/edit", files=bad_files, timeout=10.0)
        print(f"[BAD-IMAGE] status={r_bad.status_code} (ky vong 400)")
        all_ok = all_ok and r_bad.status_code == 400

        # test op rac trong plan JSON truc tiep -> khong duoc crash (bo qua op)
        with open(img_path, "rb") as f:
            files2 = {"image": (os.path.basename(img_path), f, "image/jpeg")}
            bad_plan = {"plan": [{"op": "khong_ton_tai", "params": {}},
                                  {"op": "auto_white_balance", "params": {}}]}
            data2 = {"plan": json.dumps(bad_plan)}
            r_badop = requests.post(f"{BASE_URL}/edit", files=files2, data=data2, timeout=30.0)
        print(f"[BAD-OP-IN-PLAN] status={r_badop.status_code} "
              f"X-Plan-Applied={r_badop.headers.get('X-Plan-Applied')}")
        all_ok = all_ok and r_badop.status_code == 200

    finally:
        print("[*] Dang tat server ...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        print(f"[*] Server da tat (returncode={proc.returncode})")

    time.sleep(1.0)
    netstat = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    related = [ln for ln in netstat.splitlines() if ":8123" in ln]
    still_listening = [ln for ln in related if "LISTENING" in ln]
    print(f"[NETSTAT] dong lien quan 8123 sau khi kill: {related}")
    print(f"[NETSTAT] dong con LISTENING tren 8123: {still_listening} "
          f"(TIME_WAIT la socket client cu, khong phai server con song)")
    all_ok = all_ok and (len(still_listening) == 0)

    print("-" * 60)
    print(f"KET QUA: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
