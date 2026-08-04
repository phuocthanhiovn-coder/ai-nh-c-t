"""
KIEM DINH KHUNG XUONG END-TO-END:
  anh that -> plan nhieu specialist THAT qua engine -> anh full-res dung size -> QC cham before/after.
Chung minh moi manh khop nhau va chay duoc that.
Chay: python -m ai_engine.integration_test
"""
import os
import glob
import numpy as np
import cv2

cv2.setNumThreads(2)

from ai_engine.orchestrator.engine import run_plan
from ai_engine.orchestrator.registry import REGISTRY
from ai_engine.specialists.qc_scorer.qc import score as qc_score

OUT_DIR = "outputs/integration"

# PIPELINE A — deterministic CHIN (4 con da review ky, dung duoc ngay).
# KHONG gom sky_replace: no AN TOAN (gate nội thất) nhung mask con bleed xanh len toa nha
# o ngoai that -> chua dat chat luong, xem experimental. auto_enhance cung ngoai (pilot washout).
# Kem 1 op rac de test cong an toan (op khong ton tai bi bo qua, khong crash).
PLAN_DETERMINISTIC = [
    {"op": "auto_white_balance", "params": {"wb_strength": 0.8}},
    {"op": "denoise", "params": {"denoise_strength": 0.3, "sharpen_amount": 0.3}},
    {"op": "straighten", "params": {"strength": 1.0}},
    {"op": "grass_green", "params": {"strength": 0.6}},
    {"op": "khong_ton_tai", "params": {}},
]

# PIPELINE B — them auto_enhance (checkpoint pilot CHUA CHIN): dung de kiem tra QC co BAT
# duoc output hong (washout) khong. Ky vong: anh bi bech + QC flag 'washed_out' + needs_human.
PLAN_WITH_MODEL = PLAN_DETERMINISTIC[:4] + [{"op": "auto_enhance", "params": {}}]


def panel(a_u8, b_u8, out_path, max_w=1800):
    h = min(a_u8.shape[0], b_u8.shape[0])
    def rz(x): return cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h))
    canvas = np.hstack([rz(a_u8), rz(b_u8)])
    if canvas.shape[1] > max_w:
        s = max_w / canvas.shape[1]
        canvas = cv2.resize(canvas, (int(canvas.shape[1] * s), int(canvas.shape[0] * s)))
    cv2.putText(canvas, "BEFORE", (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 255), 3)
    cv2.putText(canvas, "AFTER (pipeline)", (canvas.shape[1] // 2 + 20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 255, 60), 3)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 66)
    print("  INTEGRATION TEST: lenh -> plan -> specialist that -> anh full-res")
    print("=" * 66)
    print(f"  Registry co {len(REGISTRY)} op: {', '.join(REGISTRY.keys())}")

    # chon 2 anh: 1 ngoai that (co co/troi) + 1 noi that
    cands = sorted(glob.glob("data/pairs/before/*.jpg"))
    if not cands:
        print("[!] Khong co anh trong data/pairs/before")
        return False
    picks = []
    for pref in ("_ML_1605", "_ML_1563", "20260703-DSC1105", "_ML_1493"):
        m = [c for c in cands if os.path.basename(c).startswith(pref)]
        if m:
            picks.append(m[0])
    if not picks:
        picks = cands[:2]
    picks = picks[:2]

    all_ok = True
    qc_catches_washout = False
    for img_path in picks:
        name = os.path.basename(img_path)
        in_u8 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        in_shape = in_u8.shape[:2]
        qc_before = qc_score(in_u8.astype(np.float32) / 255.0)

        print(f"\n  === {name} ({in_shape[1]}x{in_shape[0]}) ===")
        print(f"    QC goc: overall={qc_before['overall']:.1f} flags={qc_before['flags']}")

        for tag, plan in (("A_deterministic", PLAN_DETERMINISTIC), ("B_with_model", PLAN_WITH_MODEL)):
            out_path = os.path.join(OUT_DIR, f"{tag}_{name}")
            info = run_plan(img_path, plan, out_path)
            out_u8 = cv2.imread(out_path, cv2.IMREAD_COLOR)
            size_ok = (info["in_shape"] == info["out_shape"] == out_u8.shape[:2])
            all_ok = all_ok and size_ok
            qc_after = qc_score(out_u8.astype(np.float32) / 255.0)
            panel(in_u8, out_u8, os.path.join(OUT_DIR, f"compare_{tag}_{name}"))
            # ⭐ 04/08 (ra soat vong 7) — TIEU CHI CU BI NGUOC.
            # Ban cu dat `qc_catches_washout = True` khi CHINH OUTPUT CUA MODEL bi QC
            # gan co washed_out, roi bat buoc co do phai True thi bai kiem moi PASS.
            # Tuc bai kiem nay chi PASS KHI MODEL LAM HONG ANH — no do vinh vien ke tu
            # khi model tot len (bao cao "QC bat washout: FAIL" hien nay chinh la no).
            # Do nham hai viec vao mot: (1) noi day chuoi co chay khong, (2) QC co nhay
            # khong. Nay tach: phan (1) van dung anh that o day; phan (2) kiem bang mot
            # anh BECH CO Y o duoi, khong phu thuoc checkpoint dang cai.
            pass
            print(f"    [{tag}] op={info['applied']}")
            print(f"        size {'OK' if size_ok else 'SAI'} | QC {qc_before['overall']:.1f}->{qc_after['overall']:.1f} "
                  f"| flags={qc_after['flags']} | needs_human={qc_after['needs_human']}")

    # ⭐ 04/08 (ra soat vong 7) — DO NHAY CUA QC bang ANH BECH CO Y, khong bang
    # output cua model production. Anh tong hop nay BIET TRUOC la bech (nen len 0.80,
    # bien do chi con 0.15) nen QC BAT BUOC phai bat; va no khong doi khi model doi,
    # nen bai kiem khong con tu hong moi lan model tot len.
    qc_catches_washout = False
    try:
        base = cv2.imread(SAMPLES[0] if isinstance(SAMPLES, (list, tuple)) else SAMPLES,
                          cv2.IMREAD_COLOR)
    except Exception:
        base = None
    if base is None:
        base = (np.random.RandomState(0).rand(256, 384, 3) * 255).astype(np.uint8)
    bech = np.clip(base.astype(np.float32) / 255.0 * 0.15 + 0.80, 0.0, 1.0)
    qc_bech = qc_score(bech)
    qc_catches_washout = ("washed_out" in qc_bech["flags"]) or qc_bech["needs_human"]

    print("-" * 66)
    print(f"  Khung xuong wiring : {'PASS - moi manh khop, chay end-to-end, size giu nguyen' if all_ok else 'FAIL'}")
    print(f"  QC bat anh bech    : {'PASS' if qc_catches_washout else 'FAIL - QC van bi lua!'}"
          f"  (anh tong hop biet truoc la bech: flags={qc_bech['flags']}, "
          f"needs_human={qc_bech['needs_human']}, overall={qc_bech['overall']:.1f})")
    return all_ok and qc_catches_washout


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
