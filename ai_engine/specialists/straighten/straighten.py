"""
Con "DỌC THẲNG" — nắn phối cảnh (vertical rectify) + méo ống kính nhẹ (k1).
Deterministic 100%, không model. Xem tasks/06-straighten-verticals.md.

Hợp đồng operator (Task 05 orchestrator):
    apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray cùng shape
"""

import cv2
import numpy as np

cv2.setNumThreads(2)

SMALL_DIM = 1024
MAX_ANGLE_DEG = 8.0          # giới hạn an toàn: góc nắn tối đa
ANGLE_FILTER_DEG = 15.0      # chỉ xét đoạn thẳng trong ±15° quanh phương dọc
MIN_LINES = 3                # cần >=3 đường tin cậy mới nắn
MIN_LEN_FRAC = 0.06          # đoạn thẳng phải dài >= 6% cạnh lớn nhất (ảnh nhỏ)


def resize_small(img, target_dim=SMALL_DIM):
    """Downscale ảnh giữ tỷ lệ sao cho cạnh dài nhất = target_dim. Trả về (ảnh_nhỏ, scale)."""
    h, w = img.shape[:2]
    scale = target_dim / max(h, w)
    if scale >= 1.0:
        return img.copy(), 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp), scale


def undistort_k1(img_bgr, k1):
    """Méo radial đơn giản (k1 only), camera matrix giả định fx=fy=w, cx=w/2, cy=h/2.
    Cùng công thức với ai_engine/data_pairing/undistort.py."""
    if k1 == 0.0:
        return img_bgr.copy()
    h, w = img_bgr.shape[:2]
    camera_matrix = np.array([
        [w, 0, w / 2.0],
        [0, w, h / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
    img_in = img_bgr
    if img_in.dtype != np.float32 and img_in.dtype != np.uint8:
        img_in = img_in.astype(np.float32)
    return cv2.undistort(img_in, camera_matrix, dist_coeffs)


def detect_vertical_lines(gray):
    """
    Phát hiện đoạn thẳng gần-dọc trên ảnh gray (nên là bản đã resize <=1024px cạnh dài).
    Trả về list tuple (x1, y1, x2, y2, angle_deg, length):
      - (x1,y1) luôn ở trên (y1<=y2)
      - angle_deg = góc lệch khỏi phương dọc (atan2(dx,dy)*180/pi), 0 = thẳng đứng tuyệt đối
      - đã lọc: đủ dài, góc trong ±ANGLE_FILTER_DEG
    """
    h, w = gray.shape[:2]
    min_len = max(20.0, MIN_LEN_FRAC * max(h, w))

    segments = []
    try:
        lsd = cv2.createLineSegmentDetector(0)
        detected = lsd.detect(gray)[0]
    except Exception:
        detected = None

    if detected is not None and len(detected) > 0:
        for seg in detected:
            x1, y1, x2, y2 = np.asarray(seg).reshape(-1)[:4]
            segments.append((float(x1), float(y1), float(x2), float(y2)))
    else:
        edges = cv2.Canny(gray, 50, 150)
        hough = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                 minLineLength=min_len, maxLineGap=8)
        if hough is not None:
            for seg in hough:
                x1, y1, x2, y2 = seg[0]
                segments.append((float(x1), float(y1), float(x2), float(y2)))

    lines_out = []
    for (x1, y1, x2, y2) in segments:
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_len or dy < 1e-3:
            continue
        angle = float(np.degrees(np.arctan2(dx, dy)))
        if abs(angle) > ANGLE_FILTER_DEG:
            continue
        lines_out.append((x1, y1, x2, y2, angle, length))

    return lines_out


def estimate_rectify_homography(lines, w, h):
    """
    Từ các đường gần-dọc, ước lượng vanishing point dọc -> homography đưa chúng
    về song song trục Y. Giới hạn an toàn: góc nắn tối đa MAX_ANGLE_DEG (8°) và
    dịch chuyển 4 góc ảnh <= tan(MAX_ANGLE_DEG) đường chéo ảnh; vượt ngưỡng hoặc
    <3 đường tin cậy -> trả identity.

    Trả về (H (3x3 float64), angle_deg (góc nghiêng trung vị ước lượng), applied (bool)).
    """
    identity = np.eye(3, dtype=np.float64)

    if len(lines) < MIN_LINES:
        return identity, 0.0, False

    angles = np.array([l[4] for l in lines], dtype=np.float64)
    median_angle = float(np.median(angles))

    if abs(median_angle) > MAX_ANGLE_DEG:
        return identity, median_angle, False

    # Vanishing point dọc: mỗi đường x = x0 + slope*(y - y0), slope = dx/dy
    # => vp_x - slope*vp_y = x0 - slope*y0  (tuyến tính theo (vp_x, vp_y))
    A = []
    b = []
    for (x1, y1, x2, y2, angle, length) in lines:
        dy = y2 - y1
        if abs(dy) < 1e-6:
            continue
        slope = (x2 - x1) / dy
        A.append([1.0, -slope])
        b.append(x1 - slope * y1)

    if len(A) < MIN_LINES:
        return identity, median_angle, False

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    try:
        sol, _residuals, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
        vp_x, vp_y = float(sol[0]), float(sol[1])
    except np.linalg.LinAlgError:
        return identity, median_angle, False

    if not (np.isfinite(vp_x) and np.isfinite(vp_y)):
        return identity, median_angle, False

    cx, cy = w / 2.0, h / 2.0

    # Hệ 2x2: px*vp_x + py*vp_y = -1 ; px*cx + py*cy = 0
    # (VP -> điểm tại vô cực dọc; tâm ảnh giữ nguyên vị trí, w=1)
    M = np.array([[vp_x, vp_y], [cx, cy]], dtype=np.float64)
    rhs = np.array([-1.0, 0.0], dtype=np.float64)
    det = np.linalg.det(M)
    if abs(det) < 1e-6:
        return identity, median_angle, False

    px, py = np.linalg.solve(M, rhs)

    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [px, py, 1.0],
    ], dtype=np.float64)

    # Kiểm tra an toàn trên 4 góc ảnh: dịch chuyển tối đa <= tan(MAX_ANGLE_DEG) * đường chéo
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=np.float64).T
    warped = H @ corners
    if warped[2].min() <= 1e-6:
        return identity, median_angle, False
    warped_xy = warped[:2] / warped[2]
    orig_xy = corners[:2]
    diag = float(np.hypot(w, h))
    shift_frac = np.linalg.norm(warped_xy - orig_xy, axis=0) / diag
    max_shift_frac = np.tan(np.radians(MAX_ANGLE_DEG))
    if shift_frac.max() > max_shift_frac:
        return identity, median_angle, False

    return H, median_angle, True


def analyze(img):
    """
    Chạy toàn bộ pipeline phát hiện + ước lượng ở độ phân giải nhỏ (SMALL_DIM).
    Trả về dict: {angle_deg, applied, num_lines, H_small, scale}.
    Dùng cho cả apply() lẫn báo cáo (run_samples.py) để không lặp logic.
    """
    small, scale = resize_small(img, SMALL_DIM)
    small_u8 = small
    if small_u8.dtype != np.uint8:
        small_u8 = np.clip(small_u8 * 255.0, 0, 255).astype(np.uint8) if small_u8.max() <= 1.0 + 1e-3 else np.clip(small_u8, 0, 255).astype(np.uint8)
    gray_small = cv2.cvtColor(small_u8, cv2.COLOR_BGR2GRAY)
    lines = detect_vertical_lines(gray_small)
    hs, ws = small.shape[:2]
    H_small, angle_deg, applied = estimate_rectify_homography(lines, ws, hs)
    return {
        "angle_deg": angle_deg,
        "applied": applied,
        "num_lines": len(lines),
        "H_small": H_small,
        "scale": scale,
    }


COVER_ZOOM_MAX = 1.08      # zoom bu bien toi da 8%; qua muc -> giam strength / veto
COVER_STRENGTH_DECAY = 0.7  # moi vong giam strength con 70% de thu lai


def _cover_zoom_factor(H_full, w, h):
    """Zoom z quanh TAM anh sao cho khung [0,w]x[0,h] nam TRON trong tu giac
    nguon-hop-le (4 goc anh goc map qua H_full). Tra ve z>=1 hoac None (suy bien).

    VI SAO (22/07/2026): warpPerspective + BORDER_REPLICATE de nguyen -> vung
    thieu nguon o mep thanh SOC KEO (thay o outputs/deliver_v2/edge_stages.jpg,
    panel '4 straighten'). Ghep zoom vao H truoc khi warp -> 1 lan resample duy
    nhat, khong con pixel ngoai nguon; AutoHDR cung crop-zoom nhe khi nan doc.
    """
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    ones = np.ones((4, 1))
    warped = (H_full @ np.hstack([corners, ones]).T)
    if warped[2].min() <= 1e-9:
        return None
    quad = (warped[:2] / warped[2]).T  # 4x2: tu giac nguon hop le trong toa do dich
    c = np.array([w / 2.0, h / 2.0])

    z = 1.0
    for q in corners:  # 4 goc khung dich
        d = q - c
        dist_q = np.linalg.norm(d)
        if dist_q < 1e-9:
            continue
        # giao tia (c -> q) voi 4 canh tu giac: c + t*d = A + u*(B-A)
        t_hit = None
        for i in range(4):
            A, B = quad[i], quad[(i + 1) % 4]
            e = B - A
            denom = d[0] * (-e[1]) - d[1] * (-e[0])
            if abs(denom) < 1e-12:
                continue
            rhs = A - c
            t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / denom
            u = (d[0] * rhs[1] - d[1] * rhs[0]) / denom
            if t > 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
                if t_hit is None or t < t_hit:
                    t_hit = t
        if t_hit is None:
            return None  # tam ngoai tu giac / suy bien
        z = max(z, 1.0 / t_hit)
    return z


def apply(img, params=None):
    """
    apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray cùng shape.
    params:
      - strength: 0..1 (default 1) nội suy giữa identity và homography đầy đủ.
      - k1: méo radial (default 0 = bỏ qua).
    """
    params = params or {}
    strength = float(params.get("strength", 1.0))
    strength = min(max(strength, 0.0), 1.0)
    k1 = float(params.get("k1", 0.0))

    h, w = img.shape[:2]
    working = img
    if k1 != 0.0:
        working = undistort_k1(img, k1)

    diag = analyze(working)

    if not diag["applied"] or strength <= 0.0:
        return working.copy()

    H_base = diag["H_small"]
    scale = diag["scale"]
    S = np.array([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]], dtype=np.float64)
    S_inv = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)

    # Giam dan strength toi khi zoom bu bien <= COVER_ZOOM_MAX (thua thi veto:
    # tha nghieng nhe con hon soc mep / crop qua tay).
    s = strength
    for _ in range(6):
        H_small = H_base
        if s < 1.0:
            H_small = (1.0 - s) * np.eye(3, dtype=np.float64) + s * H_base
            H_small = H_small / H_small[2, 2]
        H_full = S @ H_small @ S_inv
        z = _cover_zoom_factor(H_full, w, h)
        if z is not None and z <= COVER_ZOOM_MAX:
            break
        s *= COVER_STRENGTH_DECAY
    else:
        return working.copy()

    if z > 1.0:
        cx, cy = w / 2.0, h / 2.0
        Z = np.array([[z, 0, cx * (1.0 - z)], [0, z, cy * (1.0 - z)], [0, 0, 1]],
                     dtype=np.float64)
        H_full = Z @ H_full

    result = cv2.warpPerspective(
        working.astype(np.float32, copy=False), H_full.astype(np.float64), (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return result


# ---------------------------------------------------------------------------
# API đơn giản: ước lượng góc nghiêng + xoay lại (rotation thuần, không homography)
# ---------------------------------------------------------------------------

TILT_FILTER_DEG = 20.0   # chỉ xét đoạn thẳng lệch phương dọc trong ±20°
TILT_MIN_LINES = 3       # cần >=3 đường mới tin


def estimate_tilt(bgr):
    """Ước lượng góc nghiêng (độ) từ các đường gần thẳng đứng.
    Canny -> HoughLinesP -> lọc đoạn lệch phương dọc trong ±20° ->
    trung vị độ lệch. Trả góc (độ), + = nghiêng phải. Không đủ đường -> 0.0."""
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.ndim < 2:
        return 0.0

    small, _scale = resize_small(bgr, SMALL_DIM)
    if small.ndim == 3:
        u8 = small
        if u8.dtype != np.uint8:
            u8 = np.clip(u8 * 255.0, 0, 255).astype(np.uint8) if u8.max() <= 1.0 + 1e-3 else np.clip(u8, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = small if small.dtype == np.uint8 else np.clip(small, 0, 255).astype(np.uint8)

    h, w = gray.shape[:2]
    min_len = max(20.0, MIN_LEN_FRAC * max(h, w))
    edges = cv2.Canny(gray, 50, 150)
    hough = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=int(min_len), maxLineGap=8)
    if hough is None:
        return 0.0

    deviations = []
    for seg in hough:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(seg).reshape(-1)[:4]]
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        if np.hypot(dx, dy) < min_len or dy < 1e-3:
            continue
        angle = float(np.degrees(np.arctan2(dx, dy)))  # 0 = dọc tuyệt đối
        if abs(angle) <= TILT_FILTER_DEG:
            deviations.append(angle)

    if len(deviations) < TILT_MIN_LINES:
        return 0.0
    # đo được +a khi ảnh bị xoay CCW +a (quy ước cv2) -> tilt = -a (+ = nghiêng phải)
    return -float(np.median(deviations))


def _rotate_keep_size(img, angle_deg):
    """Xoay quanh tâm, giữ kích thước; phóng nhẹ để lấp góc đen."""
    h, w = img.shape[:2]
    a = abs(np.radians(angle_deg))
    c, s = np.cos(a), np.sin(a)
    cover = max((w * c + h * s) / w, (w * s + h * c) / h)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, cover)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def straighten(bgr, max_deg=8.0):
    """Xoay ảnh để verticals thẳng. |tilt| clamp về max_deg.
    Không đủ đường vertical (tilt=0) hoặc góc quá nhỏ -> trả ảnh gốc."""
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.ndim < 2:
        return bgr

    tilt = estimate_tilt(bgr)
    if abs(tilt) < 0.05:
        return bgr.copy()
    rot = float(np.clip(tilt, -max_deg, max_deg))
    return _rotate_keep_size(bgr, rot)


# ---------------------------------------------------------------------------
# CLI: --test [--sample p] | --in <folder> --out <outdir>
# ---------------------------------------------------------------------------

def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (max(220, 14 * len(text)), 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _run_test(sample_path=None):
    import glob as _glob
    import os

    if sample_path is None:
        cands = sorted(_glob.glob(os.path.join("data", "pairs", "before", "*.jpg")))
        if not cands:
            print("KHONG tim thay anh mau trong data/pairs/before/")
            return 1
        sample_path = cands[0]

    img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"KHONG doc duoc anh: {sample_path}")
        return 1
    print(f"Anh mau: {sample_path} ({img.shape[1]}x{img.shape[0]})")

    base_tilt = estimate_tilt(img)
    print(f"Goc nghieng san co cua anh goc: {base_tilt:+.2f} deg")

    FAKE = 4.0
    tilted = _rotate_keep_size(img, FAKE)  # xoay CCW +4 -> ky vong estimate ~ -4
    est = estimate_tilt(tilted)
    print(f"Xoay gia +{FAKE:.1f} deg -> estimate_tilt = {est:+.2f} deg (ky vong ~ {-FAKE + base_tilt:+.1f})")

    fixed = straighten(tilted)
    residual = estimate_tilt(fixed)
    print(f"Goc con du sau straighten: {residual:+.2f} deg (ky vong gan 0)")

    err = abs(est - (base_tilt - FAKE))
    if err > 1.0:
        print(f"CANH BAO: estimate_tilt lech {err:.2f} deg so voi ky vong (>1 deg) — chua dat.")
    if abs(residual) > 1.0:
        print(f"CANH BAO: goc du {residual:+.2f} deg van > 1 deg — chua dat.")

    # dai ghep [NGHIENG 4 | DA SUA | GOC]
    target_h = 600
    def _rs(im):
        sc = target_h / im.shape[0]
        return cv2.resize(im, (int(round(im.shape[1] * sc)), target_h), interpolation=cv2.INTER_AREA)
    strip = np.hstack([
        _label(_rs(tilted), f"NGHIENG +{FAKE:.0f} deg"),
        _label(_rs(fixed), "DA SUA"),
        _label(_rs(img), "GOC"),
    ])
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "straighten_test.jpg")
    cv2.imwrite(out_path, strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Da luu: {out_path}")
    print("TASK DONE")
    return 0


def _run_folder(in_dir, out_dir):
    import glob as _glob
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(_glob.glob(os.path.join(in_dir, "*.jpg")) +
                   _glob.glob(os.path.join(in_dir, "*.jpeg")) +
                   _glob.glob(os.path.join(in_dir, "*.png")))
    if not paths:
        print(f"Khong co anh trong {in_dir}")
        return 1
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"BO QUA (khong doc duoc): {p}")
            continue
        tilt = estimate_tilt(img)
        out = straighten(img)
        dst = os.path.join(out_dir, os.path.basename(p))
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"{os.path.basename(p)}: tilt={tilt:+.2f} deg -> {dst}")
    print("TASK DONE")
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Straighten verticals (BDS)")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--sample", default=None)
    ap.add_argument("--in", dest="in_dir", default=None)
    ap.add_argument("--out", dest="out_dir", default=None)
    args = ap.parse_args()

    if args.test:
        sys.exit(_run_test(args.sample))
    elif args.in_dir and args.out_dir:
        sys.exit(_run_folder(args.in_dir, args.out_dir))
    else:
        ap.print_help()
