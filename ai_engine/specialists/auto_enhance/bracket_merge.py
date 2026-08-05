# -*- coding: utf-8 -*-
"""Trộn bracket phơi sáng (exposure fusion) thành 1 ảnh đủ sáng đều.

Nguyên lý: khách chụp 1 khung bằng nhiều tấm phơi sáng khác nhau (thiếu/đủ/dư).
Module này align (chống rung tay) rồi fusion bằng Mertens (cv2.createMergeMertens)
-> 1 ảnh BGR uint8 giữ chi tiết cả vùng tối lẫn vùng sáng, để bước sau áp
operator màu + grade lên. Chạy thuần CPU, không cần torch/GPU.

CLI:
  python -m ai_engine.specialists.auto_enhance.bracket_merge --test [--sample <path.jpg>]
  python -m ai_engine.specialists.auto_enhance.bracket_merge --folder <dir> --group-size N --out <outdir>
"""

import argparse
import glob
import os
import sys

# Console Windows mặc định cp1252 không in được tiếng Việt -> ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import cv2

cv2.setNumThreads(2)

import numpy as np

# Tag EXIF ExposureTime (0x829A = 33434); IFD Exif con nằm ở 0x8769.
_TAG_EXPOSURE_TIME = 33434
_TAG_EXIF_IFD = 0x8769


def _read_image(path: str) -> np.ndarray:
    """Đọc ảnh BGR bằng cv2, raise lỗi rõ ràng nếu không đọc được."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return img


def _align_homography_veto(imgs: list) -> list:
    """Căn khớp bracket — UY QUYEN cho ai_engine.data_pairing.ingest.

    ⭐ HOP NHAT 03/08 (luat L16). Truoc day file nay giu MOT BAN SAO cua thuat toan
    trong `ingest.align_bracket_images`, va chu thich cu ngay day con ghi ro y dinh
    *"Cung fix voi data_pairing/ingest.py de 2 duong gop hanh xu giong nhau"*.
    Thuc te chung DA LECH NHAU — do tren cung bracket, cung anh:
        fp104505   ingest 4/5 tam  |  bracket_merge 5/5
        fp104525   ingest 5/5 tam  |  bracket_merge 2/5   <-- duong GIAO HANG mat 3 tam
    Moi lan va lai phai nho va hai cho, va lan nao cung quen mot cho. Day la khuon
    mau loi lon nhat cua du an (tasks/29, luat L16).
    Nay chi con MOT ban cai dat: sua mot lan la ca hai duong doi theo.

    Ban duy nhat nam o `ingest.align_bracket_images`, gom:
      - chuan hoa do sang truoc khi SO SANH (tam toi luma ~1.7 van do duoc)
      - cong HINH HOC tach DICH (AlignMTB sua duoc) khoi BIEN DANG (khong ai sua duoc)
      - nguong hieu chinh tren 53 bracket that: giu 49/49 tam chan may, bat dung
        10/10 ca xoay 0.5do / phong 1.005 / dich 60px
    """
    # import trong ham: ingest keo theo rawpy — khong bat duong giao hang phai nap
    # no ngay luc import module.
    from ai_engine.data_pairing.ingest import align_bracket_images
    return align_bracket_images(imgs)


def merge_brackets(paths: list, align: bool = True) -> np.ndarray:
    """Trộn list ảnh cùng khung khác phơi sáng -> 1 ảnh BGR uint8 fusion.

    1. Đọc bằng cv2 (BGR); khác kích thước -> resize tất cả về kích thước ảnh đầu.
    2. align=True: uỷ quyền `ingest.align_bracket_images` — similarity 4-DOF +
       cổng HÌNH HỌC (biến dạng <= 3px, dịch <= 15px) + AlignMTB. Hàm đó tự chạy
       AlignMTB ở bước cuối, ở đây KHÔNG chạy lại (04/08: MTB không idempotent).
    3. Fusion Mertens -> float [0,1] -> x255 clip uint8. Còn 1 frame (bị veto
       hết) -> trả thẳng frame đó.
    """
    if not paths or len(paths) < 2:
        raise ValueError(
            f"Bracket cần >= 2 ảnh, nhận được {0 if not paths else len(paths)}"
        )

    imgs = [_read_image(p) for p in paths]

    # Đồng nhất kích thước theo ảnh đầu tiên.
    h0, w0 = imgs[0].shape[:2]
    imgs = [
        img if img.shape[:2] == (h0, w0)
        else cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
        for img in imgs
    ]

    if align:
        # FIX 04/08: BO luot AlignMTB o day. Sau khi hop nhat (03/08),
        # _align_homography_veto uy quyen cho ingest.align_bracket_images — ma ham
        # do DA chay AlignMTB o buoc cuoi. Giu them luot nay la chay HAI LAN.
        # Khong chi ton thoi gian: MTB lam viec tren ban do nguong-trung-vi, neu
        # luot 1 da dich tam nao >= 1 pixel thi ban do luot 2 KHAC di, nen no co
        # the tinh ra dich khac 0 va them troi duoi-pixel. MTB KHONG idempotent.
        imgs = _align_homography_veto(imgs)

    if len(imgs) == 1:
        print("[bracket_merge] CẢNH BÁO: các frame khác bị veto (lệch quá) — dùng 1 frame tham chiếu.")
        return imgs[0].copy()

    fused = cv2.createMergeMertens().process(imgs)  # float32, xấp xỉ [0,1]
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


def _read_exposure_time(path: str):
    """Đọc ExposureTime (giây, float) từ EXIF bằng Pillow. Không có -> None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            val = exif.get(_TAG_EXPOSURE_TIME)
            if val is None:
                val = exif.get_ifd(_TAG_EXIF_IFD).get(_TAG_EXPOSURE_TIME)
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

# 04/08 (ra soat vong 7) — nua sau cua loi D10 trong tasks/30, chua tung duoc sua.
# `_read_image` dung cv2.imread nen KHONG doc duoc RAW. Truoc day file .arw/.cr3 chi
# don gian khong khop `_IMG_EXTS` => bi BO QUA KHONG MOT LOI NAO. Hau qua: chi mot
# thu muc RAW vao day thi hoac ra "Folder khong co anh" (thong bao sai huong, nguoi
# dung di tim file bi mat) hoac te hon — mot vai JPEG xem truoc lot vao va duoc gop
# nhu bracket that. Duong GIAO HANG khong develop RAW la mot QUYET DINH, va quyet
# dinh do phai duoc NOI RA chu khong duoc im lang.
_RAW_EXTS = (".arw", ".cr3", ".cr2", ".nef", ".dng", ".raf", ".orf", ".rw2")


def group_brackets(folder: str, group_size: int = 0) -> list:
    """Gom ảnh trong folder thành các bracket (mỗi bracket = list path, sort tên).

    - group_size > 0: cứ group_size ảnh liên tiếp = 1 bracket.
    - group_size == 0: đọc EXIF ExposureTime, gom chuỗi ảnh liên tiếp có phơi sáng
      TĂNG DẦN, reset khi giảm (1 chu kỳ = 1 bracket). Không đọc được EXIF ->
      fallback group_size=3 + in cảnh báo.
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Folder không tồn tại: {folder}")

    _tat_ca = [p for p in glob.glob(os.path.join(folder, "*")) if os.path.isfile(p)]
    paths = sorted(p for p in _tat_ca if p.lower().endswith(_IMG_EXTS))
    _raw = sorted(p for p in _tat_ca if p.lower().endswith(_RAW_EXTS))
    if _raw:
        # Noi RO thay vi bo qua im lang (xem chu thich o `_RAW_EXTS`).
        raise ValueError(
            f"Folder có {len(_raw)} file RAW ({os.path.basename(_raw[0])}...) mà đường "
            f"giao hàng này KHÔNG develop RAW được — `_read_image` dùng cv2.imread. "
            f"Hãy develop RAW trước bằng `ai_engine.data_pairing.ingest."
            f"process_cr3_to_rgb(path, no_auto_bright=True)` (cờ no_auto_bright BẮT "
            f"BUỘC cho bracket: thiếu nó là san bằng phơi sáng, mất dải sáng cửa sổ), "
            f"rồi trỏ lại folder ảnh đã develop."
        )
    if not paths:
        raise ValueError(f"Folder không có ảnh: {folder}")

    if group_size == 0:
        exposures = [_read_exposure_time(p) for p in paths]
        if any(e is None for e in exposures):
            print(
                "[bracket_merge] CẢNH BÁO: không đọc được EXIF ExposureTime của "
                "toàn bộ ảnh -> fallback gom mỗi 3 ảnh liên tiếp = 1 bracket."
            )
            group_size = 3
        else:
            # ⭐ 04/08 (ra soat toan bo) — KHONG duoc gia dinh phoi sang LUON TANG.
            # Ban cu cat chu ky moi moi khi exposures[i] <= exposures[i-1], tuc ngam
            # gia dinh may LUON ghi bracket theo thu tu TANG DAN. Rat nhieu may ghi
            # GIAM DAN (sang -> toi). Voi day giam dan thi MOI buoc deu bi coi la "chu
            # ky moi" => moi tam thanh mot bracket rieng => `merge_exposures` nhan 1
            # tam va tra thang tam do, tuc KHONG CON GOP BRACKET NUA. Anh giao ra mat
            # sach dai sang cua so — dung trieu chung ma ca du an di chua mot thang.
            # Nay xac dinh CHIEU tu buoc dau cua moi nhom, chi cat khi chieu DAO NGUOC.
            groups = []
            cur = [paths[0]]
            chieu = 0                      # 0 = chua biet, +1 = tang, -1 = giam
            for i in range(1, len(paths)):
                d = exposures[i] - exposures[i - 1]
                if d == 0:
                    cur.append(paths[i])   # phoi sang trung nhau -> van cung chu ky
                    continue
                buoc = 1 if d > 0 else -1
                if chieu == 0:
                    chieu = buoc           # nhom nay di theo chieu nay
                    cur.append(paths[i])
                elif buoc == chieu:
                    cur.append(paths[i])   # con di dung chieu -> cung chu ky
                else:
                    groups.append(cur)     # dao chieu -> chu ky moi
                    cur = [paths[i]]
                    chieu = 0
            groups.append(cur)
            # ⚠️ CHUA XU LY: kieu ghi 0 / -2 / +2 (mot so may dat EV giua truoc). Day do
            # co HAI lan dao chieu trong mot bracket nen se bi cat sai. Neu gap job kieu
            # do thi truyen group_size tuong minh thay vi de =0.
            _co = sorted({len(g) for g in groups})
            if len(_co) > 1:
                print(f"[bracket] CANH BAO: cac nhom co so tam KHONG DEU {_co} — co the "
                      f"may ghi EV kieu 0/-2/+2 (hai lan dao chieu). Neu vay hay truyen "
                      f"group_size tuong minh.", flush=True)
            return groups

    return [paths[i:i + group_size] for i in range(0, len(paths), group_size)]


# ----------------------------- CLI + tự test ------------------------------


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Vẽ nhãn chữ lên góc trên-trái ảnh (nền đen chữ trắng cho dễ đọc)."""
    out = img.copy()
    scale = max(0.6, out.shape[1] / 800.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 20, th + base + 20), (0, 0, 0), -1)
    cv2.putText(out, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def _brightness_stats(img: np.ndarray) -> dict:
    """Thống kê độ sáng: std (contrast), % pixel cháy trắng / chết đen."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n = gray.size
    return {
        "std": float(gray.std()),
        "blown_pct": 100.0 * int((gray >= 250).sum()) / n,
        "crushed_pct": 100.0 * int((gray <= 5).sum()) / n,
    }


def _run_test(sample: str) -> None:
    """Tạo bracket tổng hợp (tối/vừa/sáng) từ 1 ảnh, merge, lưu dải so sánh."""
    if sample is None:
        candidates = sorted(glob.glob(os.path.join("data", "pairs", "before", "*.jpg")))
        if not candidates:
            raise ValueError("Không tìm thấy ảnh mẫu trong data/pairs/before/*.jpg "
                             "— hãy truyền --sample <path.jpg>.")
        sample = candidates[0]
    print(f"[bracket_merge] Ảnh mẫu: {sample}")

    base = _read_image(sample)
    f = base.astype(np.float32)
    dark = np.clip(f * 0.35, 0, 255).astype(np.uint8)
    mid = base.copy()
    bright = np.clip(f * 2.6, 0, 255).astype(np.uint8)

    # Ghi 3 bản phơi sáng ra file tạm để đi đúng đường merge_brackets(paths).
    tmp_dir = os.path.join("outputs", "_bracket_test_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    trio = []
    for name, img in (("dark", dark), ("mid", mid), ("bright", bright)):
        p = os.path.join(tmp_dir, f"{name}.png")
        cv2.imwrite(p, img)
        trio.append(p)

    merged = merge_brackets(trio, align=True)

    strip = cv2.hconcat([
        _label(dark, "TOI x0.35"),
        _label(mid, "VUA x1.0"),
        _label(bright, "SANG x2.6"),
        _label(merged, "DA MERGE"),
    ])
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "bracket_test.jpg")
    if not cv2.imwrite(out_path, strip, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise IOError(f"Không ghi được {out_path}")

    s_mid = _brightness_stats(mid)
    s_mrg = _brightness_stats(merged)
    print(f"[bracket_merge] Shape ảnh merge: {merged.shape}")
    print(f"[bracket_merge] Contrast (std độ sáng): VUA={s_mid['std']:.2f}  "
          f"MERGE={s_mrg['std']:.2f}  (lệch {s_mrg['std'] - s_mid['std']:+.2f})")
    print(f"[bracket_merge] Pixel cháy trắng (>=250): VUA={s_mid['blown_pct']:.3f}%  "
          f"MERGE={s_mrg['blown_pct']:.3f}%")
    print(f"[bracket_merge] Pixel chết đen (<=5):    VUA={s_mid['crushed_pct']:.3f}%  "
          f"MERGE={s_mrg['crushed_pct']:.3f}%")
    print(f"[bracket_merge] Đã lưu dải so sánh: {out_path}")


def _run_folder(folder: str, group_size: int, out_dir: str) -> None:
    """Gom bracket trong folder -> merge từng bracket -> lưu merged_NNN.jpg q95."""
    groups = group_brackets(folder, group_size=group_size)
    os.makedirs(out_dir, exist_ok=True)
    n_ok = 0
    for i, grp in enumerate(groups):
        if len(grp) < 2:
            print(f"[bracket_merge] Bỏ qua bracket {i} (chỉ {len(grp)} ảnh): {grp}")
            continue
        merged = merge_brackets(grp, align=True)
        out_path = os.path.join(out_dir, f"merged_{n_ok:03d}.jpg")
        if not cv2.imwrite(out_path, merged, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError(f"Không ghi được {out_path}")
        print(f"[bracket_merge] Bracket {i} ({len(grp)} ảnh) -> {out_path}")
        n_ok += 1
    print(f"[bracket_merge] Xong: {n_ok}/{len(groups)} bracket đã merge -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trộn bracket phơi sáng (Mertens fusion).")
    ap.add_argument("--test", action="store_true", help="Chạy tự test bracket tổng hợp")
    ap.add_argument("--sample", default=None, help="Ảnh mẫu cho --test")
    ap.add_argument("--folder", default=None, help="Folder ảnh cần gom bracket + merge")
    ap.add_argument("--group-size", type=int, default=0,
                    help="Số ảnh mỗi bracket (0 = tự đoán theo EXIF)")
    ap.add_argument("--out", default="outputs", help="Thư mục lưu ảnh merge")
    args = ap.parse_args()

    if args.test:
        _run_test(args.sample)
    elif args.folder:
        _run_folder(args.folder, args.group_size, args.out)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
