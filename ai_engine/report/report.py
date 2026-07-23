# -*- coding: utf-8 -*-
"""Tạo báo cáo giao khách: contact-sheet TRƯỚC/SAU (JPEG) + trang HTML self-contained.

Chỉ dùng cv2 + numpy + thư viện chuẩn. Không đụng ảnh master — chỉ đọc và ghép preview.
"""
import argparse
import base64
import html
import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

# console Windows mặc định cp1252 — in tiếng Việt sẽ crash nếu không vá
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_LABEL_H = 36          # chiều cao dải nhãn BEFORE/AFTER
_GAP = 12              # khoảng cách giữa các ô
_BG = (245, 245, 245)  # nền xám nhạt (BGR)


def _read(path):
    """Đọc ảnh BGR, trả None nếu path rỗng/không đọc được (kèm cảnh báo)."""
    if not path or not os.path.isfile(path):
        print(f"[WARN] bỏ qua: không thấy file '{path}'")
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] bỏ qua: cv2 không đọc được '{path}'")
    return img


def _resize_w(img, w):
    """Resize giữ tỉ lệ về bề rộng w."""
    h = max(1, round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _labeled(img, text, w):
    """Ảnh resize về bề rộng w + dải nhãn màu phía trên."""
    img = _resize_w(img, w)
    bar = np.zeros((_LABEL_H, w, 3), np.uint8)
    bar[:] = (60, 60, 60) if text == "BEFORE" else (90, 140, 30)
    cv2.putText(bar, text, (10, _LABEL_H - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def _pair_tile(before, after, cell_w):
    """1 ô = [BEFORE | AFTER] cạnh nhau, cao bằng nhau."""
    b = _labeled(before, "BEFORE", cell_w)
    a = _labeled(after, "AFTER", cell_w)
    h = max(b.shape[0], a.shape[0])
    b = _pad_to(b, h, b.shape[1])
    a = _pad_to(a, h, a.shape[1])
    gap = np.full((h, _GAP, 3), _BG, np.uint8)
    return np.hstack([b, gap, a])


def _pad_to(img, h, w):
    """Đệm nền xám cho đủ h x w (canh trên-trái)."""
    out = np.full((h, w, 3), _BG, np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out


def make_contact_sheet(pairs: list, out_path: str, cols: int = 2, cell_w: int = 700) -> str:
    """pairs = list các (before_path, after_path). Ghép lưới: mỗi ô 1 cặp
    [BEFORE | AFTER] có nhãn, cols cặp mỗi hàng. Lưu JPEG q92 ra out_path.
    Cặp nào ảnh hỏng/None thì bỏ qua + cảnh báo. Trả out_path."""
    tiles = []
    for bp, ap in pairs:
        b, a = _read(bp), _read(ap)
        if b is None or a is None:
            continue
        tiles.append(_pair_tile(b, a, cell_w))
    if not tiles:
        raise ValueError("Không có cặp ảnh hợp lệ nào để ghép contact-sheet")

    cols = max(1, cols)
    tile_w = max(t.shape[1] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i : i + cols]
        h = max(t.shape[0] for t in row)
        row = [_pad_to(t, h, tile_w) for t in row]
        # hàng cuối thiếu ô thì đệm nền cho đủ bề rộng
        while len(row) < cols and len(tiles) > cols:
            row.append(np.full((h, tile_w, 3), _BG, np.uint8))
        gap = np.full((h, _GAP, 3), _BG, np.uint8)
        merged = row[0]
        for t in row[1:]:
            merged = np.hstack([merged, gap, t])
        rows.append(merged)

    sheet_w = max(r.shape[1] for r in rows)
    vgap = np.full((_GAP, sheet_w, 3), _BG, np.uint8)
    parts = []
    for r in rows:
        parts.append(_pad_to(r, r.shape[0], sheet_w))
        parts.append(vgap)
    sheet = np.vstack(parts[:-1])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if not cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise IOError(f"Ghi JPEG thất bại: {out_path}")
    return out_path


def _b64_jpeg(img, max_w=1400, q=85):
    """Encode ảnh -> data URI JPEG (thu nhỏ về max_w để HTML nhẹ)."""
    if img.shape[1] > max_w:
        img = _resize_w(img, max_w)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise IOError("cv2.imencode JPEG thất bại")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


_HTML_HEAD = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;background:#1b1b1f;color:#eee;margin:0;padding:24px}}
 h1{{font-size:22px;font-weight:600}} .sub{{color:#999;font-size:13px;margin-bottom:24px}}
 .pair{{max-width:1100px;margin:0 auto 40px}}
 .pair h2{{font-size:14px;font-weight:500;color:#bbb;margin:0 0 8px}}
 .cmp{{position:relative;overflow:hidden;border-radius:8px;user-select:none}}
 .cmp img{{display:block;width:100%;height:auto;pointer-events:none}}
 .cmp .after{{position:absolute;inset:0;clip-path:inset(0 0 0 50%)}}
 .cmp .bar{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;box-shadow:0 0 6px rgba(0,0,0,.7)}}
 .cmp input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}
 .tag{{position:absolute;top:10px;padding:3px 10px;border-radius:4px;font-size:12px;background:rgba(0,0,0,.55);color:#fff}}
 .tag.b{{left:10px}} .tag.a{{right:10px}}
</style></head><body>
<h1>{title}</h1><div class="sub">{count} cặp ảnh · kéo thanh trượt để so sánh TRƯỚC / SAU</div>
"""

_HTML_PAIR = """<div class="pair"><h2>{name}</h2>
 <div class="cmp">
  <img src="{before}" alt="before">
  <img class="after" src="{after}" alt="after">
  <div class="bar"></div>
  <span class="tag b">TRƯỚC</span><span class="tag a">SAU</span>
  <input type="range" min="0" max="100" value="50"
   oninput="var p=this.parentNode;p.querySelector('.after').style.clipPath='inset(0 0 0 '+this.value+'%)';p.querySelector('.bar').style.left=this.value+'%';">
 </div></div>
"""


def make_html_report(pairs: list, out_path: str, title: str = "Ảnh chỉnh AI") -> str:
    """Tạo 1 file HTML DUY NHẤT nhúng ảnh base64 data URI, mỗi cặp 1 slider
    kéo so sánh trước/sau. Mở trình duyệt xem ngay, không cần file ngoài. Trả out_path."""
    blocks = []
    for bp, ap in pairs:
        b, a = _read(bp), _read(ap)
        if b is None or a is None:
            continue
        if a.shape[:2] != b.shape[:2]:  # slider cần 2 ảnh chồng khít
            a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
        blocks.append(_HTML_PAIR.format(
            name=html.escape(os.path.basename(bp)),
            before=_b64_jpeg(b), after=_b64_jpeg(a)))
    if not blocks:
        raise ValueError("Không có cặp ảnh hợp lệ nào cho HTML report")

    doc = _HTML_HEAD.format(title=html.escape(title), count=len(blocks)) \
        + "".join(blocks) + "</body></html>\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def _collect_pairs(before_dir, after_dir, limit=None):
    """Ghép cặp cùng tên file giữa 2 thư mục."""
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    names = sorted(n for n in os.listdir(before_dir) if n.lower().endswith(exts))
    pairs = []
    for n in names:
        ap = os.path.join(after_dir, n)
        if os.path.isfile(ap):
            pairs.append((os.path.join(before_dir, n), ap))
        else:
            print(f"[WARN] không có after cùng tên cho '{n}'")
        if limit and len(pairs) >= limit:
            break
    return pairs


def _self_test():
    """Test nhanh trên data/pairs: tạo 2 file, mở lại kiểm tra, in TASK DONE."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bdir = os.path.join(root, "data", "pairs", "before")
    adir = os.path.join(root, "data", "pairs", "after")
    pairs = _collect_pairs(bdir, adir, limit=3) if os.path.isdir(bdir) else []
    if not pairs:
        print("[WARN] không có data/pairs — test bỏ qua data thật")
        sys.exit(1)

    out_jpg = os.path.join(root, "outputs", "report_test.jpg")
    out_html = os.path.join(root, "outputs", "report_test.html")
    make_contact_sheet(pairs, out_jpg, cols=1)
    make_html_report(pairs, out_html, title="Test báo cáo AI")

    assert os.path.isfile(out_jpg) and os.path.isfile(out_html)
    sheet = cv2.imread(out_jpg)
    assert sheet is not None, "contact-sheet không mở lại được bằng cv2"
    with open(out_html, encoding="utf-8") as f:
        html_txt = f.read()
    assert "data:image" in html_txt, "HTML thiếu ảnh base64 nhúng"

    print(f"contact-sheet: {out_jpg}  {sheet.shape[1]}x{sheet.shape[0]}px  "
          f"{os.path.getsize(out_jpg)/1024:.0f} KB")
    print(f"html: {out_html}  {os.path.getsize(out_html)/1024:.0f} KB  ({len(pairs)} cặp)")
    print("TASK DONE")


def main():
    p = argparse.ArgumentParser(description="Tạo contact-sheet + HTML báo cáo trước/sau")
    p.add_argument("--test", action="store_true", help="tự test trên data/pairs")
    p.add_argument("--before", help="thư mục ảnh gốc")
    p.add_argument("--after", help="thư mục ảnh đã chỉnh (cùng tên file)")
    p.add_argument("--out-jpg", default="outputs/report.jpg")
    p.add_argument("--out-html", default="outputs/report.html")
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--title", default="Ảnh chỉnh AI")
    args = p.parse_args()

    if args.test:
        _self_test()
        return
    if not (args.before and args.after):
        p.error("cần --before và --after (hoặc --test)")
    pairs = _collect_pairs(args.before, args.after)
    if not pairs:
        sys.exit("Không tìm được cặp ảnh nào cùng tên")
    print(f"{len(pairs)} cặp ảnh")
    print("contact-sheet:", make_contact_sheet(pairs, args.out_jpg, cols=args.cols))
    print("html:", make_html_report(pairs, args.out_html, title=args.title))


if __name__ == "__main__":
    main()
