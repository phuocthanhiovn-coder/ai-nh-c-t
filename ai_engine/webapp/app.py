# -*- coding: utf-8 -*-
"""Web UI local kéo-thả — demo trước/sau cho khách hàng.

Chạy server thật :  python -m ai_engine.webapp.app        (http://127.0.0.1:8760)
Tự test (không treo):  python -m ai_engine.webapp.app --selftest

Luồng xử lý 1 ảnh: upload -> cv2.imdecode -> cap chiều rộng ~1600px
-> apply_fullres (operator CH_C) -> grade_auto -> JPEG q95 -> data URL.
Model nạp ĐÚNG 1 lần lúc khởi động (global, CPU).
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request

cv2.setNumThreads(2)
torch.set_num_threads(2)

from ai_engine.specialists.auto_enhance.gpu.render_delivery import apply_fullres
from ai_engine.specialists.auto_enhance.gpu.finish_grade import grade_auto
from ai_engine.specialists.auto_enhance.bracket_deliver import load_model

ROOT = Path(__file__).resolve().parents[2]
CKPT = str(ROOT / "checkpoints" / "gpu" / "CH_C.pt")
MAX_W = 1600          # cap chiều rộng xử lý cho nhanh, vẫn nét khi xem web
JPEG_Q = 95

app = Flask(__name__)

# Nạp model 1 LẦN lúc khởi động (không nạp lại mỗi request)
MODEL, DEVICE = load_model(CKPT, device=torch.device("cpu"))


PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoHDR — Demo chỉnh ảnh BĐS</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #14161a; color: #e8e8e8; font-family: "Segoe UI", Arial, sans-serif;
         min-height: 100vh; padding: 24px; }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
  .sub { color: #9aa0a8; font-size: 13px; margin-bottom: 20px; }
  #drop { border: 2px dashed #3a4048; border-radius: 12px; padding: 48px 24px;
          text-align: center; cursor: pointer; transition: border-color .15s, background .15s;
          max-width: 960px; margin: 0 auto 20px; }
  #drop.hover { border-color: #5b8def; background: #1b2230; }
  #drop p { color: #9aa0a8; font-size: 15px; }
  #drop b { color: #e8e8e8; }
  #file { display: none; }
  #status { text-align: center; color: #f0b64a; font-size: 14px; min-height: 20px;
            margin-bottom: 16px; }
  #status.err { color: #ef6a6a; }
  .wrap { max-width: 960px; margin: 0 auto; }
  #cmp { position: relative; display: none; user-select: none; border-radius: 10px;
         overflow: hidden; background: #000; }
  #cmp img { display: block; width: 100%; height: auto; pointer-events: none; }
  #imgBefore { position: absolute; top: 0; left: 0; clip-path: inset(0 50% 0 0); }
  #bar { position: absolute; top: 0; bottom: 0; left: 50%; width: 2px; background: #fff;
         transform: translateX(-1px); pointer-events: none; }
  #knob { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
          width: 36px; height: 36px; border-radius: 50%; background: #fff; color: #14161a;
          display: flex; align-items: center; justify-content: center; font-size: 14px;
          font-weight: 700; pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,.5); }
  .tag { position: absolute; top: 10px; padding: 3px 10px; border-radius: 6px;
         background: rgba(0,0,0,.65); font-size: 12px; letter-spacing: .5px; }
  .tag.l { left: 10px; } .tag.r { right: 10px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>AutoHDR — Demo chỉnh ảnh bất động sản</h1>
  <div class="sub">Kéo-thả hoặc chọn ảnh, AI tự chỉnh, kéo thanh trượt để so sánh trước/sau.</div>

  <div id="drop">
    <p><b>Kéo &amp; thả ảnh vào đây</b><br>hoặc bấm để chọn ảnh (JPG/PNG)</p>
    <input id="file" type="file" accept="image/*">
  </div>
  <div id="status"></div>

  <div id="cmp">
    <img id="imgAfter" alt="Ảnh sau khi chỉnh">
    <img id="imgBefore" alt="Ảnh gốc">
    <div id="bar"></div>
    <div id="knob">&#8596;</div>
    <span class="tag l">TRƯỚC</span>
    <span class="tag r">SAU</span>
  </div>
</div>

<script>
const drop = document.getElementById('drop');
const fileI = document.getElementById('file');
const statusEl = document.getElementById('status');
const cmp = document.getElementById('cmp');
const imgB = document.getElementById('imgBefore');
const imgA = document.getElementById('imgAfter');
const bar = document.getElementById('bar');
const knob = document.getElementById('knob');

drop.addEventListener('click', () => fileI.click());
fileI.addEventListener('change', () => { if (fileI.files[0]) send(fileI.files[0]); });
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.add('hover'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.remove('hover'); }));
drop.addEventListener('drop', ev => {
  const f = ev.dataTransfer.files[0];
  if (f) send(f);
});

function setStatus(msg, err) {
  statusEl.textContent = msg;
  statusEl.className = err ? 'err' : '';
}

async function send(file) {
  setStatus('Đang xử lý… (CPU, xin chờ vài giây)');
  cmp.style.display = 'none';
  const fd = new FormData();
  fd.append('image', file);
  try {
    const r = await fetch('/process', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) { setStatus('Lỗi: ' + (j.error || r.status), true); return; }
    imgB.src = j.before;
    imgA.src = j.after;
    cmp.style.display = 'block';
    setPos(0.5);
    setStatus('Xong — kéo thanh trượt để so sánh.');
  } catch (e) {
    setStatus('Lỗi kết nối: ' + e, true);
  }
}

function setPos(frac) {
  frac = Math.min(1, Math.max(0, frac));
  imgB.style.clipPath = 'inset(0 ' + ((1 - frac) * 100) + '% 0 0)';
  bar.style.left = (frac * 100) + '%';
  knob.style.left = (frac * 100) + '%';
}
function posFromEvent(ev) {
  const rect = cmp.getBoundingClientRect();
  const x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
  setPos(x / rect.width);
}
let dragging = false;
cmp.addEventListener('mousedown', ev => { dragging = true; posFromEvent(ev); });
window.addEventListener('mousemove', ev => { if (dragging) posFromEvent(ev); });
window.addEventListener('mouseup', () => dragging = false);
cmp.addEventListener('touchstart', posFromEvent);
cmp.addEventListener('touchmove', posFromEvent);
</script>
</body>
</html>"""


def _to_data_url(bgr):
    """Ảnh BGR uint8 -> data URL JPEG q95."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    if not ok:
        raise RuntimeError("Encode JPEG thất bại")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def process_image(bgr):
    """Pipeline chỉnh 1 ảnh: cap ~1600px -> operator CH_C -> grade tự động."""
    h, w = bgr.shape[:2]
    if w > MAX_W:
        nh = int(round(h * MAX_W / w))
        bgr = cv2.resize(bgr, (MAX_W, nh), interpolation=cv2.INTER_AREA)
    ai = apply_fullres(MODEL, bgr, DEVICE)
    graded = grade_auto(ai)
    return bgr, graded


@app.get("/")
def index():
    """Trang demo kéo-thả."""
    return PAGE


@app.post("/process")
def process():
    """Nhận ảnh upload -> chạy pipeline -> trả data URL trước/sau."""
    f = request.files.get("image") or (next(iter(request.files.values()), None))
    if f is None:
        return jsonify(error="Thiếu file ảnh (field 'image')."), 400
    data = f.read()
    if not data:
        return jsonify(error="File rỗng."), 400
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="File không phải ảnh hợp lệ (JPG/PNG)."), 400
    try:
        before, after = process_image(bgr)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"Xử lý thất bại: {e}"), 500
    return jsonify(
        before=_to_data_url(before),
        after=_to_data_url(after),
        w=int(after.shape[1]),
        h=int(after.shape[0]),
    )


def selftest():
    """Tự test bằng Flask test client — KHÔNG mở server thật, không block."""
    client = app.test_client()

    r = client.get("/")
    html = r.get_data(as_text=True)
    ok_index = r.status_code == 200 and ("kéo" in html or "chọn ảnh" in html)
    print(f"GET /            -> {r.status_code}, chứa chữ hướng dẫn: {ok_index}")

    imgs = sorted((ROOT / "data" / "pairs" / "before").glob("*.jpg"))
    if not imgs:
        print("SELFTEST FAIL: không tìm thấy ảnh trong data/pairs/before/")
        sys.exit(1)
    with open(imgs[0], "rb") as fh:
        r2 = client.post(
            "/process",
            data={"image": (fh, imgs[0].name)},
            content_type="multipart/form-data",
        )
    ok_proc = False
    size_info = ""
    if r2.status_code == 200:
        j = r2.get_json()
        after = j.get("after", "")
        ok_proc = after.startswith("data:image/jpeg;base64,") and len(after) > 1000
        size_info = f"{j.get('w')}x{j.get('h')} px, data URL {len(after)} ký tự"
    print(f"POST /process    -> {r2.status_code}, ảnh trả về: {size_info or 'KHÔNG'}")

    if ok_index and ok_proc:
        print("SELFTEST PASS")
        print("TASK DONE")
    else:
        print("SELFTEST FAIL")
        sys.exit(1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("Demo AutoHDR chạy tại http://127.0.0.1:8760 (Ctrl+C để dừng)")
        app.run(host="127.0.0.1", port=8760, debug=False)
