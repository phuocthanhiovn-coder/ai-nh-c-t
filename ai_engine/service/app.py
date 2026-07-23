"""FastAPI service shell: upload anh -> chinh theo lenh/plan -> tra ve anh full-res.

Vo HTTP local cho orchestrator (Task 12). Chi bind 127.0.0.1 (xem run_dev.py).
"""
import json
import os
import uuid

import cv2
import numpy as np

cv2.setNumThreads(2)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from ai_engine.orchestrator.engine import run_plan
from ai_engine.orchestrator.planner import make_plan
from ai_engine.orchestrator.registry import REGISTRY, get_registry_summary
from ai_engine.specialists.qc_scorer.qc import score as qc_score

app = FastAPI(title="AutoHDR-clone service shell")

TMP_DIR = os.path.join("outputs", "service_tmp")
os.makedirs(TMP_DIR, exist_ok=True)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), media_type="text/html")

# Plan mac dinh khi khong co --command / --plan: 4 op deterministic da "chin"
# (giong PLAN_DETERMINISTIC trong ai_engine/integration_test.py).
DEFAULT_PLAN = [
    {"op": "auto_white_balance", "params": {}},
    {"op": "denoise", "params": {}},
    {"op": "straighten", "params": {}},
    {"op": "grass_green", "params": {}},
]


@app.get("/health")
def health():
    return {"status": "ok", "ops": len(REGISTRY)}


@app.get("/ops")
def ops():
    return get_registry_summary()


@app.post("/edit")
async def edit(image: UploadFile = File(...), command: str = Form(None), plan: str = Form(None)):
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    uid = uuid.uuid4().hex
    in_path = os.path.join(TMP_DIR, f"{uid}_in{ext}")
    out_path = os.path.join(TMP_DIR, f"{uid}_out.jpg")

    content = await image.read()
    with open(in_path, "wb") as f:
        f.write(content)

    try:
        img_check = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img_check is None:
            raise HTTPException(
                status_code=400,
                detail="Khong doc duoc anh upload (dinh dang khong ho tro hoac file hong).",
            )

        if plan:
            try:
                parsed = json.loads(plan)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"plan JSON khong hop le: {exc}")
            plan_list = parsed.get("plan", parsed) if isinstance(parsed, dict) else parsed
            if not isinstance(plan_list, list):
                raise HTTPException(status_code=400, detail="plan phai la list cac {op, params}.")
        elif command:
            plan_list, _source = make_plan(command)
        else:
            plan_list = DEFAULT_PLAN

        try:
            info = run_plan(in_path, plan_list, out_path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Khong ap dung duoc plan: {exc}")

        out_u8 = cv2.imread(out_path, cv2.IMREAD_COLOR)
        qc = qc_score(out_u8.astype(np.float32) / 255.0)

        with open(out_path, "rb") as f:
            out_bytes = f.read()

        headers = {
            "X-Plan-Applied": ",".join(info["applied"]),
            "X-QC-Overall": f"{qc['overall']:.2f}",
        }
        return Response(content=out_bytes, media_type="image/jpeg", headers=headers)
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)


@app.post("/qc")
async def qc_endpoint(image: UploadFile = File(...)):
    content = await image.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img_u8 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_u8 is None:
        raise HTTPException(status_code=400, detail="Khong doc duoc anh upload.")
    result = qc_score(img_u8.astype(np.float32) / 255.0)
    return JSONResponse(content=result)
