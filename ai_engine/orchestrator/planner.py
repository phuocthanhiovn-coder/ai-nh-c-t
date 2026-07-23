"""Dich lenh tu nhien (Viet/Anh) -> plan operator JSON.

Uu tien goi LLM qua ANTICODE_API_KEY. Thieu key / loi goi -> fallback rule-based (regex).
Tra ve (plan, source) voi source in {"llm", "fallback"}.
"""
import json
import os
import re
import urllib.request
import urllib.error

from .registry import REGISTRY, get_registry_summary, clamp_params

ANTICODE_BASE_URL = os.environ.get("ANTICODE_BASE_URL", "https://anticode.vn")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "cheap-ai/claude-sonnet-5")


def _build_system_prompt():
    ops_desc = []
    for entry in get_registry_summary():
        params_desc = ", ".join(
            f"{pname}({p['type']} {p['min']}..{p['max']}, default={p['default']})"
            for pname, p in entry["params"].items()
        ) or "(khong co params)"
        ops_desc.append(f"- {entry['op']}: {entry['desc']} | params: {params_desc}")

    ops_block = "\n".join(ops_desc)
    return (
        "Ban la bo dich lenh chinh anh bat dong san sang ke hoach operator JSON.\n"
        "Danh sach operator kha dung (CHI duoc dung ten trong danh sach nay):\n"
        f"{ops_block}\n\n"
        "Nhiem vu: doc lenh cua nguoi dung (tieng Viet hoac tieng Anh), chuyen thanh mot ke hoach "
        "gom danh sach cac buoc operator can ap dung TUAN TU.\n"
        "TRA VE DUY NHAT JSON theo dung dinh dang sau, KHONG them van ban giai thich:\n"
        '{"plan": [{"op": "ten_op", "params": {"ten_param": gia_tri}}, ...]}\n'
        "Neu lenh yeu cau dieu gi do khong co trong danh sach operator (vi du: xoay doc thang, xoa vat the), "
        "BO QUA yeu cau do, KHONG bia ra op moi."
    )


def _extract_json(text):
    """Lay JSON tu output LLM, chiu duoc truong hop bao trong ```json ... ``` hoac text thuan."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return text


def _validate_plan(raw_plan):
    """Loc op khong ton tai (kem warning), clamp params theo schema."""
    plan = []
    for step in raw_plan:
        op_name = step.get("op")
        if op_name not in REGISTRY:
            print(f"[WARN] planner: bo qua op la '{op_name}' (khong co trong registry).")
            continue
        params = clamp_params(op_name, step.get("params", {}) or {})
        plan.append({"op": op_name, "params": params})
    return plan


def _call_llm(command, api_key):
    system_prompt = _build_system_prompt()
    payload = {
        "model": PLANNER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": command},
        ],
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{ANTICODE_BASE_URL}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "autohdr-orchestrator/0.1",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp_body = json.loads(resp.read().decode("utf-8"))

    content = resp_body["choices"][0]["message"]["content"]
    json_text = _extract_json(content)
    parsed = json.loads(json_text)
    raw_plan = parsed.get("plan", [])
    return _validate_plan(raw_plan)


# --- Fallback rule-based (regex) ---

_FALLBACK_RULES = [
    (re.compile(r"toi|darker|\bdark\b", re.IGNORECASE), "brightness", {"amount": -0.3}),
    (re.compile(r"sang|brighter|\bbright\b", re.IGNORECASE), "brightness", {"amount": 0.3}),
    (re.compile(r"am hon|warm", re.IGNORECASE), "temperature", {"amount": 0.3}),
    (re.compile(r"lanh|cool|cold", re.IGNORECASE), "temperature", {"amount": -0.3}),
    (re.compile(r"tuong phan|contrast", re.IGNORECASE), "contrast", {"amount": 0.2}),
    (re.compile(r"ruc|vibrant|saturat", re.IGNORECASE), "saturation", {"amount": 0.3}),
    (re.compile(r"\bnet\b|sharp", re.IGNORECASE), "sharpen", {"amount": 0.2}),
    (re.compile(r"can bang trang|white balance", re.IGNORECASE), "white_balance", {"strength": 1.0}),
    (re.compile(r"tu dong|auto", re.IGNORECASE), "auto_enhance", {}),
]

_UNSUPPORTED_HINTS = [
    (re.compile(r"doc|thang|straighten|vertical", re.IGNORECASE), "straighten_verticals"),
    (re.compile(r"xoa|remove object|inpaint", re.IGNORECASE), "remove_object"),
    (re.compile(r"troi|sky replace", re.IGNORECASE), "sky_replace"),
]

_MILD_HINT = re.compile(r"nhe|slight|subtle", re.IGNORECASE)


def _strip_accents(text):
    import unicodedata
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def _fallback_plan(command):
    ascii_cmd = _strip_accents(command.lower())
    mild = bool(_MILD_HINT.search(ascii_cmd))

    plan = []
    for pattern, op_name, params in _FALLBACK_RULES:
        if pattern.search(ascii_cmd):
            final_params = dict(params)
            if mild:
                for k, v in final_params.items():
                    if isinstance(v, (int, float)):
                        final_params[k] = v * 0.5
            plan.append({"op": op_name, "params": clamp_params(op_name, final_params)})

    for pattern, hint_name in _UNSUPPORTED_HINTS:
        if pattern.search(ascii_cmd):
            print(f"[WARN] planner(fallback): lenh nhac toi '{hint_name}' nhung op nay chua ton tai. Bo qua.")

    return plan


def make_plan(command):
    """Tra ve (plan, source). source in {'llm', 'fallback'}."""
    api_key = os.environ.get("ANTICODE_API_KEY")
    if not api_key:
        print("[INFO] planner: thieu ANTICODE_API_KEY -> dung fallback rule-based.")
        return _fallback_plan(command), "fallback"

    try:
        plan = _call_llm(command, api_key)
        return plan, "llm"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
        print(f"[WARN] planner: goi LLM loi ({exc}) -> dung fallback rule-based.")
        return _fallback_plan(command), "fallback"
