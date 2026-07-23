# -*- coding: utf-8 -*-
"""Test tự động (pytest) cho các module pipeline mới.

Chạy: python -m pytest tests/test_pipeline.py -q
Chỉ dùng cv2/numpy — KHÔNG cần torch/GPU, không tải model nặng.
"""

import cv2
import numpy as np
import pytest

from ai_engine.specialists.auto_enhance.bracket_merge import (
    group_brackets,
    merge_brackets,
)
from ai_engine.specialists.auto_enhance.gpu.finish_grade import (
    _scene,
    grade,
    grade_auto,
)
from ai_engine.specialists.straighten.straighten import estimate_tilt, straighten
from ai_engine.specialists.white_balance.white_balance import auto_wb, measure_cast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scene_image(h=64, w=64, seed=0):
    """Ảnh BGR uint8 có texture (gradient + noise) để merge/grade có gì mà xử lý."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx + yy) / (h + w) * 255.0).astype(np.float32)
    img = np.stack([base * 0.9, base, base * 1.1], axis=2)
    img += rng.normal(0, 8, (h, w, 3)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _write_brackets(tmp_path, gains=(0.4, 1.0, 2.2)):
    """Ghi các bản phơi sáng khác nhau của cùng 1 cảnh ra file tạm, trả list path."""
    base = _make_scene_image().astype(np.float32)
    paths = []
    for i, g in enumerate(gains):
        p = tmp_path / f"exp_{i}.png"
        cv2.imwrite(str(p), np.clip(base * g, 0, 255).astype(np.uint8))
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# 1. bracket_merge.merge_brackets
# ---------------------------------------------------------------------------

def test_merge_brackets_basic(tmp_path):
    paths = _write_brackets(tmp_path)
    merged = merge_brackets(paths, align=False)
    assert merged.shape == (64, 64, 3)
    assert merged.dtype == np.uint8
    assert merged.min() >= 0 and merged.max() <= 255


def test_merge_brackets_two_images(tmp_path):
    paths = _write_brackets(tmp_path, gains=(0.5, 1.8))
    merged = merge_brackets(paths, align=False)
    assert merged.shape == (64, 64, 3)
    assert merged.dtype == np.uint8


def test_merge_brackets_too_few_raises(tmp_path):
    paths = _write_brackets(tmp_path, gains=(1.0,))
    with pytest.raises(ValueError):
        merge_brackets(paths)
    with pytest.raises(ValueError):
        merge_brackets([])


# ---------------------------------------------------------------------------
# 2. bracket_merge.group_brackets
# ---------------------------------------------------------------------------

def test_group_brackets_fixed_size(tmp_path):
    img = _make_scene_image(32, 32)
    for i in range(6):
        cv2.imwrite(str(tmp_path / f"img_{i:02d}.png"), img)
    groups = group_brackets(str(tmp_path), group_size=3)
    assert len(groups) == 2
    assert all(len(g) == 3 for g in groups)


def test_group_brackets_missing_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        group_brackets(str(tmp_path / "khong_ton_tai"))


# ---------------------------------------------------------------------------
# 3. white_balance.auto_wb + measure_cast
# ---------------------------------------------------------------------------

def test_auto_wb_keeps_neutral_gray_neutral():
    gray = np.full((64, 64, 3), 128, np.uint8)
    out = auto_wb(gray)
    assert out.shape == gray.shape and out.dtype == np.uint8
    cast = measure_cast(out)
    assert abs(cast["a"]) < 2.0 and abs(cast["b"]) < 2.0


def test_auto_wb_reduces_color_cast():
    # Ảnh có texture nhưng ám xanh dương rõ (kênh B trội).
    base = _make_scene_image()
    tinted = base.astype(np.float32)
    tinted[:, :, 0] = np.clip(tinted[:, :, 0] * 1.6 + 30, 0, 255)  # boost B
    tinted = tinted.astype(np.uint8)

    before = measure_cast(tinted)
    after = measure_cast(auto_wb(tinted))
    mag_before = abs(before["a"]) + abs(before["b"])
    mag_after = abs(after["a"]) + abs(after["b"])
    assert mag_after < mag_before


# ---------------------------------------------------------------------------
# 4. straighten.estimate_tilt + straighten
# ---------------------------------------------------------------------------

def _vertical_lines_image(h=512, w=512):
    img = np.full((h, w, 3), 255, np.uint8)
    for x in range(60, w - 40, 90):
        cv2.line(img, (x, 30), (x, h - 30), (0, 0, 0), 3)
    return img


def _rotate(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def test_estimate_tilt_detects_rotation():
    img = _vertical_lines_image()
    rotated = _rotate(img, 5.0)  # cv2: góc + = xoay CCW
    tilt = estimate_tilt(rotated)
    # Theo quy ước module: xoay CCW +5° -> tilt ~ -5°
    assert abs(tilt - (-5.0)) <= 1.5


def test_estimate_tilt_upright_near_zero():
    tilt = estimate_tilt(_vertical_lines_image())
    assert abs(tilt) <= 1.0


def test_straighten_flat_image_no_crash():
    flat = np.full((80, 120, 3), 130, np.uint8)
    out = straighten(flat)
    assert out.shape == flat.shape


def test_straighten_returns_same_shape_on_tilted():
    rotated = _rotate(_vertical_lines_image(), 5.0)
    out = straighten(rotated)
    assert out.shape == rotated.shape
    assert out.dtype == rotated.dtype


# ---------------------------------------------------------------------------
# 5. finish_grade.grade / grade_auto / _scene
# ---------------------------------------------------------------------------

def test_grade_auto_shape_dtype_range():
    img = _make_scene_image(96, 128, seed=7)
    out = grade_auto(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_grade_defaults_shape():
    img = _make_scene_image(64, 64, seed=3)
    out = grade(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_scene_blue_sky_is_outdoor():
    img = np.zeros((100, 100, 3), np.uint8)
    img[:50] = (230, 170, 90)    # nửa trên: trời xanh (B > R, sáng)
    img[50:] = (110, 120, 125)   # nửa dưới: xám nhạt
    assert _scene(img) == "outdoor"


def test_scene_warm_wood_is_indoor():
    img = np.zeros((100, 100, 3), np.uint8)
    img[:] = (60, 110, 170)      # gỗ ấm: R > B toàn ảnh
    assert _scene(img) == "indoor"
