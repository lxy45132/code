"""Deterministic procedural faces and natural/indoor scenes.

These images are **construction-time placeholders** until the real CelebA-HQ
and Places2 data is downloaded at T0.  They mirror the target domains:

* ``face``       frontal face on a plain backdrop
* ``portrait``   smaller face with shoulders/body framing
* ``landscape``  sky, sun/clouds, mountains, ground, optional water/trees
* ``indoor``     room with back wall, floor, window/door and furniture

No facade/window-grid archetype remains; the old facade plan is out of scope.
``save_frozen_set`` writes a small, hashed fixed-observation set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image as PILImage

from hybrid_inpaint.domain.entities import Image, Mask, Sample
from hybrid_inpaint.training.data.masks import MASK_MODES, make_mask, make_rect_mask

ARCHETYPES = ("face", "portrait", "landscape", "indoor")


def _as_uint8_bounds(value: np.ndarray) -> np.ndarray:
    return np.clip(value, 0, 255)


def _fill_rect(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: np.ndarray,
) -> None:
    height, width = canvas.shape[:2]
    x0, x1 = max(0, min(width, int(x0))), max(0, min(width, int(x1)))
    y0, y1 = max(0, min(height, int(y0))), max(0, min(height, int(y1)))
    if x0 < x1 and y0 < y1:
        canvas[y0:y1, x0:x1] = color


def _ellipse_mask(shape: tuple[int, int], cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    if rx <= 0 or ry <= 0:
        return np.zeros(shape, dtype=np.bool_)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _fill_ellipse(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    color: np.ndarray,
) -> None:
    mask = _ellipse_mask(canvas.shape[:2], cx, cy, rx, ry)
    canvas[mask] = color


def _vertical_gradient(size: int, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    column = top[None, None, :] * (1.0 - t) + bottom[None, None, :] * t
    return np.broadcast_to(column, (size, size, 3)).astype(np.float32).copy()


def _fill_below(canvas: np.ndarray, heights: np.ndarray, base: int, color: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    base = max(0, min(height, int(base)))
    for x in range(min(width, heights.shape[0])):
        y0 = max(0, min(base, int(heights[x])))
        if y0 < base:
            canvas[y0:base, x] = color


def _draw_tree(canvas: np.ndarray, x: int, base_y: int, height: int, leaf: np.ndarray, trunk: np.ndarray) -> None:
    height = max(4, height)
    for offset in range(height):
        half = max(1, int((offset / height) * height * 0.28))
        y = base_y - offset
        _fill_rect(canvas, x - half, y, x + half + 1, y + 1, leaf)
    _fill_rect(canvas, x - max(1, height // 18), base_y, x + max(2, height // 18), base_y + max(2, height // 8), trunk)


def _skin_color(rng: np.random.Generator) -> np.ndarray:
    # Warm skin-like ordering: red > green > blue.
    base = float(rng.uniform(150.0, 225.0))
    skin = np.array(
        [base, base * rng.uniform(0.68, 0.82), base * rng.uniform(0.55, 0.70)],
        dtype=np.float32,
    )
    return np.clip(skin, 0.0, 255.0)


def _render_face(size: int, rng: np.random.Generator, *, portrait: bool) -> tuple[np.ndarray, dict[str, Any]]:
    background_top = rng.integers(35, 220, size=3).astype(np.float32)
    background_bottom = rng.integers(20, 190, size=3).astype(np.float32)
    canvas = _vertical_gradient(size, background_top, background_bottom)
    canvas += rng.normal(0.0, 1.8, size=(size, size, 3)).astype(np.float32)

    cx = size * (0.5 + float(rng.uniform(-0.03, 0.03)))
    cy = size * (0.42 if portrait else 0.46)
    rx = size * (0.17 if portrait else 0.22)
    ry = size * (0.22 if portrait else 0.28)

    body_color = rng.integers(35, 190, size=3).astype(np.float32)
    _fill_ellipse(canvas, cx, size * 1.04, size * (0.46 if portrait else 0.42), size * (0.30 if portrait else 0.26), body_color)

    skin = _skin_color(rng)
    neck_top = int(cy + ry * 0.72)
    _fill_rect(canvas, int(cx - rx * 0.22), neck_top, int(cx + rx * 0.22), int(min(size, cy + ry * 1.15)), skin * 0.93)

    hair_base = float(rng.uniform(10.0, 110.0))
    hair = np.array([hair_base * 1.05, hair_base * 0.85, hair_base * 0.65], dtype=np.float32)
    hair_ry = ry * 1.18
    _fill_ellipse(canvas, cx, cy - ry * 0.06, rx * 1.12, hair_ry, hair)
    _fill_ellipse(canvas, cx, cy, rx, ry, skin)
    _fill_ellipse(canvas, cx, cy - ry * 0.72, rx * 0.94, ry * 0.34, hair)

    _fill_ellipse(canvas, cx - rx * 1.02, cy, rx * 0.13, ry * 0.18, skin * 0.94)
    _fill_ellipse(canvas, cx + rx * 1.02, cy, rx * 0.13, ry * 0.18, skin * 0.94)

    eye_y = cy - ry * 0.12
    eye_dx = rx * 0.36
    iris_color = rng.integers(30, 120, size=3).astype(np.float32)
    for sign in (-1.0, 1.0):
        ex = cx + sign * eye_dx
        _fill_ellipse(canvas, ex, eye_y, rx * 0.20, ry * 0.11, np.array([245.0, 245.0, 245.0]))
        _fill_ellipse(canvas, ex, eye_y, rx * 0.09, ry * 0.09, iris_color)
        _fill_ellipse(canvas, ex, eye_y, rx * 0.04, ry * 0.04, np.array([10.0, 10.0, 10.0]))
        _fill_rect(canvas, int(ex - rx * 0.23), int(eye_y - ry * 0.20), int(ex + rx * 0.23), int(eye_y - ry * 0.15), hair)

    _fill_ellipse(canvas, cx, cy + ry * 0.14, rx * 0.08, ry * 0.11, skin * 0.90)
    lip = rng.integers(120, 210, size=3).astype(np.float32)
    _fill_ellipse(canvas, cx, cy + ry * 0.48, rx * 0.24, ry * 0.08, lip)

    meta: dict[str, Any] = {
        "archetype": "portrait" if portrait else "face",
        "scene_type": None,
        "has_face": True,
        "face_center": [round(float(cx), 2), round(float(cy), 2)],
        "face_size": [round(float(rx), 2), round(float(ry), 2)],
        "symmetry_axis_x": round(float(cx), 2),
    }
    return canvas, meta


def _render_landscape(size: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    sky_top = rng.integers(55, 150, size=3).astype(np.float32)
    sky_bottom = rng.integers(155, 245, size=3).astype(np.float32)
    canvas = _vertical_gradient(size, sky_top, sky_bottom)
    canvas += rng.normal(0.0, 1.5, size=(size, size, 3)).astype(np.float32)

    horizon = int(size * float(rng.uniform(0.55, 0.70)))
    sun_color = rng.integers(210, 255, size=3).astype(np.float32)
    _fill_ellipse(
        canvas,
        float(rng.uniform(0.12, 0.88) * size),
        float(rng.uniform(0.08, 0.30) * size),
        size * 0.055,
        size * 0.055,
        sun_color,
    )

    cloud_color = np.array([235.0, 238.0, 245.0], dtype=np.float32)
    for _ in range(int(rng.integers(3, 7))):
        ccx = float(rng.uniform(0.0, 1.0) * size)
        ccy = float(rng.uniform(0.04, 0.45) * horizon)
        crx = float(rng.uniform(0.05, 0.14) * size)
        cry = crx * float(rng.uniform(0.28, 0.55))
        _fill_ellipse(canvas, ccx, ccy, crx, cry, cloud_color)

    xs = np.arange(size, dtype=np.float32)
    for layer in range(2):
        base = horizon - int(layer * size * 0.04)
        amplitude = size * float(rng.uniform(0.07, 0.17))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        frequency = float(rng.uniform(1.0, 2.8))
        heights = base - amplitude * (0.5 + 0.5 * np.sin(xs / size * 2.0 * np.pi * frequency + phase))
        color = rng.integers(35, 120, size=3).astype(np.float32)
        _fill_below(canvas, heights, base, color)

    ground_color = rng.integers(35, 150, size=3).astype(np.float32)
    canvas[horizon:, :, :] = ground_color
    has_water = bool(rng.integers(0, 2))
    if has_water:
        water_start = horizon + int((size - horizon) * 0.35)
        water_color = rng.integers(35, 140, size=3).astype(np.float32)
        canvas[water_start:, :, :] = water_color
        for row in range(water_start, size, max(2, size // 40)):
            canvas[row : row + 1, :, :] = np.clip(water_color + 25.0, 0, 255)

    has_trees = bool(rng.integers(0, 2))
    if has_trees:
        leaf = rng.integers(20, 100, size=3).astype(np.float32)
        trunk = rng.integers(35, 90, size=3).astype(np.float32)
        for _ in range(int(rng.integers(3, 9))):
            x = int(rng.uniform(0.02, 0.98) * size)
            base_y = int(rng.uniform(horizon + 0.02 * size, size * 0.96))
            height = int(rng.uniform(0.10, 0.25) * size)
            _draw_tree(canvas, x, base_y, height, leaf, trunk)

    return canvas, {
        "archetype": "landscape",
        "scene_type": "landscape",
        "has_face": False,
        "horizon_y": int(horizon),
        "has_water": has_water,
        "has_mountains": True,
        "has_trees": has_trees,
    }


def _render_indoor(size: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    wall_top = rng.integers(95, 205, size=3).astype(np.float32)
    wall_bottom = np.clip(wall_top * 0.86, 0, 255)
    canvas = _vertical_gradient(size, wall_top, wall_bottom)
    canvas += rng.normal(0.0, 1.5, size=(size, size, 3)).astype(np.float32)

    horizon = int(size * float(rng.uniform(0.55, 0.70)))
    floor_color = rng.integers(35, 125, size=3).astype(np.float32)
    canvas[horizon:, :, :] = floor_color

    left_wall = np.clip(wall_top * 0.72, 0, 255)
    right_wall = np.clip(wall_top * 0.78, 0, 255)
    for y in range(horizon, size):
        inset = int((y - horizon) / max(1, size - horizon) * size * 0.14)
        if inset > 0:
            _fill_rect(canvas, 0, y, inset, y + 1, left_wall)
            _fill_rect(canvas, size - inset, y, size, y + 1, right_wall)

    has_window = bool(rng.integers(0, 2))
    has_door = bool(rng.integers(0, 2))
    if has_window:
        window_w = int(size * float(rng.uniform(0.16, 0.28)))
        window_h = int(size * float(rng.uniform(0.18, 0.30)))
        x0 = int(rng.uniform(0.10, 0.55) * size)
        y0 = int(rng.uniform(0.16, 0.36) * horizon)
        frame = rng.integers(35, 100, size=3).astype(np.float32)
        sky = rng.integers(150, 245, size=3).astype(np.float32)
        _fill_rect(canvas, x0, y0, x0 + window_w, y0 + window_h, frame)
        _fill_rect(canvas, x0 + 3, y0 + 3, x0 + window_w - 3, y0 + window_h - 3, sky)
        _fill_rect(canvas, x0 + window_w // 2 - 1, y0 + 3, x0 + window_w // 2 + 2, y0 + window_h - 3, frame)
        _fill_rect(canvas, x0 + 3, y0 + window_h // 2 - 1, x0 + window_w - 3, y0 + window_h // 2 + 2, frame)

    if has_door:
        door_w = int(size * 0.16)
        door_h = int(size * 0.34)
        x0 = int(rng.uniform(0.55, 0.80) * size)
        y0 = horizon - door_h
        door_color = rng.integers(30, 90, size=3).astype(np.float32)
        _fill_rect(canvas, x0, y0, x0 + door_w, horizon, door_color)
        _fill_ellipse(canvas, x0 + door_w * 0.82, y0 + door_h * 0.52, max(1, size * 0.008), max(1, size * 0.008), np.array([200.0, 190.0, 120.0]))

    sofa_color = rng.integers(40, 170, size=3).astype(np.float32)
    sofa_w = int(size * float(rng.uniform(0.28, 0.42)))
    sofa_h = int(size * 0.14)
    sx = int(rng.uniform(0.08, 0.55) * size)
    sy = int(size * float(rng.uniform(0.80, 0.88)))
    _fill_rect(canvas, sx, sy, sx + sofa_w, sy + sofa_h, sofa_color)
    _fill_rect(canvas, sx, sy - int(sofa_h * 0.45), sx + sofa_w, sy, np.clip(sofa_color * 1.08, 0, 255))

    return canvas, {
        "archetype": "indoor",
        "scene_type": "indoor",
        "has_face": False,
        "horizon_y": int(horizon),
        "has_window": has_window,
        "has_door": has_door,
    }


_RENDERERS = {
    "face": lambda size, rng: _render_face(size, rng, portrait=False),
    "portrait": lambda size, rng: _render_face(size, rng, portrait=True),
    "landscape": _render_landscape,
    "indoor": _render_indoor,
}


def generate_synthetic_image(
    archetype: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate one uint8 RGB image and its structural metadata."""
    if archetype not in _RENDERERS:
        raise ValueError(f"unknown archetype: {archetype!r}; expected one of {ARCHETYPES}")
    if size < 32 or size % 16 != 0:
        raise ValueError("size must be >= 32 and a multiple of 16")

    canvas, meta = _RENDERERS[archetype](size, rng)
    canvas = _as_uint8_bounds(canvas).astype(np.uint8)
    return canvas, meta


@dataclass
class SyntheticInpaintDataset:
    """A reproducible, on-the-fly synthetic inpainting dataset."""

    size: int = 256
    length: int = 256
    seed: int = 2026
    archetypes: Sequence[str] = ARCHETYPES
    mask_modes: Sequence[str] = MASK_MODES
    min_ratio: float = 0.1
    max_ratio: float = 0.4

    def __post_init__(self) -> None:
        if self.size < 32 or self.size % 16 != 0:
            raise ValueError("size must be >= 32 and a multiple of 16")
        if self.length < 1:
            raise ValueError("length must be >= 1")
        if not self.archetypes:
            raise ValueError("archetypes must not be empty")
        if any(archetype not in ARCHETYPES for archetype in self.archetypes):
            raise ValueError(f"unknown archetype in {self.archetypes!r}")
        if not self.mask_modes or any(mode not in MASK_MODES for mode in self.mask_modes):
            raise ValueError(f"mask_modes must be a non-empty subset of {MASK_MODES}")
        if not (0.0 < self.min_ratio <= self.max_ratio < 1.0):
            raise ValueError("require 0 < min_ratio <= max_ratio < 1")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Sample:
        if not 0 <= index < self.length:
            raise IndexError(f"index {index} out of range [0, {self.length})")

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        archetype = self.archetypes[index % len(self.archetypes)]
        image, meta = generate_synthetic_image(archetype, self.size, rng)

        target_ratio = float(rng.uniform(self.min_ratio, self.max_ratio))
        mode = self.mask_modes[int(rng.integers(0, len(self.mask_modes)))]

        mask: np.ndarray | None = None
        tolerance = max(0.05, target_ratio * 0.5)
        for _ in range(20):
            candidate = make_mask(self.size, target_ratio, rng, mode=mode)
            ratio = float(candidate.mean())
            if 0.0 < ratio < 1.0 and abs(ratio - target_ratio) <= tolerance:
                mask = candidate
                break
        if mask is None:
            mask = make_rect_mask(self.size, target_ratio, rng)

        if not 0.0 < float(mask.mean()) < 1.0:
            raise RuntimeError("synthetic mask generation produced an empty or all-hole mask")

        sample_meta = {
            **meta,
            "index": int(index),
            "seed": int(self.seed),
            "mask_mode": mode,
            "target_ratio": round(target_ratio, 6),
            "actual_ratio": round(float(mask.mean()), 6),
        }
        return Sample(
            sample_id=f"syn-{index:06d}",
            image=Image(image),
            mask=Mask(mask),
            meta=sample_meta,
        )


def frozen_samples(size: int = 256, count: int = 16, seed: int = 2026) -> list[Sample]:
    """Return a deterministic set of samples for fixed-observation tests."""
    dataset = SyntheticInpaintDataset(size=size, length=count, seed=seed)
    samples: list[Sample] = []
    for index in range(count):
        sample = dataset[index]
        samples.append(
            Sample(
                sample_id=f"frozen-{index:04d}",
                image=sample.image,
                mask=sample.mask,
                meta={**sample.meta, "frozen": True},
            )
        )
    return samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_frozen_set(
    output_dir: str | Path,
    *,
    size: int = 256,
    count: int = 16,
    seed: int = 2026,
) -> dict[str, Any]:
    """Write frozen images, masks and a hashed JSON manifest."""
    root = Path(output_dir)
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "contract_version": 1,
        "size": int(size),
        "count": int(count),
        "seed": int(seed),
        "samples": [],
    }
    for index, sample in enumerate(frozen_samples(size=size, count=count, seed=seed)):
        image_path = images_dir / f"{index:04d}.png"
        mask_path = masks_dir / f"{index:04d}.png"
        PILImage.fromarray(sample.image.data, mode="RGB").save(image_path)
        PILImage.fromarray(sample.mask.to_uint8(), mode="L").save(mask_path)
        manifest["samples"].append(
            {
                "sample_id": sample.sample_id,
                "image": str(image_path.relative_to(root)).replace("\\", "/"),
                "mask": str(mask_path.relative_to(root)).replace("\\", "/"),
                "image_sha256": _sha256(image_path),
                "mask_sha256": _sha256(mask_path),
                "meta": sample.meta,
            }
        )

    (root / "frozen.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
