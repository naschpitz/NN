#!/usr/bin/env python3
"""
Generate synthetic OOD images: pure black/white/gray, solid RGB blocks,
random noise, blurred noise. Mirrors common user-mistake inputs that
softmax would happily classify with high confidence.

Usage: make-synthetic-ood.py OUT_DIR
"""
import os
import sys
import random
from pathlib import Path

from PIL import Image, ImageFilter
import numpy as np


SIZE = 450  # matches train-app-9 inputShape (3 x 450 x 450); OK for any size since the model resizes


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "JPEG", quality=92)


def solid(color, name, out: Path) -> None:
    img = Image.new("RGB", (SIZE, SIZE), color)
    save(img, out / f"solid_{name}.jpg")


def gradient(out: Path) -> None:
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, SIZE, dtype=np.uint8)
    img = Image.fromarray(arr)
    save(img, out / "gradient_red_horiz.jpg")
    arr2 = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    arr2[:, :, 2] = np.linspace(255, 0, SIZE, dtype=np.uint8)[:, None]
    img2 = Image.fromarray(arr2)
    save(img2, out / "gradient_blue_vert.jpg")


def random_noise(rng: random.Random, idx: int, out: Path) -> None:
    seed = rng.randrange(2**32)
    np_rng = np.random.default_rng(seed)
    arr = np_rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    save(img, out / f"noise_uniform_{idx:03d}.jpg")


def random_lowfreq(rng: random.Random, idx: int, out: Path) -> None:
    """Random noise blurred — mimics blurry / out-of-focus user photos."""
    seed = rng.randrange(2**32)
    np_rng = np.random.default_rng(seed)
    arr = np_rng.integers(0, 256, size=(SIZE // 16, SIZE // 16, 3), dtype=np.uint8)
    img = Image.fromarray(arr).resize((SIZE, SIZE), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=8))
    save(img, out / f"noise_blurred_{idx:03d}.jpg")


def half_split(out: Path) -> None:
    """Half black / half white — sharp edge, no skin-like features."""
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    arr[:, : SIZE // 2, :] = 0
    arr[:, SIZE // 2 :, :] = 255
    save(Image.fromarray(arr), out / "split_black_white.jpg")


def checker(out: Path) -> None:
    n = 16
    cell = SIZE // n
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                arr[i * cell : (i + 1) * cell, j * cell : (j + 1) * cell, :] = 255
    save(Image.fromarray(arr), out / "checker_bw.jpg")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: make-synthetic-ood.py OUT_DIR", file=sys.stderr)
        sys.exit(1)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # Solid colors.
    solid((0, 0, 0), "black", out)
    solid((255, 255, 255), "white", out)
    solid((128, 128, 128), "gray", out)
    solid((255, 0, 0), "red", out)
    solid((0, 255, 0), "green", out)
    solid((0, 0, 255), "blue", out)
    solid((255, 255, 0), "yellow", out)
    solid((255, 0, 255), "magenta", out)
    solid((0, 255, 255), "cyan", out)
    solid((230, 200, 170), "skin_like", out)  # plausible-skin-tone solid

    # Gradients / patterns.
    gradient(out)
    half_split(out)
    checker(out)

    # Random uniform noise.
    for i in range(20):
        random_noise(rng, i, out)

    # Blurred low-frequency noise (out-of-focus).
    for i in range(20):
        random_lowfreq(rng, i, out)

    count = sum(1 for _ in out.glob("*.jpg"))
    print(f"Generated {count} synthetic OOD images in {out}")


if __name__ == "__main__":
    main()
