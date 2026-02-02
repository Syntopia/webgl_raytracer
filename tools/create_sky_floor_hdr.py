#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path


def float_to_rgbe(r: float, g: float, b: float) -> bytes:
    v = max(r, g, b)
    if v < 1e-32:
        return bytes([0, 0, 0, 0])
    m, e = math.frexp(v)
    scale = m * 256.0 / v
    def to_byte(x: float) -> int:
        return max(0, min(255, int(x * scale + 0.5)))
    return bytes([
        to_byte(r),
        to_byte(g),
        to_byte(b),
        max(0, min(255, int(e + 128))),
    ])


def encode_rle_channel(channel: list[int]) -> bytes:
    out = bytearray()
    n = len(channel)
    i = 0
    while i < n:
        run_val = channel[i]
        run_len = 1
        while i + run_len < n and run_len < 127 and channel[i + run_len] == run_val:
            run_len += 1
        if run_len >= 4:
            out.append(128 + run_len)
            out.append(run_val)
            i += run_len
            continue

        start = i
        i += 1
        while i < n:
            run_val = channel[i]
            run_len = 1
            while i + run_len < n and run_len < 127 and channel[i + run_len] == run_val:
                run_len += 1
            if run_len >= 4:
                break
            i += 1
            if i - start >= 128:
                break
        count = i - start
        out.append(count)
        out.extend(channel[start:i])
    return bytes(out)


def encode_scanline_rgbe(pixels: list[bytes], width: int) -> bytes:
    out = bytearray()
    out.extend([2, 2, (width >> 8) & 0xFF, width & 0xFF])
    channels = [[], [], [], []]
    for pix in pixels:
        for c in range(4):
            channels[c].append(pix[c])
    for c in range(4):
        out.extend(encode_rle_channel(channels[c]))
    return bytes(out)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env_dir = repo_root / "assets" / "env"
    if not env_dir.is_dir():
        raise SystemExit(f"Missing env directory: {env_dir}")

    width = 64
    height = 32
    hdr_path = env_dir / "blue_sky_brown_floor.hdr"
    header = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {} +X {}\n".format(height, width)

    with hdr_path.open("wb") as f:
        f.write(header.encode("ascii"))
        for y in range(height):
            t = y / max(1, height - 1)
            pixels = []
            for _x in range(width):
                if t < 0.5:
                    k = t / 0.5
                    r = 0.15 + 0.35 * k
                    g = 0.25 + 0.45 * k
                    b = 0.60 + 0.35 * k
                else:
                    k = (t - 0.5) / 0.5
                    r = 0.35 - 0.10 * k
                    g = 0.22 - 0.08 * k
                    b = 0.12 - 0.04 * k
                pixels.append(float_to_rgbe(r, g, b))
            f.write(encode_scanline_rgbe(pixels, width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
