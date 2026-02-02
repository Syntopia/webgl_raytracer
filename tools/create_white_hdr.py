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
    return bytes([
        int(r * scale + 0.5),
        int(g * scale + 0.5),
        int(b * scale + 0.5),
        int(e + 128),
    ])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env_dir = repo_root / "assets" / "env"
    if not env_dir.is_dir():
        raise SystemExit(f"Missing env directory: {env_dir}")

    hdr_path = env_dir / "white.hdr"
    header = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n"
    rgbe = float_to_rgbe(1.0, 1.0, 1.0)
    with hdr_path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rgbe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
