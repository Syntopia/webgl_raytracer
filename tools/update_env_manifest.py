#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


def prettify_name(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_\d+k$", "", stem, flags=re.IGNORECASE)
    parts = stem.split("_")
    return " ".join(part.capitalize() for part in parts if part)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env_dir = repo_root / "assets" / "env"
    if not env_dir.is_dir():
        raise SystemExit(f"Missing env directory: {env_dir}")

    hdr_files = sorted(p.name for p in env_dir.glob("*.hdr"))
    if not hdr_files:
        raise SystemExit(f"No .hdr files found in {env_dir}")

    manifest = [{"file": name, "name": prettify_name(name)} for name in hdr_files]
    manifest_path = env_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
