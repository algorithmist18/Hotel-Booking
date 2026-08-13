"""
compress_models.py — shrink pickled scikit-learn models for GitHub / Streamlit.

Random forests (and other tree ensembles) pickle to large, *uncompressed*
files by default. GitHub rejects any single file over 100 MB, which blocks the
push and stops Streamlit Cloud from ever seeing your models.

This script loads every model in a folder and re-saves it with joblib
compression. Predictions are unchanged — only the on-disk size shrinks (usually
5-10x for a forest).

Usage
-----
    # compress every .pkl/.joblib in hotel_ai_models/ in place (default level 3)
    python compress_models.py

    # point at a different folder / stronger compression
    python compress_models.py --dir hotel_ai_models --level 6

    # write compressed copies into a new folder instead of overwriting
    python compress_models.py --out compressed_models

    # strongest ratio (slower): xz/lzma
    python compress_models.py --method xz --level 9

Notes
-----
* Run this in the SAME environment you trained in — loading a pickle needs a
  compatible scikit-learn version.
* Compression affects disk size only, not RAM at inference time.
* If a forest is *still* over 100 MB after xz level 9, reduce the model itself
  (fewer trees / smaller max_depth) and re-export — see the tip printed at the
  end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

LIMIT_MB = 100  # GitHub hard limit per file


def human(n_bytes: int) -> str:
    mb = n_bytes / (1024 * 1024)
    return f"{mb:.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="hotel_ai_models",
                        help="Folder containing the model files (default: hotel_ai_models)")
    parser.add_argument("--out", default=None,
                        help="Optional output folder. Default: overwrite in place.")
    parser.add_argument("--method", default="zlib",
                        choices=["zlib", "gzip", "bz2", "xz", "lzma", "lz4"],
                        help="Compression codec (default: zlib). 'xz' gives the best ratio.")
    parser.add_argument("--level", type=int, default=3,
                        help="Compression level 1-9 (default: 3). Higher = smaller + slower.")
    args = parser.parse_args()

    src_dir = Path(args.dir)
    if not src_dir.is_dir():
        print(f"ERROR: folder not found: {src_dir.resolve()}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else src_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for ext in ("*.pkl", "*.joblib", "*.sav")
        for p in src_dir.glob(ext)
    )
    if not files:
        print(f"No .pkl/.joblib/.sav files found in {src_dir.resolve()}")
        return 1

    compress = (args.method, args.level)
    print(f"Compressing {len(files)} file(s) with {args.method} level {args.level}\n")

    any_still_too_big = False
    for path in files:
        before = path.stat().st_size
        try:
            obj = joblib.load(path)
        except Exception as exc:
            print(f"  ✗ {path.name}: could not load ({exc})")
            print("    -> run this in the same scikit-learn version you trained with.")
            continue

        target = out_dir / path.name
        joblib.dump(obj, target, compress=compress)
        after = target.stat().st_size

        ratio = before / after if after else 0
        flag = ""
        if after > LIMIT_MB * 1024 * 1024:
            flag = "  ⚠️ STILL OVER 100 MB"
            any_still_too_big = True
        print(f"  ✓ {path.name}: {human(before)} -> {human(after)}  "
              f"({ratio:.1f}x smaller){flag}")

    print("\nDone.")
    if any_still_too_big:
        print(
            "\nSome files are still over 100 MB. Options:\n"
            "  • Try a stronger codec:  python compress_models.py --method xz --level 9\n"
            "  • Reduce the model when you train/export it, e.g.:\n"
            "        RandomForestClassifier(n_estimators=100, max_depth=20)\n"
            "    then joblib.dump(model, 'model.pkl', compress=3)\n"
            "  • Or use Git LFS for the large file."
        )
    else:
        print("\nAll files are under 100 MB — safe to commit and push:")
        print(f"    git add {out_dir}/")
        print('    git commit -m "Compress trained models"')
        print("    git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
