"""List an original texture's exact colours and show where each sits (letter map).

    python3 palette.py <png> [...]
"""
import string
import sys

import numpy as np
from PIL import Image

LET = string.ascii_letters + string.digits


def palette(path):
    a = np.asarray(Image.open(path).convert("RGBA"))
    a = a[: a.shape[1]]
    keys = [tuple(int(v) for v in px) for px in a.reshape(-1, 4)]
    cols = sorted(set(k for k in keys if k[3] > 0), key=lambda c: -(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]))
    return a, cols


if __name__ == "__main__":
    for p in sys.argv[1:]:
        a, cols = palette(p)
        idx = {c: LET[i % len(LET)] for i, c in enumerate(cols)}
        print(p.split("/")[-1], a.shape[:2], len(cols), "colours")
        for c in cols:
            n = int(((a == np.array(c)).all(-1)).sum())
            print(f"  {idx[c]} #{c[0]:02x}{c[1]:02x}{c[2]:02x} a{c[3]:3d}  x{n}")
        for y in range(a.shape[0]):
            print("  " + "".join(idx.get(tuple(int(v) for v in a[y, x]), ".") for x in range(a.shape[1])))
