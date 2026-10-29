"""Print an original texture as a coordinate grid for tracing designs by hand.

Each texel: a lightness digit 0-9 (CIE L*/10) and a hue letter
(r o y g c b p m = red..magenta, n = neutral/grey); '.' is transparent.

    python3 trace.py <png> [...]
"""
import sys

import cv2
import numpy as np
from PIL import Image

HUES = "roygcbpm"


def cell(px):
    if px[3] < 128:
        return " . "
    rgb = px[:3].astype(np.float32).reshape(1, 1, 3) / 255
    L, a, b = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0]
    c = np.hypot(a, b)
    hue = "n" if c < 12 else HUES[int(((np.degrees(np.arctan2(b, a)) + 360 + 22.5) % 360) // 45)]
    return f"{min(9, int(L / 10))}{hue} "


for path in sys.argv[1:]:
    a = np.asarray(Image.open(path).convert("RGBA"))
    a = a[: a.shape[1]]
    print(path.split("/")[-1])
    print("    " + "".join(f"{x:<3d}" for x in range(a.shape[1])))
    for y in range(a.shape[0]):
        print(f"{y:2d}  " + "".join(cell(a[y, x]) for x in range(a.shape[1])))
