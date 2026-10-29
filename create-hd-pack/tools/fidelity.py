"""Does an HD item still read as the original at inventory size?

Shrinks each HD texture to the original's size the way the game's mipmaps do
(alpha-weighted 2x2 averages) and compares it with the original:
  iou      silhouette overlap (alpha > 0.5)
  fill     HD silhouette area / original's
  dL       mean lightness shift (CIE L*, HD - original), over the original's pixels
  dC       mean chroma shift (saturation; negative = washed out)
  dE       mean colour difference after a slight blur (layout of light/dark/colour)
  fringe   share of HD edge texels with partial alpha (soft halos when extruded in hand)

    python3 fidelity.py <src_assets> <pack_dir> <handbuilt_dir> [--kind item] [--top N]
"""
import os
import sys

import cv2
import numpy as np

from ingame_preview import load, mips


def lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)


def score(orig, hd):
    chain = mips(hd)
    small = next(m for m in chain if m.shape[0] <= orig.shape[0])
    if small.shape[0] != orig.shape[0]:
        small = cv2.resize(small, orig.shape[:2][::-1], interpolation=cv2.INTER_AREA)
    ao, ah = orig[..., 3] > 0.5, small[..., 3] > 0.5
    union = ao | ah
    iou = (ao & ah).sum() / max(union.sum(), 1)
    fill = ah.sum() / max(ao.sum(), 1)
    lo, lh = lab(orig[..., :3]), lab(small[..., :3])
    both = ao & ah
    dL = float((lh[..., 0] - lo[..., 0])[both].mean()) if both.any() else 0.0
    chroma = lambda l: np.hypot(l[..., 1], l[..., 2])
    dC = float((chroma(lh) - chroma(lo))[both].mean()) if both.any() else 0.0
    blur = lambda l: cv2.GaussianBlur(l, (0, 0), 0.7)
    dE = float(np.linalg.norm(blur(lh) - blur(lo), axis=-1)[both].mean()) if both.any() else 99.0
    a = hd[..., 3]
    edge = (a > 0) & (cv2.dilate((a == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    fringe = float(((a > 0) & (a < 0.99) & edge).sum() / max(edge.sum(), 1))
    return dict(iou=iou, fill=fill, dL=dL, dC=dC, dE=dE, fringe=fringe)


def main(src, pack, hand, kind="item", top=60):
    rows = []
    root = os.path.join(hand, "assets")
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".png") or f.endswith(("_n.png", "_s.png")):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            ns, path = rel.split(os.sep, 1)
            if f"/{kind}/" not in "/" + path.replace(os.sep, "/"):
                continue
            o = load(f"{src}/{ns}/{path}")
            h = load(f"{pack}/assets/{ns}/{path}")
            rows.append((f"{ns}:{f[:-4]}", score(o, h)))
    rows.sort(key=lambda r: -r[1]["dE"])
    print(f"{'item':44s} {'iou':>5s} {'fill':>5s} {'dL':>6s} {'dC':>6s} {'dE':>5s} {'fringe':>6s}")
    for name, s in rows[:top]:
        print(f"{name:44s} {s['iou']:5.2f} {s['fill']:5.2f} {s['dL']:6.1f} {s['dC']:6.1f} {s['dE']:5.1f} {s['fringe']:6.2f}")
    arr = np.array([[s[k] for k in ("iou", "fill", "dL", "dC", "dE", "fringe")] for _, s in rows])
    print(f"\n{len(rows)} textures. median iou {np.median(arr[:, 0]):.2f}  fill {np.median(arr[:, 1]):.2f}  "
          f"dL {np.median(arr[:, 2]):.1f}  dC {np.median(arr[:, 3]):.1f}  dE {np.median(arr[:, 4]):.1f}  "
          f"fringe {np.median(arr[:, 5]):.2f}")
    return rows


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "item"
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 60
    args = [a for a in args if a not in (kind, str(top))]
    main(*args[:3], kind=kind, top=top)
