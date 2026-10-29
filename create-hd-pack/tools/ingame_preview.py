"""How item textures look in the game, side by side with the originals.

Per item, one row:
  inventory slot at GUI scale 3 and 2 (original | HD), drawn the way the game
  draws atlas sprites: GL_NEAREST_MIPMAP_LINEAR over Minecraft-style mipmaps
  (2x2 alpha-weighted averages), so a 64px icon in a 32px slot shows mip 1;
  in hand: the generated item model, an extruded sprite (1/16 thick, a side
  wall on every non-transparent texel), magnified with nearest filtering as in
  first person, lit like vanilla (front 1.0, sides 0.6-0.8).

    python3 ingame_preview.py <src_assets> <pack_dir> <dest.png> ns/textures/item/x.png ...
"""
import math
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

PANEL, SLOT, DARK, LIGHT = (198, 198, 198), (139, 139, 139), (55, 55, 55), (255, 255, 255)


def load(path):
    a = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255
    return a[: a.shape[1]]                                    # first frame of an animation strip


def mips(tex):
    """Minecraft-style mip chain: each level averages 2x2 texels, colour
    weighted by alpha (so transparent texels don't darken edges)."""
    out = [tex]
    while out[-1].shape[0] > 1:
        t = out[-1]
        h = t.shape[0] // 2
        q = t[: h * 2, : h * 2].reshape(h, 2, h, 2, 4)
        a = q[..., 3:4]
        rgb = (q[..., :3] * a).sum((1, 3)) / np.maximum(a.sum((1, 3)), 1e-6)
        out.append(np.concatenate([rgb, a.mean((1, 3))], -1))
    return out


def sample_gui(tex, px):
    """The sprite drawn into a px x px screen square (GL_NEAREST_MIPMAP_LINEAR)."""
    lod = max(0.0, math.log2(tex.shape[0] / px))
    chain = mips(tex)
    lo = min(int(math.floor(lod)), len(chain) - 1)
    hi = min(lo + 1, len(chain) - 1)
    f = lod - math.floor(lod)
    near = lambda t: cv2.resize(t, (px, px), interpolation=cv2.INTER_NEAREST)
    return near(chain[lo]) * (1 - f) + near(chain[hi]) * f


def slot(tex, scale):
    """An inventory slot (18x18 GUI px) at a GUI scale, item composited (translucent sheet)."""
    s = 18 * scale
    img = np.zeros((s, s, 3), np.float32)
    img[:] = np.array(SLOT) / 255
    img[:scale, :] = np.array(DARK) / 255
    img[:, :scale] = np.array(DARK) / 255
    img[-scale:, scale:] = np.array(LIGHT) / 255
    img[scale:, -scale:] = np.array(LIGHT) / 255
    it = sample_gui(tex, 16 * scale)
    a = it[..., 3:4]
    img[scale:-scale, scale:-scale] = img[scale:-scale, scale:-scale] * (1 - a) + it[..., :3] * a
    return img


def in_hand(tex, size=300, yaw=38, tilt=12):
    """Extruded sprite seen at an angle, nearest-magnified (first person)."""
    n = tex.shape[0]
    rgb, a = tex[..., :3], tex[..., 3]
    solid = a > 0                                             # the game builds walls for any alpha > 0
    k = size * 0.62 / n
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(tilt))
    # texture (u, v) -> screen: compress u by the yaw, shear v a little for perspective
    M = np.array([[k * cy, 0, size * 0.5 - k * cy * n / 2],
                  [k * cy * sy, k, size * 0.5 - k * n / 2 - k * cy * sy * n / 2]], np.float32)
    depth = n / 16.0                                          # 1/16 of the item thick
    dvec = np.array([k * math.sin(math.radians(yaw)), -k * 0.15])
    out = np.zeros((size, size, 3), np.float32)
    out[:] = (0.38, 0.5, 0.62)                                # sky-ish backdrop
    acc = np.zeros((size, size), np.float32)
    steps = max(2, int(abs(dvec[0]) * depth))
    warp = lambda img, off: cv2.warpAffine(img, M + np.array([[0, 0, off[0]], [0, 0, off[1]]], np.float32),
                                           (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
    for i in range(steps, 0, -1):                             # side walls, back to front
        off = dvec * depth * i / steps
        side = warp(np.dstack([rgb * 0.68, solid.astype(np.float32) * a]), off)
        sa = side[..., 3:4]
        out = out * (1 - sa) + side[..., :3] * sa
    front = warp(np.dstack([rgb, a]), (0, 0))
    fa = front[..., 3:4]
    return out * (1 - fa) + front[..., :3] * fa


def to8(img):
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def row(orig, hd, name):
    cells = [slot(orig, 3), slot(hd, 3), slot(orig, 2), slot(hd, 2)]
    hand_o, hand_h = in_hand(orig, 220), in_hand(hd, 220)
    h = 220
    w = sum(c.shape[1] for c in cells) + 2 * 220 + 8 * 7
    canvas = np.zeros((h + 18, w, 3), np.float32)
    canvas[:] = np.array(PANEL) / 255
    x = 8
    for c in cells:
        y = (h - c.shape[0]) // 2
        canvas[y:y + c.shape[0], x:x + c.shape[1]] = c
        x += c.shape[1] + 8
    for c in (hand_o, hand_h):
        canvas[:h, x:x + c.shape[1]] = c
        x += c.shape[1] + 8
    im = Image.fromarray(to8(canvas))
    ImageDraw.Draw(im).text((8, h + 3), name, fill=(40, 40, 40))
    return im


def tile(orig, hd, name):
    """Compact cell: original slot | HD slot (GUI scale 3) over the HD in hand."""
    a, b = slot(orig, 3), slot(hd, 3)
    hand = in_hand(hd, 150)
    w = max(a.shape[1] * 2 + 8, 150)
    cell = np.zeros((a.shape[0] + 150 + 22, w, 3), np.float32)
    cell[:] = np.array(PANEL) / 255
    cell[:a.shape[0], :a.shape[1]] = a
    cell[:b.shape[0], a.shape[1] + 8:a.shape[1] * 2 + 8] = b
    cell[a.shape[0] + 4:a.shape[0] + 154, :150] = hand
    im = Image.fromarray(to8(cell))
    ImageDraw.Draw(im).text((2, a.shape[0] + 156), name[:24], fill=(40, 40, 40))
    return im


def grid(src, pack, dest, items, cols=6):
    cells = []
    for it in items:
        ns, rel = it.split("/", 1)
        cells.append(tile(load(f"{src}/{ns}/{rel}"), load(f"{pack}/assets/{ns}/{rel}"), rel.split("/")[-1][:-4]))
    cw, ch = cells[0].width + 10, cells[0].height + 6
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * min(cols, len(cells)), ch * rows), PANEL)
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * cw, (i // cols) * ch))
    sheet.save(dest)
    print(dest)


def main(src, pack, dest, items):
    rows = []
    for it in items:
        ns, rel = it.split("/", 1)
        rows.append(row(load(f"{src}/{ns}/{rel}"), load(f"{pack}/assets/{ns}/{rel}"), it.split("/")[-1][:-4]))
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows)), PANEL)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(dest)
    print(dest)


if __name__ == "__main__":
    if sys.argv[1] == "--grid":
        grid(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
