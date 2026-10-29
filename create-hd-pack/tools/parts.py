"""Reusable detail parts: mitred frames, bolts, rivets, planks, plates."""
import numpy as np
from texgen import (N, Material, blur, fbm, vnoise, worley, sstep, round_profile,
                    grid, lerp, col, scratches, _rng)

XX, YY = grid()


def edge_dist():
    """Distance (px) to the nearest texture edge."""
    return np.minimum(np.minimum(XX, N - XX), np.minimum(YY, N - YY))


def frame(mat, width, metal, seed, top=1.0, outer_bevel=7, inner_bevel=5,
          miter_groove=True, brushed=True, dents=0.0):
    """Mitred metal frame around the texture border.

    metal: dict(light, dark, f0, smooth, porosity)
    Returns frame mask.
    """
    d = edge_dist()
    din = width - d
    mask = sstep(-0.5, 0.5, din)
    prof = np.minimum(round_profile(d, outer_bevel), round_profile(din, inner_bevel))
    h = top * (0.55 + 0.45 * prof)

    # which bar a pixel belongs to: horizontal (top/bottom) or vertical
    dy_edge = np.minimum(YY, N - YY)
    dx_edge = np.minimum(XX, N - XX)
    horiz = dy_edge < dx_edge
    if miter_groove:
        miter = np.abs(dy_edge - dx_edge)
        h -= 0.10 * (1 - np.clip(miter / 1.6, 0, 1)) * mask

    # brushed grain running along each bar
    if brushed:
        bh = fbm(3, 96, 3, seed)          # streaks along x
        bv = fbm(96, 3, 3, seed + 7)      # streaks along y
        brush = np.where(horiz, bh, bv)
    else:
        brush = fbm(12, 12, 4, seed)
    tone = 0.55 * brush + 0.45 * fbm(6, 6, 4, seed + 3)
    h += (tone - 0.5) * 0.02 * mask
    if dents:
        w = worley(100, seed + 11, jitter=0.8)
        h -= dents * (1 - sstep(0, 12, w)) ** 2 * mask

    alb = lerp(metal["dark"], metal["light"], np.clip(tone * 1.3 - 0.15, 0, 1))
    sm = metal["smooth"] + (brush - 0.5) * 0.15
    mat.put(mask, albedo=alb, height=h, smooth=sm, f0=metal["f0"], porosity=metal.get("porosity", 0))
    return mask


def hex_bolt(mat, cx, cy, r, base_h, metal, washer, seed, slot=False):
    r_ = _rng(seed)
    ang = r_.uniform(0, np.pi / 3)
    px, py = XX - cx, YY - cy
    rr = np.hypot(px, py)

    # washer
    wr = r * 1.45
    wmask = sstep(wr + 0.6, wr - 0.6, rr)
    wh = base_h + 0.05 + 0.04 * round_profile(wr - rr, 2.0)
    mat.put(wmask, albedo=lerp(washer["dark"], washer["light"], 0.5 + 0.3 * np.cos(np.arctan2(py, px) * 2 + 1)),
            height=wh, smooth=washer["smooth"], f0=washer["f0"])

    # hexagon head
    d = np.full_like(rr, -1e9)
    for k in range(3):
        a = ang + k * np.pi / 3
        d = np.maximum(d, np.abs(px * np.cos(a) + py * np.sin(a)))
    hd = r - d
    hmask = sstep(-0.6, 0.6, hd)
    head_h = base_h + 0.09 + 0.16 * round_profile(hd, 2.5)
    # domed/chamfered top
    head_h -= 0.05 * np.clip(rr / r, 0, 1) ** 2
    shade = np.clip(0.55 + 0.35 * np.cos(np.arctan2(py, px) - ang * 3), 0, 1)
    alb = lerp(metal["dark"], metal["light"], shade * 0.6 + 0.3)
    mat.put(hmask, albedo=alb, height=head_h, smooth=metal["smooth"], f0=metal["f0"])
    if slot:
        sa = ang + np.pi / 6
        sd = np.abs(-px * np.sin(sa) + py * np.cos(sa))
        smask = sstep(1.6, 0.8, sd) * sstep(r * 0.8, r * 0.6, rr)
        mat.put(smask, albedo=metal["dark"] * 0.6, height=base_h + 0.08)
    # contact shadow around the washer
    mat.extra_ao *= 1 - 0.35 * sstep(wr + 5, wr, rr) * (1 - wmask)


def rivet(mat, cx, cy, r, base_h, metal):
    rr = np.hypot(XX - cx, YY - cy)
    m = sstep(r + 0.6, r - 0.6, rr)
    dome = np.sqrt(np.clip(1 - (rr / r) ** 2, 0, 1))
    shade = np.clip(0.5 + 0.5 * ((-(XX - cx) - (YY - cy)) / r), 0, 1)
    mat.put(m, albedo=lerp(metal["dark"], metal["light"], 0.35 + 0.5 * shade * dome),
            height=base_h + 0.03 + 0.10 * dome, smooth=metal["smooth"], f0=metal["f0"])
    mat.extra_ao *= 1 - 0.3 * sstep(r + 3, r, rr) * (1 - m)


def planks(mat, x0, y0, x1, y1, count, vertical, base_h, wood, seed,
           nails=None, knots=True):
    """Wooden planks filling a rectangle; grain runs along the plank."""
    r = _rng(seed)
    across = (XX - x0) if vertical else (YY - y0)
    along = (YY - y0) if vertical else (XX - x0)
    span_across = (x1 - x0) if vertical else (y1 - y0)
    span_along = (y1 - y0) if vertical else (x1 - x0)
    inside = ((XX >= x0) & (XX < x1) & (YY >= y0) & (YY < y1)).astype(float)

    pw = span_across / count
    idx = np.clip(np.floor(across / pw), 0, count - 1).astype(int)
    pos = across - idx * pw  # px into plank

    if vertical:
        g_fine = fbm(48, 2, 4, seed, 0.55)
        g_warp = fbm(4, 2, 3, seed + 5)
    else:
        g_fine = fbm(2, 48, 4, seed, 0.55)
        g_warp = fbm(2, 4, 3, seed + 5)

    alb = np.zeros((N, N, 3))
    h = np.full((N, N), base_h)
    for i in range(count):
        sel = idx == i
        off = int(r.integers(0, N))
        gf = np.roll(g_fine, off, axis=0 if vertical else 1)
        gw = np.roll(g_warp, off, axis=0 if vertical else 1)
        # growth rings: curved bands across the plank
        centre = r.uniform(0.2, 0.8) * pw
        ring_coord = np.abs(pos - centre) * 0.22 + gw * 6 + np.sin(along / span_along * np.pi * r.uniform(1, 2.5)) * 1.5
        # latewood: soft darker bands rather than bright lines
        late = sstep(0.35, 0.95, 0.5 + 0.5 * np.sin(ring_coord * 2 * np.pi))
        grain = np.clip(0.75 * gf + 0.25 * (1 - late), 0, 1)
        tint = r.uniform(-0.07, 0.07)
        c = lerp(wood["dark"], wood["light"], np.clip(gf * 1.1 + 0.05 + tint - late * 0.28, 0, 1))
        c = c * (1 + r.uniform(-0.06, 0.06))
        # knot
        if knots and r.random() < 0.55:
            kx = r.uniform(0.3, 0.7) * pw
            ky = r.uniform(0.15, 0.85) * span_along
            kd = np.hypot((pos - kx) * 1.0, (along - ky) * 0.45)
            km = sstep(9, 2, kd)
            kr = 0.5 + 0.5 * np.sin(kd * 1.4)
            c = lerp(c, wood["dark"] * (0.6 + 0.4 * kr[..., None]), km * 0.85)
            grain = grain - km * 0.3
        alb[sel] = c[sel]
        crown = 0.035 * (1 - ((pos / pw) * 2 - 1) ** 2)
        hp = base_h + crown + r.uniform(-0.015, 0.015) - (1 - gf) * 0.018 - late * 0.006
        h[sel] = hp[sel]

    # seams between planks (V grooves)
    edge = np.minimum(pos, pw - pos)
    groove = (1 - np.clip(edge / 3.2, 0, 1)) ** 1.6
    h -= 0.13 * groove
    alb = alb * (1 - 0.45 * groove)[..., None]
    sm = wood["smooth"] + (g_fine - 0.5) * 0.1 - groove * 0.1
    mat.put(inside, albedo=alb, height=h, smooth=sm, f0=0 * h + wood.get("f0", 10), porosity=wood.get("porosity", 12))

    if nails:
        margin, nr, nmetal = nails
        for i in range(count):
            c_across = (i + 0.5) * pw
            for c_along in (margin, span_along - margin):
                if vertical:
                    rivet(mat, x0 + c_across, y0 + c_along, nr, base_h + 0.02, nmetal)
                else:
                    rivet(mat, x0 + c_along, y0 + c_across, nr, base_h + 0.02, nmetal)
    return inside
