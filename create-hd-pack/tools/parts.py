"""Reusable detail parts: connected mitred frames, bolts, rivets, planks.

Everything that can continue onto a neighbouring block (planks, ribs, grain,
brushing) is periodic over the 256px tile, so connected tiles meet seamlessly.
"""
import numpy as np
from texgen import (N, Material, blur, fbm, vnoise, worley, sstep, round_profile,
                    grid, lerp, col, scratches, _rng)
from ct import Ctx

XX, YY = grid()
ISOLATED = Ctx(False, False, False, False)


def edge_dist():
    """Distance (px) to the nearest texture edge."""
    return np.minimum(np.minimum(XX, N - XX), np.minimum(YY, N - YY))


def frame_layout(ctx, width):
    """Frame mask pieces for a connection context.

    Returns (mask, horiz, outer_d, miter_d, joints):
      horiz   - pixel belongs to a horizontal bar (for brushing direction)
      outer_d - distance to the block's free outer edges (outer bevel)
      miter_d - distance to mitre seams
      joints  - (cx, cy) centres of corner joints that get a bolt
    """
    w = width
    top = (YY < w) & (not ctx.up)
    bot = (YY >= N - w) & (not ctx.down)
    lef = (XX < w) & (not ctx.left)
    rig = (XX >= N - w) & (not ctx.right)
    tl_i, tr_i, bl_i, br_i = ctx.inner()
    corners = {
        "tl": ((XX < w) & (YY < w), XX, YY),
        "tr": ((XX >= N - w) & (YY < w), N - XX, YY),
        "bl": ((XX < w) & (YY >= N - w), XX, N - YY),
        "br": ((XX >= N - w) & (YY >= N - w), N - XX, N - YY),
    }
    mask = top | bot | lef | rig
    horiz = top | bot
    big = np.full((N, N), 1e9)
    outer_d = big.copy()
    if not ctx.up:
        outer_d = np.minimum(outer_d, YY)
    if not ctx.down:
        outer_d = np.minimum(outer_d, N - YY)
    if not ctx.left:
        outer_d = np.minimum(outer_d, XX)
    if not ctx.right:
        outer_d = np.minimum(outer_d, N - XX)
    miter_d = big.copy()
    joints = []
    outer_flags = {"tl": not ctx.up and not ctx.left, "tr": not ctx.up and not ctx.right,
                   "bl": not ctx.down and not ctx.left, "br": not ctx.down and not ctx.right}
    inner_flags = {"tl": tl_i, "tr": tr_i, "bl": bl_i, "br": br_i}
    centres = {"tl": (w / 2, w / 2), "tr": (N - w / 2, w / 2), "bl": (w / 2, N - w / 2), "br": (N - w / 2, N - w / 2)}
    for k, (sq, lx, ly) in corners.items():
        if outer_flags[k]:
            # outer corner: mitre from the block corner; bar that enters from
            # the side is the horizontal one
            horiz = np.where(sq, ly < lx, horiz)
            miter_d = np.where(sq, np.abs(lx - ly) / np.sqrt(2), miter_d)
            joints.append(centres[k])
        elif inner_flags[k]:
            # inner corner post: joins the neighbours' bars, which meet at this
            # tile's corner point; round the bevel around that point
            mask = mask | sq
            horiz = np.where(sq, lx < ly, horiz)
            miter_d = np.where(sq, np.abs(lx - ly) / np.sqrt(2), miter_d)
            outer_d = np.where(sq, np.minimum(outer_d, np.hypot(lx, ly)), outer_d)
            joints.append(centres[k])
    return mask, horiz, outer_d, miter_d, joints


def dist_to(region, ctx, radius=10):
    """Distance (px, capped at radius) from each pixel to `region`.

    Beyond connected sides the region is extended by edge replication (the
    neighbour continues the same layout); beyond free sides it is empty.
    """
    p = radius + 1
    reg = np.pad(region, p, mode="edge")
    if not ctx.up:
        reg[:p, :] = False
    if not ctx.down:
        reg[-p:, :] = False
    if not ctx.left:
        reg[:, :p] = False
    if not ctx.right:
        reg[:, -p:] = False
    d = np.full((N, N), float(radius))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r = np.hypot(dx, dy)
            if r >= radius:
                continue
            shifted = reg[p + dy:p + dy + N, p + dx:p + dx + N]
            d = np.where(shifted & (r < d), r, d)
    return d


def frame(mat, width, metal, seed, ctx=ISOLATED, top=1.0, outer_bevel=7, inner_bevel=5,
          miter_groove=True, dents=0.0):
    """Mitred, brushed metal frame on every unconnected side of the tile."""
    fm, horiz, outer_d, miter_d, joints = frame_layout(ctx, width)
    panel = ~fm
    din = dist_to(panel, ctx, radius=inner_bevel + 3) - 0.5
    mask = fm.astype(float)
    prof = np.minimum(round_profile(outer_d, outer_bevel), round_profile(din, inner_bevel))
    h = top * (0.55 + 0.45 * prof)
    if miter_groove:
        h -= 0.10 * (1 - np.clip(miter_d / 1.2, 0, 1)) * mask

    # brushed grain running along each bar (periodic, so bars continue across blocks)
    bh = fbm(3, 96, 3, seed)          # streaks along x
    bv = fbm(96, 3, 3, seed + 7)      # streaks along y
    brush = np.where(horiz, bh, bv)
    tone = 0.55 * brush + 0.45 * fbm(6, 6, 4, seed + 3)
    h += (tone - 0.5) * 0.02 * mask
    if dents:
        w = worley(100, seed + 11, jitter=0.8)
        h -= dents * (1 - sstep(0, 12, w)) ** 2 * mask

    alb = lerp(metal["dark"], metal["light"], np.clip(tone * 1.3 - 0.15, 0, 1))
    sm = metal["smooth"] + (brush - 0.5) * 0.15
    mat.put(mask, albedo=alb, height=h, smooth=sm, f0=metal["f0"], porosity=metal.get("porosity", 0))
    return mask, joints


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


def wrap_d(a, b):
    """Shortest distance between coordinates on the 256px torus."""
    d = np.abs(a - b) % N
    return np.minimum(d, N - d)


def planks(mat, pitch, offset, vertical, base_h, wood, seed, knots=True):
    """Planks covering the whole tile, periodic over 256px.

    Plank seams fall at offset + k*pitch (mod 256); 256 must divide by pitch.
    Grain runs along the plank. The frame is painted over this afterwards.
    """
    assert N % pitch == 0
    count = N // pitch
    r = _rng(seed)
    across = ((XX if vertical else YY) - offset) % N
    along = YY if vertical else XX
    idx = np.floor(across / pitch).astype(int)
    pos = across - idx * pitch  # px into plank

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
        # growth rings: curved bands across the plank (wave count is an
        # integer per 256px so the figure tiles along the plank)
        centre = r.uniform(0.2, 0.8) * pitch
        waves = int(r.integers(1, 3))
        ring_coord = np.abs(pos - centre) * 0.22 + gw * 6 + np.sin(along / N * 2 * np.pi * waves) * 1.5
        # latewood: soft darker bands rather than bright lines
        late = sstep(0.35, 0.95, 0.5 + 0.5 * np.sin(ring_coord * 2 * np.pi))
        grain = np.clip(0.75 * gf + 0.25 * (1 - late), 0, 1)
        tint = r.uniform(-0.07, 0.07)
        c = lerp(wood["dark"], wood["light"], np.clip(gf * 1.1 + 0.05 + tint - late * 0.28, 0, 1))
        c = c * (1 + r.uniform(-0.06, 0.06))
        if knots and r.random() < 0.55:
            kx = r.uniform(0.3, 0.7) * pitch
            ky = r.uniform(0, N)
            kd = np.hypot(pos - kx, wrap_d(along, ky) * 0.45)
            km = sstep(9, 2, kd)
            kr = 0.5 + 0.5 * np.sin(kd * 1.4)
            c = lerp(c, wood["dark"] * (0.6 + 0.4 * kr[..., None]), km * 0.85)
        alb[sel] = c[sel]
        crown = 0.035 * (1 - ((pos / pitch) * 2 - 1) ** 2)
        hp = base_h + crown + r.uniform(-0.015, 0.015) - (1 - gf) * 0.018 - late * 0.006
        h[sel] = hp[sel]

    # seams between planks (V grooves)
    edge = np.minimum(pos, pitch - pos)
    groove = (1 - np.clip(edge / 3.2, 0, 1)) ** 1.6
    h -= 0.13 * groove
    alb = alb * (1 - 0.45 * groove)[..., None]
    sm = wood["smooth"] + (g_fine - 0.5) * 0.1 - groove * 0.1
    mat.put(np.ones((N, N)), albedo=alb, height=h, smooth=sm, f0=0 * h + wood.get("f0", 10),
            porosity=wood.get("porosity", 12))


def plank_nails(mat, ctx, pitch, offset, vertical, frame_w, margin, r, base_h, metal):
    """Nails near each frame bar that a plank end runs into.

    Nails sitting on a block seam are drawn at both x=0 and x=256 so each
    neighbour paints its half.
    """
    centres = [(offset + pitch / 2 + k * pitch) % N for k in range(N // pitch)]
    centres = centres + [c + N for c in centres if c < r + 1] + [c - N for c in centres if c > N - r - 1]
    if vertical:
        ends = [(frame_w + margin, not ctx.up), (N - frame_w - margin, not ctx.down)]
        lo_free, hi_free = not ctx.left, not ctx.right
    else:
        ends = [(frame_w + margin, not ctx.left), (N - frame_w - margin, not ctx.right)]
        lo_free, hi_free = not ctx.up, not ctx.down
    for a, bar in ends:
        if not bar:
            continue
        for c in centres:
            # skip nails hidden under a side bar
            if (lo_free and c < frame_w) or (hi_free and c > N - frame_w):
                continue
            if vertical:
                rivet(mat, c, a, r, base_h, metal)
            else:
                rivet(mat, a, c, r, base_h, metal)
