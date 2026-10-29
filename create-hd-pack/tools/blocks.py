"""Sample Create block textures (first look)."""
import numpy as np
from texgen import (N, Material, derive, save, fbm, vnoise, worley, blur, sstep,
                    round_profile, lerp, col, scratches, IRON, COPPER, ALBEDO_METAL, DIELECTRIC)
from parts import XX, YY, edge_dist, frame, hex_bolt, rivet, planks

# ------------------------------------------------------------- palettes ---
ANDESITE = dict(light=col(176, 184, 174), dark=col(112, 120, 113), f0=IRON, smooth=0.48)
BRASS = dict(light=col(246, 205, 112), dark=col(176, 118, 48), f0=ALBEDO_METAL, smooth=0.72)
COPPER_M = dict(light=col(232, 136, 98), dark=col(158, 72, 48), f0=COPPER, smooth=0.62)
STEEL = dict(light=col(150, 150, 154), dark=col(70, 70, 76), f0=IRON, smooth=0.55)
DARK_IRON = dict(light=col(118, 118, 122), dark=col(62, 62, 68), f0=IRON, smooth=0.42)
SPRUCE = dict(light=col(168, 118, 70), dark=col(96, 62, 34), smooth=0.30, porosity=12)
DARK_OAK = dict(light=col(104, 68, 38), dark=col(46, 28, 15), smooth=0.34, porosity=12)

FW = 32  # frame width: 2 vanilla px, matching Create's casing layout


def casing(frame_metal, seed, panel_fn, bolt_metal=None, dents=0.0):
    m = Material()
    panel_fn(m)
    frame(m, FW, frame_metal, seed, dents=dents)
    bm = bolt_metal or frame_metal
    c = FW / 2
    for i, (bx, by) in enumerate(((c, c), (N - c, c), (c, N - c), (N - c, N - c))):
        hex_bolt(m, bx, by, 7.5, 0.98, bm, frame_metal, seed + 31 * i)
    return m


def andesite_casing():
    def panel(m):
        planks(m, FW, FW, N - FW, N - FW, 4, True, 0.42, SPRUCE, 11,
               nails=(14, 3.6, STEEL))
    return casing(ANDESITE, 100, panel, bolt_metal=STEEL), dict(edge_wear=0.5)


def brass_casing():
    def panel(m):
        planks(m, FW, FW, N - FW, N - FW, 4, False, 0.42, DARK_OAK, 21,
               nails=(14, 3.6, BRASS))
    return casing(BRASS, 200, panel, dents=0.006), dict(edge_wear=0.45)


def copper_casing():
    def panel(m):
        inside = ((XX >= FW) & (XX < N - FW) & (YY >= FW) & (YY < N - FW)).astype(float)
        ribs = 6
        u = (XX - FW) / (N - 2 * FW) * ribs
        f = u - np.floor(u)
        prof = np.sin(f * np.pi) ** 0.6           # rounded corrugation
        grime = fbm(8, 8, 5, 41)
        s = scratches(60, 42, length=(6, 30), angle=np.pi / 2)
        h = 0.36 + 0.10 * prof + (grime - 0.5) * 0.015 - s * 0.01
        alb = lerp(STEEL["dark"] * 0.85, STEEL["light"] * 0.8, np.clip(0.25 + prof * 0.45 + (grime - 0.5) * 0.5, 0, 1))
        alb = lerp(alb, col(150, 150, 156), s * 0.25)
        m.put(inside, albedo=alb, height=h, smooth=0.45 + s * 0.2 - (1 - prof) * 0.15, f0=IRON)
    return casing(COPPER_M, 300, panel, dents=0.006), dict(edge_wear=0.5)


def industrial_iron_block():
    m = Material()
    d = edge_dist()
    tone = 0.6 * fbm(10, 10, 5, 51) + 0.4 * fbm(48, 48, 3, 55)
    hammer = worley(144, 52, jitter=0.9)
    dent = (1 - sstep(0, 13, hammer)) ** 2
    s = scratches(55, 53, length=(5, 22))
    h = 0.55 + 0.45 * round_profile(d, 6) - dent * 0.012 + (tone - 0.5) * 0.015 - s * 0.01
    # inset panel line at 1.5 vanilla px
    line = np.abs(d - 24)
    h -= 0.16 * (1 - np.clip(line / 2.5, 0, 1)) ** 1.5
    alb = lerp(DARK_IRON["dark"], DARK_IRON["light"], np.clip(0.2 + (tone - 0.5) * 0.7 + 0.35, 0, 1))
    alb = lerp(alb, col(140, 140, 146), s * 0.3)
    # faint rust bloom in the panel groove
    rust = sstep(0.5, 0.75, fbm(16, 16, 4, 54)) * (1 - np.clip(line / 7, 0, 1))
    alb = lerp(alb, col(118, 70, 44), rust * 0.65)
    m.put(np.ones((N, N)), albedo=alb, height=h, smooth=DARK_IRON["smooth"] + s * 0.15 - rust * 0.25, f0=IRON)
    m.f0 = np.where(rust > 0.5, DIELECTRIC + 30, m.f0)
    m.porosity = rust * 30
    for i, (rx, ry) in enumerate(((12, 12), (N - 12, 12), (12, N - 12), (N - 12, N - 12))):
        rivet(m, rx, ry, 5.5, 0.97, STEEL)
    return m, dict(edge_wear=0.4)


def brass_block():
    m = Material()
    d = edge_dist()
    u = ((XX + YY) % N).astype(int)  # diagonal coordinate, still tiles
    fine = fbm(128, 1, 2, 62)[0][u]
    bands = 0.5 + 0.5 * np.sin((XX + YY) / N * 2 * np.pi * 3 + (fbm(4, 4, 3, 65) - 0.5) * 2.5)
    tone = fbm(6, 6, 5, 63)
    h = 0.62 + 0.38 * round_profile(d, 9) + (fine - 0.5) * 0.008
    t = np.clip(0.35 + 0.35 * bands + 0.25 * (fine - 0.5) + 0.3 * (tone - 0.5), 0, 1)
    alb = lerp(BRASS["dark"], BRASS["light"], t)
    # soft tarnish creeping in from the rim
    tarnish = (1 - np.clip(d / 20, 0, 1)) * 0.3 + sstep(0.62, 0.8, fbm(8, 8, 4, 66)) * 0.15
    alb = lerp(alb, col(130, 86, 36), tarnish)
    m.put(np.ones((N, N)), albedo=alb, height=h, smooth=0.8 + (fine - 0.5) * 0.12 - tarnish * 0.35, f0=ALBEDO_METAL)
    return m, dict(edge_wear=0.35)


def _rod(axis_vertical=True):
    """Shaft (create:block/axis): only u 6..10 of the 16px texture is used."""
    m = Material()
    m.alpha[:] = 0
    x0, x1 = 6 * 16, 10 * 16
    w = x1 - x0
    inside = ((XX >= x0) & (XX < x1)).astype(float)
    t = np.clip((XX - x0) / w, 0, 1) * 2 - 1
    cyl = np.sqrt(np.clip(1 - t ** 2, 0, 1))
    brush = fbm(64, 2, 3, 71)
    tone = fbm(8, 4, 4, 72)
    # machined collars every 4 vanilla px
    collar = np.abs(((YY / 64.0) % 1.0) - 0.5) * 64
    ring = (1 - np.clip(collar / 3.0, 0, 1)) ** 1.5
    h = 0.55 + 0.4 * cyl - ring * 0.08 + (brush - 0.5) * 0.015
    shade = 0.35 + 0.65 * cyl * (0.75 + 0.25 * -t)
    alb = lerp(ANDESITE["dark"] * 0.9, ANDESITE["light"], np.clip(shade * (0.85 + 0.3 * (tone - 0.5)) + (brush - 0.5) * 0.25, 0, 1))
    alb = alb * (1 - 0.35 * ring)[..., None]
    m.put(inside, albedo=alb, height=h, smooth=0.55 + (brush - 0.5) * 0.2, f0=IRON)
    m.alpha = inside
    return m, dict(edge_wear=0.3)


def axis():
    return _rod()


def axis_top():
    m = Material()
    m.alpha[:] = 0
    cx = cy = N / 2
    rr = np.hypot(XX - cx, YY - cy)
    sq = (np.abs(XX - cx) < 32) & (np.abs(YY - cy) < 32)
    inside = sq.astype(float)
    # end face: square stock with a turned centre and a chamfered rim
    d = np.minimum(32 - np.abs(XX - cx), 32 - np.abs(YY - cy))
    tone = fbm(8, 8, 4, 81)
    lathe = 0.5 + 0.5 * np.sin(rr * 1.6 + (tone - 0.5) * 3)
    h = 0.6 + 0.4 * round_profile(d, 6) - 0.12 * sstep(20, 17, rr) + 0.06 * sstep(6, 4, rr)
    alb = lerp(ANDESITE["dark"], ANDESITE["light"], np.clip(0.4 + 0.25 * lathe * sstep(21, 17, rr) + (tone - 0.5) * 0.4, 0, 1))
    m.put(inside, albedo=alb, height=h, smooth=0.5 + 0.2 * sstep(21, 17, rr), f0=IRON)
    m.alpha = inside
    return m, dict(edge_wear=0.4)


SAMPLES = {
    "andesite_casing": andesite_casing,
    "brass_casing": brass_casing,
    "copper_casing": copper_casing,
    "industrial_iron_block": industrial_iron_block,
    "brass_block": brass_block,
    "axis": axis,
    "axis_top": axis_top,
}


def build(outdir):
    out = {}
    for name, fn in SAMPLES.items():
        m, opts = fn()
        maps = derive(m, **opts)
        save(name, maps, outdir)
        out[name] = maps
        print("built", name)
    return out


if __name__ == "__main__":
    import sys
    build(sys.argv[1])
