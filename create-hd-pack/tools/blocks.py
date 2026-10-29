"""Create block textures."""
import numpy as np
from texgen import (N, Material, derive, save, fbm, vnoise, worley, blur, sstep,
                    round_profile, lerp, col, scratches, IRON, COPPER, ALBEDO_METAL, DIELECTRIC)
from parts import XX, YY, ISOLATED, edge_dist, frame, hex_bolt, rivet, planks, plank_nails
from ct import omni_tiles, ctm_tiles

# ------------------------------------------------------------- palettes ---
ANDESITE = dict(light=col(176, 184, 174), dark=col(112, 120, 113), f0=IRON, smooth=0.48)
BRASS = dict(light=col(246, 205, 112), dark=col(176, 118, 48), f0=ALBEDO_METAL, smooth=0.72)
COPPER_M = dict(light=col(232, 136, 98), dark=col(158, 72, 48), f0=COPPER, smooth=0.62)
STEEL = dict(light=col(150, 150, 154), dark=col(70, 70, 76), f0=IRON, smooth=0.55)
DARK_IRON = dict(light=col(118, 118, 122), dark=col(62, 62, 68), f0=IRON, smooth=0.42)
SPRUCE = dict(light=col(168, 118, 70), dark=col(96, 62, 34), smooth=0.30, porosity=12)
DARK_OAK = dict(light=col(104, 68, 38), dark=col(46, 28, 15), smooth=0.34, porosity=12)

FW = 32  # frame width: 2 vanilla px, matching Create's casing layout
PITCH, OFFSET = 64, 32  # plank/rib seams at 32 + 64k: one seam on each inner frame edge


def casing(ctx, frame_metal, seed, panel_fn, bolt_metal=None, dents=0.0):
    m = Material()
    panel_fn(m, ctx)
    _, joints = frame(m, FW, frame_metal, seed, ctx=ctx, dents=dents)
    bm = bolt_metal or frame_metal
    for i, (bx, by) in enumerate(joints):
        hex_bolt(m, bx, by, 7.5, 0.98, bm, frame_metal, seed + 31 * i)
    return m


CASING_OPTS = dict(edges="reflect")


def andesite_casing(ctx=ISOLATED):
    def panel(m, ctx):
        planks(m, PITCH, OFFSET, True, 0.42, SPRUCE, 11)
        plank_nails(m, ctx, PITCH, OFFSET, True, FW, 14, 3.6, 0.44, STEEL)
    return casing(ctx, ANDESITE, 100, panel, bolt_metal=STEEL), dict(edge_wear=0.5, **CASING_OPTS)


def brass_casing(ctx=ISOLATED):
    def panel(m, ctx):
        planks(m, PITCH, OFFSET, False, 0.42, DARK_OAK, 21)
        plank_nails(m, ctx, PITCH, OFFSET, False, FW, 14, 3.6, 0.44, BRASS)
    return casing(ctx, BRASS, 200, panel, dents=0.006), dict(edge_wear=0.45, **CASING_OPTS)


def copper_casing(ctx=ISOLATED):
    def panel(m, ctx):
        u = (XX - FW) / 32.0                      # 32px ribs, periodic over 256
        f = u - np.floor(u)
        prof = np.sin(f * np.pi) ** 0.6           # rounded corrugation
        grime = fbm(8, 8, 5, 41)
        s = scratches(60, 42, length=(6, 30), angle=np.pi / 2)
        h = 0.36 + 0.10 * prof + (grime - 0.5) * 0.015 - s * 0.01
        alb = lerp(STEEL["dark"] * 0.85, STEEL["light"] * 0.8, np.clip(0.25 + prof * 0.45 + (grime - 0.5) * 0.5, 0, 1))
        alb = lerp(alb, col(150, 150, 156), s * 0.25)
        m.put(np.ones((N, N)), albedo=alb, height=h, smooth=0.45 + s * 0.2 - (1 - prof) * 0.15, f0=IRON)
    return casing(ctx, COPPER_M, 300, panel, dents=0.006), dict(edge_wear=0.5, **CASING_OPTS)


# Create 6 casings that use the 8x8 OMNIDIRECTIONAL sheet (AllSpriteShifts)
CT_CASINGS = {"andesite_casing": andesite_casing, "brass_casing": brass_casing, "copper_casing": copper_casing}


def build_connected(name, outdir):
    """Write <name>_connected.png (+_n, _s): 8x8 tiles of 256px, row-major."""
    fn = CT_CASINGS[name]
    sheets = [np.zeros((8 * N, 8 * N, 4), np.uint8) for _ in range(3)]
    # unused slots: flat normal at full height, non-emissive specular, so no
    # shader reads stray emission or depth from them through mipmaps
    sheets[1][...] = (128, 128, 255, 255)
    sheets[2][...] = (0, DIELECTRIC, 0, 255)
    for i, ctx in omni_tiles().items():
        m, opts = fn(ctx)
        maps = derive(m, **opts)
        y, x = (i // 8) * N, (i % 8) * N
        for sh, mp in zip(sheets, maps):
            sh[y:y + N, x:x + N] = mp
    save(name + "_connected", sheets, outdir)
    return sheets


def plate_dist(ctx):
    """Distance to the edge of the connected plate: free sides and inner corner points."""
    d = np.full((N, N), 1e9)
    if not ctx.up:
        d = np.minimum(d, YY)
    if not ctx.down:
        d = np.minimum(d, N - YY)
    if not ctx.left:
        d = np.minimum(d, XX)
    if not ctx.right:
        d = np.minimum(d, N - XX)
    for flag, (cx, cy) in zip(ctx.inner(), ((0, 0), (N, 0), (0, N), (N, N))):
        if flag:
            d = np.minimum(d, np.hypot(XX - cx, YY - cy))
    return d


def industrial_iron_block(ctx=ISOLATED, top=False):
    """Side texture; top=True gives industrial_iron_block_top (cube_column end), which
    adds vertical ribbing inside the panel line like Create's original."""
    m = Material()
    d = plate_dist(ctx)
    tone = 0.6 * fbm(10, 10, 5, 51) + 0.4 * fbm(48, 48, 3, 55)
    hammer = worley(144, 52, jitter=0.9)
    dent = (1 - sstep(0, 13, hammer)) ** 2
    s = scratches(55, 53, length=(5, 22))
    h = 0.55 + 0.45 * round_profile(d, 6) - dent * 0.012 + (tone - 0.5) * 0.015 - s * 0.01
    # inset panel line 1.5 vanilla px in from the plate edge (follows the whole connected plate)
    line = np.abs(d - 24)
    h -= 0.16 * (1 - np.clip(line / 2.5, 0, 1)) ** 1.5
    if top:
        u = (XX / 16.0) % 1.0                      # 16px ribs, periodic over 256
        rib = np.sin(u * np.pi) ** 0.8
        h += 0.035 * rib * sstep(28, 34, d)
    alb = lerp(DARK_IRON["dark"], DARK_IRON["light"], np.clip(0.2 + (tone - 0.5) * 0.7 + 0.35, 0, 1))
    alb = lerp(alb, col(140, 140, 146), s * 0.3)
    if top:
        alb = alb * (0.9 + 0.12 * rib * sstep(28, 34, d))[..., None]
    # faint rust bloom in the panel groove
    rust = sstep(0.5, 0.75, fbm(16, 16, 4, 54)) * (1 - np.clip(line / 7, 0, 1))
    alb = lerp(alb, col(118, 70, 44), rust * 0.65)
    m.put(np.ones((N, N)), albedo=alb, height=h, smooth=DARK_IRON["smooth"] + s * 0.15 - rust * 0.25, f0=IRON)
    m.f0 = np.where(rust > 0.5, DIELECTRIC + 30, m.f0)
    m.porosity = rust * 30
    # rivets: plate corners, inner corners and the middle of each free edge
    pts = []
    for flag, p in zip((not ctx.up and not ctx.left, not ctx.up and not ctx.right,
                        not ctx.down and not ctx.left, not ctx.down and not ctx.right),
                       ((12, 12), (N - 12, 12), (12, N - 12), (N - 12, N - 12))):
        if flag:
            pts.append(p)
    for flag, p in zip(ctx.inner(), ((9, 9), (N - 9, 9), (9, N - 9), (N - 9, N - 9))):
        if flag:
            pts.append(p)
    for free, p in ((not ctx.up, (N / 2, 12)), (not ctx.down, (N / 2, N - 12)),
                    (not ctx.left, (12, N / 2)), (not ctx.right, (N - 12, N / 2))):
        if free:
            pts.append(p)
    for rx, ry in pts:
        rivet(m, rx, ry, 5.5, 0.97, STEEL)
    return m, dict(edge_wear=0.4, edges="reflect")


def industrial_iron_block_top(ctx=ISOLATED):
    return industrial_iron_block(ctx, top=True)


def build_ctm(name, fn, outdir_assets, block_id):
    """Continuity / OptiFine CTM: 47 tiles + _n/_s each, and the .properties file.

    Matches the sprite (matchTiles) as well as the block, so only the faces that
    use create:block/<name> are replaced (cube_column blocks have separate ends).
    """
    import os
    folder = f"{outdir_assets}/create/optifine/ctm/{name}"
    os.makedirs(folder, exist_ok=True)
    tiles = {}
    for i, ctx in ctm_tiles().items():
        m, opts = fn(ctx)
        maps = derive(m, **opts)
        save(str(i), maps, folder)
        tiles[i] = maps
    with open(f"{folder}/{name}.properties", "w") as f:
        f.write(f"# Continuity connected textures (OptiFine CTM format)\n"
                f"method=ctm\nmatchBlocks={block_id}\nmatchTiles=create:block/{name}\n"
                f"tiles=0-46\nconnect=block\n")
    return tiles


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
    "industrial_iron_block_top": industrial_iron_block_top,
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
    for name in CT_CASINGS:
        sheets = build_connected(name, outdir)
        # tile 0 (no connections) must match the plain texture
        assert all((sh[:N, :N] == mp).all() for sh, mp in zip(sheets, out[name]))
        out[name + "_connected"] = sheets
        print("built", name + "_connected")
    assets = outdir.split("/create/textures/block")[0]
    for name, fn in (("industrial_iron_block", industrial_iron_block),
                     ("industrial_iron_block_top", industrial_iron_block_top)):
        tiles = build_ctm(name, fn, assets, "create:industrial_iron_block")
        # tile 0 (no connections) must match the plain texture
        assert all((t == mp).all() for t, mp in zip(tiles[0], out[name]))
        out[name + "_ctm"] = tiles
        print("built", name, "continuity ctm")
    return out


if __name__ == "__main__":
    import sys
    build(sys.argv[1])
