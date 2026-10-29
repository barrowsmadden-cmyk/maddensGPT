"""Hand-made Create block textures (machines, storage, kinetics).

Each design is drawn with the block kit (handmade/blockkit.py) in the
original texture's texel coordinates, traced from the original (its colour
regions and their positions) and coloured from its own texels, at the
casings' density (16 HD px per original texel). DESIGNS maps texture paths
(under assets/create/textures/) to drawing functions.
"""
import numpy as np

from handmade.blockkit import Kit, mat
from texgen import IRON, GOLD, COPPER, ALBEDO_METAL, DIELECTRIC


# ------------------------------------------------------------- vault ---
# All 12 vault textures (small/medium/large x front/side/top/bottom; the medium
# and large ones are Create's 4x4 connected-texture sheets) share one palette:
# light steel frame bars, a green-grey rim, violet-grey grooves and a blue
# steel plate. One colour map paints them all, the layout staying Create's.

VAULT_FRAME = {(0xbc, 0xb6, 0xae), (0x9f, 0x9b, 0x93), (0x84, 0x82, 0x77), (0x69, 0x69, 0x62)}
VAULT_RIM = {(0x53, 0x57, 0x51), (0x41, 0x48, 0x44), (0x30, 0x3a, 0x38), (0x25, 0x2f, 0x2d)}
VAULT_PLATE = {(0x56, 0x6b, 0x81), (0x4b, 0x5e, 0x71), (0x49, 0x56, 0x69), (0x46, 0x4e, 0x61), (0x44, 0x48, 0x5a),
               (0x41, 0x41, 0x50)}
VAULT_GROOVE = {(0x3d, 0x3c, 0x48), (0x39, 0x38, 0x42), (0x35, 0x34, 0x3d), (0x31, 0x30, 0x37), (0x2b, 0x2b, 0x31),
                (0x25, 0x25, 0x2b)}


def tone(cs):
    """(light, dark) of a colour set, 0-1 RGB."""
    c = sorted(cs, key=lambda v: 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2])
    return np.array(c[-1]) / 255, np.array(c[0]) / 255


def vault(k, bottom=False):
    """Faces: light steel rails over one blue steel plate whose light and shade
    (the cylinder's darker side) come from the original, smoothed; the rim sits
    low; bolt specks become round bolts. Bottoms: green-grey girders raised over
    a dark back."""
    frame = mat("metal", *tone(VAULT_FRAME), smooth=0.5, brush=0.05)
    plate = mat("metal", *tone(VAULT_PLATE), f0=ALBEDO_METAL, smooth=0.6, brush=0.025)
    rim = mat("metal", *tone(VAULT_RIM), smooth=0.4, brush=0.03)
    if bottom:
        spec = {"frame": dict(M=frame, h=0.85, bevel=0.5, shade=0.6),
                "girder": dict(M=rim, h=0.7, bevel=0.45, shade=0.5, smooth_px=0.7),
                "back": dict(M=mat("metal", *tone(VAULT_GROOVE), smooth=0.35), h=0.35, bevel=0.3, shade=0.6)}
        groups = (("frame", VAULT_FRAME), ("girder", VAULT_RIM), ("back", VAULT_PLATE | VAULT_GROOVE))
    else:
        spec = {"frame": dict(M=frame, h=0.9, bevel=0.55, shade=0.55),
                "plate": dict(M=plate, h=0.6, bevel=0.45, shade=1.0),
                "rim": dict(M=rim, h=0.3, bevel=0.3, shade=0.5)}
        groups = (("frame", VAULT_FRAME), ("rim", VAULT_RIM), ("plate", VAULT_PLATE | VAULT_GROOVE))

    def classify(c):
        return next((name for name, cs in groups if c in cs), groups[-1][0])
    k.classes(classify, spec, bolts={"frame"})


def vault_bottom(k):
    vault(k, bottom=True)


# ------------------------------------------------------ Create materials ---
# Create's machines share one material vocabulary. classify_create() sorts an
# original colour into it by hue and lightness; machine designs start from that
# and override per texture (a colour, a region, extra geometry).

import cv2 as _cv2


def lab(c):
    return _cv2.cvtColor(np.array([[c]], np.float32) / 255, _cv2.COLOR_RGB2LAB)[0, 0]


def classify_create(c):
    """Create's materials by lightness, colourfulness and hue (CIE Lab): copper is
    the red-orange family (hue < 50, including its dark shading), spruce and dark
    oak the browns (hue 50-100), brass the bright yellows, dark reds are recesses."""
    L, a, b = lab(c)
    chroma = float(np.hypot(a, b))
    hue = float(np.degrees(np.arctan2(b, a))) % 360
    if L < 13:
        return "hole"
    if chroma < 9:
        return "metal" if L > 38 else "darkmetal"
    if hue < 20 or hue > 330:
        return "other"                      # dark reds: pump and drain interiors, redstone
    if hue < 50 and chroma > 18:
        return "copper"
    if 72 <= hue <= 100 and chroma > 40 and L > 60:
        return "brass"
    if 45 <= hue <= 100:
        return "wood"
    if 180 <= hue <= 290:
        return "blue"
    return "other"


def create_spec(k, groups, overrides=None):
    """Materials for the classes present, toned from the texture's own texels in each class."""
    th, tw = k.src.shape[:2]
    rgb8 = np.round(k.src[..., :3] * 255).astype(int)
    by = {}
    for y in range(th):
        for x in range(tw):
            if k.src[y, x, 3] > 0.5:
                by.setdefault(groups(tuple(rgb8[y, x])), []).append(k.src[y, x, :3])
    def tones_of(px):
        px = np.array(px)
        L = px @ np.array([0.2126, 0.7152, 0.0722])
        o = np.argsort(L)
        px = px[o]
        n = len(px)
        return px[int(n * 0.6):].mean(0), px[: max(1, int(n * 0.4))].mean(0)
    base = {
        "metal": dict(kind="metal", f0=IRON, smooth=0.5, h=0.9, bevel=0.5, brush=0.05),
        "darkmetal": dict(kind="metal", f0=IRON, smooth=0.4, h=0.74, bevel=0.3, brush=0.04, round=True),
        "hole": dict(kind="dark", f0=DIELECTRIC, smooth=0.1, h=0.1, bevel=0.8, round=True),
        "wood": dict(kind="wood", f0=DIELECTRIC, smooth=0.28, h=0.6, bevel=0.3, porosity=12),
        "brass": dict(kind="metal", f0=ALBEDO_METAL, smooth=0.7, h=0.85, bevel=0.5, brush=0.04),
        "copper": dict(kind="metal", f0=COPPER, smooth=0.6, h=0.85, bevel=0.5, brush=0.04),
        "blue": dict(kind="metal", f0=ALBEDO_METAL, smooth=0.55, h=0.7, bevel=0.4, brush=0.03),
        "other": dict(kind="flat", f0=DIELECTRIC, smooth=0.3, h=0.6, bevel=0.3, shade=0.4),
    }
    spec = {}
    for name, px in by.items():
        b = dict(base.get(name, base["other"]))
        b.update((overrides or {}).get(name, {}))
        light, dark = tones_of(px)
        M = mat(b.pop("kind"), light, dark, f0=b.pop("f0"), smooth=b.pop("smooth"), porosity=b.pop("porosity", 0),
                **({"brush": b.pop("brush")} if "brush" in b else {}))
        spec[name] = dict(M=M, shade=b.pop("shade", 0.85), **b)
    return spec


COLOURED = {"wood", "copper", "brass", "other", "arch", "blue"}


def machine(k, overrides=None, groups=classify_create, bolts=("metal",)):
    """A Create machine face from its own colour regions, in Create's materials."""
    spec = create_spec(k, groups, overrides)
    for host in COLOURED:                       # absorbed shading may land in a class with no texels of its own
        spec.setdefault(host, spec.get("other") or next(iter(spec.values())))
    k.classes(groups, spec, bolts=set(bolts), absorb={"darkmetal": COLOURED, "hole": {"wood", "copper", "brass", "other"},
                                                      "metal": {"blue"}})


DESIGNS = {f"textures/block/vault/vault_{f}_{n}.png": vault_bottom if f == "bottom" else vault
           for f in ("front", "side", "top", "bottom") for n in ("small", "medium", "large")}
def cast_metal(k):
    """Girders, roller frames, bogeys: speckled cast/rolled steel -- one surface whose
    tone and gentle relief follow the original, its dark specks are shading not holes."""
    Lsrc = k.src[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    hi = Lsrc[k.src[..., 3] > 0.5].max() - 0.01                 # rivets: the texture's lightest colour

    def groups(c):
        g = classify_create(c)
        if g in ("metal", "darkmetal", "hole"):
            return "rivet" if (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255 >= hi else "cast"
        return g
    machine(k, overrides={"cast": dict(kind="metal", h=0.75, bevel=0.0, shade=1.0, relief=0.1, brush=0.04,
                                       smooth=0.45, f0=IRON),
                          "rivet": dict(kind="metal", h=0.85, bevel=0.3, shade=0.6, brush=0.02, smooth=0.6,
                                        f0=IRON)}, groups=groups, bolts=("rivet",))


def cast_stone(k, kind="metal"):
    """Basin and chute: dense cast andesite. The original's grey texels are one
    surface whose tone and gentle relief follow the artist's light pattern (rim
    bands, the emblem ring); speckle stays tone, never bumps."""
    def groups(c):
        g = classify_create(c)
        return "cast" if g in ("metal", "darkmetal") else g
    machine(k, overrides={"cast": dict(kind=kind, h=0.75, bevel=0.0, shade=1.0, relief=0.12, brush=0.015,
                                       smooth=0.32, f0=IRON if kind == "metal" else DIELECTRIC),
                          "hole": dict(round=False)}, groups=groups, bolts=())


def basin(k):
    """Basin (32px atlas). Outer wall: dense cast andesite, raised rim band, a
    square emblem ring with a dark centre, thicker corner feet, a recessed foot
    line. Top: a raised rim ring round the darker wall tops. Inner wall and
    floor: darker cast, lit toward the top."""
    cast = mat("metal", *k.tones((2, 6, 14, 12)), smooth=0.32, brush=0.015)
    light = mat("metal", *k.tones((0, 0, 16, 2), (5, 1, 11, 2)), smooth=0.36, brush=0.015)
    dark = mat("metal", *k.tones((18, 18, 30, 30)), smooth=0.25, brush=0.01)
    inner = mat("metal", *k.tones((18, 2, 30, 13)), smooth=0.28, brush=0.01)
    # outer wall (0,0)-(16,14) and its foot strip (2,14)-(14,16)
    k.rect(0, 0, 16, 14, cast, h=0.6, bevel=0)
    k.rect(0, 0, 16, 2.2, light, h=0.72, bevel=0.4)
    k.rect(0, 2.2, 1.6, 13, cast, h=0.66, bevel=0.35)
    k.rect(14.4, 2.2, 16, 13, cast, h=0.66, bevel=0.35)
    k.frame(5.4, 2.6, 10.6, 7.8, 1.2, light, h=0.7, bevel=0.35)
    k.rect(6.8, 4.0, 9.2, 6.4, dark, h=0.5, bevel=0.3)
    k.rect(0, 12.6, 16, 14, dark, h=0.48, bevel=0.25)
    k.shade_rect(0, 0, 16, 14, 0.5)
    k.rect(2, 14, 14, 16, cast, h=0.6, bevel=0.2)
    # top: rim ring (0,16)-(16,32) round darker wall tops
    k.rect(0, 16, 16, 32, dark, h=0.5, bevel=0)
    k.frame(0, 16, 16, 32, 2.2, light, h=0.72, bevel=0.45)
    k.shade_rect(0, 16, 16, 32, 0.4)
    # inner wall (18,0)-(30,14) and floor (18,18)-(30,30)
    k.rect(18, 0, 30, 14, inner, h=0.55, bevel=0)
    k.shade_rect(18, 0, 30, 14, 0.9)
    k.rect(18, 18, 30, 30, dark, h=0.45, bevel=0)
    k.shade_rect(18, 18, 30, 30, 0.6)


def millstone(k):
    """Millstone (32px atlas). Grinding stone top with a raised square ring and a
    centre pit; light stone bands; the base band with its vertical slits; a
    wooden hopper round a dark hole."""
    stone = mat("stone", *k.tones((1, 1, 11, 11)), smooth=0.2)
    pale = mat("stone", *k.tones((0, 26, 16, 32)), smooth=0.2)
    side = mat("stone", *k.tones((0, 12, 12, 17), (0, 20, 18, 26)), smooth=0.2)
    pit = mat("dark", *k.tones((5, 5, 7, 7)))
    # grinding top (0,0)-(12,12)
    k.rect(0, 0, 12, 12, stone, h=0.6, bevel=0)
    k.frame(1.5, 1.5, 10.5, 10.5, 1.4, stone, h=0.72, bevel=0.35)
    k.rect(4.6, 4.6, 7.4, 7.4, pit, h=0.3, bevel=0.5)
    k.shade_rect(0, 0, 12, 12, 0.6)
    # stone sides and rotating band (0,12)-(21,26)
    k.rect(0, 12, 12, 17, side, h=0.6, bevel=0.2)
    k.rect(0, 17, 18, 20, side, h=0.6, bevel=0.2)
    k.rect(0, 20, 21, 26, side, h=0.6, bevel=0.2)
    for y in (14.5, 23.0):
        k.groove(np.abs(k.y - y) + 99 * ((k.x > 21) | ((k.y < 12) | (k.y > 26))), depth=0.05)
    k.shade_rect(0, 12, 21, 26, 0.6)
    # base band (0,26)-(16,32) with slits
    k.rect(0, 26, 16, 32, pale, h=0.65, bevel=0.2)
    notch = mat("stone", *k.tones((0, 12, 12, 17)), smooth=0.18)
    for x in (2.7, 6.7, 9.3, 12.9):
        k.rect(x, 28.4, x + 0.6, 31.2, notch, h=0.52, bevel=0.25)
    k.shade_rect(0, 26, 16, 32, 0.5)
    # wooden hopper (20,4)-(32,16) and its sides (20,16)-(32,20)
    wood = mat("wood", *k.tones((20, 4, 32, 16)), porosity=12)
    k.rect(20, 4, 32, 20, wood, h=0.6, bevel=0.2, dir="h")
    k.frame(20, 4, 32, 16, 2.4, wood, h=0.66, bevel=0.3)
    k.rect(23.4, 7.4, 28.6, 12.6, mat("dark", (0.16, 0.12, 0.08), (0.02, 0.02, 0.02)), h=0.1, bevel=0.8)
    k.shade_rect(20, 4, 32, 20, 0.5)


def fan_side(k):
    """Encased fan side: wooden panels, each with a raised lighter arch (the
    fan's housing), in the andesite frame."""
    def groups(c):
        g = classify_create(c)
        if g == "wood" and lab(c)[0] > 34:
            return "arch"
        return g
    machine(k, overrides={"arch": dict(kind="wood", h=0.68, bevel=0.35, porosity=12, smooth=0.3, shade=0.6,
                                       smooth_px=0.6)},
            groups=groups)


MACHINE_TEXTURES = """
gearbox gearbox_top brass_gearbox depot_top depot_side mechanical_press_top mechanical_press_side mechanical_press_bottom
mechanical_press_head mechanical_press_pole fan_casing fan_side fan_blades funnel/andesite_funnel_frame mixer_base_side
mixer_head basin millstone andesite_casing_piston andesite_casing_short andesite_casing_very_short piston_bottom
piston_inner crushing_wheel_insert crushing_wheel_plates chute chute_hole encased_pipe spout copper_underside
item_drain_side item_drain_top pump mechanical_saw_top_no_slot mechanical_drill_top encased_chain_drive
blaze_heater_brazier fluid_valve valve_open chute_large chute_smart chute_smart_top funnel/funnel_open crafter_side
crafter_thingies smooth_dark_log_top engine flywheel weighted_ejector_casing cogwheel cogwheel_axis large_cogwheel
waterwheel_metal windmill_bearing_side mechanical_bearing_side clutch_off clutch_on gearshift_off gearshift_on
""".split()
for _n in MACHINE_TEXTURES:
    DESIGNS[f"textures/block/{_n}.png"] = machine
for _n in ("chute", "chute_large", "chute_smart", "chute_smart_top"):
    DESIGNS[f"textures/block/{_n}.png"] = cast_stone
DESIGNS["textures/block/basin.png"] = basin
DESIGNS["textures/block/millstone.png"] = millstone
DESIGNS["textures/block/fan_side.png"] = fan_side


# ------------------------------------------------- copper decor, tanks ---
ROOT = None   # assets root, for designs that read a sibling texture (set by the caller)


def _copper_f0(c):
    """Bare copper is the copper metal; patina (green/teal) and dark grime are dielectric."""
    L, a, b = lab(tuple(np.asarray(c) * 255))
    hue = float(np.degrees(np.arctan2(b, a))) % 360
    chroma = float(np.hypot(a, b))
    if chroma > 14 and (hue < 60 or hue > 330):
        return COPPER
    return DIELECTRIC


def _copper_smooth(c):
    return 0.62 if _copper_f0(c) == COPPER else 0.3


def copper_pattern(k):
    """Copper tiles, shingles, roof tops and the copper fluid tank: the shape from
    the clean (unoxidised) pattern, the colour - patina and all - from this one."""
    import re as _re
    from families import load_rgba
    base = None
    if ROOT and k.path:
        bp = _re.sub(r"(exposed_|weathered_|oxidized_)", "", k.path)
        if bp != k.path:
            try:
                base = load_rgba(f"{ROOT}/create/{bp}")
            except Exception:
                base = None
    name = k.path.split("/")[-1]
    opts = {}
    if "roof_top" in name:
        opts = dict(groove_q=0.1, relief=0.14, groove_blur=0.3)
    elif name.startswith("fluid_tank"):
        opts = dict(groove_q=0.15, relief=0.1, groove_blur=0.25)
    k.pattern(base=base, metal_of=_copper_f0, smooth_of=_copper_smooth, **opts)


for _st in ("", "exposed_", "weathered_", "oxidized_"):
    for _n in ("roof_top", "tiles", "shingles", "tiles_top_connected", "shingles_top_connected"):
        DESIGNS[f"textures/block/copper/{_st}copper_{_n}.png"] = copper_pattern
for _n in ("fluid_tank", "fluid_tank_top", "fluid_tank_inner", "fluid_tank_connected", "fluid_tank_top_connected",
           "fluid_tank_inner_connected"):
    DESIGNS[f"textures/block/{_n}.png"] = copper_pattern


def cloth(k):
    """Table cloths, seat cushions, sail canvas: crisp original colours, woven surface."""
    k.cloth()


def _painted_f0(c):
    """Painted wood with brass fittings: brass is metal, the paint is not."""
    L, a, b = lab(tuple(np.asarray(c) * 255))
    hue = float(np.degrees(np.arctan2(b, a))) % 360
    return ALBEDO_METAL if (72 <= hue <= 100 and np.hypot(a, b) > 40 and L > 55) else DIELECTRIC


def _painted_smooth(c):
    return 0.7 if _painted_f0(c) == ALBEDO_METAL else 0.38


def painted(k):
    """Toolboxes: one layout in sixteen paints -- the colours and seams are each
    original's, the brass latches metal, the paint satin."""
    k.pattern(metal_of=_painted_f0, smooth_of=_painted_smooth, groove_q=0.12, relief=0.06, groove_blur=0.3)


def rose_quartz_lamp(k):
    """Rose quartz lamp: polished quartz panes (the original's facets and sparkle,
    shading kept) in a raised steel lattice; the panes glow when powered."""
    def groups(c):
        L, a, b = lab(c)
        hue = float(np.degrees(np.arctan2(b, a))) % 360
        pink = np.hypot(a, b) > 9 and (hue > 280 or hue < 40)
        return "quartz" if pink else "frame"
    machine(k, overrides={"frame": dict(kind="metal", f0=IRON, smooth=0.55, h=0.85, bevel=0.4, brush=0.03,
                                        smooth_px=0.45),
                          "quartz": dict(kind="flat", f0=40, smooth=0.88, h=0.62, bevel=0.3, shade=1.0)},
            groups=groups, bolts=())
    if "powered" in k.path:
        L = k.m.albedo @ np.array([0.2126, 0.7152, 0.0722])
        k.m.emissive = np.where(k.m.f0 == 40, np.clip((L - 0.15) * 2.0, 0.2, 0.9), 0)


POSTBOX_IRON = {(33, 27, 21), (40, 34, 28), (48, 44, 41), (69, 63, 59), (80, 74, 70)}   # in all 16 postboxes
POSTBOX_STEEL = {(96, 99, 95), (112, 117, 112), (130, 135, 132)}
POSTBOX_FLAG = {(106, 32, 51), (131, 40, 54), (144, 45, 54), (156, 49, 55)}


def painted_planks(k):
    """Postboxes: painted planks (each box's own paint, the boards' shading
    kept, grain under the paint), dark iron bands, steel latch and the red flag
    -- the shared parts are the colours common to all sixteen boxes."""
    def groups(c):
        if (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) < 14:
            return "hole"
        if c in POSTBOX_IRON:
            return "band"
        if c in POSTBOX_STEEL:
            return "metal"
        if c in POSTBOX_FLAG and "red" not in (k.path or ""):
            return "flag"
        return "paint"
    machine(k, overrides={"band": dict(kind="metal", f0=IRON, smooth=0.4, h=0.82, bevel=0.35, brush=0.03),
                          "flag": dict(kind="flat", f0=DIELECTRIC, smooth=0.5, h=0.8, bevel=0.3),
                          "paint": dict(kind="wood", f0=DIELECTRIC, smooth=0.42, h=0.68, bevel=0.25, shade=1.0)},
            groups=groups, bolts=("band", "metal"))


def cardboard(k):
    """Cardboard: the original's colours crisp (tape, corrugation lines), a fibrous paper surface."""
    k.cloth(weave=0.0, smooth=0.25)


def two_tone_casing(k, frame_q=None):
    """Casings drawn in one hue (shadow steel, refined radiance): the frame is the
    lighter band round the edge, the panel the rest; layout from the original."""
    a = k.src[..., 3] > 0.5
    Lsrc = k.src[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    edge = np.zeros(a.shape, bool)
    h, w = a.shape
    for y in range(h):
        for x in range(w):
            if min(x % 16, y % 16, 15 - x % 16, 15 - y % 16) < 2:
                edge[y, x] = True
    cut = (np.median(Lsrc[a & edge]) + np.median(Lsrc[a & ~edge])) / 2 if (a & ~edge).any() else np.median(Lsrc[a])
    lighter_frame = np.median(Lsrc[a & edge]) >= np.median(Lsrc[a & ~edge])

    def groups(c):
        L = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255
        return "frame" if (L >= cut) == lighter_frame else "panel"
    machine(k, overrides={"frame": dict(kind="metal", f0=ALBEDO_METAL, smooth=0.6, h=0.9, bevel=0.5, brush=0.04),
                          "panel": dict(kind="metal", f0=ALBEDO_METAL, smooth=0.5, h=0.6, bevel=0.35, brush=0.03,
                                        shade=1.0)}, groups=groups, bolts=())


def radiance_casing(k):
    """Refined radiance casing: pale steel frame round a glowing white panel."""
    two_tone_casing(k)
    L = k.m.albedo @ np.array([0.2126, 0.7152, 0.0722])
    k.m.emissive = np.clip((L - 0.72) * 3.0, 0, 0.6)


def register_specials(plan):
    """Families whose look is a material pattern rather than machine parts."""
    for t in plan:
        if t["ns"] != "create" or t["kind"] != "block" or t["frames"] > 1:
            continue
        name = t["path"][len("textures/block/"):-4]
        if name in ("table_cloth/andesite", "table_cloth/brass", "table_cloth/copper"):
            continue                                        # metal "cloths": the machine painter
        if name.startswith(("table_cloth/", "seat/top_", "seat/side_", "sail/canvas_")):
            DESIGNS[t["path"]] = cloth
        elif name.startswith(("girder", "metal_girder", "roller_metal", "roller_wheel", "bogey/")):
            DESIGNS[t["path"]] = cast_metal
        elif name.startswith("refined_radiance_casing"):
            DESIGNS[t["path"]] = radiance_casing
        elif name.startswith("shadow_steel_casing"):
            DESIGNS[t["path"]] = two_tone_casing
        elif "cardboard" in name:
            DESIGNS[t["path"]] = cardboard
        elif name.startswith("toolbox/"):
            DESIGNS[t["path"]] = painted
        elif name.startswith("post_box/"):
            DESIGNS[t["path"]] = painted_planks
        elif name.startswith("rose_quartz_lamp"):
            DESIGNS[t["path"]] = rose_quartz_lamp
        elif name in ("bracket_metal", "bracket_plate_metal", "copycat_base"):
            DESIGNS[t["path"]] = cast_stone


# Everything else Create draws on its machines and fittings (funnels, tunnels,
# belts, pipes, tanks, copper roofing, seats, sails, bogeys...) starts from the
# machine painter; hand-made textures elsewhere (casings, shafts, brass block,
# industrial iron: blocks.py) and the decorative stone palettes are not here.
HAND_BUILT = {"axis", "axis_top", "brass_block", "zinc_block", "andesite_casing", "brass_casing", "copper_casing",
              "industrial_iron_block", "industrial_iron_block_top", "railway_casing", "railway_casing_side"}
SKIP_DIRS = ("palettes/",)
# natural or glowing surfaces, and the blaze's face: not machine parts, left to the automatic remaster
SKIP_NAMES = ("zinc_ore", "deepslate_zinc_ore", "raw_zinc_block", "refined_radiance_block", "blaze_", "experience_block")
PPU_HIGH, PPU_LOW = 16, 8      # most-seen textures at the casings' density, the rest at half
USAGE = {}                     # texture path -> number of models using it (set by register_machines)


def register_machines(plan, usage):
    """Add the machine painter for every model-used Create block texture not designed by hand."""
    register_specials(plan)
    for t in plan:
        if t["ns"] != "create" or t["kind"] != "block" or t["frames"] > 1:
            continue
        path = t["path"]
        name = path[len("textures/block/"):-4]
        if path in DESIGNS or name in HAND_BUILT or name.endswith("_connected") and name[:-10] in HAND_BUILT:
            continue
        if any(name.startswith(d) for d in SKIP_DIRS) or name.startswith(SKIP_NAMES):
            continue
        n = len({u.model for u in usage.get("create:" + path[len("textures/"):-4], [])})
        if n == 0:
            continue
        USAGE[path] = n
        DESIGNS[path] = machine
    # connected-texture sheets aren't referenced by models (the CT system swaps them in):
    # each one gets its base texture's painter, so joined blocks match single ones
    for t in plan:
        if t["ns"] != "create" or t["kind"] != "block" or t["frames"] > 1:
            continue
        p = t["path"]
        for suf in ("_connected.png", "_top_connected.png"):
            if p.endswith(suf) and p not in DESIGNS:
                base = p[: -len(suf)] + ".png"
                name = base[len("textures/block/"):-4]
                if base in DESIGNS and name not in HAND_BUILT:
                    DESIGNS[p] = DESIGNS[base]
                    USAGE[p] = USAGE.get(base, 99)


MAX_SIDE = 1024               # keeps the block atlas inside one 16384^2 page with the rest of the pack
HIGH_USE = 8                  # textures on at least this many models get the full density


def ppu_of(path, width=16):
    """Hand-designed and most-used textures at 16 px per texel, the rest at 8,
    no texture beyond MAX_SIDE (the 2048 casing sheets are blocks.py's)."""
    p = PPU_HIGH if USAGE.get(path, 99) >= HIGH_USE else PPU_LOW
    return max(1, min(p, MAX_SIDE // max(width, 1)))


def design_of(path, ns):
    return DESIGNS.get(path) if ns == "create" else None


def render(path, src, ppu=None):
    k = Kit(src, ppu=ppu or ppu_of(path, max(src.shape[:2])))
    k.path = path
    DESIGNS[path](k)
    return k.finish()
