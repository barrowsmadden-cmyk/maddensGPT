"""Pack generated LabPBR maps into the interactive three.js viewer page.

three.js wants OpenGL (Y+) normals and an AO/roughness/metal map, so this
converts from the LabPBR files exactly as written to the pack.
"""
import base64
import io
import json
import sys

import numpy as np
from PIL import Image

from ct import CTM_SPRITE_INDEX_MAP, omni_tiles
from render import tile_view

BLOCKS = [
    ("andesite_casing", "Andesite Casing", "cube"),
    ("brass_casing", "Brass Casing", "cube"),
    ("copper_casing", "Copper Casing", "cube"),
    ("industrial_iron_block", "Industrial Iron Block", "cube"),
    ("brass_block", "Brass Block", "cube"),
    ("axis", "Shaft", "shaft"),
]
CT = {
    "andesite_casing": ("omni", "Create built-in CT, andesite_casing_connected.png (8×8)"),
    "brass_casing": ("omni", "Create built-in CT, brass_casing_connected.png (8×8)"),
    "copper_casing": ("omni", "Create built-in CT, copper_casing_connected.png (8×8)"),
    "industrial_iron_block": ("ctm", "Continuity, optifine/ctm/industrial_iron_block + _top (47 tiles each)"),
}
# cube_column blocks: top/bottom texture
ENDS = {"industrial_iron_block": "industrial_iron_block_top"}
METALS = {230: "iron", 231: "gold", 234: "copper", 255: "albedo metal"}


def uri(img, fmt="PNG", **kw):
    buf = io.BytesIO()
    if fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, fmt, optimize=True, **kw)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def load(texdir, name):
    return [np.asarray(Image.open(f"{texdir}/{name}{s}.png").convert("RGBA")) for s in ("", "_n", "_s")]


def three_maps(a8, n8, s8, compact=False):
    """compact: JPEG colour/ORM (no alpha) to keep the big connected sheets small."""
    nx = n8[..., 0] / 127.5 - 1
    ny = n8[..., 1] / 127.5 - 1
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
    normal = np.dstack([n8[..., 0], 255 - n8[..., 1], (nz * 0.5 + 0.5) * 255]).round().astype(np.uint8)
    rough = 255 - s8[..., 0].astype(int)  # three's roughness is perceptual
    metal = np.where(s8[..., 1] >= 230, 255, 0)
    orm = np.dstack([n8[..., 2], rough, metal]).astype(np.uint8)
    if compact:
        return dict(
            albedo=uri(Image.fromarray(a8, "RGBA"), "JPEG", quality=90),
            normal=uri(Image.fromarray(normal, "RGB")),
            orm=uri(Image.fromarray(orm, "RGB"), "JPEG", quality=92),
            height=uri(Image.fromarray(n8[..., 3], "L"), "JPEG", quality=92),
        )
    return dict(
        albedo=uri(Image.fromarray(a8, "RGBA")),
        normal=uri(Image.fromarray(normal, "RGB")),
        orm=uri(Image.fromarray(orm, "RGB")),
        height=uri(Image.fromarray(n8[..., 3], "L")),
    )


VIEW_TILE, GUTTER = 128, 16
CELL = VIEW_TILE + 2 * GUTTER


def _shrink(maps):
    """256px tile -> 128px for the web viewer, renormalising the normal xy."""
    out = []
    for i, arr in enumerate(maps):
        a = np.asarray(Image.fromarray(arr, "RGBA").resize((VIEW_TILE, VIEW_TILE), Image.LANCZOS)).copy()
        if i == 1:
            nx, ny = a[..., 0] / 127.5 - 1, a[..., 1] / 127.5 - 1
            ln = np.maximum(np.hypot(nx, ny), 1)
            a[..., 0] = ((nx / ln + 1) * 127.5).round().astype(np.uint8)
            a[..., 1] = ((ny / ln + 1) * 127.5).round().astype(np.uint8)
        out.append(a)
    return out


def viewer_sheet(tiles):
    """8x8 cells of CELL px: each tile shrunk on its own and wrapped in a GUTTER-px
    border of its own edge pixels, so filtering and mipmaps never pull in the
    next tile (which would draw seams between connected blocks)."""
    sheets = [np.zeros((8 * CELL, 8 * CELL, 4), np.uint8) for _ in range(3)]
    for i, maps in tiles.items():
        y, x = (i // 8) * CELL, (i % 8) * CELL
        for sh, mp in zip(sheets, _shrink(maps)):
            sh[y:y + CELL, x:x + CELL] = np.pad(mp, ((GUTTER, GUTTER), (GUTTER, GUTTER), (0, 0)), mode="edge")
    return sheets


def omni_sheet_tiles(texdir, name):
    sheet = load(texdir, name + "_connected")
    n = sheet[0].shape[0] // 8
    return {i: [m[(i // 8) * n:(i // 8 + 1) * n, (i % 8) * n:(i % 8 + 1) * n] for m in sheet] for i in omni_tiles()}


def ctm_folder_tiles(folder):
    return {i: load(folder, str(i)) for i in range(47)}


def thumbs(maps):
    a8, n8, s8 = maps
    flat = lambda arr: Image.fromarray(arr.astype(np.uint8), "RGB").resize((160, 160), Image.LANCZOS)
    bg = np.full(a8.shape[:2] + (3,), 40, float)
    a = a8[..., 3:4] / 255.0
    return dict(
        albedo=uri(flat(a8[..., :3] * a + bg * (1 - a)), "JPEG", quality=88),
        n=uri(flat(np.dstack([n8[..., 0], n8[..., 1], np.full_like(n8[..., 0], 255)]) * a + bg * (1 - a)), "JPEG", quality=88),
        s=uri(flat(np.dstack([s8[..., 0], s8[..., 1], np.clip(s8[..., 2].astype(int) * 3, 0, 255)]) * a + bg * (1 - a)), "JPEG", quality=88),
        lit=uri(tile_view(maps).convert("RGB").resize((320, 320), Image.LANCZOS) if a8[..., 3].min() == 255 else
                Image.alpha_composite(Image.new("RGBA", (256, 256), (40, 40, 40, 255)), tile_view(maps)).convert("RGB").resize((320, 320), Image.LANCZOS), "JPEG", quality=90),
    )


def stats(maps):
    a8, n8, s8 = maps
    vis = a8[..., 3] > 0
    ids = sorted(set(int(v) for v in np.unique(s8[..., 1][vis]) if v >= 230))
    return dict(
        metals=[f"{v} {METALS.get(v, '')}".strip() for v in ids],
        smooth=[int(np.percentile(s8[..., 0][vis], 5)), int(np.percentile(s8[..., 0][vis], 95))],
        porosity=int(s8[..., 2][vis].max()),
    )


def main(texdir, template, out):
    data = {}
    for name, label, kind in BLOCKS:
        maps = load(texdir, name)
        entry = dict(label=label, kind=kind, file=name, tex=three_maps(*maps), thumbs=thumbs(maps), stats=stats(maps))
        if kind == "shaft":
            top = load(texdir, "axis_top")
            entry["topTex"] = three_maps(*top)
        if name in ENDS:
            entry["endTex"] = three_maps(*load(texdir, ENDS[name]))
        if name in CT:
            ctype, note = CT[name]
            if ctype == "omni":
                tiles = omni_sheet_tiles(texdir, name)
            else:
                tiles = ctm_folder_tiles(texdir.replace("textures/block", "optifine/ctm/" + name))
            entry["ct"] = dict(type=ctype, note=note, cell=CELL, gutter=GUTTER, size=8 * CELL,
                               tex=three_maps(*viewer_sheet(tiles), compact=True))
            if name in ENDS:
                end = ctm_folder_tiles(texdir.replace("textures/block", "optifine/ctm/" + ENDS[name]))
                entry["ct"]["endTex"] = three_maps(*viewer_sheet(end), compact=True)
        data[name] = entry
    html = (open(template).read().replace("/*__TEX__*/null", json.dumps(data))
            .replace("/*__CTM_MAP__*/null", json.dumps(CTM_SPRITE_INDEX_MAP)))
    open(out, "w").write(html)
    print(out, f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
