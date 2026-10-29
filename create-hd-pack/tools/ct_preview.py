"""Assemble a wall of connected casings the way Create picks tiles, then light it."""
import sys
import numpy as np
from PIL import Image
from ct import Ctx, omni_index, ctm_index
from render import tile_view

N = 256
WALL = ["11110",
        "11110",
        "11011",
        "11111"]


def ctx_at(grid, r, c):
    def on(rr, cc):
        return 0 <= rr < len(grid) and 0 <= cc < len(grid[0]) and grid[rr][cc] == "1"
    up, down, left, right = on(r - 1, c), on(r + 1, c), on(r, c - 1), on(r, c + 1)
    return Ctx(up, down, left, right,
               up and left and on(r - 1, c - 1), up and right and on(r - 1, c + 1),
               down and left and on(r + 1, c - 1), down and right and on(r + 1, c + 1))


def assemble(texdir, name, grid=WALL):
    sheets = [np.asarray(Image.open(f"{texdir}/{name}_connected{s}.png").convert("RGBA")) for s in ("", "_n", "_s")]
    H, W = len(grid), len(grid[0])
    out = [np.zeros((H * N, W * N, 4), np.uint8) for _ in range(3)]
    out[1][..., :] = (128, 128, 255, 255)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != "1":
                continue
            i = omni_index(ctx_at(grid, r, c))
            ty, tx = (i // 8) * N, (i % 8) * N
            for o, s in zip(out, sheets):
                o[r * N:(r + 1) * N, c * N:(c + 1) * N] = s[ty:ty + N, tx:tx + N]
    return out


def assemble_ctm(folder, grid=WALL):
    """Same as assemble() but with Continuity's 47 separate tile files."""
    tiles = {}
    H, W = len(grid), len(grid[0])
    out = [np.zeros((H * N, W * N, 4), np.uint8) for _ in range(3)]
    out[1][..., :] = (128, 128, 255, 255)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != "1":
                continue
            i = ctm_index(ctx_at(grid, r, c))
            if i not in tiles:
                tiles[i] = [np.asarray(Image.open(f"{folder}/{i}{s}.png").convert("RGBA")) for s in ("", "_n", "_s")]
            for o, t in zip(out, tiles[i]):
                o[r * N:(r + 1) * N, c * N:(c + 1) * N] = t
    return out


def lit_small(maps):
    lit = tile_view(maps)
    bg = Image.new("RGBA", lit.size, (24, 24, 28, 255))
    bg.alpha_composite(lit)
    return bg.resize((lit.width // 2, lit.height // 2), Image.LANCZOS)


if __name__ == "__main__":
    texdir, dest = sys.argv[1], sys.argv[2]
    imgs = []
    for name in ("andesite_casing", "brass_casing", "copper_casing"):
        imgs.append(lit_small(assemble(texdir, name)))
    ctm_dir = texdir.replace("textures/block", "optifine/ctm/industrial_iron_block")
    imgs.append(lit_small(assemble_ctm(ctm_dir)))
    W = sum(i.width for i in imgs) + 16 * (len(imgs) + 1)
    sheet = Image.new("RGB", (W, imgs[0].height + 32), (24, 24, 28))
    x = 16
    for im in imgs:
        sheet.paste(im.convert("RGB"), (x, 16))
        x += im.width + 16
    sheet.save(dest)
    print(dest, sheet.size)
