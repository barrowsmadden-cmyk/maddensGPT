"""Measure discontinuities across block seams in an assembled connected wall.

For each map channel, measures the "dip" at a seam: how far the two columns (rows)
that meet at the seam sit from the average of the columns just outside them, and
compares it with the same measure at ordinary columns. A ratio near 1 means the
seam is invisible; well above 1 means a visible line.
"""
import sys
import numpy as np
from ct_preview import assemble, assemble_ctm, WALL

N = 256


def seam_ratios(maps, grid=WALL):
    a8, n8, s8 = (m.astype(float) for m in maps)
    chans = {"albedo": a8[..., :3].mean(-1), "normal_x": n8[..., 0], "normal_y": n8[..., 1],
             "ao": n8[..., 2], "height": n8[..., 3], "smooth": s8[..., 0]}
    H, W = len(grid), len(grid[0])
    out = {}
    for name, ch in chans.items():
        def dip_v(x, rows):  # seam between columns x and x+1
            inner = (ch[rows, x] + ch[rows, x + 1]) / 2
            outer = (ch[rows, x - 1] + ch[rows, x + 2]) / 2
            return np.abs(inner - outer).mean()

        def dip_h(y, cols):
            inner = (ch[y, cols] + ch[y + 1, cols]) / 2
            outer = (ch[y - 1, cols] + ch[y + 2, cols]) / 2
            return np.abs(inner - outer).mean()

        vs, vi, hs, hi = [], [], [], []
        for r in range(H):
            for c in range(W):
                if grid[r][c] != "1":
                    continue
                rows = slice(r * N + 40, (r + 1) * N - 40)
                cols = slice(c * N + 40, (c + 1) * N - 40)
                if c + 1 < W and grid[r][c + 1] == "1":
                    x = (c + 1) * N - 1
                    vs.append(dip_v(x, rows))
                    vi.append(np.mean([dip_v(x - k, rows) for k in (11, 23, 37)]))
                if r + 1 < H and grid[r + 1][c] == "1":
                    y = (r + 1) * N - 1
                    hs.append(dip_h(y, cols))
                    hi.append(np.mean([dip_h(y - k, cols) for k in (11, 23, 37)]))
        out[name] = (np.mean(vs) / max(np.mean(vi), 1e-6), np.mean(hs) / max(np.mean(hi), 1e-6))
    return out


if __name__ == "__main__":
    texdir = sys.argv[1]
    sets = {n: assemble(texdir, n) for n in ("andesite_casing", "brass_casing", "copper_casing")}
    sets["industrial_iron_block"] = assemble_ctm(texdir.replace("textures/block", "optifine/ctm/industrial_iron_block"))
    for name, maps in sets.items():
        r = seam_ratios(maps)
        print(f"{name:24s}", "  ".join(f"{k}: v{v:4.1f} h{h:4.1f}" for k, (v, h) in r.items()))
