"""Hand-made glass: clear, 16 stained and tinted glass blocks.

Vanilla layout: 1px frame, pane interior, two diagonal glints in opposite
corners. Alpha follows the source exactly in kind: clear glass is a cutout
texture (interior stays alpha 0), stained/tinted glass keep their translucent
frame/interior alpha levels.
"""
import re

import cv2
import numpy as np

from texgen import Material, derive, fbm, sstep, round_profile
from families import grid

PATTERN = re.compile(r"^((white|orange|magenta|light_blue|yellow|lime|pink|gray|light_gray|cyan|purple|blue|brown|"
                     r"green|red|black)_stained_glass|glass|tinted_glass)$")


def glass_of(path, ns):
    if ns != "minecraft" or not path.startswith("textures/block/"):
        return False
    return bool(PATTERN.match(path.split("/")[-1][:-4]))


def render(path, src, size, ns="minecraft"):
    s = size / 16
    xx, yy = grid(size)
    ux, uy = xx / s, yy / s
    rgb, a = src[..., :3], src[..., 3]
    frame_px = np.zeros(a.shape, bool)
    frame_px[0, :] = frame_px[-1, :] = frame_px[:, 0] = frame_px[:, -1] = True
    frame_col = rgb[frame_px].mean(0)
    frame_a = a[frame_px].mean()
    inner = ~frame_px
    pane_a = np.median(a[inner])
    pane_col = rgb[inner & (a > 0)].mean(0) if (a[inner] > 0).any() else frame_col
    glint_px = inner & (a > pane_a + 0.05)
    glint_col = rgb[glint_px].mean(0) if glint_px.any() else np.minimum(frame_col * 1.3, 1)
    glint_a = a[glint_px].mean() if glint_px.any() else 1.0
    cutout = (a < 0.02).any()

    # frame: 1 vanilla px border with a rounded inner bevel
    d_edge = np.minimum(np.minimum(ux, 16 - ux), np.minimum(uy, 16 - uy))
    frame = d_edge < 1.0
    bevel = round_profile(1.0 - d_edge, 0.45) * frame

    # glints: two long diagonal streaks per corner, anti-aliased
    glints = np.zeros((size * 4, size * 4), np.uint8)
    k = s * 4
    for (x0, y0, x1, y1, w) in ((1.7, 5.2, 5.2, 1.7, 0.42), (2.0, 7.0, 7.0, 2.0, 0.24),
                                 (10.8, 14.3, 14.3, 10.8, 0.42), (9.0, 14.0, 14.0, 9.0, 0.24)):
        cv2.line(glints, (int(x0 * k), int(y0 * k)), (int(x1 * k), int(y1 * k)), 255,
                 thickness=max(1, int(w * k)), lineType=cv2.LINE_AA)
    g = cv2.resize(glints.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) / 255.0
    g = g * ~frame

    ripple = fbm(5, 5, 4, 3, size=size)
    col = np.where(frame[..., None], frame_col[None, None, :] * (0.85 + 0.3 * bevel[..., None]), pane_col[None, None, :])
    col = col * (1 + 0.04 * (ripple[..., None] - 0.5))
    col = col * (1 - g[..., None]) + glint_col[None, None, :] * g[..., None]
    if cutout:
        alpha = np.where(frame, 1.0, 0.0)
        alpha = np.maximum(alpha, (g > 0.35).astype(float))
    else:
        alpha = np.where(frame, frame_a, pane_a + 0.03 * (ripple - 0.5))
        alpha = alpha * (1 - g) + glint_a * g
    m = Material(size)
    m.albedo = np.clip(col, 0, 1)
    m.alpha = np.clip(alpha, 0, 1)
    m.height = np.where(frame, 0.92 + 0.07 * bevel, 0.9 + 0.01 * ripple)
    m.smooth = np.where(frame, 0.7, 0.96)
    m.f0[:] = 10
    m.porosity[:] = 0
    return derive(m, edge_wear=0.0, ao_strength=0.4, edges="wrap", normal_strength=3.0)
