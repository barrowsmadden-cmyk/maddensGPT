"""Redraw pixel art as clean shapes at high resolution.

At 16 px a sword blade or an ingot edge can only be a staircase. Here every
colour region is traced along its pixel corners, the staircases are simplified
into straight segments (Douglas-Peucker), gentle bends are rounded into curves
(Chaikin, deliberate corners kept sharp) and the shapes are filled anti-aliased
at the target size. Colour inside each region is a smooth version of the
original's shading.
"""
import cv2
import numpy as np

from remaster import material_regions, merge_speckles, srgb_to_lab, blur_pad

SUPER = 4          # supersampling for anti-aliased fills
EPSILON = 0.85     # simplification tolerance, in source pixels: a 1px staircase
                   # (zigzag amplitude ~0.7px) becomes one straight edge
CORNER_DEG = 52    # turns sharper than this stay corners; gentler ones become curves


def trace(mask):
    """Outlines of a binary source mask in pixel-corner coordinates.
    Returns (contours, hierarchy) with contours as float arrays of (x, y)."""
    up = np.kron(mask.astype(np.uint8), np.ones((4, 4), np.uint8))
    cs, hier = cv2.findContours(up, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        # contour runs through pixel centres of the 4x image; move to corners
        out.append((c[:, 0, :].astype(float) + 0.5) / 4.0)
    return out, hier


def simplify(poly):
    if len(poly) < 4:
        return poly
    p = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), EPSILON, True)[:, 0, :].astype(float)
    return p if len(p) >= 3 else poly


def smooth_curves(poly, iterations=3):
    """Chaikin corner cutting, but vertices that turn sharply stay put."""
    for _ in range(iterations):
        n = len(poly)
        if n < 4:
            return poly
        prev, nxt = np.roll(poly, 1, 0), np.roll(poly, -1, 0)
        a, b = poly - prev, nxt - poly
        cosang = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)
        turn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        out = []
        for i in range(n):
            if turn[i] > CORNER_DEG:
                out.append(poly[i])
            else:
                out.append(0.75 * poly[i] + 0.25 * prev[i])
                out.append(0.75 * poly[i] + 0.25 * nxt[i])
        poly = np.array(out)
    return poly


def vector_labels(labels, scale):
    """Label map at `scale` x with every region redrawn as clean shapes.

    Regions are simplified independently, so where two simplified outlines
    disagree each output pixel goes to the region covering most of it; pixels
    no outline covers take the region whose (blurred) coverage is strongest.
    Returns (labels_big, {id: coverage})."""
    h, w = labels.shape
    H, W = h * scale, w * scale
    ids = [i for i in np.unique(labels) if i >= 0]
    covs = {i: coverage(labels == i, (H, W), scale) for i in ids}
    if not ids:
        return np.full((H, W), -1), covs
    stack = np.stack([covs[i] for i in ids])
    best = stack.argmax(0)
    out = np.array(ids)[best]
    total = stack.max(0)
    hole = total < 0.25
    if hole.any():
        soft = np.stack([cv2.GaussianBlur(c, (0, 0), scale * 0.5) for c in stack])
        out = np.where(hole, np.array(ids)[soft.argmax(0)], out)
    return out, covs


def _thin_paths(comp):
    """If a component is a 1px-wide line (no 2x2 block), return its pixel-centre
    path(s) as lists of (x, y); diagonal steps that cut an L corner are dropped so
    a staircase reads as one diagonal. Returns None for solid components."""
    if (comp[:-1, :-1] & comp[1:, :-1] & comp[:-1, 1:] & comp[1:, 1:]).any():
        return None
    ys, xs = np.nonzero(comp)
    pts = set(zip(ys.tolist(), xs.tolist()))
    if len(pts) < 3:
        return None
    nb = {}
    for (y, x) in pts:
        orth = [(y + dy, x + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (y + dy, x + dx) in pts]
        diag = []
        for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = (y + dy, x + dx)
            # skip a diagonal if the two pixels already touch through an orthogonal one
            if q in pts and (y + dy, x) not in pts and (y, x + dx) not in pts:
                diag.append(q)
        nb[(y, x)] = orth + diag
    if any(len(v) > 2 for v in nb.values()):
        # branched line: draw every link as its own stroke
        segs = []
        for p_, qs in nb.items():
            for q in qs:
                if p_ < q:
                    segs.append([(p_[1] + 0.5, p_[0] + 0.5), (q[1] + 0.5, q[0] + 0.5)])
        return segs
    start = next((p_ for p_, v in nb.items() if len(v) == 1), next(iter(pts)))
    path, prev, cur = [start], None, start
    while True:
        nxt = [q for q in nb[cur] if q != prev and q not in path]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)
    return [[(x + 0.5, y + 0.5) for y, x in path]]


def coverage(mask, size_hw, scale):
    """Anti-aliased coverage (0..1) of the vectorised mask at the output size.
    Solid areas are traced, simplified and filled; 1px-wide lines are redrawn as
    straight strokes one source pixel wide."""
    H, W = size_hw
    canvas = np.zeros((H * SUPER, W * SUPER), np.uint8)
    n, comps = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    solid = np.zeros_like(mask, dtype=bool)
    k = scale * SUPER
    for c in range(1, n):
        comp = comps == c
        paths = _thin_paths(comp)
        if paths is None:
            solid |= comp
            continue
        for path in paths:
            pts = np.array(path, float)
            if len(pts) > 2:
                pts = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), 0.75, False)[:, 0, :].astype(float)
            pl = np.round(pts * k).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pl], False, 255, thickness=max(1, int(round(k * 0.95))), lineType=cv2.LINE_8)
            for x, y in pts[[0, -1]]:  # round caps
                cv2.circle(canvas, (int(round(x * k)), int(round(y * k))), max(1, int(round(k * 0.47))), 255, -1)
    contours, hier = trace(solid)
    if not contours:
        small = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
        return small.astype(float) / 255.0
    polys = [np.round(smooth_curves(simplify(c)) * scale * SUPER).astype(np.int32).reshape(-1, 1, 2) for c in contours]
    # RETR_CCOMP: parent == -1 are outer boundaries, others are holes
    for i, p in enumerate(polys):
        if hier[0][i][3] == -1:
            cv2.fillPoly(canvas, [p], 255, lineType=cv2.LINE_8)
    for i, p in enumerate(polys):
        if hier[0][i][3] != -1:
            cv2.fillPoly(canvas, [p], 0, lineType=cv2.LINE_8)
    small = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    return small.astype(float) / 255.0


DEFAULT_MERGES = ((2, 28.0), (3, 12.0))  # (min size, max dE): fold specks into surroundings


def part_labels(rgb, solid, max_regions=7, merges=DEFAULT_MERGES):
    labels = material_regions(rgb, solid, max_regions=max_regions)
    lab = srgb_to_lab(rgb)
    for size, de in merges:
        labels = merge_speckles(labels, size, lab, max_de=de)
    return labels


def vectorize(rgba, scale, max_regions=7, merges=DEFAULT_MERGES):
    """rgba float (h, w, 4). Returns (rgb, alpha, labels, region coverage list)."""
    h, w = rgba.shape[:2]
    H, W = h * scale, w * scale
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    solid = alpha > 0.5
    # single pixels are usually shading at 16px; fold them into their surroundings
    # unless they clearly stand out (glints, eyes, rivets)
    labels = part_labels(rgb, solid, max_regions, merges)
    ids = [i for i in np.unique(labels) if i >= 0]
    out_alpha = coverage(solid, (H, W), scale)
    lab_big, covs = vector_labels(labels, scale)
    lab_big = np.where(out_alpha > 0.02, lab_big, -1)

    # smooth shading per region from the original colours (no bleeding across regions)
    col = np.zeros((H, W, 3))
    s = scale * 0.7
    for i in ids:
        m = (labels == i).astype(np.float32)
        mb = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        num = np.stack([blur_pad(cv2.resize((rgb[..., k] * m).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST), s)
                        for k in range(3)], -1)
        den = blur_pad(mb, s)[..., None]
        reg = np.clip(num / np.maximum(den, 1e-4), 0, 1)
        sel = lab_big == i
        col[sel] = reg[sel]
    # anti-alias region boundaries: soften colour across a 1px seam
    col = np.stack([cv2.GaussianBlur(col[..., k], (0, 0), 0.6) * 0.5 + col[..., k] * 0.5 for k in range(3)], -1)
    return col, out_alpha, lab_big, covs
