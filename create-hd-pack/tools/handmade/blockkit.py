"""Hand-made block textures at any size: a small drawing kit.

Coordinates are the original texture's texels (texel (x, y) spans x..x+1,
y..y+1); the canvas is `ppu` HD pixels per original texel (16 = the casings'
density). Primitives paint material layers (albedo, height, smoothness,
LabPBR F0/metal id, porosity) into a texgen Material, later ones on top, with
bevelled edges from their distance-to-edge; colours come from the original
texture (`k.tones(...)` samples texels by letter or rect), so every machine
keeps Create's own palette. `k.finish()` derives the PBR maps.

Material kinds:
  metal  brushed streaks along the part (bars along their length), bevelled
  wood   planks with grain along `dir`, darker seams between planks
  stone  mottled, speckled (andesite, basin, millstone)
  dark   recesses and holes: near-black, the far inner wall catching light
  glass  smooth, tinted, keeps the original's alpha
  flat   uniform with faint noise (paint, cloth, rubber)
"""
import math

import cv2
import numpy as np

from texgen import Material, derive, fbm, worley, sstep, round_profile, lerp, IRON, GOLD, COPPER, \
    ALBEDO_METAL, DIELECTRIC

LUMW = np.array([0.2126, 0.7152, 0.0722])


class Mat(dict):
    """A material: kind + light/dark tones (0-1 RGB) + PBR settings."""


def mat(kind, light, dark, f0=None, smooth=None, porosity=0, **kw):
    f0d = {"metal": IRON, "wood": DIELECTRIC, "stone": DIELECTRIC, "dark": DIELECTRIC, "glass": 12,
           "flat": DIELECTRIC}[kind]
    smd = {"metal": 0.55, "wood": 0.28, "stone": 0.22, "dark": 0.15, "glass": 0.94, "flat": 0.3}[kind]
    return Mat(kind=kind, light=np.asarray(light, float), dark=np.asarray(dark, float),
               f0=f0d if f0 is None else f0, smooth=smd if smooth is None else smooth, porosity=porosity, **kw)


class Kit:
    def __init__(self, src, ppu=16, seed=1):
        self.src = src
        self.th, self.tw = src.shape[:2]
        self.ppu = ppu
        self.H, self.W = self.th * ppu, self.tw * ppu
        self.m = Material(self.H, self.W)
        # cutouts stay cut out: the canvas takes the original's alpha, texel for texel
        self.m.alpha = cv2.resize(src[..., 3].astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(float)
        self.x, self.y = (xx + 0.5) / ppu, (yy + 0.5) / ppu       # texel coordinates of pixel centres
        self.seed = seed
        self.path = None
        self._noise = {}

    # ---------------------------------------------------------- colours ---
    def texels(self, x0, y0, x1, y1):
        return self.src[int(y0):int(y1), int(x0):int(x1), :3].reshape(-1, 3)

    def tones(self, *rects, lift=0.0):
        """(light, dark) from the original texels in the given rects (x0, y0, x1, y1),
        taken as the 85th / 15th lightness percentiles, so the palette is Create's."""
        px = np.concatenate([self.texels(*r) for r in rects])
        L = px @ LUMW
        o = np.argsort(L)
        px = px[o]
        n = len(px)
        light = px[int(n * 0.7):].mean(0) if n > 2 else px.max(0)
        dark = px[: max(1, int(n * 0.3))].mean(0)
        return np.clip(light * (1 + lift), 0, 1), np.clip(dark * (1 + lift), 0, 1)

    # ---------------------------------------------------------- shapes ----
    def d_rect(self, x0, y0, x1, y1):
        return np.minimum(np.minimum(self.x - x0, x1 - self.x), np.minimum(self.y - y0, y1 - self.y))

    def d_circle(self, cx, cy, r):
        return r - np.hypot(self.x - cx, self.y - cy)

    def d_poly(self, pts):
        m = np.zeros((self.H, self.W), np.uint8)
        cv2.fillPoly(m, [np.round(np.array(pts) * self.ppu - 0.5).astype(np.int32)], 1)
        din = cv2.distanceTransform(m, cv2.DIST_L2, 5) / self.ppu
        dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) / self.ppu
        return np.where(m > 0, din, -dout)

    def _cov(self, d):
        """Anti-aliased coverage from a signed distance (texels, + inside)."""
        return np.clip(d * self.ppu + 0.5, 0, 1)

    # ---------------------------------------------------------- noise -----
    def noise(self, sx, sy, seed):
        key = (sx, sy, seed)
        if key not in self._noise:
            n = max(self.H, self.W)
            f = fbm(sx, sy, 4, seed, size=n)[: self.H, : self.W]
            self._noise[key] = (f - f.mean()) / (f.std() + 1e-6)
        return self._noise[key]

    # ---------------------------------------------------------- painting --
    def paint(self, d, M, h=0.6, bevel=0.6, dir="h", dome=0.0, seed=0, mask=None, alpha=None):
        """Paint material M where the signed distance d > 0 (bevel in texels)."""
        cov = self._cov(d) if mask is None else mask
        dd = np.maximum(d, 0)
        prof = round_profile(dd * self.ppu, bevel * self.ppu) if bevel > 0 else np.ones_like(dd)
        k = M["kind"]
        s = self.seed * 101 + seed
        span = max(self.tw, self.th)
        if k == "metal":
            # fine brushed streaks along the part: few noise cells along it, many across
            along = self.noise(span * 6, span * 0.2, s) if dir == "v" else self.noise(span * 0.2, span * 6, s)
            mott = self.noise(span * 0.3, span * 0.3, s + 3)
            br = M.get("brush", 0.05)
            t = np.clip(0.6 + br * along + 0.6 * br * mott, 0, 1)
            hgt = h - 0.25 * (1 - prof) + 0.0008 * along
            smooth = M["smooth"] + 0.05 * along
        elif k == "wood":
            g = self.noise(span * 6, span * 0.3, s) if dir == "v" else self.noise(span * 0.3, span * 6, s)
            warp = self.noise(span * 0.15, span * 0.15, s + 5)
            t = np.clip(0.5 + 0.22 * g + 0.12 * warp, 0, 1)
            hgt = h - 0.2 * (1 - prof) - 0.01 * np.clip(-g, 0, None)
            smooth = M["smooth"] + 0.04 * g
        elif k == "stone":
            mott = self.noise(span * 0.5, span * 0.5, s)
            fine = self.noise(span * 3, span * 3, s + 1)
            t = np.clip(0.5 + 0.12 * mott + 0.07 * fine, 0, 1)
            hgt = h - 0.2 * (1 - prof) + 0.003 * fine
            smooth = M["smooth"] + 0.03 * fine
        elif k == "dark":
            # the far (upper) inner wall of a hole catches light; the rest falls to black
            t = np.clip(0.15 + 0.6 * (1 - prof), 0, 1)
            hgt = h - 0.15 * prof
            smooth = np.full_like(dd, M["smooth"])
        elif k == "glass":
            t = np.clip(0.6 + 0.1 * self.noise(span * 0.5, span * 0.5, s), 0, 1)
            hgt = np.full_like(dd, h)
            smooth = np.full_like(dd, M["smooth"])
        else:
            t = np.clip(0.5 + 0.1 * self.noise(span * 1.5, span * 1.5, s), 0, 1)
            hgt = h - 0.15 * (1 - prof)
            smooth = np.full_like(dd, M["smooth"])
        if dome:
            hgt = hgt + dome * np.clip(dd / max(dd.max(), 1e-6), 0, 1) ** 0.5
        alb = lerp(M["dark"], M["light"], t)
        self.m.put(cov, albedo=alb, height=hgt, smooth=smooth, f0=M["f0"], porosity=M["porosity"])
        if alpha is not None:
            self.m.alpha = self.m.alpha * (1 - cov) + alpha * cov
        if M.get("emissive"):
            self.m.emissive = self.m.emissive * (1 - cov) + M["emissive"] * cov
        return cov

    def fill(self, M, h=0.5, **kw):
        return self.paint(np.full((self.H, self.W), 99.0), M, h=h, bevel=0, **kw)

    def rect(self, x0, y0, x1, y1, M, h=0.6, bevel=0.6, **kw):
        return self.paint(self.d_rect(x0, y0, x1, y1), M, h=h, bevel=bevel, **kw)

    def circle(self, cx, cy, r, M, h=0.6, bevel=0.6, **kw):
        return self.paint(self.d_circle(cx, cy, r), M, h=h, bevel=bevel, **kw)

    def poly(self, pts, M, h=0.6, bevel=0.6, **kw):
        return self.paint(self.d_poly(pts), M, h=h, bevel=bevel, **kw)

    def frame(self, x0, y0, x1, y1, w, M, h=0.9, bevel=0.6, seed=0):
        """Mitred square frame of bars `w` wide inside the rect, brushed along each bar."""
        d_out = self.d_rect(x0, y0, x1, y1)
        d_in = self.d_rect(x0 + w, y0 + w, x1 - w, y1 - w)
        d = np.minimum(d_out, -d_in)
        horiz = np.minimum(self.y - y0, y1 - self.y) <= np.minimum(self.x - x0, x1 - self.x)
        cov_h = self.paint(np.where(horiz, d, -1), M, h=h, bevel=bevel, dir="h", seed=seed)
        cov_v = self.paint(np.where(~horiz, d, -1), M, h=h, bevel=bevel, dir="v", seed=seed + 1)
        # mitre: a hairline groove where horizontal and vertical bars meet
        diag = np.abs(np.abs(self.x - (x0 + x1) / 2) - np.abs(self.y - (y0 + y1) / 2) -
                      ((x1 - x0) - (y1 - y0)) / 2)
        groove = (d > 0) & (diag < 0.6 / self.ppu * 2)
        self.m.height = np.where(groove, self.m.height - 0.05, self.m.height)
        return np.maximum(cov_h, cov_v)

    def hole(self, d, M, h=0.15):
        return self.paint(d, M, h=h, bevel=1.2)

    def bolt(self, cx, cy, r, M, h=0.95, hexagon=False):
        """Round (or hex) bolt head: domed, lit rim, contact shadow round it."""
        dx, dy = self.x - cx, self.y - cy
        rr = np.hypot(dx, dy)
        if hexagon:
            dd = np.full_like(rr, -1e9)
            for k in range(3):
                a = 0.3 + k * math.pi / 3
                dd = np.maximum(dd, np.abs(dx * math.cos(a) + dy * math.sin(a)))
            d = r - dd
        else:
            d = r - rr
        cov = self._cov(d)
        dome = np.sqrt(np.clip(1 - (rr / r) ** 2, 0, 1))
        shade = np.clip(0.5 + 0.5 * (-(dx + dy) / (r * 1.4)), 0, 1)
        alb = lerp(M["dark"], M["light"], 0.3 + 0.6 * shade * (0.4 + 0.6 * dome))
        self.m.put(cov, albedo=alb, height=h + 0.08 * dome, smooth=M["smooth"] + 0.1, f0=M["f0"])
        self.m.extra_ao *= 1 - 0.35 * sstep(r * 1.6, r, rr) * (1 - cov)
        return cov

    def line(self, x0, y0, x1, y1, w, M, h=0.6, bevel=0.3, **kw):
        """A straight bar from (x0, y0) to (x1, y1), `w` texels wide."""
        px, py = self.x - x0, self.y - y0
        vx, vy = x1 - x0, y1 - y0
        ln = math.hypot(vx, vy) or 1.0
        t = np.clip((px * vx + py * vy) / ln ** 2, 0, 1)
        dist = np.hypot(px - t * vx, py - t * vy)
        return self.paint(w / 2 - dist, M, h=h, bevel=bevel, dir="h" if abs(vx) > abs(vy) else "v", **kw)

    def groove(self, d_line, depth=0.06, dark=0.75):
        """Darken and lower a hairline (d_line: distance to the line, texels)."""
        g = np.clip(1 - d_line / (0.9 / self.ppu * 3), 0, 1)
        self.m.height -= depth * g
        self.m.albedo *= (1 - (1 - dark) * g)[..., None]

    def shade_rect(self, x0, y0, x1, y1, strength=0.8, blur=0.6):
        """Carry the original's light and shade inside a rect onto what's painted there
        (smoothly: the artist's gradients, not its texel noise)."""
        L = self.src[..., :3] @ LUMW
        Lup = cv2.GaussianBlur(cv2.resize(L.astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_CUBIC),
                               (0, 0), self.ppu * blur)
        sub = L[int(y0):int(y1), int(x0):int(x1)]
        sel = (self.x >= x0) & (self.x < x1) & (self.y >= y0) & (self.y < y1)
        f = 1 + strength * (Lup / max(sub.mean(), 1e-3) - 1)
        self.m.albedo = np.where(sel[..., None], np.clip(self.m.albedo * f[..., None], 0, 1), self.m.albedo)

    def pattern(self, base=None, groove_q=0.22, groove_depth=0.08, relief=0.08, h=0.7, blur=0.35,
                metal_of=None, smooth_of=None, brush=0.03, bevel=0.4, groove_blur=0.45, groove_dark=0.12):
        """Single-material patterned surfaces (copper tiles, shingles, roofs, tank
        plates). Colour: this texture's own, smoothly upscaled (patina patches stay
        where the artist put them). Shape: from `base` (the clean pattern; this
        texture if None): its darkest lines sink as rounded grooves, the faces keep
        the artist's slope as gentle relief. metal_of(rgb 0-1) -> LabPBR F0/metal id
        per pixel, smooth_of(rgb) -> smoothness."""
        base = self.src if base is None else base
        up = lambda a, interp: cv2.resize(a.astype(np.float32), (self.W, self.H), interpolation=interp)
        col = np.dstack([cv2.GaussianBlur(up(self.src[..., c], cv2.INTER_CUBIC), (0, 0), self.ppu * blur)
                         for c in range(3)])
        Lb = base[..., :3] @ LUMW
        a = base[..., 3] > 0.5
        lo, hi = np.percentile(Lb[a], 2), np.percentile(Lb[a], 98)
        Ln = np.clip((Lb - lo) / max(hi - lo, 1e-6), 0, 1)
        Lup = cv2.GaussianBlur(up(Ln, cv2.INTER_CUBIC), (0, 0), self.ppu * 0.45)
        groove = cv2.GaussianBlur(up(Ln <= groove_q, cv2.INTER_NEAREST), (0, 0), self.ppu * groove_blur) > 0.5
        d = cv2.distanceTransform((~groove).astype(np.uint8), cv2.DIST_L2, 5) / self.ppu
        prof = round_profile(d * self.ppu, bevel * self.ppu)
        span = max(self.tw, self.th)
        streak = self.noise(span * 0.2, span * 6, self.seed * 31 + 5)
        alb = np.clip(col * (1 + brush * streak)[..., None], 0, 1)
        hgt = h + relief * (Lup - 0.5) - groove_depth * (1 - prof)
        alb = alb * (1 - groove_dark * (1 - prof))[..., None]      # grooves a little darker at the bottom
        flat = alb.reshape(-1, 3)
        f0 = np.array([metal_of(c) for c in np.round(col[::8, ::8].reshape(-1, 3) * 64) / 64]) if metal_of else None
        if metal_of:
            # metal id per pixel (evaluated on a coarse grid, then nearest-upscaled: ids must not blend)
            g = f0.reshape(col[::8, ::8].shape[:2])
            f0 = cv2.resize(g.astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            sm = cv2.resize(np.array([smooth_of(c) for c in np.round(col[::8, ::8].reshape(-1, 3) * 64) / 64],
                                     np.float32).reshape(g.shape), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        else:
            f0, sm = DIELECTRIC, 0.4
        cov = up(self.src[..., 3] > 0.5, cv2.INTER_NEAREST)
        self.m.put(cov, albedo=alb, height=hgt, smooth=sm + 0.04 * streak, f0=f0)
        return groove

    def cloth(self, weave=1.0, relief=0.05, h=0.7, smooth=0.12):
        """Fabric (table cloths, seats, sails): the original's colours kept crisp,
        texel for texel (checks and stripes stay sharp), with a fine woven surface
        and a soft pile."""
        up = cv2.resize(self.src[..., :3].astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        p = self.ppu / 4.0                                       # four threads per texel
        wx = np.sin(self.x * self.ppu / p * np.pi)
        wy = np.sin(self.y * self.ppu / p * np.pi)
        over = (np.floor(self.x * 4) + np.floor(self.y * 4)) % 2
        wv = np.where(over > 0, wx, wy)
        span = max(self.tw, self.th)
        pile = self.noise(span * 1.2, span * 1.2, self.seed * 7 + 11)
        alb = np.clip(up * (1 + 0.03 * weave * wv + 0.012 * pile)[..., None], 0, 1)
        hgt = h + 0.0015 * weave * wv + 0.003 * pile
        cov = cv2.resize((self.src[..., 3] > 0.5).astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        self.m.put(cov, albedo=alb, height=hgt, smooth=smooth + 0.03 * pile, f0=DIELECTRIC, porosity=40)

    # ---------------------------------------------------------- classes ---
    def classes(self, classify, spec, shade=0.7, bolts=None, seed=0, absorb=None):
        """Paint the original's colour regions as materials.

        classify(rgb 0-255 tuple) -> class name (or None to leave the canvas)
        spec[name] = dict(M=material, h=height, bevel=texels, dir="auto"|"h"|"v", shade=0..1)
        Straight texel edges stay crisp; each region bevels down only where it meets
        a lower one; metal/wood streaks follow each run's direction; the original's
        own light and shade inside a region is kept (`shade` strength); classes in
        `bolts` have their small specks (<= 4 texels) redrawn as round bolts.
        """
        th, tw = self.th, self.tw
        rgb8 = np.round(self.src[..., :3] * 255).astype(int)
        names = np.empty((th, tw), object)
        for y in range(th):
            for x in range(tw):
                names[y, x] = classify(tuple(rgb8[y, x])) if self.src[y, x, 3] > 0.5 else None
        if absorb:
            # thin or small runs of a class bordered mostly by one coloured material are that
            # material's shading (Create shades wood/copper/brass with dark low-colour texels)
            for cls, hosts in absorb.items():
                tmc = names == cls
                cnt, lab, stats, _ = cv2.connectedComponentsWithStats(tmc.astype(np.uint8), connectivity=8)
                for i in range(1, cnt):
                    x, y, w, h, area = stats[i]
                    if (min(w, h) > 2 and area > 6) or (cls == "hole" and area > 2):
                        continue                         # true openings stay open
                    comp = lab == i
                    ring = (cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~comp
                    nb = [m for m in names[ring] if m is not None]
                    if not nb:
                        continue
                    top = max(set(nb), key=nb.count)
                    if top in hosts and nb.count(top) >= 0.6 * len(nb):
                        names[comp] = top
        order = sorted({n for n in names.ravel() if n is not None}, key=lambda n: spec[n]["h"])
        up = lambda a: cv2.resize(a.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        # the original's luminance, smoothly upscaled, for carrying its shading over
        L = self.src[..., :3] @ LUMW
        Lup = cv2.GaussianBlur(cv2.resize(L.astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_CUBIC),
                               (0, 0), self.ppu * 0.6)
        bolt_specks = []
        for n in order:
            sp = spec[n]
            tm = names == n
            if bolts and n in bolts:
                cnt, lab, stats, cents = cv2.connectedComponentsWithStats(tm.astype(np.uint8), connectivity=4)
                specks = [i for i in range(1, cnt) if stats[i, cv2.CC_STAT_AREA] <= 4]
                # a few deliberate specks are bolts; many are surface speckle (girders, grime)
                if 0 < len(specks) <= 10:
                    for i in specks:
                        tm = tm & (lab != i)
                        bolt_specks.append((cents[i][0] + 0.5, cents[i][1] + 0.5,
                                            max(0.55, 0.5 * np.sqrt(stats[i, cv2.CC_STAT_AREA])), n))
            if not tm.any():
                continue
            mask = up(tm) > 0
            if sp.get("round"):
                mask = self._roundify(tm, mask)
            if sp.get("smooth_px"):
                # staircase diagonals -> straight, lone speckles dissolve: blur the mask, re-threshold
                mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), self.ppu * sp["smooth_px"]) > 0.5
            left = (up(tm) > 0) & ~mask
            if left.any():
                # pixels the region gives up take the material that surrounds the region
                ring = cv2.dilate(tm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                around = [m for m in names[ring & ~tm] if m is not None and m != n]
                if around:
                    sur = max(set(around), key=around.count)
                    self.paint(np.where(left, 9.0, -1.0), spec[sur]["M"], h=spec[sur]["h"], bevel=0,
                               seed=seed + 3)
            # bevel only toward lower regions: distance to the nearest pixel that is lower
            higher_or_same = np.array([[v is None for v in row] for row in names])  # no bevel toward unused space
            for n2 in order:
                if spec[n2]["h"] >= sp["h"]:
                    higher_or_same |= names == n2
            hm = up(higher_or_same) > 0
            if sp.get("smooth_px") or sp.get("round"):
                hm = (hm & ~(up(tm) > 0)) | mask
            d = cv2.distanceTransform(hm.astype(np.uint8), cv2.DIST_L2, 5) / self.ppu
            d = np.where(mask, d, -1.0)
            # direction of each run: horizontal where the texel run along x is the longer
            dirn = sp.get("dir", "auto")
            if dirn == "auto":
                hr = np.zeros((th, tw)); vr = np.zeros((th, tw))
                for y in range(th):
                    run = 0
                    for x in range(tw + 1):
                        if x < tw and tm[y, x]:
                            run += 1
                        else:
                            hr[y, x - run:x] = run; run = 0
                for x in range(tw):
                    run = 0
                    for y in range(th + 1):
                        if y < th and tm[y, x]:
                            run += 1
                        else:
                            vr[y - run:y, x] = run; run = 0
                horiz = up(hr >= vr) > 0
                ch = self.paint(np.where(mask & horiz, d, -1), sp["M"], h=sp["h"], bevel=sp.get("bevel", 0.5),
                                dir="h", seed=seed + 7 * len(n))
                cv_ = self.paint(np.where(mask & ~horiz, d, -1), sp["M"], h=sp["h"], bevel=sp.get("bevel", 0.5),
                                 dir="v", seed=seed + 7 * len(n) + 1)
                cov = np.maximum(ch, cv_)
            else:
                cov = self.paint(d, sp["M"], h=sp["h"], bevel=sp.get("bevel", 0.5), dir=dirn, seed=seed + 7 * len(n))
            if sp.get("relief"):
                # cast surfaces: gentle height from the original's own light pattern
                Lr = cv2.GaussianBlur(cv2.resize(L.astype(np.float32), (self.W, self.H),
                                              interpolation=cv2.INTER_CUBIC), (0, 0), self.ppu * 0.3)
                lo, hi = np.percentile(L[tm], 5), np.percentile(L[tm], 95)
                rel = np.clip((Lr - lo) / max(hi - lo, 1e-3), 0, 1)
                self.m.height = np.where(cov > 0.5, self.m.height + sp["relief"] * (rel - 0.5), self.m.height)
            st = sp.get("shade", shade)
            if st:
                mean = L[tm].mean()
                f = 1 + st * (Lup / max(mean, 1e-3) - 1)
                self.m.albedo = np.where((cov > 0.5)[..., None], np.clip(self.m.albedo * f[..., None], 0, 1),
                                         self.m.albedo)
        for cx, cy, r, n in bolt_specks:
            self.bolt(cx, cy, r, spec[n]["M"], h=spec[n]["h"] + 0.05)
        return names

    def _roundify(self, tm, mask):
        """Round regions drawn as staircases become true circles: a near-square
        component becomes a disc, or a ring if it encloses something."""
        out = mask.copy()
        cnt, lab, stats, cents = cv2.connectedComponentsWithStats(tm.astype(np.uint8), connectivity=8)
        for i in range(1, cnt):
            x, y, w, h, area = stats[i]
            if area < 4 or not (0.75 <= w / h <= 1.33):
                continue
            comp = lab == i
            # a round shape drawn in texels leaves its bounding box's corners empty;
            # squares and square rings fill them and must stay square
            corners = [comp[y, x], comp[y, x + w - 1], comp[y + h - 1, x], comp[y + h - 1, x + w - 1]]
            if any(corners) or w < 4:
                continue
            cx, cy = x + w / 2, y + h / 2
            r_out = (w + h) / 4
            filled = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            inner_area = (w * h * np.pi / 4) - area                 # what the ring encloses
            ring = inner_area > 0.25 * w * h
            region = cv2.resize(comp.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0
            out &= ~region
            rr = np.hypot(self.x - cx, self.y - cy)
            if ring:
                r_in = np.sqrt(max(inner_area, 0) / np.pi)
                out |= (rr <= r_out) & (rr >= r_in)
            else:
                out |= rr <= r_out
        return out

    def _extend_into_cutouts(self):
        """Carry each painted island's height and colour outward into unused
        (transparent) space, so atlas islands and cutout edges get no false cliff
        in the normals/AO and mipmaps don't bleed dark into their edges."""
        empty = (self.m.alpha < 0.5).astype(np.uint8)
        if not empty.any() or empty.all():
            return
        _, labels = cv2.distanceTransformWithLabels(empty, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        src_idx = np.zeros(labels.max() + 1, np.int64)
        flat = np.arange(empty.size).reshape(empty.shape)
        src_idx[labels[empty == 0]] = flat[empty == 0]
        near = src_idx[labels]
        sel = empty > 0
        self.m.height[sel] = self.m.height.ravel()[near[sel]]
        for c in range(3):
            ch = self.m.albedo[..., c]
            ch[sel] = ch.ravel()[near[sel]]

    def finish(self, edge_wear=0.35, height_scale=0.85, normal_strength=6.0, edges="reflect"):
        self.m.height = np.clip(self.m.height, 0.0, None)
        self._extend_into_cutouts()
        top = self.m.height.max()
        if top * height_scale > 1.0:
            self.m.height = self.m.height / top * (1.0 / height_scale)
        return derive(self.m, normal_strength=normal_strength, edge_wear=edge_wear, edges=edges,
                      height_scale=height_scale)
