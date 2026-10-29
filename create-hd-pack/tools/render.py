"""Offline LabPBR preview renderer: lit flat tiles and isometric blocks.

Reads the maps exactly as a LabPBR shader would (DirectX normals, AO in _n.b,
height in _n.a, smoothness/F0 in _s) so the preview checks the encoding too.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GAMMA = 2.2


def decode(a8, n8, s8):
    alb = (a8[..., :3] / 255.0) ** GAMMA
    alpha = a8[..., 3] / 255.0
    nx = n8[..., 0] / 255.0 * 2 - 1
    ny = n8[..., 1] / 255.0 * 2 - 1
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
    ao = n8[..., 2] / 255.0
    height = n8[..., 3] / 255.0
    smooth = s8[..., 0] / 255.0
    g = s8[..., 1].astype(float)
    metal = g >= 230
    f0 = np.where(metal[..., None], alb, (g / 255.0)[..., None] * np.ones(3))
    return dict(alb=alb, alpha=alpha, n=np.dstack([nx, ny, nz]), ao=ao, h=height,
                smooth=smooth, metal=metal, f0=f0)


def env(r):
    """Simple sky/ground environment, r = world dirs (..., 3), y up."""
    up = np.clip(r[..., 1], -1, 1)
    sky = np.array([0.55, 0.68, 0.9])
    horizon = np.array([0.9, 0.85, 0.75])
    ground = np.array([0.18, 0.15, 0.12])
    t = np.clip(up, 0, 1)[..., None]
    c = np.where(up[..., None] > 0, horizon + (sky - horizon) * np.sqrt(t), horizon + (ground - horizon) * np.clip(-up * 3, 0, 1)[..., None])
    return c


def self_shadow(h, l_ts, depth_px=64.0, steps=48):
    """Height-field shadowing along the light direction in texture space.

    Decoded height 0..1 spans LabPBR's 25% of a block = 64px at 256x.
    """
    lx, ly, lz = l_ts
    lxy = np.hypot(lx, ly)
    if lxy < 1e-4 or lz <= 0:
        return np.ones_like(h)
    dx, dy = lx / lxy, ly / lxy
    rise = lz / lxy / depth_px  # height units gained per px travelled
    shadow = np.zeros_like(h)
    hh = h
    for k in range(1, steps + 1):
        sx, sy = int(round(dx * k)), int(round(dy * k))
        sample = np.roll(np.roll(hh, -sy, 0), -sx, 1)
        occ = sample - (hh + rise * k)
        shadow = np.maximum(shadow, occ)
    return np.clip(1 - shadow * 25, 0, 1)


def shade(d, T, B, Nw, L, V, sun=3.2, amb=0.55):
    """Shade one face. T/B/N: world vectors for texture +u, +v (down), normal."""
    T, B, Nw, L, V = (np.asarray(x, float) for x in (T, B, Nw, L, V))
    n = d["n"]
    nw = n[..., :1] * T + n[..., 1:2] * B + n[..., 2:3] * Nw
    nw /= np.linalg.norm(nw, axis=-1, keepdims=True)
    H = (L + V) / np.linalg.norm(L + V)
    nl = np.clip((nw * L).sum(-1), 0, 1)
    nv = np.clip((nw * V).sum(-1), 1e-3, 1)
    nh = np.clip((nw * H).sum(-1), 0, 1)
    vh = np.clip((V * H).sum(), 0, 1)
    rough = np.clip((1 - d["smooth"]) ** 2, 0.02, 1)
    a2 = rough ** 2
    D = a2 / (np.pi * (nh ** 2 * (a2 - 1) + 1) ** 2)
    k = rough / 2
    G = (nl / (nl * (1 - k) + k)) * (nv / (nv * (1 - k) + k))
    F = d["f0"] + (1 - d["f0"]) * (1 - vh) ** 5
    spec = (D * G / (4 * np.maximum(nl, 1e-3) * nv))[..., None] * F
    kd = (~d["metal"]).astype(float)[..., None]
    diffuse = d["alb"] * kd / np.pi

    l_ts = (float(L @ T), float(L @ B), float(L @ Nw))
    sh = self_shadow(d["h"], l_ts)[..., None]
    ao = d["ao"][..., None]

    R = 2 * (nw * V).sum(-1, keepdims=True) * nw - V
    e = env(R)
    e_avg = np.array([0.5, 0.52, 0.55])
    e = e + (e_avg - e) * rough[..., None]
    Fr = d["f0"] + (np.maximum(1 - rough[..., None], d["f0"]) - d["f0"]) * (1 - nv[..., None]) ** 5
    ambient_spec = e * Fr * ao * (0.6 + 0.4 * sh)
    sky_light = env(nw) * amb
    c = (diffuse + spec) * sun * nl[..., None] * sh + d["alb"] * kd * sky_light * ao * 0.9 + ambient_spec
    c = c / (1 + c)  # Reinhard
    c = np.clip(c * 1.25, 0, 1) ** (1 / GAMMA)
    return np.dstack([c, d["alpha"]])


def tile_view(maps, light=(-0.55, -0.6, 0.58)):
    """Face-on render, raking light from the upper-left."""
    d = decode(*maps)
    L = np.array(light) / np.linalg.norm(light)
    # Treat texture space as world: +x right, +y down, +z out of screen.
    # env() expects y-up, so flip y via a world basis: up = -B.
    T, B, Nw = np.array([1., 0, 0]), np.array([0., -1, 0]), np.array([0., 0, 1])
    Lw = L[0] * T + L[1] * B + L[2] * Nw
    img = shade(d, T, B, Nw, Lw, Nw)
    return Image.fromarray((img * 255).round().astype(np.uint8), "RGBA")


# --------------------------------------------------------- iso renderer ---

VIEW = np.array([1.0, 0.82, 1.0]); VIEW /= np.linalg.norm(VIEW)
RIGHT = np.array([1.0, 0, -1.0]); RIGHT /= np.linalg.norm(RIGHT)
UPS = np.cross(VIEW, RIGHT); UPS /= np.linalg.norm(UPS)
SUN = np.array([0.35, 0.9, 0.55]); SUN /= np.linalg.norm(SUN)


def _proj(p, scale, origin):
    p = np.asarray(p, float)
    return np.array([origin[0] + scale * (p @ RIGHT), origin[1] - scale * (p @ UPS)])


def render_box(faces, size=420, scale=None, frm=(0, 0, 0), to=(16, 16, 16), bg=(0, 0, 0, 0)):
    """faces: dict face -> (maps, uv(u0,v0,u1,v1) in 0..16). Faces: up, south, east."""
    scale = scale or size / 30.0
    canvas = Image.new("RGBA", (size, size), bg)
    centre = (np.array(frm) + np.array(to)) / 2
    origin = (size / 2 - scale * (centre @ RIGHT), size / 2 + scale * (centre @ UPS))
    x0, y0, z0 = frm
    x1, y1, z1 = to
    defs = {
        # origin corner (texture top-left), T (u dir, full length), B (v dir), N
        "up": ((x0, y1, z0), (x1 - x0, 0, 0), (0, 0, z1 - z0), (0, 1, 0)),
        "south": ((x0, y1, z1), (x1 - x0, 0, 0), (0, -(y1 - y0), 0), (0, 0, 1)),
        "east": ((x1, y1, z1), (0, 0, -(z1 - z0)), (0, -(y1 - y0), 0), (1, 0, 0)),
    }
    for face in ("up", "south", "east"):
        if face not in faces:
            continue
        maps, uv = faces[face]
        O, Tv, Bv, Nv = (np.array(v, float) for v in defs[face])
        shaded = shade(decode(*maps), Tv / np.linalg.norm(Tv), Bv / np.linalg.norm(Bv), Nv, SUN, VIEW)
        img = Image.fromarray((shaded * 255).round().astype(np.uint8), "RGBA")
        res = img.width / 16.0
        u0, v0, u1, v1 = uv
        crop = img.crop((int(u0 * res), int(v0 * res), int(u1 * res), int(v1 * res)))
        # screen = S0 + a*U + b*Vv, with (a,b) in crop pixels
        S0 = _proj(O, scale, origin)
        U = (_proj(O + Tv, scale, origin) - S0) / crop.width
        Vv = (_proj(O + Bv, scale, origin) - S0) / crop.height
        M = np.array([[U[0], Vv[0]], [U[1], Vv[1]]])
        Mi = np.linalg.inv(M)
        # PIL wants input = A * output + c
        a, b = Mi[0]
        c_, d_ = Mi[1]
        off = -Mi @ S0
        warped = crop.transform(canvas.size, Image.AFFINE, (a, b, off[0], c_, d_, off[1]), resample=Image.BICUBIC)
        canvas.alpha_composite(warped)
    return canvas


def cube(maps, size=420, top=None, side=None):
    full = (0, 0, 16, 16)
    return render_box({"up": (top or maps, full), "south": (side or maps, full), "east": (side or maps, full)}, size)
