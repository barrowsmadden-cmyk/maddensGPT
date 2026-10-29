"""Minecraft JSON block models: resolve, index texture usage, render.

    resolve(model_id, assets)          -> (elements, textures) after following parents
    texture_usage(assets, ns)          -> {texture_id: [Use(model, elem, face, uv, size, rot)]}
    render(model_id, assets, tex_root) -> RGBA image of the model, isometric, vanilla face shading

Model UVs are in 0..16 over the whole texture (texture_size is Blockbench
metadata). Face UV defaults, face rotation and element rotation follow
vanilla's FaceBakery conventions.
"""
import json
import math
import os
from collections import defaultdict, namedtuple

import numpy as np
from PIL import Image

Use = namedtuple("Use", "model elem face uv size rot")
FACES = ("north", "south", "west", "east", "up", "down")


def _split(mid, default_ns="minecraft"):
    mid = mid.lstrip("#")
    ns, path = mid.split(":", 1) if ":" in mid else (default_ns, mid)
    return ns, path


def _load(assets, mid):
    ns, path = _split(mid)
    p = os.path.join(assets, ns, "models", path + ".json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def resolve(mid, assets):
    """Elements (from the nearest model that has them) and the merged texture
    map with #references resolved."""
    textures, elements, seen = {}, None, set()
    cur = mid
    while cur and cur not in seen:
        seen.add(cur)
        m = _load(assets, cur)
        if m is None:
            break
        for k, v in m.get("textures", {}).items():
            textures.setdefault(k, v)
        if elements is None and "elements" in m:
            elements = m["elements"]
        cur = m.get("parent")
        if cur and ":" not in cur and not cur.startswith("block/") and not cur.startswith("item/"):
            cur = _split(mid)[0] + ":" + cur
    def deref(v, depth=0):
        while isinstance(v, str) and v.startswith("#") and depth < 10:
            v = textures.get(v[1:], v)
            depth += 1
        return v
    return elements or [], {k: deref(v) for k, v in textures.items()}


def default_uv(face, f, t):
    x1, y1, z1 = f
    x2, y2, z2 = t
    return {"down": [x1, 16 - z2, x2, 16 - z1], "up": [x1, z1, x2, z2],
            "north": [16 - x2, 16 - y2, 16 - x1, 16 - y1], "south": [x1, 16 - y2, x2, 16 - y1],
            "west": [z1, 16 - y2, z2, 16 - y1], "east": [16 - z2, 16 - y2, 16 - z1, 16 - y1]}[face]


def face_size(face, f, t):
    dx, dy, dz = (abs(b - a) for a, b in zip(f, t))
    return {"down": (dx, dz), "up": (dx, dz), "north": (dx, dy), "south": (dx, dy),
            "west": (dz, dy), "east": (dz, dy)}[face]


def texture_usage(assets, ns="create"):
    """Every texture used by `ns` block models, with the UV rect and model-space
    size (in 1/16 block) of each face that uses it."""
    usage = defaultdict(list)
    root = os.path.join(assets, ns, "models", "block")
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            mid = f"{ns}:" + os.path.relpath(os.path.join(dirpath, fn), os.path.join(assets, ns, "models"))[:-5]
            elements, textures = resolve(mid, assets)
            for ei, el in enumerate(elements):
                for face, fd in el.get("faces", {}).items():
                    tex = textures.get(fd.get("texture", "").lstrip("#"), fd.get("texture"))
                    if not isinstance(tex, str) or tex.startswith("#"):
                        continue
                    tns, tpath = _split(tex)
                    uv = fd.get("uv") or default_uv(face, el["from"], el["to"])
                    usage[f"{tns}:{tpath}"].append(Use(mid, ei, face, tuple(uv), face_size(face, el["from"], el["to"]),
                                                       fd.get("rotation", 0)))
    return usage


# ------------------------------------------------------------------ render ---

def _rot(v, axis, ang, origin):
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    x, y, z = (v[i] - origin[i] for i in range(3))
    if axis == "x":
        y, z = y * c - z * s, y * s + z * c
    elif axis == "y":
        x, z = x * c + z * s, -x * s + z * c
    else:
        x, y = x * c - y * s, x * s + y * c
    return (x + origin[0], y + origin[1], z + origin[2])


def _corners(face, f, t):
    """Face corners in texture order: top-left, top-right, bottom-right, bottom-left
    (as seen from outside, matching the default UV layout)."""
    x1, y1, z1 = f
    x2, y2, z2 = t
    return {"north": [(x2, y2, z1), (x1, y2, z1), (x1, y1, z1), (x2, y1, z1)],
            "south": [(x1, y2, z2), (x2, y2, z2), (x2, y1, z2), (x1, y1, z2)],
            "west": [(x1, y2, z1), (x1, y2, z2), (x1, y1, z2), (x1, y1, z1)],
            "east": [(x2, y2, z2), (x2, y2, z1), (x2, y1, z1), (x2, y1, z2)],
            "up": [(x1, y2, z1), (x2, y2, z1), (x2, y2, z2), (x1, y2, z2)],
            "down": [(x1, y1, z2), (x2, y1, z2), (x2, y1, z1), (x1, y1, z1)]}[face]


LIGHT_TS = np.array([-0.45, -0.55, 0.70]) / np.linalg.norm([-0.45, -0.55, 0.70])  # texture space, y down


def load_texture(tex_root, tex_id, cache):
    if tex_id in cache:
        return cache[tex_id]
    ns, path = _split(tex_id)
    roots = [tex_root] if isinstance(tex_root, str) else tex_root
    p = next((os.path.join(r, ns, "textures", path + ".png") for r in roots
              if os.path.exists(os.path.join(r, ns, "textures", path + ".png"))), "")
    img = None
    if os.path.exists(p):
        a = np.asarray(Image.open(p).convert("RGBA")).astype(np.float32) / 255
        img = a[: a.shape[1]] if a.shape[0] > a.shape[1] else a          # first animation frame
        npath = p[:-4] + "_n.png"
        if os.path.exists(npath):
            # light the albedo with the normal map (LabPBR, DirectX: green = down) and its AO,
            # so the preview shows relief the way a shader will
            nm = np.asarray(Image.open(npath).convert("RGBA")).astype(np.float32) / 255
            nm = nm[: nm.shape[1]] if nm.shape[0] > nm.shape[1] else nm
            if nm.shape[:2] == img.shape[:2]:
                nx, ny = nm[..., 0] * 2 - 1, nm[..., 1] * 2 - 1
                nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
                dot = nx * LIGHT_TS[0] + ny * LIGHT_TS[1] + nz * LIGHT_TS[2]
                f = np.clip(0.25 + 0.75 * dot / LIGHT_TS[2], 0.2, 1.35) * (0.45 + 0.55 * nm[..., 2])
                img = img.copy()
                img[..., :3] = np.clip(img[..., :3] * f[..., None], 0, 1)
    cache[tex_id] = img
    return img


def render(mid, assets, tex_root, size=256, yaw=-45, pitch=30, ss=2, override=None):
    """Render a model isometrically: orthographic, vanilla face shading, nearest
    sampling. `tex_root` is an assets dir holding the textures to use;
    `override` maps texture ids to arrays."""
    elements, textures = resolve(mid, assets)
    cache = dict(override or {})
    n = size * ss
    col = np.zeros((n, n, 4), np.float32)
    zb = np.full((n, n), -1e9, np.float32)
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))

    def project(v):
        x, y, z = v[0] - 8, v[1] - 8, v[2] - 8
        x, z = x * cy - z * sy, x * sy + z * cy                        # yaw about y
        y, z = y * cp - z * sp, y * sp + z * cp                        # pitch about x
        k = n / 34.0
        return (n / 2 + x * k, n / 2 - y * k, z)

    for el in elements:
        f, t = el["from"], el["to"]
        r = el.get("rotation")
        for face, fd in el.get("faces", {}).items():
            tex_id = textures.get(fd.get("texture", "").lstrip("#"), fd.get("texture"))
            if not isinstance(tex_id, str):
                continue
            img = load_texture(tex_root, tex_id if ":" in tex_id else "minecraft:" + tex_id, cache)
            if img is None:
                continue
            corners = _corners(face, f, t)
            normal = {"north": (0, 0, -1), "south": (0, 0, 1), "west": (-1, 0, 0), "east": (1, 0, 0),
                      "up": (0, 1, 0), "down": (0, -1, 0)}[face]
            if r:
                corners = [_rot(c, r["axis"], r["angle"], r["origin"]) for c in corners]
                normal = _rot(normal, r["axis"], r["angle"], (0, 0, 0))
            nx, ny, nz = normal
            shade = 1.0 if not el.get("shade", True) else \
                0.6 * nx * nx + 0.8 * nz * nz + (1.0 if ny > 0 else 0.5) * ny * ny
            P = [project(c) for c in corners]
            # back-face cull: screen-space winding
            (ax, ay, _), (bx, by, _), (cx_, cy_, _) = P[0], P[1], P[2]
            if (bx - ax) * (cy_ - ay) - (by - ay) * (cx_ - ax) <= 0:
                continue
            uv = fd.get("uv") or default_uv(face, f, t)
            rot = fd.get("rotation", 0)
            h, w = img.shape[:2]
            o = np.array(P[0][:2]); e1 = np.array(P[1][:2]) - o; e2 = np.array(P[3][:2]) - o
            M = np.array([e1, e2]).T
            if abs(np.linalg.det(M)) < 1e-6:
                continue
            inv = np.linalg.inv(M)
            xs = [p[0] for p in P]; ys = [p[1] for p in P]
            x0, x1 = max(0, int(min(xs))), min(n, int(math.ceil(max(xs))))
            y0, y1 = max(0, int(min(ys))), min(n, int(math.ceil(max(ys))))
            if x1 <= x0 or y1 <= y0:
                continue
            gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
            st = np.einsum("ij,jhw->ihw", inv, np.stack([gx - o[0], gy - o[1]]))
            s, tt = st[0], st[1]
            inside = (s >= 0) & (s <= 1) & (tt >= 0) & (tt <= 1)
            if not inside.any():
                continue
            for _ in range((rot // 90) % 4):                         # texture rotated clockwise on the face
                s, tt = tt, 1 - s
            u = (uv[0] + (uv[2] - uv[0]) * s) / 16 * w
            v = (uv[1] + (uv[3] - uv[1]) * tt) / 16 * h
            ui = np.clip(np.floor(u).astype(int), 0, w - 1)
            vi = np.clip(np.floor(v).astype(int), 0, h - 1)
            texel = img[vi, ui]
            za, zb_, zc = P[0][2], P[1][2], P[3][2]
            depth = za + (zb_ - za) * st[0] + (zc - za) * st[1]
            ok = inside & (texel[..., 3] > 0.1) & (depth > zb[y0:y1, x0:x1])
            sub = col[y0:y1, x0:x1]
            sub[ok, :3] = texel[ok, :3] * shade
            sub[ok, 3] = 1.0
            zb[y0:y1, x0:x1][ok] = depth[ok]
    out = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8), "RGBA")
    return out.resize((size, size), Image.LANCZOS) if ss > 1 else out
