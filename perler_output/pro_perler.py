#!/usr/bin/env python3
"""Toddler head → perler beads (CIEDE2000, lip-locked palette, clean edges)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from rembg import remove as rembg_remove

SRC = Path(
    "/home/ubuntu/.cursor/projects/workspace/assets/"
    "E0758CA1-23E5-463B-A249-271BB347ADD9_L0_001.jpg"
)
# Fallback: relative path if assets move
if not SRC.exists():
    SRC = Path(__file__).resolve().parent.parent / "assets" / SRC.name

OUT = Path(__file__).resolve().parent
ART = Path("/opt/cursor/artifacts")
if ART.is_dir():
    OUT_DIRS = [OUT, ART]
else:
    OUT_DIRS = [OUT]

PALETTE = [
    ("Black", "#121212"),
    ("Dark Brown", "#2A1810"),
    ("Espresso", "#3E2416"),
    ("Walnut", "#6E4028"),
    ("Cocoa", "#5A3220"),
    ("Brown", "#4A2C18"),
    ("Ivory", "#FFF8F0"),
    ("Cream", "#FFE8D0"),
    ("Light Peach", "#FFD2B0"),
    ("Peach", "#F0C09A"),
    ("Apricot", "#E4B090"),
    ("Warm Tan", "#D8A078"),
    ("Tan", "#C89068"),
    ("Caramel", "#B87850"),
    ("Shadow Tan", "#9A6848"),
    ("Deep Shadow", "#785040"),
    ("Lip Pink", "#D06068"),
    ("Hot Rose", "#B84850"),
    ("Deep Rose", "#903040"),
    ("White", "#FFFFFF"),
]
LIP = {"Lip Pink", "Hot Rose", "Deep Rose"}
HAIR = {"Black", "Dark Brown", "Espresso", "Walnut", "Cocoa", "Brown"}
SKIN = {
    "Ivory", "Cream", "Light Peach", "Peach", "Apricot",
    "Warm Tan", "Tan", "Caramel", "Shadow Tan", "Deep Shadow",
}


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c / 255.0, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(rgb.astype(np.float64))
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ M.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = lab1[:, None, 0], lab1[:, None, 1], lab1[:, None, 2]
    L2, a2, b2 = lab2[None, :, 0], lab2[None, :, 1], lab2[None, :, 2]
    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    Cbar = 0.5 * (C1 + C2)
    Cbar7 = Cbar**7
    G = 0.5 * (1 - np.sqrt(Cbar7 / (Cbar7 + 25**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((C1p * C2p) == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbar = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hbar = np.where(
        (C1p * C2p) == 0,
        hsum,
        np.where(np.abs(h1p - h2p) > 180, np.where(hsum < 360, hsum + 360, hsum - 360) / 2, hsum / 2),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hbar - 30))
        + 0.24 * np.cos(np.radians(2 * hbar))
        + 0.32 * np.cos(np.radians(3 * hbar + 6))
        - 0.20 * np.cos(np.radians(4 * hbar - 63))
    )
    dtheta = 30 * np.exp(-(((hbar - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbarp**7 / (Cbarp**7 + 25**7))
    Sl = 1 + (0.015 * (Lbar - 50) ** 2) / np.sqrt(20 + (Lbar - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def cut_head(src: Path) -> np.ndarray:
    im = Image.open(src).convert("RGB")
    W, H = im.size
    crop = im.crop((0, 0, int(W * 0.48), int(H * 0.56)))
    rgba = np.array(rembg_remove(crop))
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3].astype(np.float32)
    mx, mn = rgb.max(2), rgb.min(2)
    sat = (mx - mn) / np.maximum(mx, 1)
    a[(mx > 240) & (sat < 0.06)] = 0
    a[a < 40] = 0
    am = (a > 128).astype(np.uint8) * 255
    am = cv2.morphologyEx(am, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(am, 8)
    if n > 1:
        am = np.where(lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))), 255, 0).astype(np.uint8)
    am = cv2.morphologyEx(am, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    hh, ww = am.shape
    gray = rgb.mean(2)
    dark = (gray < 65) & (am > 0)
    rows = (am > 0).sum(1)
    drow = dark.sum(1)
    cut_y = hh
    for y in range(int(hh * 0.50), hh):
        if rows[y] < 8:
            continue
        if drow[y] / max(1, rows[y]) > 0.38:
            cut_y = y
            break
    am[cut_y:] = 0
    am = cv2.morphologyEx(am, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    am = cv2.morphologyEx(am, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    inv = 255 - am
    n2, l2, s2, _ = cv2.connectedComponentsWithStats(inv, 8)
    border = set()
    for x in range(ww):
        if inv[0, x]:
            border.add(l2[0, x])
        if inv[hh - 1, x]:
            border.add(l2[hh - 1, x])
    for y in range(hh):
        if inv[y, 0]:
            border.add(l2[y, 0])
        if inv[y, ww - 1]:
            border.add(l2[y, ww - 1])
    for i in range(1, n2):
        if i not in border and s2[i, cv2.CC_STAT_AREA] < 5000:
            am[l2 == i] = 255
    a = cv2.GaussianBlur(am.astype(np.float32), (0, 0), 0.8)
    rgba[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
    ys, xs = np.where(rgba[:, :, 3] > 18)
    return rgba[max(0, ys.min() - 2) : min(hh, ys.max() + 3), max(0, xs.min() - 2) : min(ww, xs.max() + 3)]


def alpha_resize(rgba, nw, nh, thr=0.55):
    h, w = rgba.shape[:2]
    rgb0 = rgba[:, :, :3].astype(np.float64)
    al = rgba[:, :, 3].astype(np.float64) / 255
    out = np.zeros((nh, nw, 3))
    oa = np.zeros((nh, nw))
    for y in range(nh):
        y0 = int(np.floor(y * h / nh))
        y1 = min(h, max(y0 + 1, int(np.floor((y + 1) * h / nh))))
        for x in range(nw):
            x0 = int(np.floor(x * w / nw))
            x1 = min(w, max(x0 + 1, int(np.floor((x + 1) * w / nw))))
            ba = al[y0:y1, x0:x1]
            wt = ba.sum()
            oa[y, x] = wt / max(1, ba.size)
            if wt > 1e-6:
                out[y, x] = (rgb0[y0:y1, x0:x1] * ba[..., None]).sum((0, 1)) / wt
    return np.clip(out, 0, 255), oa >= thr


def clean_mask(mask):
    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))) if n > 1 else m > 0


def nearest(rgb, mask, pal_lab, allow):
    idx = np.full(mask.shape, -1, np.int32)
    ys, xs = np.where(mask)
    labs = rgb_to_lab(rgb[ys, xs])
    sub = pal_lab[allow]
    out = np.empty(len(ys), np.int32)
    for s in range(0, len(ys), 4000):
        e = min(len(ys), s + 4000)
        out[s:e] = allow[delta_e2000(labs[s:e], sub).argmin(1)]
    idx[ys, xs] = out
    return idx


def remove_isolated(idx, mask, min_n=2):
    out = idx.copy()
    H, W = idx.shape
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if not mask[y, x]:
                continue
            c = idx[y, x]
            neigh = idx[y - 1 : y + 2, x - 1 : x + 2]
            nm = mask[y - 1 : y + 2, x - 1 : x + 2]
            if int(((neigh == c) & nm).sum()) - 1 < min_n:
                vals = neigh[nm]
                vals = vals[vals >= 0]
                if len(vals) == 0:
                    continue
                vals = vals[vals != c] if (vals != c).any() else vals
                out[y, x] = Counter(vals.tolist()).most_common(1)[0][0]
    return out


def majority(idx, mask):
    out = idx.copy()
    nxt = out.copy()
    H, W = idx.shape
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if not mask[y, x]:
                continue
            vals = out[y - 1 : y + 2, x - 1 : x + 2][mask[y - 1 : y + 2, x - 1 : x + 2]]
            vals = vals[vals >= 0]
            if len(vals) < 5:
                continue
            mode, cnt = Counter(vals.tolist()).most_common(1)[0]
            if cnt >= 5 and int((vals == out[y, x]).sum()) <= 2:
                nxt[y, x] = mode
    return nxt


def save_all(name: str, img: Image.Image):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / name)


def main():
    if not SRC.exists():
        raise SystemExit(f"Source image not found: {SRC}")
    print("cutout…")
    rgba = cut_head(SRC)
    Image.fromarray(rgba).save(OUT / "head_clean_rgba.png")
    wbg = Image.new("RGB", (rgba.shape[1], rgba.shape[0]), (255, 255, 255))
    wbg.paste(Image.fromarray(rgba), mask=Image.fromarray(rgba[:, :, 3]))
    wbg.save(OUT / "head_clean_white.jpg", quality=95)
    if ART.is_dir():
        Image.fromarray(rgba).save(ART / "head_clean_rgba.png")
        wbg.save(ART / "head_clean_white.jpg", quality=95)

    rgb = rgba[:, :, :3].copy()
    aa = rgba[:, :, 3]
    sm = cv2.bilateralFilter(rgb, 7, 40, 7)
    lum = rgb.mean(2)
    keep = (lum < 100) & (aa > 128)
    blend = sm.copy()
    blend[keep] = rgb[keep]
    rgba2 = rgba.copy()
    rgba2[:, :, :3] = blend
    pil = Image.fromarray(rgba2[:, :, :3])
    pil = ImageEnhance.Color(pil).enhance(1.12)
    pil = ImageEnhance.Contrast(pil).enhance(1.10)
    pil = ImageEnhance.Brightness(pil).enhance(1.03)
    rgba2[:, :, :3] = np.array(pil)

    h0, w0 = rgba2.shape[:2]
    nh = 192
    nw = max(1, int(round(w0 * (nh / h0))))
    if nw > 200:
        nw = 200
        nh = max(1, int(round(h0 * (nw / w0))))
    print("grid", nw, nh)

    rgb_f, mask = alpha_resize(rgba2, nw, nh, 0.55)
    mask = clean_mask(mask)
    edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    mask &= ~((rgb_f.mean(2) > 238) & edge)
    mask = clean_mask(mask)

    names = [n for n, _ in PALETTE]
    pal_rgb = np.stack([hex_to_rgb(h) for _, h in PALETTE], 0)
    pal_lab = rgb_to_lab(pal_rgb)
    lip_i = np.array([i for i, n in enumerate(names) if n in LIP], np.int32)
    hair_i = np.array([i for i, n in enumerate(names) if n in HAIR], np.int32)
    skin_i = np.array([i for i, n in enumerate(names) if n in SKIN], np.int32)
    base_i = np.array([i for i, n in enumerate(names) if n not in LIP], np.int32)

    idx = nearest(rgb_f, mask, pal_lab, base_i)
    ys, xs = np.where(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    yy, xx = np.mgrid[0:nh, 0:nw]
    red = (r - g) + 0.35 * (r - b)

    win = mask & (yy > cy * 1.03) & (yy < cy * 1.20) & (xx > cx * 0.80) & (xx < cx * 1.20) & (r > 150) & (r > g + 14) & (r > b + 10)
    if win.sum() < 6:
        win = mask & (yy > cy * 1.00) & (yy < cy * 1.24) & (xx > cx * 0.74) & (xx < cx * 1.26) & (r > 145) & (r > g + 12)
    lips = np.zeros_like(mask)
    if win.sum():
        thr = max(20.0, float(np.percentile(red[win], 62)))
        mm = ((win) & (red >= thr)).astype(np.uint8) * 255
        mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mm, 8)
        best, bs = None, 1e18
        maxa = max(60, int(0.009 * mask.sum()))
        for i in range(1, n):
            ar = st[i, cv2.CC_STAT_AREA]
            if ar < 5 or ar > maxa:
                continue
            yys, xxs = np.where(lab == i)
            sc = (yys.mean() - cy * 1.12) ** 2 + (xxs.mean() - cx) ** 2 + 0.04 * ar
            if sc < bs:
                bs, best = sc, i
        if best is None and n > 1:
            best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if best is not None:
            lips = lab == best
        lips = cv2.dilate(lips.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), 1) > 0
        lips &= mask
        cap = max(30, int(0.0065 * mask.sum()))
        if lips.sum() > cap:
            yys, xxs = np.where(lips)
            sc = red[yys, xxs]
            k = np.argsort(-sc)[:cap]
            nl = np.zeros_like(lips)
            nl[yys[k], xxs[k]] = True
            lips = cv2.morphologyEx(nl.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))) > 0
    print("lips", int(lips.sum()))
    if lips.sum():
        d = delta_e2000(rgb_to_lab(rgb_f[lips]), pal_lab[lip_i])
        idx[lips] = lip_i[d.argmin(1)]

    hair = mask & (yy < cy * 0.80) & ((r + g + b) / 3 < 100)
    hair |= mask & (((xx < cx * 0.40) | (xx > cx * 1.52)) & ((r + g + b) / 3 < 112) & (yy < cy * 1.18))
    hair &= ~lips
    if hair.sum():
        d = delta_e2000(rgb_to_lab(rgb_f[hair]), pal_lab[hair_i])
        idx[hair] = hair_i[d.argmin(1)]
    eyes = mask & (yy > cy * 0.68) & (yy < cy * 0.93) & ((r + g + b) / 3 < 75) & ~lips
    if eyes.sum():
        idx[eyes] = names.index("Black")

    bad = mask & ~lips & np.isin(idx, lip_i)
    if bad.sum():
        d = delta_e2000(rgb_to_lab(rgb_f[bad]), pal_lab[skin_i])
        idx[bad] = skin_i[d.argmin(1)]

    prot = lips | hair | eyes
    idx = remove_isolated(idx, mask, 2)
    idx = majority(idx, mask & ~prot)
    idx = remove_isolated(idx, mask, 1)
    if lips.sum():
        d = delta_e2000(rgb_to_lab(rgb_f[lips]), pal_lab[lip_i])
        idx[lips] = lip_i[d.argmin(1)]
    bad = mask & ~lips & np.isin(idx, lip_i)
    if bad.sum():
        d = delta_e2000(rgb_to_lab(rgb_f[bad]), pal_lab[skin_i])
        idx[bad] = skin_i[d.argmin(1)]

    er = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    for y, x in zip(*np.where(mask & (er == 0))):
        if mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].sum() <= 2:
            mask[y, x] = False
            idx[y, x] = -1

    bead = np.zeros((nh, nw, 4), np.uint8)
    for i, col in enumerate(pal_rgb):
        sel = idx == i
        bead[sel, :3] = np.clip(col, 0, 255).astype(np.uint8)
        bead[sel, 3] = 255
    ys, xs = np.where(bead[:, :, 3] > 0)
    bead = bead[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    nh, nw = bead.shape[:2]
    total = int((bead[:, :, 3] > 0).sum())
    used = Counter(tuple(map(int, bead[y, x, :3])) for y in range(nh) for x in range(nw) if bead[y, x, 3])
    print("FINAL", nw, nh, total, "colors", len(used))

    cell = 5
    prev = Image.new("RGB", (nw * cell, nh * cell), (255, 255, 255))
    px = prev.load()
    for y in range(nh):
        for x in range(nw):
            if not bead[y, x, 3]:
                continue
            rr, gg, bb = map(int, bead[y, x, :3])
            for dy in range(cell - 1):
                for dx in range(cell - 1):
                    px[x * cell + dx, y * cell + dy] = (rr, gg, bb)
    for d in OUT_DIRS:
        prev.save(d / "perler-preview.png")
        prev.save(d / f"拼豆预览_写实_{nw}x{nh}.png")

    rgb_to_name = {tuple(int(x) for x in hex_to_rgb(h)): n for n, h in PALETTE}
    colors = sorted(used, key=lambda c: (-used[c], c))
    step = 6
    pat_w = max(nw * step + 24, 520)
    legend_h = 36 + 20 * ((len(colors) + 1) // 2)
    pat = Image.new("RGB", (pat_w, nh * step + 70 + legend_h), (255, 255, 255))
    draw = ImageDraw.Draw(pat)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_s = ImageFont.load_default()
    draw.text((12, 12), f"拼豆图纸 {nw}x{nh}  |  {total} beads  |  {len(colors)} colors  |  CIEDE2000", fill=(25, 25, 25), font=font)
    for y in range(nh):
        for x in range(nw):
            if not bead[y, x, 3]:
                continue
            rr, gg, bb = map(int, bead[y, x, :3])
            x0, y0 = 12 + x * step, 42 + y * step
            draw.rectangle([x0, y0, x0 + 4, y0 + 4], fill=(rr, gg, bb))
    ly = 48 + nh * step
    col_w = pat_w // 2
    for i, col in enumerate(colors):
        cx_ = 12 + (i % 2) * col_w
        cy_ = ly + (i // 2) * 20
        draw.rectangle([cx_, cy_, cx_ + 14, cy_ + 14], fill=col, outline=(70, 70, 70))
        draw.text((cx_ + 20, cy_), f"{rgb_to_name.get(col, str(col))}  ×{used[col]}", fill=(35, 35, 35), font=font_s)
    for d in OUT_DIRS:
        pat.save(d / "perler-pattern-full.png")
        pat.save(d / f"拼豆图纸_写实_{nw}x{nh}.png")

    ref = Image.open(OUT / "head_clean_white.jpg").convert("RGB")
    ref_s = ref.resize((300, int(300 * ref.size[1] / ref.size[0])), Image.Resampling.LANCZOS)
    pw = 460
    ph = int(460 * prev.size[1] / prev.size[0])
    prev_s = prev.resize((pw, ph), Image.Resampling.NEAREST)
    comp = Image.new("RGB", (300 + pw + 50, max(ref_s.size[1], ph) + 55), (255, 255, 255))
    d = ImageDraw.Draw(comp)
    d.text((10, 10), "干净原图（仅头部）", fill=(30, 30, 30), font=font)
    d.text((340, 10), f"拼豆预览 {nw}x{nh}", fill=(30, 30, 30), font=font)
    comp.paste(ref_s, (10, 38))
    comp.paste(prev_s, (340, 38))
    for dd in OUT_DIRS:
        comp.save(dd / "perler-compare.png")

    lines = [f"# 拼豆材料清单 {nw}x{nh}", f"总豆数: {total}", f"颜色数: {len(colors)}", ""]
    for col in colors:
        lines.append(f"{rgb_to_name.get(col, str(col))}\tRGB{col}\t{used[col]}")
    text = "\n".join(lines)
    for dd in OUT_DIRS:
        (dd / "perler-materials.txt").write_text(text, encoding="utf-8")
    np.savez(OUT / "bead_grid.npz", bead=bead)
    print(text)
    print("done")


if __name__ == "__main__":
    main()
