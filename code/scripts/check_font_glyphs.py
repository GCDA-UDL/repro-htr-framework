#!/usr/bin/env python3
"""Glyph diagnostics of the curated font sets. For every font folder the first .ttf/.otf file is loaded with PIL and each
test character is rendered alone at the same origin. Reported per font: (1) UNICASE letters, x/X pairs drawn with the
same glyph (the font cannot show case, so image and transcription disagree in case); (2) STAMP-like glyphs, renders
made of dense tiny text (demo fonts put a 'PERSONAL USE ONLY' stamp in some character slots); (3) PLACEHOLDER glyphs,
three or more different characters sharing one identical render; (4) EMPTY renders. Per set: the fonts to look at.
Read-only. Usage: python code/scripts/check_font_glyphs.py [set ...]
   default sets: fonts/KidsCurated fonts_sel/School_ca fonts_sel/Handwritten25_ca
   (paths relative to code/framework; e.g. fonts/Handwritten for the whole base category, a few minutes)
"""
import glob, hashlib, os, sys
from PIL import Image, ImageDraw, ImageFont, ImageOps

os.chdir(os.path.expanduser("~/workrepo-htr-synthetic-framework/code/framework"))
SETS = sys.argv[1:] or ["fonts/KidsCurated", "fonts_sel/School_ca", "fonts_sel/Handwritten25_ca"]
LOWER = "abcdefghijklmnopqrstuvwxyzàèéíïòóúüçáñ"
UPPER = LOWER.upper()
OTHER = "0123456789.,;:!?'\"-()º·¿¡"
CHARS = LOWER + UPPER + OTHER
SIZE = 48


def render(font, c):
    """(hash of ink bitmap + position, ink fraction, mean black/white transitions per row, width) or None if empty."""
    asc, desc = font.getmetrics()
    try:
        l, t, r, b = font.getbbox(c)
    except Exception:
        return None
    w, h = max(1, r) + 60, asc + desc + 60
    im = Image.new("L", (w, h), 255)
    ImageDraw.Draw(im).text((30, 30), c, font=font, fill=0)
    box = ImageOps.invert(im).getbbox()
    if box is None:
        return None
    crop = im.crop(box)
    px = crop.load()
    cw, ch = crop.size
    dark = sum(1 for y in range(ch) for x in range(cw) if px[x, y] < 128)
    trans = sum(1 for y in range(ch) for x in range(1, cw) if (px[x, y] < 128) != (px[x - 1, y] < 128))
    key = hashlib.md5(crop.tobytes() + bytes(str(box), "ascii")).hexdigest()
    return key, dark / (cw * ch), trans / ch, cw


for s in SETS:
    folders = sorted(d for d in glob.glob(os.path.join(s, "*")) if os.path.isdir(d))
    print(f"\n=== {s}: {len(folders)} font folders ===")
    unicase_fonts, stamp_fonts = [], []
    for d in folders:
        files = sorted(glob.glob(os.path.join(d, "*.[to]tf")) + glob.glob(os.path.join(d, "*.[TO]TF")))
        if not files:
            print(f"  {os.path.basename(d)}: no .ttf/.otf file"); continue
        try:
            font = ImageFont.truetype(files[0], SIZE)
        except Exception as e:
            print(f"  {os.path.basename(d)}: cannot load {os.path.basename(files[0])} ({e})"); continue
        r = {c: render(font, c) for c in CHARS}
        empty = [c for c in CHARS if r[c] is None]
        uni = [lo for lo, up in zip(LOWER, UPPER) if r[lo] is not None and r[up] is not None and r[lo][0] == r[up][0]]
        stampy = [c for c in CHARS if r[c] is not None and r[c][1] > 0.2 and r[c][2] >= 8 and r[c][3] >= 0.8 * SIZE]
        groups = {}
        for c, v in r.items():
            if v is not None:
                groups.setdefault(v[0], []).append(c)
        dup = ["".join(g) for g in groups.values() if len(g) >= 3]
        flags = []
        if len(uni) >= 20:
            flags.append(f"UNICASE ({len(uni)}/{len(LOWER)} letter pairs identical)"); unicase_fonts.append(os.path.basename(d))
        elif uni:
            flags.append(f"{len(uni)} identical case pairs: {''.join(uni)}")
        if stampy:
            flags.append(f"STAMP-like glyphs: {''.join(stampy)}"); stamp_fonts.append(os.path.basename(d))
        if dup:
            flags.append("PLACEHOLDER (same render): " + " | ".join(dup))
            if os.path.basename(d) not in stamp_fonts:
                stamp_fonts.append(os.path.basename(d))
        if empty:
            flags.append(f"empty: {''.join(empty)}")
        print(f"  {os.path.basename(d):32s} {'; '.join(flags) if flags else 'ok'}")
    print(f"  -> unicase fonts: {len(unicase_fonts)} {unicase_fonts}")
    print(f"  -> fonts with stamp / placeholder glyphs: {len(stamp_fonts)} {stamp_fonts}")
