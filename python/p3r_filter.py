"""
P3R-style photo filter
----------------------
ภาพคนสมจริง + โทนขาวดำหลายระดับ + พื้นหลังฟ้าสด (แรงบันดาลใจจาก Persona 3 Reload)

AI ใช้แค่ตัดพื้นหลัง (rembg) ส่วนที่เหลือเป็น OpenCV ล้วน

ติดตั้ง:
    pip install opencv-python numpy rembg onnxruntime

ใช้งาน:
    python p3r_filter.py input.jpg output.png
    python p3r_filter.py input.jpg output.png --smooth 0.4 --levels 7 --shards 30
"""

import argparse
import cv2
import numpy as np


# ---------------------------------------------------------------- 1. ตัดพื้นหลัง
def get_mask(img_bgr, band=30, protect=None):
    """คืน mask 0..1 (float32) ของตัวคน"""
    try:
        from rembg import remove, new_session
        session = new_session("u2net_human_seg")  # โมเดลสำหรับคนโดยเฉพาะ
        rgba = remove(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                      session=session, post_process_mask=True)
        mask = rgba[:, :, 3].astype(np.float32) / 255.0
    except ImportError:
        # สำรอง: GrabCut (ไม่ต้องใช้ AI แต่แม่นน้อยกว่า)
        h, w = img_bgr.shape[:2]
        m = np.zeros((h, w), np.uint8)
        rect = (int(w * .05), int(h * .02), int(w * .9), int(h * .97))
        cv2.grabCut(img_bgr, m, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((m == 1) | (m == 3), 1.0, 0.0).astype(np.float32)
    mask = refine_edges(img_bgr, mask, band, protect)
    # ขอบนุ่มเล็กน้อย ไม่ให้ดูเหมือนตัดแปะ
    return cv2.GaussianBlur(mask, (0, 0), 1.0)


def refine_edges(img_bgr, mask, band_px=30, protect_px=None):
    """
    ลบพื้นหลังเดิมที่ติดมาตามช่องว่างของผมหยิก/ขอบตัว:
    พิกเซลที่สีใกล้สีพื้นหลังเดิม และอยู่ใกล้ขอบ จะถูกลดค่า mask
    (เช็คเฉพาะแถบขอบ ไม่ไปกัดผิวด้านใน)

    protect_px: ลบได้เฉพาะส่วนที่ "บาง" (ช่องผม, ขอบ halo)
    ส่วนที่หนากว่า ~2*protect_px เช่น จมูก ริมฝีปาก แก้มที่โดนแสง
    จะไม่ถูกลบ แม้สีจะใกล้ผนัง (0 = ปิดการป้องกัน)
    """
    if band_px <= 0:
        return mask
    if protect_px is None:
        protect_px = max(3, round(max(mask.shape) * 0.005))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_px = lab[mask < 0.05]
    if bg_px.size < 300:
        return mask
    bg_ref = np.median(bg_px, axis=0)
    dist = np.linalg.norm(lab - bg_ref, axis=2)
    key = np.clip((dist - 10) / 18, 0, 1)  # 0 = เหมือนพื้นหลัง

    # แถบขอบ: ระยะจากพื้นหลังไม่เกิน band_px
    outside = (mask < 0.5).astype(np.uint8)
    d = cv2.distanceTransform(1 - outside, cv2.DIST_L2, 5)
    near = np.clip(1 - d / band_px, 0, 1)
    removed = near * (1 - key)

    if protect_px > 0:
        # ส่วนที่จะถูกลบ ถ้าหนาพอที่ kernel วงกลมขนาดนี้ใส่ได้ = เนื้อจริง ไม่ใช่ช่องผม
        region = ((removed > 0.3) & (mask > 0.5)).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * protect_px + 1, 2 * protect_px + 1))
        thick = cv2.morphologyEx(region, cv2.MORPH_OPEN, k)
        thick = cv2.dilate(thick, np.ones((3, 3), np.uint8)).astype(np.float32)
        thick = cv2.GaussianBlur(thick, (0, 0), 1.5)
        # ปกป้องเฉพาะส่วนหนาที่สี "ไม่ใช่ผนังแท้" (halo ผนังจริงสีห่างแค่ ~0-8)
        not_wall = np.clip((dist - 8) / 8, 0, 1)
        removed *= 1 - np.clip(thick * 1.5, 0, 1) * not_wall

    # ลบได้เฉพาะกลุ่มพิกเซลที่ (ก) ต่อกับพื้นหลังจริง หรือ (ข) อยู่ติดผม/ส่วนมืด
    # กันจุดไฮไลต์บนจมูก/หน้าผากที่สีสว่างใกล้ผนังไม่ให้กลายเป็นรู
    region = ((removed > 0.3) & (mask > 0.5)).astype(np.uint8)
    n, lbl = cv2.connectedComponents(region, connectivity=8)
    if n > 1:
        seed = cv2.dilate(outside, np.ones((5, 5), np.uint8))
        dark = (lab[:, :, 0] < bg_ref[0] * 0.55).astype(np.uint8)
        seed = seed | cv2.dilate(dark, np.ones((7, 7), np.uint8))
        ok = np.zeros(n, bool)
        ok[np.unique(lbl[(seed > 0) & (region > 0)])] = True
        ok[0] = True
        bad = (~ok[lbl]).astype(np.float32)
        bad = cv2.GaussianBlur(cv2.dilate(bad, np.ones((3, 3), np.uint8)), (0, 0), 1.0)
        removed *= 1 - np.clip(bad, 0, 1)
    return mask * (1 - removed)


# ---------------------------------------------------------------- 2. ลบริ้วรอยบางส่วน
def soften_skin(img_bgr, strength=0.45):
    """
    ลบริ้วรอย/รูขุมขนแค่บางส่วน:
    - bilateral filter ทำให้ผิวเรียบแต่ขอบยังคม
    - ผสมกลับกับภาพเดิมตาม strength (0 = ไม่ทำ, 1 = เรียบสุด)
    - คืน texture ละเอียดบางส่วนกลับมา เพื่อไม่ให้ผิวดูเป็นพลาสติก
    """
    smooth = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=45, sigmaSpace=9)
    smooth = cv2.bilateralFilter(smooth, d=7, sigmaColor=30, sigmaSpace=7)
    out = cv2.addWeighted(img_bgr, 1 - strength, smooth, strength, 0)

    # high-frequency detail (texture ผิว) คืนมา 35%
    f = img_bgr.astype(np.float32)
    detail = f - cv2.GaussianBlur(f, (0, 0), 1.5)
    out = out.astype(np.float32) + detail * 0.35 * (1 - strength)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 3. โทนแสงเงา
def tone_map(gray, mask, levels=6, posterize=0.55,
             black=22, white=238, gamma=1.0):
    """
    - CLAHE เพิ่มมิติเฉพาะจุด (ไม่ให้ทั้งภาพมืดหรือสว่างแบน)
    - auto-level จากเฉพาะพื้นที่ตัวคน
    - posterize แบบนุ่ม: ผสมระดับที่ quantize กับค่าต่อเนื่องเดิม
      ได้ก้อนเงาแบบกราฟิก แต่ยังมี gradient ให้ดูสมจริง
    - จำกัด black/white point ไม่ให้ดำสนิทหรือขาวโพลน
    """
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    g = clahe.apply(gray).astype(np.float32)

    # auto-level จาก percentile ของตัวคน
    fg = g[mask > 0.5]
    if fg.size > 100:
        lo, hi = np.percentile(fg, 2), np.percentile(fg, 98)
        g = (g - lo) / max(hi - lo, 1) * 255
    g = np.clip(g, 0, 255) / 255.0
    g = g ** gamma

    # soft posterize: ใช้ smoothstep ระหว่างขั้น เพื่อไม่ให้มีเส้นแบ่งแข็งเกิน
    # quantize จากภาพที่เรียบกว่า เพื่อให้ขอบก้อนเงาเนียน ไม่เป็นจุดกระจาย
    gs = cv2.bilateralFilter((g * 255).astype(np.uint8), 9, 30, 9)
    gs = cv2.GaussianBlur(gs, (0, 0), 1.2).astype(np.float32) / 255.0
    steps = levels - 1
    x = gs * steps
    base = np.floor(x)
    frac = x - base
    t = np.clip((frac - 0.35) / 0.3, 0, 1)
    t = t * t * (3 - 2 * t)
    q = (base + t) / steps

    out = q * posterize + g * (1 - posterize)
    out = black + out * (white - black)
    return out.astype(np.float32)


# ---------------------------------------------------------------- 4. เส้นหมึกบาง ๆ
def ink_lines(gray, amount=0.35):
    """เส้นขอบเข้มบาง ๆ จาก adaptive threshold (amount 0 = ปิด)"""
    if amount <= 0:
        return np.ones_like(gray, np.float32)
    g = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 11, 9)
    edges = cv2.GaussianBlur(edges, (0, 0), 0.8).astype(np.float32) / 255.0
    return 1 - (1 - edges) * amount


# ---------------------------------------------------------------- 5. พื้นหลัง
def hex_to_bgr(h):
    h = h.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.float32)


def make_background(h, w, color="#12ACF4", shards=25, seed=7):
    base = hex_to_bgr(color)
    bg = np.ones((h, w, 3), np.float32) * base

    # vignette อ่อน ๆ ให้พื้นมีมิติเล็กน้อย
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h * .4) / h) ** 2)
    bg *= (1.06 - 0.18 * d)[..., None]

    # เศษกระจกสีฟ้าอ่อน
    rng = np.random.default_rng(seed)
    layer = np.zeros_like(bg)
    alpha = np.zeros((h, w), np.float32)
    s = min(h, w)
    for _ in range(shards):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.008, 0.035) * s
        n = rng.integers(3, 6)
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        pts = np.stack([cx + r * rng.uniform(.5, 1.3, n) * np.cos(ang),
                        cy + r * rng.uniform(.5, 1.3, n) * np.sin(ang)], 1)
        tint = np.array([255, rng.uniform(230, 255), rng.uniform(140, 210)])
        a = np.zeros((h, w), np.float32)
        cv2.fillPoly(a, [pts.astype(np.int32)], float(rng.uniform(.25, .6)))
        layer += a[..., None] * tint
        alpha = np.maximum(alpha, a)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.7)
    bg = bg * (1 - alpha[..., None]) + layer  # layer = สี x alpha แล้ว
    return np.clip(bg, 0, 255)


# ---------------------------------------------------------------- 6. รวมทั้งหมด
def p3r_filter(img_bgr, smooth=0.45, levels=6, posterize=0.55,
               ink=0.35, tint=0.12, bg_color="#12ACF4", shards=25,
               black=22, white=238, gamma=1.0, detail=0.9, band=30, protect=None):
    mask = get_mask(img_bgr, band, protect)
    soft = soften_skin(img_bgr, smooth)
    gray = cv2.cvtColor(soft, cv2.COLOR_BGR2GRAY)

    # กันขอบขาว (halo): พิกเซลแถวขอบปนสีพื้นหลังเดิม
    # จึงเติมค่าจากด้านในตัวคนแทน ก่อนคำนวณโทน
    inner = (mask > 0.95).astype(np.uint8)
    band = ((mask > 0.02) & (mask <= 0.95)).astype(np.uint8)
    band = cv2.dilate(band, np.ones((5, 5), np.uint8)) & (1 - inner)
    gray = cv2.inpaint(gray, band * 255, 5, cv2.INPAINT_TELEA)
    # หด mask เข้าไปนิดเดียว ให้ขอบคม
    mask = cv2.GaussianBlur(cv2.erode(mask, np.ones((5, 5), np.uint8)), (0, 0), 1.0)

    tone = tone_map(gray, mask, levels, posterize, black, white, gamma)
    tone *= ink_lines(gray, ink)

    # คืนรายละเอียดเส้นผม/ผ้าจากภาพต้นฉบับ (high-pass) ให้ภาพไม่ลอย
    g0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g0 - cv2.GaussianBlur(g0, (0, 0), 2.5)
    tone = np.clip(tone + hp * detail, black * 0.6, white)

    # ใส่สีฟ้าจาง ๆ ในเงาและโทนกลาง ให้ตัวคนกลืนกับพื้นหลัง
    person = np.repeat(tone[..., None], 3, axis=2)
    shadow_w = (1 - tone / 255.0)[..., None]
    blue = hex_to_bgr(bg_color)
    person = person * (1 - tint * shadow_w) + (blue * 0.35) * tint * shadow_w

    h, w = img_bgr.shape[:2]
    bg = make_background(h, w, bg_color, shards)

    m = mask[..., None]
    out = person * m + bg * (1 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="P3R-style photo filter")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--smooth", type=float, default=0.45, help="ลบริ้วรอย 0-1")
    p.add_argument("--levels", type=int, default=6, help="จำนวนระดับเงา")
    p.add_argument("--posterize", type=float, default=0.55,
                   help="0 = สมจริงต่อเนื่อง, 1 = ก้อนเงาแบบกราฟิก")
    p.add_argument("--ink", type=float, default=0.35, help="ความเข้มเส้นขอบ 0-1")
    p.add_argument("--tint", type=float, default=0.12, help="สีฟ้าในเงา 0-1")
    p.add_argument("--black", type=int, default=22, help="จุดดำสุด (ยิ่งมากยิ่งไม่ดำสนิท)")
    p.add_argument("--white", type=int, default=238, help="จุดขาวสุด (ยิ่งน้อยยิ่งไม่ขาวโพลน)")
    p.add_argument("--gamma", type=float, default=1.0, help=">1 มืดลง, <1 สว่างขึ้น")
    p.add_argument("--detail", type=float, default=0.9, help="รายละเอียดผม/ผ้า 0-2")
    p.add_argument("--bg", default="#12ACF4", help="สีพื้นหลัง")
    p.add_argument("--shards", type=int, default=25, help="จำนวนเศษกระจก (0 = ปิด)")
    p.add_argument("--band", type=int, default=30, help="ระยะลบพื้นหลังเดิมที่ขอบ px (0 = ปิด)")
    p.add_argument("--protect", type=int, default=None, help="ปกป้องจมูก/ส่วนหนา px (0 = ปิด, ไม่ใส่ = อัตโนมัติ)")
    p.add_argument("--max-size", type=int, default=2000)
    a = p.parse_args()

    img = cv2.imread(a.input)
    if img is None:
        raise SystemExit(f"เปิดไฟล์ไม่ได้: {a.input}")
    s = a.max_size / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    out = p3r_filter(img, a.smooth, a.levels, a.posterize, a.ink, a.tint,
                     a.bg, a.shards, a.black, a.white, a.gamma, a.detail, a.band, a.protect)
    cv2.imwrite(a.output, out)
    print("saved:", a.output)


if __name__ == "__main__":
    main()
