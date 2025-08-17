# -*- coding: utf-8 -*-
"""
CV + OCR utilities:
- Phân tích cấu trúc bản vẽ
- Sinh box phụ thuộc ảnh (cạnh/dòng/contours)
- Ước lượng mm/px từ số liệu kích thước bằng Tesseract
- Xử lý robust để chạy được ngay cả khi chỉ có 1 ảnh
"""

import os
import hashlib
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# OCR
import pytesseract


# -----------------------------
# Tesseract helpers
# -----------------------------
def ensure_tesseract_ready(verbose=False) -> str:
    """
    Đặt pytesseract.tesseract_cmd nếu cần (Windows).
    Trả về version (nếu có).
    """
    exe_candidates = [
        os.environ.get("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "tesseract",  # rely on PATH
    ]
    for p in exe_candidates:
        if not p:
            continue
        try:
            pytesseract.pytesseract.tesseract_cmd = p
            ver = pytesseract.get_tesseract_version()
            if verbose:
                print(f"DEBUG: Tesseract forced: {p} (v{ver})")
            return str(ver)
        except Exception:
            continue
    if verbose:
        print("DEBUG: Tesseract not found in common locations; OCR may fail.")
    return ""


# -----------------------------
# Basic image analysis
# -----------------------------
def create_image_hash(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def analyze_image_features(image_path: str) -> Dict:
    im = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # edges + lines
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=120, maxLineGap=8)
    lines = [] if lines is None else lines[:,0,:].tolist()

    # contours (để tìm khối chữ nhật)
    cnts, _ = cv2.findContours((edges>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    H, W = gray.shape[:2]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 200:   # bỏ nhiễu
            continue
        # chuẩn hoá field (x, y, z, sx, sy, sz, az) với z,sz tạm thời dummy
        components.append([float(x+0.5*w), float(y+0.5*h), 450.0, float(w), float(h), 20.0, 0.0])

    return {
        "image_path": image_path,
        "image_hash": create_image_hash(image_path),
        "W": W, "H": H,
        "edges": edges,
        "lines": lines,
        "components": components,
    }


def generate_image_dependent_boxes(features: Dict, max_boxes=12) -> List[Tuple[str, List[float]]]:
    """
    Chuyển các contours/rectangles thành Bbox list ưu tiên theo:
    - kích thước lớn trước (các khối chính)
    - phân bố đều theo không gian
    """
    comps = features.get("components", [])
    # sort theo area desc
    comps = sorted(comps, key=lambda c: c[3]*c[4], reverse=True)
    comps = comps[:max_boxes]
    # sanitize (ép mm/thickness hợp lệ + round đẹp)
    def _smart_round(v): return float(np.round(v, 1))
    out = []
    for i, c in enumerate(comps):
        x,y,z,sx,sy,sz,az = c
        z  = min(800.0, max(200.0, float(z)))
        sz = min(100.0, max(10.0,  float(sz)))
        out.append((f"bbox_{i}", [_smart_round(x), _smart_round(y), _smart_round(z),
                                  _smart_round(sx), _smart_round(sy), _smart_round(20.0), 0.0]))
    return out


# -----------------------------
# OCR numbers + simple viz
# -----------------------------
def _preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    # nhẹ nhàng: blur nhẹ + adaptive threshold
    g = cv2.medianBlur(gray, 3)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 35, 7)
    # mở/đóng để liền nét
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    return th


def _ocr_tokens(image_path: str) -> Tuple[List[Tuple[str, int, int, int, int]], str]:
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    th = _preprocess_for_ocr(im)
    cfg = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(th, config=cfg, output_type=pytesseract.Output.DICT)
    tokens = []
    for i, txt in enumerate(data["text"]):
        if not txt or txt.strip() == "":
            continue
        conf = int(data["conf"][i]) if str(data["conf"][i]).isdigit() else -1
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        tokens.append((txt.strip(), x, y, w, h))
    # lưu debug
    os.makedirs("output/ocr_debug", exist_ok=True)
    dbg_path = "output/ocr_debug/last_ocr.png"
    vis = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    for t, x, y, w, h in tokens:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 1)
        cv2.putText(vis, t, (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,0), 1, cv2.LINE_AA)
    cv2.imwrite(dbg_path, vis)
    return tokens, dbg_path


def _extract_integers(tokens: List[Tuple[str,int,int,int,int]]) -> List[Tuple[int,int,int,int,int]]:
    ints = []
    for t,x,y,w,h in tokens:
        # loại "QS-01" -> lấy phần số độc lập
        m = None
        # thuần số
        if re_fullmatch(r"\d{2,4}", t):
            m = t
        else:
            # tách số đứng riêng trong chuỗi (ví dụ 1014)
            import re
            r = re.findall(r"\b\d{2,4}\b", t)
            if r:
                m = r[0]
        if m is not None:
            v = int(m)
            # bỏ số quá nhỏ (0..20) thường không phải mm
            if v < 20: 
                continue
            ints.append((v,x,y,w,h))
    return ints

# tiny regex helper
import re
def re_fullmatch(pat, s):
    return re.fullmatch(pat, s) is not None


# -----------------------------
# Estimate scale mm/px
# -----------------------------
def estimate_scale_mm_per_px(image_path: str, features: Dict=None) -> Dict:
    """
    Trả về dict:
      {'sx': mm_per_px_x or None, 'sy': ..., 'conf_x': int, 'conf_y': int, 'tokens': N, 'merged_numbers': K, 'debug_path': path, 'source': 'ocr'|'catalog'}
    Chiến lược:
      1) OCR tất cả số 2..4 chữ số.
      2) Pair số với "đoạn thẳng Hough" gần nó theo phương ngang/dọc để ước lượng pixel-length.
      3) mm/px = value_mm / length_px ; lấy median của đủ nhiều cặp.
      4) Nếu không đủ bằng chứng -> thử "catalog pairing" (mọi cặp số nằm cùng hàng/cột), đánh dấu source='catalog'.
    """
    ensure_tesseract_ready(verbose=False)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"sx": None, "sy": None, "conf_x": 0, "conf_y": 0, "tokens": 0, "merged_numbers": 0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tokens, dbg = _ocr_tokens(image_path)
    ints = _extract_integers(tokens)

    # Hough lines để đo px-length theo chiều ngang/dọc
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=140, maxLineGap=6)
    lines = [] if lines is None else lines[:,0,:].tolist()
    horizontals = [(x1,y1,x2,y2) for (x1,y1,x2,y2) in lines if abs(y2-y1) <= 2 and abs(x2-x1) >= 20]
    verticals   = [(x1,y1,x2,y2) for (x1,y1,x2,y2) in lines if abs(x2-x1) <= 2 and abs(y2-y1) >= 20]

    def _nearest_len(px, py, pool):
        # lấy đoạn thẳng gần nhất theo hướng tương ứng
        best = None; bestd = 999999
        for (x1,y1,x2,y2) in pool:
            # distance point->segment (x roughly within range & y khoảng cách nhỏ)
            if pool is horizontals:
                if abs(py - y1) > 25:  # cùng hàng
                    continue
                # lấy đoạn chồng x
                if px < min(x1,x2)-20 or px > max(x1,x2)+20:
                    continue
                L = abs(x2-x1)
            else:
                if abs(px - x1) > 25:
                    continue
                if py < min(y1,y2)-20 or py > max(y1,y2)+20:
                    continue
                L = abs(y2-y1)
            d = min(abs(px-x1)+abs(py-y1), abs(px-x2)+abs(py-y2))
            if d < bestd and L > 30:
                bestd = d; best = L
        return best

    ratios_x, ratios_y = [], []
    for v,x,y,w,h in ints:
        cx, cy = x + w//2, y + h//2
        Lx = _nearest_len(cx, cy, horizontals)
        if Lx is not None:
            ratios_x.append(float(v) / float(Lx))
        Ly = _nearest_len(cx, cy, verticals)
        if Ly is not None:
            ratios_y.append(float(v) / float(Ly))

    source = "ocr"
    conf_x = len(ratios_x)
    conf_y = len(ratios_y)

    if conf_x < 3 and conf_y < 3:
        # fallback: ghép mọi cặp số nằm gần cùng hàng/cột để suy length_px ~ |dx| or |dy|
        # (độ tin cậy thấp hơn)
        for i in range(len(ints)):
            vi, xi, yi, wi, hi = ints[i]
            cxi, cyi = xi+wi//2, yi+hi//2
            for j in range(i+1, len(ints)):
                vj, xj, yj, wj, hj = ints[j]
                cxj, cyj = xj+wj//2, yj+hj//2
                if abs(cyi-cyj) <= 15 and abs(cxj-cxi) >= 30:
                    Lx = abs(cxj-cxi)
                    ratios_x.append(abs(vj-vi)/float(Lx) if vj!=vi else None)
                if abs(cxj-cxi) <= 15 and abs(cyj-cyi) >= 30:
                    Ly = abs(cyj-cyi)
                    ratios_y.append(abs(vj-vi)/float(Ly) if vj!=vi else None)
        ratios_x = [r for r in ratios_x if r and r>0]
        ratios_y = [r for r in ratios_y if r and r>0]
        source = "catalog"
        conf_x = len(ratios_x)
        conf_y = len(ratios_y)

    def _robust_median(arr):
        if not arr: return None
        arr = sorted(arr)
        m = arr[len(arr)//2] if len(arr)%2 else 0.5*(arr[len(arr)//2-1]+arr[len(arr)//2])
        # clamp vào khoảng mm/px hợp lý cho cabinet (0.3..5.0)
        return float(max(0.3, min(5.0, m)))

    sx = _robust_median(ratios_x)
    sy = _robust_median(ratios_y)

    return {
        "sx": sx, "sy": sy, "conf_x": int(conf_x), "conf_y": int(conf_y),
        "tokens": int(len(tokens)), "merged_numbers": int(len(ints)),
        "debug_path": dbg, "source": source
    }
