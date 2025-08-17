# -*- coding: utf-8 -*-
"""
Inference pipeline theo tinh thần CAD2Program:
- Nhận 1 ảnh CAD (PNG/JPG)
- Gọi VLM (Qwen2-VL) để sinh "program" ở dạng DSL
- Hậu xử lý + CV/OCR để chuẩn hoá và tăng độ tin cậy
"""

import os, re, sys, random, argparse, torch
from typing import List, Tuple
from PIL import Image

# ====== VLM (HuggingFace Transformers) ======
from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig

# ====== CV helpers ======
from cv_analysis import (
    analyze_image_features,
    generate_image_dependent_boxes,
    estimate_scale_mm_per_px,
    ensure_tesseract_ready,
)

# -----------------------------
# Prompt template theo paper (image->program)
# -----------------------------
SYS_PROMPT = """You are a CAD parametric model extraction assistant. Analyze the engineering drawing and extract geometric components with parametric model information.

Output ONLY the program between <PROGRAM> and </PROGRAM>.
Format line 1..N (exact):
bbox_N = Bbox(pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, angle_z)
model_N = <model_#########>(optional_args)

Rules:
- pos_x, pos_y, scale_x, scale_y: PIXELS from the given image
- pos_z, scale_z: MILLIMETERS (pos_z: 200..800, scale_z: 10..100)
- angle_z: usually 0
- 12 UNIQUE components with DIFFERENT coordinates
- Generate unique 9-digit model IDs (#########) for each model line
"""

USER_PROMPT = """CRITICAL: Analyze THIS image (size: {W}x{H} px). Extract exactly 12 components you really see (shelves, frames, doors, dimension blocks).
VARY coordinates for each component; do not repeat template values.

<PROGRAM>
</PROGRAM>
"""

# -----------------------------
# Parse & post-process VLM text + CV/OCR
# -----------------------------
def parse_and_make_absolute(raw: str, W: int, H: int, max_boxes=12, image_path: str = None) -> str:
    _float = r"-?\d+(?:\.\d+)?"
    RE_BBOX  = re.compile(rf"^(bbox_\d+)\s*=\s*[Bb]box\(\s*({_float})\s*,\s*({_float})\s*,\s*({_float})\s*,\s*({_float})\s*,\s*({_float})\s*,\s*({_float})\s*,\s*({_float})\s*\)\s*$")
    RE_MODEL = re.compile(r"(model_\d+)\s*=\s*<?(model_\d+)>?\s*\(([^)]*)\)")

    boxes_raw, models_raw = [], []
    for ln in [l.strip() for l in raw.splitlines() if l.strip()]:
        m = RE_BBOX.match(ln)
        if m:
            name = m.group(1)
            vals = list(map(float, m.groups()[1:]))
            boxes_raw.append((name, vals))
            continue
        m = RE_MODEL.match(ln)
        if m:
            n, op, args = m.groups()
            if "<" in ln and "operation" in ln:  # bỏ placeholder lạ
                continue
            models_raw.append((n, op, args or ""))

    boxes_raw = boxes_raw[:max_boxes]

    def _smart_round(val: float) -> float:
        if abs(val - round(val + 0.5)) < 0.1:
            return float(round(val + 0.5) - 0.5)
        return round(float(val), 1)

    def _convert_vals(vals):
        x, y, z, sx, sy, sz, az = vals
        norm_like = (0.0 <= x <= 1.2) and (0.0 <= y <= 1.2) and (0.0 <= sx <= 1.2) and (0.0 <= sy <= 1.2)
        if norm_like:
            x  = x * W
            y  = y * H
            z  = 200 + (z % 1.0) * 600       # 200–800 mm
            sx = max(10.0, sx * W)
            sy = max(10.0, sy * H)
            sz = 10 + (sz % 1.0) * 90        # 10–100 mm
        z  = float(min(800.0, max(200.0, z)))
        sz = float(min(100.0, max(10.0,  sz)))
        return [_smart_round(x), _smart_round(y), _smart_round(z),
                _smart_round(sx), _smart_round(sy), _smart_round(sz), _smart_round(az)]

    vlm_boxes = [(f"bbox_{i}", _convert_vals(vals)) for i, (_, vals) in enumerate(boxes_raw)]

    # ---- Lấy box từ CV (ưu tiên) ----
    cv_boxes = []
    cv_features = None
    if image_path:
        cv_features = analyze_image_features(image_path)
        img_hash = cv_features.get("image_hash", "NA")
        print(f"DEBUG: Image hash: {img_hash}, CV components: {len(cv_features.get('components', []))}")
        cv_boxes = generate_image_dependent_boxes(cv_features, max_boxes)
        print(f"DEBUG: CV generated {len(cv_boxes)} image-dependent boxes")

    fixed_boxes = (cv_boxes if len(cv_boxes) > 0 else vlm_boxes)[:max_boxes]

    # ---- Sort ổn định & rename tuần tự ----
    coords = [c for _, c in fixed_boxes]
    order = sorted(range(len(coords)), key=lambda i: (coords[i][1], coords[i][0]))  # (y,x)
    fixed_boxes = [(f"bbox_{i}", coords[j]) for i, j in enumerate(order)]

    # ---- Làm tròn cuối + clamp + ép góc 0.0 ----
    def _final(vals):
        x, y, z, sx, sy, sz, az = vals
        z  = min(800.0, max(200.0, float(z)))
        sz = min(100.0, max(10.0,  float(sz)))
        az = 0.0
        return [_smart_round(x), _smart_round(y), _smart_round(z),
                _smart_round(sx), _smart_round(sy), _smart_round(sz), az]
    fixed_boxes = [(nm, _final(cs)) for nm, cs in fixed_boxes]

    # ---- Cụm hai "cửa/ngăn lớn" theo area để suy NKA/NKB ----
    big_idxs = [i for i, (_, c) in enumerate(fixed_boxes) if (c[3] >= 200 and c[4] >= 200)]
    def _kmeans2(idxs):
        if len(idxs) < 2: return [], []
        xs = [fixed_boxes[i][1][0] for i in idxs]
        c1, c2 = min(xs), max(xs)
        for _ in range(5):
            left  = [i for i in idxs if abs(fixed_boxes[i][1][0]-c1) <= abs(fixed_boxes[i][1][0]-c2)]
            right = [i for i in idxs if i not in left]
            if not left or not right: break
            c1 = sum(fixed_boxes[i][1][0] for i in left)/len(left)
            c2 = sum(fixed_boxes[i][1][0] for i in right)/len(right)
        return left, right

    left_idx, right_idx = _kmeans2(big_idxs)
    if not left_idx or not right_idx:
        medx = sorted([c[1][0] for c in fixed_boxes])[len(fixed_boxes)//2] if fixed_boxes else 0.0
        left_idx  = [i for i in big_idxs if fixed_boxes[i][1][0] <= medx] or left_idx
        right_idx = [i for i in big_idxs if fixed_boxes[i][1][0] >  medx] or right_idx

    # danh sách width (px) của hai cụm
    left_w  = [fixed_boxes[i][1][3] for i in left_idx]
    right_w = [fixed_boxes[i][1][3] for i in right_idx]

    def _safe_mean(arr, fb=380.0): return float(sum(arr)/len(arr)) if arr else fb
    def _pretty_int(v: float):      return int(round(round(v, 1) / 5.0) * 5)

    mean_left_px  = _safe_mean(left_w,  380.0)
    mean_right_px = _safe_mean(right_w, 380.0)

    # --- scale động từ OCR + (có thể) Hough ---
    scale = None
    if image_path:
        try:
            scale = estimate_scale_mm_per_px(image_path, cv_features)
        except Exception as e:
            print(f"DEBUG: scale estimation skipped ({e})")
            scale = None

    mm_per_px_x, conf_x = None, 0
    if isinstance(scale, dict):
        mm_per_px_x = scale.get("sx")
        conf_x = int(scale.get("conf_x", 0))
    print(f"DEBUG: scale_x(mm/px)={mm_per_px_x}, conf_x={conf_x}")

    if (mm_per_px_x is not None) and (0.2 < float(mm_per_px_x) < 10.0):
        mm_per_px_x = float(mm_per_px_x)
        NKA = _pretty_int(mean_left_px  * mm_per_px_x)
        NKB = _pretty_int(mean_right_px * mm_per_px_x)
    else:
        NKA = _pretty_int(mean_left_px)
        NKB = _pretty_int(mean_right_px)

    # ---- Sinh model IDs + gán tham số cho box lớn ----
    used_ids = set()
    def _new_model_id():
        while True:
            mid = random.randint(100_000_000, 999_999_999)
            if mid not in used_ids:
                used_ids.add(mid)
                return mid

    idxs_by_area = sorted(
        range(len(fixed_boxes)),
        key=lambda i: fixed_boxes[i][1][3] * fixed_boxes[i][1][4],
        reverse=True
    )
    param_idxs = [i for i in idxs_by_area if i in (left_idx + right_idx)][:4]

    fixed_models = []
    for i in range(len(fixed_boxes)):
        mid = _new_model_id()
        if i in param_idxs:
            sz = int(round(fixed_boxes[i][1][5]))
            cfg = f"N=2, NKA={NKA}, NKB={NKB}, BT={sz}"
            fixed_models.append((f"model_{i}", f"model_{mid}", cfg))
        else:
            fixed_models.append((f"model_{i}", f"model_{mid}", ""))

    # ---- Build output ----
    out_lines = []
    for nm, (x,y,z,sx,sy,sz,az) in fixed_boxes:
        out_lines.append(f"{nm} = Bbox({x}, {y}, {z}, {sx}, {sy}, {sz}, {az})")
    for nm, op, args in fixed_models:
        out_lines.append(f"{nm} = <{op}>({args})" if args.strip() else f"{nm} = <{op}>()")
    return "\n".join(out_lines)

# -----------------------------
# VLM call
# -----------------------------
def run_vlm(image_path: str, device: str = "auto"):
    model_id = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto" if device != "cpu" else None,
    )

    im = Image.open(image_path).convert("RGB")
    W, H = im.size

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text",  "text": USER_PROMPT.format(W=W, H=H)},
        ]},
    ]

    # 1) render chat template → string
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # 2) tokenize + encode ảnh
    inputs = processor(text=[prompt], images=[im], return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    gen_cfg = GenerationConfig(
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
    )

    with torch.no_grad():
        generated = model.generate(**inputs, generation_config=gen_cfg)

    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    m = re.search(r"<PROGRAM>\s*(.*?)\s*</PROGRAM>", text, flags=re.S)
    raw_block = m.group(1).strip() if m else text.strip()
    return raw_block, (W, H)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="input/sample.png")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"[ERR] Image not found: {image_path}")
        sys.exit(1)

    ensure_tesseract_ready(verbose=True)

    print("==================================================")
    raw, (W, H) = run_vlm(image_path)
    print("=== RAW MODEL OUTPUT ===")
    print(raw[:1000] if len(raw) > 1000 else raw)

    program = parse_and_make_absolute(raw, W, H, max_boxes=12, image_path=image_path)
    print("\n=== FINAL PROGRAM ===")
    print(program)

    # Lưu để render 3D
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "program.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(program + "\n")

    print("[OK] Saved program -> output/program.txt")

if __name__ == "__main__":
    main()
