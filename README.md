# CAD2Program – Phân tích bản vẽ tủ và lắp ráp 3D

Dự án chuyển một ảnh bản vẽ kỹ thuật (CAD) của tủ/kệ thành một "chương trình" DSL mô tả các khối (bbox + model), sau đó lắp ráp và render mô hình 3D. Hệ thống kết hợp mô hình Ngôn ngữ-Thị giác (VLM) với Computer Vision + OCR để đảm bảo kết quả thực sự phụ thuộc vào ảnh (image-dependent), tránh việc lặp lại mẫu từ prompt.

## Điểm nổi bật
- **Kết hợp VLM + CV/OCR**: VLM (Qwen2-VL) sinh chương trình thô; CV (OpenCV) tìm cạnh/dòng/contours để tạo bbox theo ảnh; OCR (Tesseract) ước lượng mm/px từ số đo trên bản vẽ.
- **Kết quả phụ thuộc ảnh**: Mỗi ảnh đầu vào cho chương trình/bbox khác nhau dựa trên đặc trưng thật (đã kiểm chứng qua hash ảnh và số lượng component).
- **Lắp ráp 3D tự động**: Từ DSL → phân loại khối (hull, shelf, vách đứng, lưng tủ, cặp cửa) → lắp trong lòng tủ theo gap/mm-per-px → render PNG/OBJ.

## Cấu trúc thư mục
```
project/
├── input/                  # Ảnh đầu vào (PNG/JPG)
├── output/
│   ├── program.txt         # Program cuối sau inference
│   ├── logs/               # Log inference (chứa dòng mm/px)
│   ├── ocr_debug/          # Ảnh debug OCR
│   └── runs/
│       └── <tên_ảnh>/      # Kết quả theo từng ảnh: program + render + obj
├── scripts/
│   ├── inference.py        # Gọi VLM + chuẩn hoá + ghép CV/OCR
│   ├── cv_analysis.py      # Phân tích edges/lines/contours, OCR mm/px
│   └── render3d.py         # Parse program, assemble, render, export OBJ
├── assets/                 # Tài nguyên bổ trợ (nếu có)
├── run.sh                  # Chạy toàn trình (Git Bash/WSL)
├── requirements.txt        # Thư viện Python cần cài
└── README.md               # Tài liệu này
```

## Cài đặt
1) **Python packages**
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
2) **Tesseract OCR (bắt buộc cho OCR)**
- Windows: cài từ https://github.com/tesseract-ocr/tesseract
- Nếu không có trong PATH, code sẽ tự thử các đường dẫn:
  - `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- Có thể đặt biến môi trường `TESSERACT_CMD` trỏ thẳng tới tesseract.exe.

3) **PyTorch** (CPU hoặc CUDA)
- CPU-only: `pip install torch`
- CUDA 12.1 (ví dụ):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Cách chạy
### Cách 1: Từng bước (Windows PowerShell)
1) Inference (sinh `output/program.txt`):
```powershell
python scripts\inference.py --image input\sample.png
```
2) Render 3D + xuất OBJ:
```powershell
python scripts\render3d.py ^
  --program output\program.txt ^
  --mm-per-px 1.8 ^
  --assemble ^
  --out output\renders\render.png ^
  --obj output\renders\model.obj
```

### Cách 2: Toàn trình (Git Bash/WSL)
```bash
bash run.sh input/sample.png
```
Kết quả theo ảnh sẽ nằm trong `output/runs/<tên_ảnh>/`: `program.txt`, `render_assembled.png`, `render_exploded.png` (nếu bật), `model.obj`.

> Lưu ý: `run.sh` sẽ tự bắt mm/px từ log inference (chuỗi `scale_x(mm/px)=...`). Nếu không bắt được, sẽ dùng fallback trong script hoặc bạn có thể ép bằng `-m <mm_per_px>`.

## DSL (định dạng chương trình)
- Mỗi component gồm 2 dòng:
```
bbox_N = Bbox(pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, angle_z)
model_N = <model_#########>(optional_args)
```
- Ý nghĩa:
  - `pos_x, pos_y, scale_x, scale_y`: đơn vị pixel (lấy từ ảnh)
  - `pos_z, scale_z`: đơn vị millimeter (z: 200–800, thickness: 10–100)
  - `angle_z`: thường là 0
  - `model_#########`: ID ngẫu nhiên 9 chữ số; với cửa đôi có thể có tham số `N=2, NKA, NKB, BT`.

Các hàm liên quan:
- Parse chương trình: `parse_program()` trong `scripts/render3d.py`
- Phân loại sơ cấp: `classify()` trong `scripts/render3d.py`
- Lắp ráp 3D: `CabinetAssembler.build()` trong `scripts/render3d.py`

## Bên trong pipeline
- `scripts/inference.py`
  - `run_vlm()`: gọi Qwen2-VL qua HuggingFace Transformers.
  - `parse_and_make_absolute()`: parse VLM output, ưu tiên bbox từ CV (`cv_analysis.py`), ước lượng mm/px bằng OCR, sắp xếp/làm tròn/kẹp, sinh model IDs và tham số cửa đôi cho các bbox lớn.
- `scripts/cv_analysis.py`
  - `analyze_image_features()`: edges (Canny), lines (Hough), contours → `components`.
  - `generate_image_dependent_boxes()`: contours → danh sách bbox theo ảnh (ưu tiên dùng nếu có).
  - `estimate_scale_mm_per_px()`: OCR số + ghép với đoạn thẳng gần nhất để suy ra mm/px; fallback kiểu catalog.
- `scripts/render3d.py`
  - Map pixel→mm trong lòng tủ (tôn trọng `side_gap`, `top_gap`, `bot_gap`, `front_gap`, `back_gap`).
  - Đặt kệ (SHELF), vách đứng (VERT_PANEL), lưng tủ (BACK), và cặp cửa (DOOR_PAIR) với `--door-gap`.
  - Render PNG bằng `pyrender`, export OBJ bằng `trimesh`.

## Tuỳ chọn hữu ích (render3d.py)
- `--assemble`: lắp ráp theo logic nội thất (khuyến nghị bật).
- `--mm-per-px`: ép mm/px nếu muốn bỏ qua giá trị tự động.
- `--front-gap`, `--back-gap`, `--door-gap`: điều chỉnh các khe hở.
- `--hull-alpha`, `--alpha`, `--hide-hull`: kiểm soát hiển thị vỏ tủ và phần còn lại.
- `--explode`: đặt khoảng cách tách khối để xem exploded view.

## Sự cố thường gặp
- **Tesseract không tìm thấy**: Cài Tesseract và/hoặc set `TESSERACT_CMD`. Kiểm tra `ensure_tesseract_ready(verbose=True)` trong `scripts/cv_analysis.py` để xem debug.
- **Thiếu OpenGL khi render**: `pyrender` cần context OpenGL. Trên server không có GUI, dùng `OffscreenRenderer` (đã dùng). Nếu vẫn lỗi, đảm bảo `PyOpenGL`, `pyglet` đúng phiên bản và driver đồ hoạ phù hợp.
- **Thiếu mm/px**: Kiểm tra log inference để thấy `scale_x(mm/px)=...`. Có thể ép qua `run.sh -m <giá trị>` hoặc flag `--mm-per-px` khi chạy `render3d.py`.
- **Torch/VLM nặng**: Nếu GPU không đủ, dùng bản Torch CPU và model nhỏ (mặc định `Qwen/Qwen2-VL-2B-Instruct`).

