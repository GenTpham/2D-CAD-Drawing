#!/usr/bin/env bash
set -Eeuo pipefail

# -------------------------------
# Defaults (có thể đổi bằng flags)
# -------------------------------
OUT_ROOT="output/runs"
MM_FALLBACK="1.80"         # dùng khi không bắt được mm/px từ log
FRONT_GAP="6"
BACK_GAP="3"
DOOR_GAP="3"
HULL_ALPHA="0.25"
ALPHA="1.0"
EXPLODE="120"              # ảnh exploded phụ (đặt 0 nếu không muốn)

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options] <image1> [image2 ...]
Options:
  -o <outdir>     Thư mục gốc output (mặc định: ${OUT_ROOT})
  -m <mm_per_px>  Ép mm/px (bỏ qua giá trị auto từ OCR)
  -f <mm>         FRONT_GAP (default ${FRONT_GAP})
  -b <mm>         BACK_GAP  (default ${BACK_GAP})
  -d <mm>         DOOR_GAP  (default ${DOOR_GAP})
  -e <mm>         EXPLODE distance cho ảnh exploded (default ${EXPLODE})
  -h              Help
Ví dụ:
  bash run.sh -m 1.869 input/sample.png
  bash run.sh input/a.png input/b.png
EOF
}

# Parse flags
USER_MM=""
while getopts ":o:m:f:b:d:e:h" opt; do
  case "$opt" in
    o) OUT_ROOT="$OPTARG" ;;
    m) USER_MM="$OPTARG" ;;
    f) FRONT_GAP="$OPTARG" ;;
    b) BACK_GAP="$OPTARG" ;;
    d) DOOR_GAP="$OPTARG" ;;
    e) EXPLODE="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done
shift $((OPTIND-1))

if [ "$#" -lt 1 ]; then
  usage; exit 1
fi

# Activate venv if present (Windows Git Bash or Unix)
if [ -f ".venv/Scripts/activate" ]; then
  # Windows venv (Git Bash)
  # shellcheck disable=SC1091
  source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
  # Unix venv
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$OUT_ROOT" "output/logs"

# --- helper: trích mm/px từ log inference (chuỗi 'scale_x(mm/px)=...') ---
extract_mm_from_log () {
  local log_file="$1"
  local mm
  # bắt số cuối cùng xuất hiện
  mm="$(grep -Eo 'scale_x\(mm/px\)=[0-9. ]+' "$log_file" | tail -n1 | sed -E 's/.*=([0-9. ]+)/\1/' | tr -d ' ')"
  echo "${mm:-}"
}

# --- loop qua các ảnh ---
for img in "$@"; do
  if [ ! -f "$img" ]; then
    echo "[ERR] Không thấy ảnh: $img" >&2
    continue
  fi

  tag="$(basename "$img")"
  tag="${tag%.*}"

  run_dir="${OUT_ROOT}/${tag}"
  mkdir -p "$run_dir"

  echo "=================="
  echo "[1/3] Inference: $img"
  log_path="output/logs/${tag}_inference.log"
  # chạy và log
  python scripts/inference.py --image "$img" | tee "$log_path" >/dev/tty

  # copy program.txt để “đóng băng” theo từng ảnh
  if [ ! -f "output/program.txt" ]; then
    echo "[ERR] Chưa thấy output/program.txt sau inference." >&2
    exit 3
  fi
  cp "output/program.txt" "${run_dir}/program.txt"

  # Lấy mm/px
  mm_px="$USER_MM"
  if [ -z "$mm_px" ]; then
    mm_px="$(extract_mm_from_log "$log_path")"
  fi
  if [ -z "$mm_px" ]; then
    mm_px="$MM_FALLBACK"
    echo "[WARN] Không bắt được mm/px từ log. Dùng fallback: ${mm_px}"
  else
    echo "[INFO] mm/px = ${mm_px}"
  fi

  echo "[2/3] Render assembled -> ${run_dir}/render_assembled.png"
  python scripts/render3d.py \
    --program "${run_dir}/program.txt" \
    --mm-per-px "${mm_px}" \
    --assemble \
    --front-gap "${FRONT_GAP}" \
    --back-gap "${BACK_GAP}" \
    --door-gap "${DOOR_GAP}" \
    --hull-alpha "${HULL_ALPHA}" \
    --alpha "${ALPHA}" \
    --out "${run_dir}/render_assembled.png" \
    --obj "${run_dir}/model.obj"

  if [ "${EXPLODE}" != "0" ]; then
    echo "[3/3] Render exploded -> ${run_dir}/render_exploded.png"
    python scripts/render3d.py \
      --program "${run_dir}/program.txt" \
      --mm-per-px "${mm_px}" \
      --assemble \
      --explode "${EXPLODE}" \
      --filter-outside 0 \
      --hull-alpha "${HULL_ALPHA}" \
      --alpha "${ALPHA}" \
      --out "${run_dir}/render_exploded.png"
  fi

  echo "[OK] Done: ${tag}"
  echo "     - Program: ${run_dir}/program.txt"
  echo "     - Assembled PNG: ${run_dir}/render_assembled.png"
  if [ "${EXPLODE}" != "0" ]; then
    echo "     - Exploded PNG: ${run_dir}/render_exploded.png"
  fi
  echo "     - OBJ: ${run_dir}/model.obj"
done

echo "== All done =="
