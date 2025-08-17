# scripts/render3d.py
# -*- coding: utf-8 -*-
import argparse, os, re, math, random
from typing import List, Dict, Optional
import numpy as np

# 3D
import trimesh
from trimesh.creation import box as make_box
import pyrender
from PIL import Image

# --------------------------
# Regex & utils
# --------------------------
FLOAT = r"-?\d+(?:\.\d+)?"
RE_BBOX  = re.compile(
    rf"^(bbox_\d+)\s*=\s*Bbox\(\s*({FLOAT})\s*,\s*({FLOAT})\s*,\s*({FLOAT})\s*,\s*({FLOAT})\s*,\s*({FLOAT})\s*,\s*({FLOAT})\s*,\s*({FLOAT})\s*\)\s*$",
    re.I)
RE_MODEL = re.compile(r"^(model_\d+)\s*=\s*<\s*(model_(\d+))\s*>\s*\((.*?)\)\s*$", re.I)
RE_ARGKV = re.compile(r"([A-Za-z_]+)\s*=\s*([0-9]+(?:\.[0-9]+)?)")

def clamp(v, lo, hi): return max(lo, min(hi, v))

def nice_color(seed):
    rnd = random.Random(int(seed) if isinstance(seed, (int, np.integer)) else hash(seed))
    return np.array([rnd.random()*0.7+0.3, rnd.random()*0.7+0.3, rnd.random()*0.7+0.3], dtype=float)

# --------------------------
# Program parser
# --------------------------
class Node:
    def __init__(self, name, x,y,z, sx,sy,sz, az):
        self.name = name
        self.x,self.y,self.z = float(x),float(y),float(z)
        self.sx,self.sy,self.sz = float(sx),float(sy),float(sz)
        self.az = float(az)
        self.model_name: Optional[str] = None
        self.model_id:   Optional[int] = None
        self.params: Dict[str,float] = {}

def parse_program(text:str):
    nodes: Dict[str,Node] = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: continue
        m = RE_BBOX.match(ln)
        if m:
            nm,x,y,z,sx,sy,sz,az = m.groups()
            nodes[nm] = Node(nm,x,y,z,sx,sy,sz,az)
            continue
        m = RE_MODEL.match(ln)
        if m:
            slot, model_tag, mid, args = m.groups()
            idx = int(slot.split("_")[1])
            bbox_key = f"bbox_{idx}"
            if bbox_key not in nodes:
                nodes[bbox_key] = Node(bbox_key,0,0,400,100,100,18,0)
            n = nodes[bbox_key]
            n.model_name = model_tag
            n.model_id   = int(mid)
            params = {}
            for k,v in RE_ARGKV.findall(args or ""):
                params[k.upper()] = float(v)
            n.params = params
    # ordered by index
    return [nodes[k] for k in sorted(nodes.keys(), key=lambda s:int(s.split("_")[1]))]

# --------------------------
# Primitive classifier (paper-like)
# --------------------------
PRIM_HULL       = "HULL"
PRIM_SHELF      = "SHELF"
PRIM_VERT_PANEL = "VERT_PANEL"
PRIM_BACK       = "BACK"
PRIM_DOOR_PAIR  = "DOOR_PAIR"

def classify(nodes: List[Node]) -> Dict[str, str]:
    kinds: Dict[str,str] = {}
    if not nodes: return kinds
    areas = [(n.name, n.sx*n.sy) for n in nodes]
    hull_name = max(areas, key=lambda t:t[1])[0]
    kinds[hull_name] = PRIM_HULL
    hull_w = [n for n in nodes if n.name==hull_name][0].sx
    hull_h = [n for n in nodes if n.name==hull_name][0].sy

    for n in nodes:
        if n.name == hull_name: 
            continue
        p = {k.upper():v for k,v in n.params.items()}
        if ("NKA" in p) or ("NKB" in p) or (p.get("N",0)>=2):
            kinds[n.name] = PRIM_DOOR_PAIR
            continue
        ar = n.sx / max(1.0, n.sy)
        # thin plates as shelf
        if ar > 3.0 and n.sy < 0.35*hull_h:
            kinds[n.name] = PRIM_SHELF
        elif ar < 0.33 and n.sx < 0.35*hull_w:
            kinds[n.name] = PRIM_VERT_PANEL
        else:
            if n.sy > 0.6*hull_h and n.sx > 0.6*hull_w:
                kinds[n.name] = PRIM_BACK
            else:
                kinds[n.name] = PRIM_SHELF
    return kinds

# --------------------------
# Geometry helpers
# --------------------------
def mesh_box(w, h, d, color, name, alpha=1.0):
    # color: RGB (3) hoặc RGBA (4), alpha ưu tiên tham số 'alpha' nếu color chỉ có 3 phần tử
    col = np.asarray(color, dtype=float).reshape(-1)
    if col.size == 4:
        rgba = np.array([col[0], col[1], col[2], col[3]], dtype=float)
    else:
        rgba = np.array([col[0], col[1], col[2], float(alpha)], dtype=float)

    geom = make_box(extents=[w, h, d])
    mesh = trimesh.Trimesh(vertices=geom.vertices, faces=geom.faces, process=False)
    rgba255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba255, (mesh.vertices.shape[0], 1))
    mesh.metadata["name"] = name
    mesh.metadata["rgba"] = rgba
    return mesh


def auto_camera_pose(bounds):
    """
    Trả về pose 4x4 đưa camera nhìn tâm cảnh; không phụ thuộc look_at của trimesh.
    """
    c = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    radius = float(np.linalg.norm(extent)) * 0.6 + 1e-6
    dist = 2.2 * radius
    eye = c + np.array([dist, dist*0.45, dist], dtype=float)
    up  = np.array([0.0, 1.0, 0.0], dtype=float)
    f = (c - eye); f /= (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up); s /= (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    pose = np.eye(4)
    pose[0,:3] = s; pose[1,:3] = u; pose[2,:3] = -f
    pose[:3, 3] = eye
    return pose

def filter_inside_hull(nodes: List[Node], hull: Optional[Node], margin_ratio: float) -> List[Node]:
    if hull is None or margin_ratio <= 0.0: 
        return nodes[:]  # không lọc
    x0, y0 = hull.x, hull.y
    x1, y1 = hull.x + hull.sx, hull.y + hull.sy
    dx, dy = hull.sx*margin_ratio, hull.sy*margin_ratio
    x0 -= dx; y0 -= dy; x1 += dx; y1 += dy
    kept = []
    for n in nodes:
        if n is hull:
            kept.append(n); continue
        p = {k.upper():v for k,v in n.params.items()}
        if ("NKA" in p) or ("NKB" in p) or (p.get("N",0)>=2):
            kept.append(n); continue
        cx, cy = n.x, n.y
        if (x0 <= cx <= x1) and (y0 <= cy <= y1):
            kept.append(n)
    return kept

# --------------------------
# Assembler
# --------------------------
class CabinetAssembler:
    def __init__(self, nodes: List[Node], mm_per_px: float, assemble: bool=True,
                 side_gap: float=2.0, top_gap: float=2.0, bot_gap: float=2.0,
                 front_gap: float=1.5, back_gap: float=1.0, epsilon: float=0.2,
                 hull_alpha: float=1.0, rest_alpha: float=1.0, hide_hull: bool=False,
                 filter_outside: float=0.0, door_gap: float=2.0):
        self.nodes = nodes
        self.kinds = classify(nodes)
        self.mm_per_px = mm_per_px
        self.assemble = assemble
        self.side_gap = side_gap
        self.top_gap  = top_gap
        self.bot_gap  = bot_gap
        self.front_gap= front_gap
        self.back_gap = back_gap
        self.eps = epsilon
        self.hull_alpha = hull_alpha
        self.rest_alpha = rest_alpha
        self.hide_hull  = hide_hull
        self.door_gap = float(door_gap)

        self.hull = self._get_hull()
        # tuỳ chọn lọc node ngoài hull (mặc định không lọc)
        self.nodes = filter_inside_hull(self.nodes, self.hull, filter_outside)
        self.kinds = classify(self.nodes)

        self.BT   = float(self.hull.sz) if self.hull else 18.0
        if self.hull:
            self.W = self.hull.sx * mm_per_px
            self.H = self.hull.sy * mm_per_px
            self.D = max(200.0, float(self.hull.z))
        else:
            self.W=self.H=900.0; self.D=350.0

        self.innerW = max(10.0, self.W - 2*self.BT)
        self.innerH = max(10.0, self.H - 2*self.BT)
        self.innerD = max(10.0, self.D - 1*self.BT)

    def _get_hull(self)->Optional[Node]:
        if not self.nodes: return None
        areas = [(n, n.sx*n.sy) for n in self.nodes]
        return max(areas, key=lambda t:t[1])[0]

    def build(self)->trimesh.Scene:
        scene = trimesh.Scene()
        # 1) HULL
        if self.hull and not self.hide_hull:
            hull = mesh_box(self.W, self.H, self.D, np.array([0.9,0.4,0.4]), "HULL", alpha=self.hull_alpha)
            scene.add_geometry(hull, node_name="HULL")

        hx, hy = self.hull.x, self.hull.y
        hsx, hsy = self.hull.sx, self.hull.sy
        def map_xy_to_mm(px, py):
            nx = clamp((px - hx) / max(1e-6, hsx), 0.0, 1.0)
            ny = clamp((py - hy) / max(1e-6, hsy), 0.0, 1.0)
            x_mm = -self.W/2 + self.BT + self.side_gap + nx*(self.innerW - 2*self.side_gap)
            y_mm =  self.H/2 - self.BT - self.top_gap - ny*(self.innerH - self.top_gap - self.bot_gap)
            return x_mm, y_mm

        # 2) Others
        z_eps_front = 0.1   # đẩy nhẹ để không dính vào hull
        z_eps_back  = 0.1

        stats = {"HULL":1 if (self.hull and not self.hide_hull) else 0,
                 "BACK":0,"SHELF":0,"VERT_PANEL":0,"DOOR_PAIR":0,"PLATE":0}

        for n in self.nodes:
            if n is self.hull: 
                continue
            kind = self.kinds.get(n.name, PRIM_SHELF)
            color = nice_color(n.model_id or hash(n.name))

            if kind == PRIM_BACK:
                w = self.innerW; h = self.innerH; d = max(6.0, self.BT*0.6)
                m = mesh_box(w,h,d, color, f"{n.name}_BACK", alpha=self.rest_alpha)
                m.apply_translation([0, 0, -self.D/2 + d/2 + self.back_gap + z_eps_back])
                scene.add_geometry(m); stats["BACK"] += 1
                continue

            if kind == PRIM_SHELF:
                th = max(6.0, float(n.sz))
                w = self.innerW - 2*self.side_gap
                h = th
                depth = self.innerD - self.front_gap - self.back_gap
                # chỉ lấy Y từ ảnh; X luôn ở tâm lòng tủ
                _, y_mm = map_xy_to_mm(n.x, n.y)

                m = mesh_box(w, h, depth, color, f"{n.name}_SHELF", alpha=self.rest_alpha)
                m.apply_translation([0, -h/2 + self.eps, 0])   # top-align
                m.apply_translation([0, y_mm, 0])              # << X=0
                scene.add_geometry(m); stats["SHELF"] += 1
                continue

            if kind == PRIM_VERT_PANEL:
                t = max(8.0, float(n.sz))
                w = t; h = self.innerH
                depth = self.innerD - self.front_gap - self.back_gap
                x_mm, _ = map_xy_to_mm(n.x, n.y)
                x_min = -self.innerW/2 + self.side_gap + t/2
                x_max =  self.innerW/2 - self.side_gap - t/2
                x_mm = clamp(x_mm, x_min, x_max)              # << kẹp X trong lòng tủ

                m = mesh_box(w, h, depth, color, f"{n.name}_VERT", alpha=self.rest_alpha)
                m.apply_translation([x_mm, 0, 0])
                scene.add_geometry(m); stats["VERT_PANEL"] += 1
                continue

            if kind == PRIM_DOOR_PAIR:
                p = n.params
                door_th = max(12.0, float(p.get("BT", n.sz)))
                NKA = float(p.get("NKA", n.sx*self.mm_per_px/2.0))
                NKB = float(p.get("NKB", n.sx*self.mm_per_px/2.0))
                door_h = self.innerH - self.top_gap - self.bot_gap

                # phân bổ bề rộng theo lòng tủ và chừa khe
                left_w  = clamp(NKA, 50.0, self.W)
                right_w = clamp(NKB, 50.0, self.W)
                total   = max(1e-6, left_w + right_w)
                avail   = max(10.0, self.innerW - 2*self.side_gap - self.door_gap)
                scale   = avail / total
                left_w  *= scale
                right_w *= scale

                gap = max(0.0, self.door_gap)
                left_w  = max(30.0, left_w  - gap/2.0)
                right_w = max(30.0, right_w - gap/2.0)

                # ĐẶT Ở NGOÀI HULL (flush/nhô nhẹ ra ngoài)
                z_front = self.D/2 + door_th/2 - self.front_gap + self.eps

                # tâm mỗi cánh (đối xứng qua giữa)
                xl = -avail/2 + left_w/2
                xr =  avail/2 - right_w/2

                # 2 màu khác nhau cho dễ nhận biết
                colL = np.array([0.85, 0.45, 0.45], dtype=float)   # đỏ nhạt
                colR = np.array([0.45, 0.55, 0.88], dtype=float)   # xanh tím

                mL = mesh_box(left_w,  door_h, door_th, colL, f"{n.name}_DOOR_L", alpha=self.rest_alpha)
                mR = mesh_box(right_w, door_h, door_th, colR, f"{n.name}_DOOR_R", alpha=self.rest_alpha)
                mL.apply_translation([xl, 0, z_front])
                mR.apply_translation([xr, 0, z_front])
                scene.add_geometry(mL); scene.add_geometry(mR)

                # Thanh mỏng làm đường phân chia (khe ở giữa)
                if gap > 0.0:
                    split_depth = min(door_th*0.6, 2.0)  # mỏng để thấy line
                    mid = mesh_box(gap, door_h, split_depth, np.array([0.08,0.08,0.08,1.0]),
                                f"{n.name}_SPLIT")
                    mid.apply_translation([0.0, 0.0, z_front + 0.5*split_depth + 0.05])
                    scene.add_geometry(mid)
                continue

            # fallback plate
            th = max(6.0, float(n.sz))
            w = self.innerW - 2*self.side_gap
            depth = self.innerD - self.front_gap - self.back_gap
            x_mm, y_mm = map_xy_to_mm(n.x, n.y)
            m = mesh_box(w, th, depth, color, f"{n.name}_PLATE", alpha=self.rest_alpha)
            m.apply_translation([x_mm, y_mm, z_eps_front - z_eps_back])
            scene.add_geometry(m); stats["PLATE"] += 1

        print("[Assembler] parts:", stats)
        return scene

# --------------------------
# Renderer & exporters
# --------------------------
def render_scene(scene: trimesh.Scene, out_png: str):
    pr_scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.25,0.25,0.25,1.0])
    # materials with alpha
    for tm in scene.geometry.values():
        rgba = tm.metadata.get("rgba", np.array([0.8,0.8,0.8,1.0], dtype=float))
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])),
            metallicFactor=0.05, roughnessFactor=0.8, alphaMode='BLEND'
        )
        pr_scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False))

    pr_scene.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4))
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    pr_scene.add(cam, pose=auto_camera_pose(scene.bounds))

    r = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=960)
    color, _ = r.render(pr_scene)
    Image.fromarray(color).save(out_png)

def export_obj(scene:trimesh.Scene, path_obj:str):
    meshes = [geom.copy() for _, geom in scene.geometry.items()]
    if not meshes: return
    union = trimesh.util.concatenate(meshes)
    union.export(path_obj)

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True, help="path to program.txt")
    ap.add_argument("--out", default="output/render3d.png")
    ap.add_argument("--mm-per-px", type=float, default=1.8)
    ap.add_argument("--assemble", action="store_true", help="assemble like paper")
    ap.add_argument("--front-gap", type=float, default=1.5)
    ap.add_argument("--back-gap",  type=float, default=1.0)
    ap.add_argument("--explode", type=float, default=0.0, help="explode distance (mm)")
    ap.add_argument("--obj", type=str, default=None, help="export OBJ path")
    ap.add_argument("--alpha", type=float, default=1.0, help="alpha for non-hull parts (0..1)")
    ap.add_argument("--hull-alpha", type=float, default=1.0, help="alpha for hull (0..1)")
    ap.add_argument("--hide-hull", action="store_true", help="do not render the hull")
    ap.add_argument("--filter-outside", type=float, default=0.0, help="0=không lọc; >0 lọc node ngoài hull theo tỉ lệ bbox")
    ap.add_argument("--door-gap", type=float, default=2.0,
                    help="khoảng hở giữa 2 cánh (mm) để nhìn rõ đường phân chia")
    args = ap.parse_args()

    with open(args.program, "r", encoding="utf-8") as f:
        text = f.read()
    nodes = parse_program(text)

    assembler = CabinetAssembler(
        nodes, mm_per_px=args.mm_per_px, assemble=args.assemble,
        front_gap=args.front_gap, back_gap=args.back_gap,
        hull_alpha=args.hull_alpha, rest_alpha=args.alpha,
        hide_hull=args.hide_hull, filter_outside=args.filter_outside,
        door_gap=args.door_gap
    )
    scene = assembler.build()

    if args.explode > 0:
        step = float(args.explode)
        for i, name in enumerate(list(scene.geometry.keys())):
            T = np.eye(4); T[0,3] = (i%3 - 1)*step; T[2,3] = (i//3 - 1)*step*0.6
            scene.geometry[name].apply_transform(T)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    render_scene(scene, args.out)
    print(f"[OK] PNG saved: {args.out}")

    if args.obj:
        os.makedirs(os.path.dirname(args.obj), exist_ok=True)
        export_obj(scene, args.obj)
        print(f"[OK] OBJ saved: {args.obj}")

if __name__ == "__main__":
    main()
