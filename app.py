import io, os, math, base64, time, datetime
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Query, Header, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from ultralytics import YOLO
import csv

# ===== 파일 경로 =====
DISTRICT_CSV = os.getenv("DISTRICT_CSV", "data/district_scores.csv")
DONG_CSV     = os.getenv("DONG_CSV", "data/dong_scores.csv")

# ==== Inference 설정 ====
MODEL_PATH  = os.getenv("MODEL_PATH", "weights/best.pt")
AGG_DEFAULT = os.getenv("AGG_DEFAULT", "max")  # max | mean | p90

# 1차(빠른) 추론
CONF_THRES  = float(os.getenv("CONF_THRES", 0.50))
IOU_THRES   = float(os.getenv("IOU_THRES", 0.50))
IMG_SIZE    = int(os.getenv("IMG_SIZE", "1280"))
MAX_DET     = int(os.getenv("MAX_DET", "300"))

# 2차(민감) 자동 재시도
FALLBACK_ENABLE      = int(os.getenv("FALLBACK_ENABLE", "1"))
CONF_FALLBACK        = float(os.getenv("CONF_FALLBACK", "0.28"))
IOU_FALLBACK         = float(os.getenv("IOU_FALLBACK",  "0.45"))
IMG_SIZE_FALLBACK    = int(os.getenv("IMG_SIZE_FALLBACK", "1536"))
PREPROC_FALLBACK     = os.getenv("PREPROC_FALLBACK", "autocontrast")  

# 위험도 계산 파라미터
RISK_SCALE  = float(os.getenv("RISK_SCALE", 2.0))
RISK_BIAS   = float(os.getenv("RISK_BIAS", 0.00))
RISK_EMPTY  = float(os.getenv("RISK_EMPTY", 5.0))   

# 노이즈 억제(너무 작은 검출 제거)
MIN_AREA_RATIO = float(os.getenv("MIN_AREA_RATIO", "0.000001"))

# 오버레이 (base64 PNG 크기 제한)
OVERLAY_MAX_W = int(os.getenv("OVERLAY_MAX_W", "1280"))

# 보안(백↔AI 내부용 키)
AI_API_KEY  = os.getenv("AI_API_KEY", "")

# ==== 모델/클래스 정의 ====
CLASS_KEYS = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
W = { "longitudinal_crack": 0.7, "transverse_crack": 0.7, "alligator_crack": 1.1, "pothole": 1.3 }
NMAX = {"longitudinal_crack":15, "transverse_crack":15, "alligator_crack":10, "pothole":5}
AMAX = {"longitudinal_crack":0.04, "transverse_crack":0.04, "alligator_crack":0.05, "pothole":0.03}
COLORS = {
    "longitudinal_crack": (0,153,255),
    "transverse_crack":   (0,200,120),
    "alligator_crack":    (255,0,0),
    "pothole":            (255,165,0),
}
BBOX2MASK = {"longitudinal_crack":0.12, "transverse_crack":0.12, "alligator_crack":0.25, "pothole":0.65}
MAX_BOX_RATIO = {"longitudinal_crack":0.18, "transverse_crack":0.18, "alligator_crack":0.35, "pothole":0.60}

def sigmoid(x: float) -> float:
    return 1/(1+math.exp(-x))

# ==== 앱/모델 초기화 ====
app = FastAPI(title="Sinkhole Risk AI (inference + baseline combine)")

# YOLO 스킵 모드(테스트용)
if os.getenv("SKIP_YOLO", "0") == "1":
    class _DummyModel:
        names = {0: "pothole"}
        def predict(self, *args, **kwargs):
            class R:
                boxes = type("B", (), {"cls": [], "conf": [], "xyxy": []})()
                masks = None
            return [R()]
    model = _DummyModel()
else:
    model = YOLO(MODEL_PATH)

# 모델 라벨 읽기
MODEL_NAMES: Dict[int, str] = {}
try:
    if isinstance(model.names, dict):
        MODEL_NAMES = {int(k): v for k, v in model.names.items()}
    elif isinstance(model.names, list):
        MODEL_NAMES = {i: n for i, n in enumerate(model.names)}
except Exception:
    MODEL_NAMES = {}

_NAME_CANON = {
    "pothole": "pothole",
    "alligator_crack": "alligator_crack",
    "transverse_crack": "transverse_crack",
    "longitudinal_crack": "longitudinal_crack",
}
def _name_to_key(raw: str) -> Optional[str]:
    k = raw.strip().lower().replace("-", " ").replace("/", " ")
    k = "_".join(k.split())
    if k in _NAME_CANON:
        return _NAME_CANON[k]
    if k in {"alligator","alligator_cracks"}: return "alligator_crack"
    if k in {"transverse"}: return "transverse_crack"
    if k in {"longitudinal"}: return "longitudinal_crack"
    return None

# ==== 권한 ====
def _check_key(x_ai_key: Optional[str] = Header(None)):
    if AI_API_KEY and x_ai_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="invalid X-AI-Key")

# ==== 유틸 ====
def _polygon_area(poly: List[List[float]]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        area += x1*y2 - x2*y1
    return abs(area) / 2.0

def _encode_overlay_png(im: Image.Image) -> str:
    w, h = im.size
    if w > OVERLAY_MAX_W:
        nh = int(h * (OVERLAY_MAX_W / w))
        im = im.resize((OVERLAY_MAX_W, nh))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _preprocess_for_fallback(im: Image.Image) -> Image.Image:
    if PREPROC_FALLBACK == "autocontrast":
        im2 = ImageOps.autocontrast(im, cutoff=1)
        im2 = im2.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        return im2
    return im

def _draw_and_collect(im: Image.Image, res) -> Tuple[dict, dict, list, Image.Image]:
    Wimg, Himg = im.size
    counts = {k:0 for k in W.keys()}
    areas  = {k:0.0 for k in W.keys()}
    detections: List[Dict[str, Any]] = []
    overlay = im.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    if getattr(res, "masks", None) is not None and res.masks is not None:
        polys = res.masks.xy
        for i, cls_idx in enumerate(res.boxes.cls.tolist()):
            raw_name = MODEL_NAMES.get(int(cls_idx), str(cls_idx))
            c = _name_to_key(raw_name)
            if not c or c not in W:
                continue
            conf = float(res.boxes.conf[i])
            poly = [(float(x), float(y)) for x, y in polys[i]]
            a = _polygon_area(poly) / (Wimg * Himg)
            if a < MIN_AREA_RATIO:
                continue
            detections.append({"type": c, "confidence": conf, "polygon": poly})
            areas[c] += max(0.0, a)
            counts[c] += 1
            col = COLORS.get(c, (0,255,0))
            draw.polygon(poly, fill=(col[0], col[1], col[2], 40), outline=(col[0], col[1], col[2], 200))
    else:
        for box, cls_idx, det_conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            raw_name = MODEL_NAMES.get(int(cls_idx), str(cls_idx))
            c = _name_to_key(raw_name)
            if not c or c not in W:
                continue
            x1, y1, x2, y2 = box
            box_ratio = max(0.0, (x2 - x1) * (y2 - y1) / (Wimg * Himg))
            if box_ratio < MIN_AREA_RATIO:
                continue
            box_ratio = min(box_ratio, MAX_BOX_RATIO.get(c, 1.0))
            used_area = box_ratio * BBOX2MASK.get(c, 0.6)
            areas[c] += used_area
            counts[c] += 1
            detections.append({"type": c, "confidence": float(det_conf), "bbox": [float(x1), float(y1), float(x2), float(y2)]})
            col = COLORS.get(c, (0,255,0))
            draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
            draw.rectangle([x1, y1, x2, y2], fill=(col[0], col[1], col[2], 40))

    return counts, areas, detections, overlay

def _infer_once(im: Image.Image, conf_thres: float, iou_thres: float, img_size: int):
    res = model.predict(
        im, verbose=False, conf=conf_thres, iou=iou_thres, imgsz=img_size, max_det=MAX_DET
    )[0]
    return _draw_and_collect(im, res)

def _crop_tiles(im: Image.Image, grid: int = 2) -> List[Tuple[Image.Image, Tuple[int,int]]]:
    W, H = im.size
    tiles = []
    w = W // grid
    h = H // grid
    for gy in range(grid):
        for gx in range(grid):
            left, upper = gx*w, gy*h
            right, lower = (gx+1)*w if gx<grid-1 else W, (gy+1)*h if gy<grid-1 else H
            tiles.append((im.crop((left, upper, right, lower)), (left, upper)))
    return tiles

def _infer_tiled(im: Image.Image, grid: int = 2,
                 conf: float = 0.22, iou: float = 0.45, imgsz: int = 1280):
    Wimg, Himg = im.size
    counts = {k:0 for k in W.keys()}
    areas  = {k:0.0 for k in W.keys()}
    detections = []
    overlay = im.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    tiles = _crop_tiles(im, grid)
    for tile, (ox, oy) in tiles:
        res = model.predict(tile, verbose=False, conf=conf, iou=iou, imgsz=imgsz, max_det=MAX_DET)[0]

        if getattr(res, "masks", None) is not None and res.masks is not None:
            polys = res.masks.xy
            for i, cls_idx in enumerate(res.boxes.cls.tolist()):
                raw = MODEL_NAMES.get(int(cls_idx), str(cls_idx))
                c = _name_to_key(raw)
                if not c or c not in W: continue
                poly = [(float(x)+ox, float(y)+oy) for x,y in polys[i]]
                a = _polygon_area(poly) / (Wimg*Himg)
                if a < MIN_AREA_RATIO: continue
                counts[c] += 1; areas[c] += a
                detections.append({"type": c, "confidence": float(res.boxes.conf[i]), "polygon": poly})
                col = COLORS.get(c, (0,255,0))
                draw.polygon(poly, fill=(col[0], col[1], col[2], 40), outline=(col[0], col[1], col[2], 200))
        else:
            for box, cls_idx, det_conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                raw = MODEL_NAMES.get(int(cls_idx), str(cls_idx))
                c = _name_to_key(raw)
                if not c or c not in W: continue
                x1, y1, x2, y2 = box
                x1 += ox; y1 += oy; x2 += ox; y2 += oy
                box_ratio = max(0.0, (x2-x1)*(y2-y1)/(Wimg*Himg))
                if box_ratio < MIN_AREA_RATIO: continue
                box_ratio = min(box_ratio, MAX_BOX_RATIO.get(c,1.0))
                used_area = box_ratio * BBOX2MASK.get(c, 0.6)
                counts[c] += 1; areas[c] += used_area
                detections.append({"type": c, "confidence": float(det_conf), "bbox": [float(x1), float(y1), float(x2), float(y2)]})
                col = COLORS.get(c, (0,255,0))
                draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
                draw.rectangle([x1, y1, x2, y2], fill=(col[0], col[1], col[2], 40))

    return counts, areas, detections, overlay

def analyze_image(raw: bytes, conf_thres: float, iou_thres: float, img_size: int, return_overlay: bool=False) -> Dict[str, Any]:
    t0 = time.time()
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")

    c1, a1, d1, ov1 = _infer_once(im, conf_thres, iou_thres, img_size)
    use_counts, use_areas, use_dets, use_overlay = c1, a1, d1, ov1
    used_pass = "fast"

    if FALLBACK_ENABLE and sum(c1.values()) == 0:
        im2 = _preprocess_for_fallback(im)
        c2, a2, d2, ov2 = _infer_once(im2, CONF_FALLBACK, IOU_FALLBACK, IMG_SIZE_FALLBACK)
        if sum(c2.values()) > 0:
            s1, s2 = sum(c1.values()), sum(c2.values())
            area1, area2 = sum(a1.values()), sum(a2.values())
            if s2 > s1 or (s2 == s1 and area2 > area1):
                use_counts, use_areas, use_dets, use_overlay = c2, a2, d2, ov2
                used_pass = "fallback"

    if sum(use_counts.values()) == 0:
        c3, a3, d3, ov3 = _infer_tiled(im, grid=2, conf=0.22, iou=0.45, imgsz=1280)
        if sum(c3.values()) > 0:
            use_counts, use_areas, use_dets, use_overlay = c3, a3, d3, ov3
            used_pass = "tiled"

    if sum(use_counts.values()) == 0 and all(v == 0.0 for v in use_areas.values()):
        out = {
            "risk_percent": float(RISK_EMPTY),
            "explanations": [], "detections": [],
            "used_pass": used_pass, "latency_ms": int((time.time()-t0)*1000),
        }
        if return_overlay:
            out["preview_overlay_png"] = _encode_overlay_png(use_overlay)
        return out

    score = RISK_BIAS
    explanations = []
    for c in W.keys():
        cn = min(use_counts[c] / max(NMAX[c], 1e-6), 1.0)
        an = min(use_areas[c]  / max(AMAX[c], 1e-6), 1.0)
        contrib = W[c] * (cn + an) / 2.0
        score += contrib
        if use_counts[c] > 0 or use_areas[c] > 0:
            explanations.append({"type": c, "count": use_counts[c], "area_ratio": use_areas[c]})

    risk_percent = max(0.0, min(100.0, 100 * sigmoid(RISK_SCALE * score)))

    out = {
        "risk_percent": risk_percent,
        "explanations": explanations,
        "detections": use_dets,
        "used_pass": used_pass,
        "latency_ms": int((time.time()-t0)*1000),
    }
    if return_overlay:
        out["preview_overlay_png"] = _encode_overlay_png(use_overlay)
    return out

# ==== 집계 유틸 ====
def _percentile(values: List[float], p: float = 90) -> float:
    if not values: return 0.0
    vals = sorted(values)
    p = max(0.0, min(100.0, float(p)))
    k = max(0, min(len(vals)-1, int(math.ceil(p/100.0 * len(vals))) - 1))
    return float(vals[k])

def _aggregate_risks(risks: List[float], mode: str = "max") -> float:
    mode = (mode or "max").lower()
    if not risks: return 0.0
    if mode == "mean": return float(sum(risks) / len(risks))
    if mode in ("p90","p95"): return _percentile(risks, p=90 if mode=="p90" else 95)
    return float(max(risks))

def _load_csv_rows(path: str) -> List[dict]:
    # ★ utf-8-sig 로 BOM 안전하게 제거
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    # ★ gu/dong 를 로드 시점에 표준화
    for r in rows:
        if "gu" in r and r["gu"] is not None:
            r["gu"] = _n(r["gu"])
        if "dong" in r and r["dong"] is not None:
            r["dong"] = _n(r["dong"])
    return rows


def _load_dong_rows() -> List[dict]:
    return _load_csv_rows(DONG_CSV)

def _to_int_safe(v) -> Optional[int]:
    if v in (None, "", "NaN"): return None
    try:
        return int(round(float(v)))
    except:
        return None
    
# ===============================
# 이름 정규화/매칭 유틸
# ===============================
import re, unicodedata
from difflib import get_close_matches

def _n(s: str) -> str:
    if not s: return ""
    s = str(s).replace("\ufeff","")          # BOM 제거
    s = unicodedata.normalize("NFC", s)      # 유니코드 정규화
    s = re.sub(r"\s+", " ", s).strip()       # 공백 정리
    return s

def _strip_paren(s: str) -> str:
    return re.sub(r"\(.*?\)", "", s).strip()

def _norm_gu(s: str) -> str:
    return _n(_strip_paren(s))

_DOT = re.compile(r"[·\.]")

def _dong_aliases(d: str) -> set:
    """상계3·4동 → {상계3동, 상계4동} 같은 별칭 생성"""
    base = _n(_strip_paren(d))
    out = {base}
    if "동" in base and _DOT.search(base):
        head = base.split("동")[0]
        m = re.match(r"^([가-힣A-Za-z]+)([0-9·\.]+)$", head)
        if m:
            pref, nums = m.groups()
            for tok in _DOT.split(nums):
                tok = tok.strip()
                if tok: out.add(f"{pref}{tok}동")
    if "제" in base:
        out.add(re.sub(r"제(\d+)동$", r"\1동", base))  # 제1동 → 1동
    return out

def _eq_gu(a: str, b: str) -> bool:
    return _norm_gu(a) == _norm_gu(b)

def _eq_dong(a: str, b: str) -> bool:
    A, B = _n(_strip_paren(a)), _n(_strip_paren(b))
    if A == B: return True
    return bool(_dong_aliases(A) & _dong_aliases(B))


# ===============================
#   ▽▽▽  정성 점수 저장 (SQLite)
# ===============================
from sqlalchemy import create_engine, Column, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

DB_URL = os.getenv("DB_URL", "sqlite:///data.db")
Base = declarative_base()
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class BaselineScore(Base):
    __tablename__ = "baseline_scores"
    district    = Column(String(64), primary_key=True)
    groundwater = Column(Float, default=None)
    subway      = Column(Float, default=None)
    incident    = Column(Float, default=None)
    old         = Column(Float, default=None)
    updated_at  = Column(DateTime)  # 생성/업데이트 시에 직접 세팅

Base.metadata.create_all(engine)

_GRADE2PCT = {1:10.0, 2:30.0, 3:50.0, 4:70.0, 5:90.0}
def _grade_to_percent(g: Optional[float]) -> Optional[float]:
    if g is None: return None
    try:
        gi = int(round(float(g)))
        return _GRADE2PCT.get(gi)
    except Exception:
        return None

def _final_grade(p: float) -> int:
    return 1 if p < 20 else 2 if p < 40 else 3 if p < 60 else 4 if p < 80 else 5

# ===== Helper: 동/구 베이스라인 로드 =====
def _get_dong_baseline(district: str, dong: str) -> dict:
    for r in _load_dong_rows():
        if _eq_gu(r.get("gu",""), district) and _eq_dong(r.get("dong",""), dong):
            return dict(
                incident=_to_int_safe(r.get("incident_grade")),
                subway=_to_int_safe(r.get("subway_grade")),
                groundwater=_to_int_safe(r.get("groundwater")),
                old=_to_int_safe(r.get("old")),
            )
    raise HTTPException(status_code=404, detail="dong not found in dong_scores.csv")


def _get_district_baseline_from_db(district: str) -> Optional[dict]:
    with SessionLocal() as s:
        row = s.query(BaselineScore).filter_by(district=district).first()
        if not row:
            return None
        return dict(
            groundwater=row.groundwater,
            subway=row.subway,
            incident=row.incident,
            old=row.old
        )

def _get_district_baseline_from_csv(district: str) -> Optional[dict]:
    """
    district_scores.csv에서 등급(1~5) 가져와 baseline 구성.
    """
    for r in _load_csv_rows(DISTRICT_CSV):
        if r.get("district") == district:
            return dict(
                incident=_to_int_safe(r.get("incident_grade")),
                subway=_to_int_safe(r.get("subway_grade")),
                groundwater=_to_int_safe(r.get("groundwater")),
                old=_to_int_safe(r.get("old")),
            )
    return None

# ===============================
#            엔드포인트
# ===============================
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model": os.path.basename(MODEL_PATH),
        "model_names": [MODEL_NAMES[i] for i in sorted(MODEL_NAMES.keys())],
        "class_keys": CLASS_KEYS,
        "params": {
            "fast": {"conf": CONF_THRES, "iou": IOU_THRES, "imgsz": IMG_SIZE, "max_det": MAX_DET},
            "fallback": {"enabled": bool(FALLBACK_ENABLE), "conf": CONF_FALLBACK, "iou": IOU_FALLBACK, "imgsz": IMG_SIZE_FALLBACK, "preproc": PREPROC_FALLBACK},
            "risk": {"scale": RISK_SCALE, "bias": RISK_BIAS, "empty": RISK_EMPTY, "min_area_ratio": MIN_AREA_RATIO},
            "overlay": {"max_w": OVERLAY_MAX_W},
            "batch": {"agg_default": AGG_DEFAULT},
        }
    }

@app.post("/infer_batch", dependencies=[Depends(_check_key)])
async def infer_batch(
    images: List[UploadFile] = File(...),          # 최대 3장
    text: Optional[str] = Form(None),
    conf: Optional[float] = Query(None),
    iou: Optional[float] = Query(None),
    imgsz: Optional[int] = Query(None),
    overlay: int = Query(0),
    lite: int = Query(0),
    agg: str = Query(AGG_DEFAULT)
):
    if not images:
        raise HTTPException(status_code=400, detail="no images")
    if len(images) > 3:
        raise HTTPException(status_code=400, detail="up to 3 images allowed")

    conf_v = float(conf) if conf is not None else CONF_THRES
    iou_v  = float(iou)  if iou  is not None else IOU_THRES
    img_v  = int(imgsz)  if imgsz is not None else IMG_SIZE

    per_results, risks = [], []
    for uf in images:
        raw = await uf.read()
        out = analyze_image(raw, conf_v, iou_v, img_v, return_overlay=bool(overlay))
        per_results.append({
            "filename": uf.filename,
            "risk_percent": out.get("risk_percent", 0.0),
            "used_pass": out.get("used_pass"),
            "latency_ms": out.get("latency_ms"),
            "detections": out.get("detections", []) if not lite else [],
            "preview_overlay_png": out.get("preview_overlay_png") if overlay else None
        })
        risks.append(float(out.get("risk_percent", 0.0)))

    combined = _aggregate_risks(risks, mode=agg)
    if lite:
        return JSONResponse({"risk_percent": combined})
    return JSONResponse({"combined_risk_percent": combined, "aggregate_mode": agg, "items": per_results})

# ====== 정성 점수 관리 ======
@app.post("/districts/baseline/upsert", dependencies=[Depends(_check_key)])
def upsert_baseline(item: dict = Body(...)):
    """
    JSON 예:
    {
      "district":"중구",
      "groundwater":3, "subway":4, "incident":3, "old":2
    }
    일부 필드만 보내도 됨(부분 수정).
    """
    name = item.get("district")
    if not name:
        raise HTTPException(status_code=400, detail="district required")

    with SessionLocal() as s:
        row = s.query(BaselineScore).filter_by(district=name).first()
        if not row:
            row = BaselineScore(district=name)
            s.add(row)
        for k in ["groundwater","subway","incident","old"]:
            if k in item and item[k] is not None:
                setattr(row, k, float(item[k]))
        row.updated_at = datetime.datetime.utcnow()
        s.commit()
        out = {
            "district": row.district,
            "groundwater": row.groundwater,
            "subway": row.subway,
            "incident": row.incident,
            "old": row.old,
            "updated_at": row.updated_at.isoformat()+"Z"
        }
    return {"status":"ok","baseline":out}

@app.get("/districts/{name}/baseline", dependencies=[Depends(_check_key)])
def get_baseline(name: str):
    with SessionLocal() as s:
        row = s.query(BaselineScore).filter_by(district=name).first()
        if not row:
            raise HTTPException(status_code=404, detail="baseline not found")
        return {
            "district": row.district,
            "groundwater": row.groundwater,
            "subway": row.subway,
            "incident": row.incident,
            "old": row.old,
            "updated_at": row.updated_at.isoformat()+"Z"
        }

# ====== 이미지 + 정성 점수 결합 ======
@app.post("/combine_risk", dependencies=[Depends(_check_key)])
async def combine_risk(
    district: Optional[str] = Form(None),
    dong: Optional[str] = Form(None),                      # ★ 동 단위 지원
    images: Optional[List[UploadFile]] = File(None),       # 0~3장
    agg: str = Query(AGG_DEFAULT),

    # 가중치(예시 기본값)
    w_image: float = Form(40),
    w_groundwater: float = Form(15),
    w_subway: float = Form(15),
    w_incident: float = Form(15),
    w_old: float = Form(15),

    lite: int = Query(0)
):
    # 1) baseline 로드
    baseline = None
    if district and dong:
        # dong_scores.csv에서 5지표(incident/subway/groundwater/old) 등급 읽기
        baseline = _get_dong_baseline(district, dong)
    elif district:
        # DB → 없으면 district_scores.csv 폴백
        baseline = _get_district_baseline_from_db(district) or _get_district_baseline_from_csv(district)

    # 2) 이미지 위험도
    img_percent = None
    items = []
    if images:
        if len(images) > 3:
            raise HTTPException(status_code=400, detail="up to 3 images allowed")
        risks = []
        for uf in images:
            raw = await uf.read()
            out = analyze_image(raw, CONF_THRES, IOU_THRES, IMG_SIZE, return_overlay=False)
            rp = float(out.get("risk_percent", 0.0))
            risks.append(rp)
            items.append({"filename": uf.filename, "risk_percent": rp, "used_pass": out.get("used_pass")})
        img_percent = _aggregate_risks(risks, mode=agg) if risks else None

    # 3) 정성 → 퍼센트
    def g2p(v): return _grade_to_percent(v) if v is not None else None
    gw = g2p((baseline or {}).get("groundwater")) if baseline else None
    sw = g2p((baseline or {}).get("subway")) if baseline else None
    ic = g2p((baseline or {}).get("incident")) if baseline else None
    od = g2p((baseline or {}).get("old")) if baseline else None

    # 4) 가중 평균(존재하는 항목만)
    parts, weights, labels = [], [], []
    if img_percent is not None:
        parts.append(img_percent); weights.append(w_image); labels.append("image")
    for v,w,name in [(gw,w_groundwater,"groundwater"),
                     (sw,w_subway,"subway"),
                     (ic,w_incident,"incident"),
                     (od,w_old,"old")]:
        if v is not None and w and w>0:
            parts.append(v); weights.append(w); labels.append(name)

    if not parts:
        raise HTTPException(status_code=400, detail="no image or baseline for this area")

    weights_n = [w/sum(weights) for w in weights]
    combined = float(sum(p*w for p,w in zip(parts,weights_n)))
    grade = _final_grade(combined)

    resp = {
        "district": district,
        "dong": dong,
        "combined_risk_percent": round(combined,2),
        "combined_grade_1_to_5": grade,
        "aggregate_mode": agg,
        "weights_used": {
            "image": w_image, "groundwater": w_groundwater,
            "subway": w_subway, "incident": w_incident, "old": w_old
        }
    }
    if not lite:
        resp["image_part"] = {"img_percent": img_percent, "items": items} if img_percent is not None else None
        resp["baseline_part"] = {
            "groundwater_percent": gw, "subway_percent": sw, "incident_percent": ic, "old_percent": od
        } if baseline else None

    return resp

# ====== CSV 조회 ======
@app.get("/district_scores")
def list_scores():
    return _load_csv_rows(DISTRICT_CSV)

@app.get("/district_scores/{gu}")
def get_score(gu: str):
    for row in _load_csv_rows(DISTRICT_CSV):
        if row.get("district") == gu:
            return row
    raise HTTPException(status_code=404, detail="district not found")

@app.get("/dong_scores")
def list_dong_scores():
    return _load_dong_rows()

@app.get("/dong_scores/{gu}")
def list_dong_scores_by_gu(gu: str):
    rows = _load_dong_rows()
    # ★ _eq_gu 로 비교
    out = [r for r in rows if _eq_gu(r.get("gu",""), gu)]
    if not out:
        raise HTTPException(status_code=404, detail="no dongs for this district")
    return out


@app.get("/dong_scores/{gu}/{dong}")
def get_dong_score(gu: str, dong: str):
    rows = _load_dong_rows()
    # 먼저 동일 구만 필터
    cand = [r for r in rows if _eq_gu(r.get("gu",""), gu)]

    # 동 정규화 매칭
    for r in cand:
        if _eq_dong(r.get("dong",""), dong):
            return r

    # 근사치 제안(디버깅 도움)
    names = [ _n(_strip_paren(r.get("dong",""))) for r in cand ]
    suggestions = get_close_matches(_n(_strip_paren(dong)), names, n=5, cutoff=0.6)
    raise HTTPException(status_code=404, detail={"msg": f"dong not found in '{gu}'", "suggestions": suggestions})

