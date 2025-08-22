# app.py
import io, os, math, base64, time
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Query, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from ultralytics import YOLO

# =========================
# 설정 (환경변수로 조정)
# =========================
MODEL_PATH  = os.getenv("MODEL_PATH", "weights/best.pt")
AGG_DEFAULT = os.getenv("AGG_DEFAULT", "max")


# 1차(빠른) 추론 파라미터
CONF_THRES  = float(os.getenv("CONF_THRES", 0.50))
IOU_THRES   = float(os.getenv("IOU_THRES", 0.50))
IMG_SIZE    = int(os.getenv("IMG_SIZE", "1280"))
MAX_DET     = int(os.getenv("MAX_DET", "300"))

# 2차(민감) 자동 재시도 파라미터
FALLBACK_ENABLE      = int(os.getenv("FALLBACK_ENABLE", "1"))  # 1=활성
CONF_FALLBACK        = float(os.getenv("CONF_FALLBACK", "0.28"))
IOU_FALLBACK         = float(os.getenv("IOU_FALLBACK",  "0.45"))
IMG_SIZE_FALLBACK    = int(os.getenv("IMG_SIZE_FALLBACK", "1536"))
PREPROC_FALLBACK     = os.getenv("PREPROC_FALLBACK", "autocontrast")  # autocontrast/none

# 위험도 계산 파라미터
RISK_SCALE  = float(os.getenv("RISK_SCALE", 2.0))
RISK_BIAS   = float(os.getenv("RISK_BIAS", 0.00))
RISK_EMPTY  = float(os.getenv("RISK_EMPTY", 5.0))   # 검출 0개일 때 고정 %

# 노이즈 억제(너무 작은 검출 제거) — 얇은 크랙이면 낮춰주세요
MIN_AREA_RATIO = float(os.getenv("MIN_AREA_RATIO", "0.000001"))  # 0.0001% = 1e-6

# 오버레이 (base64 PNG 크기 제한)
OVERLAY_MAX_W = int(os.getenv("OVERLAY_MAX_W", "1280"))

# 보안(백↔AI 내부용 키)
AI_API_KEY  = os.getenv("AI_API_KEY", "")

# 4 클래스 (키는 내부명; 가중치/색 등에서 사용)
CLASS_KEYS = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]

# 위험도 가중치/정규화
W = {
    "longitudinal_crack": 0.7,
    "transverse_crack":   0.7,
    "alligator_crack":    1.1,
    "pothole":            1.3,
}
NMAX = {  # count 정규화 상한
    "longitudinal_crack": 15,
    "transverse_crack":   15,
    "alligator_crack":    10,
    "pothole":             5,
}
AMAX = {  # 면적비(이미지 대비) 정규화 상한
    "longitudinal_crack": 0.04,
    "transverse_crack":   0.04,
    "alligator_crack":    0.05,
    "pothole":            0.03,
}

# 시각화 색상 (RGB)
COLORS = {
    "longitudinal_crack": (0, 153, 255),
    "transverse_crack":   (0, 200, 120),
    "alligator_crack":    (255, 0, 0),
    "pothole":            (255, 165, 0),
}

# bbox → 면적 근사 계수 + 박스비율 상한
BBOX2MASK = {
    "longitudinal_crack": 0.12,   # 얇은 선형
    "transverse_crack":   0.12,
    "alligator_crack":    0.25,   # 망상형
    "pothole":            0.65,   # 포트홀은 박스와 근사
}
MAX_BOX_RATIO = {
    "longitudinal_crack": 0.18,
    "transverse_crack":   0.18,
    "alligator_crack":    0.35,
    "pothole":            0.60,
}

def sigmoid(x: float) -> float:
    return 1/(1+math.exp(-x))

# =========================
# 앱/모델 초기화
# =========================
app = FastAPI(title="Sinkhole Risk AI (inference-only)")
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

# 모델 이름 → 내부 키로 정규화
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
    # 허용할 변형
    if k in {"alligator", "alligator_cracks"}: return "alligator_crack"
    if k in {"transverse"}:                    return "transverse_crack"
    if k in {"longitudinal"}:                  return "longitudinal_crack"
    return None

# =========================
# 권한(옵션)
# =========================
def _check_key(x_ai_key: Optional[str] = Header(None)):
    if AI_API_KEY and x_ai_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="invalid X-AI-Key")

# =========================
# 유틸
# =========================
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
        im2 = ImageOps.autocontrast(im, cutoff=1)  # 밝기 히스토그램 양끝 1% 컷
        im2 = im2.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        return im2
    return im

def _draw_and_collect(im: Image.Image, res) -> Tuple[dict, dict, list, Image.Image]:
    """YOLO 결과에서 counts/areas/detections/overlay를 얻음 (이름 기반 매핑)"""
    Wimg, Himg = im.size
    counts = {k:0 for k in W.keys()}
    areas  = {k:0.0 for k in W.keys()}
    detections: List[Dict[str, Any]] = []
    overlay = im.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    # segmentation이 있으면 폴리곤 면적 사용
    if getattr(res, "masks", None) is not None and res.masks is not None:
        polys = res.masks.xy  # list[ndarray(N,2)]
        for i, cls_idx in enumerate(res.boxes.cls.tolist()):
            raw_name = MODEL_NAMES.get(int(cls_idx), str(cls_idx))
            c = _name_to_key(raw_name)
            if not c or c not in W:
                continue
            conf = float(res.boxes.conf[i])
            poly = [(float(x), float(y)) for x, y in polys[i]]
            a = _polygon_area(poly) / (Wimg * Himg)
            if a < MIN_AREA_RATIO:
                continue  # 너무 작은 노이즈 제거
            detections.append({"type": c, "confidence": conf, "polygon": poly})
            areas[c] += max(0.0, a)
            counts[c] += 1
            col = COLORS.get(c, (0,255,0))
            draw.polygon(poly, fill=(col[0], col[1], col[2], 40), outline=(col[0], col[1], col[2], 200))
    else:
        # bbox만 있을 때: 면적 근사
        for box, cls_idx, det_conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            raw_name = MODEL_NAMES.get(int(cls_idx), str(cls_idx))
            c = _name_to_key(raw_name)
            if not c or c not in W:
                continue
            x1, y1, x2, y2 = box
            box_ratio = max(0.0, (x2 - x1) * (y2 - y1) / (Wimg * Himg))
            if box_ratio < MIN_AREA_RATIO:
                continue  # 아주 작은 박스 제거
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
        im,
        verbose=False,
        conf=conf_thres,
        iou=iou_thres,
        imgsz=img_size,
        max_det=MAX_DET
    )[0]
    return _draw_and_collect(im, res)

# ---- 2x2 타일링 백업 ----
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
    """타일 추론 결과를 원본 좌표계로 병합"""
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
                x1 += ox; y1 += oy; x2 += ox; y2 += oy   # 타일 → 원본 좌표
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
# ---- /타일링 백업 ----

def analyze_image(raw: bytes, conf_thres: float, iou_thres: float, img_size: int, return_overlay: bool=False) -> Dict[str, Any]:
    t0 = time.time()
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")

    # 1차: 기본(빠른)
    c1, a1, d1, ov1 = _infer_once(im, conf_thres, iou_thres, img_size)
    use_counts, use_areas, use_dets, use_overlay = c1, a1, d1, ov1
    used_pass = "fast"

    # 2차: 비었으면(검출 0) 더 민감 + 전처리
    if FALLBACK_ENABLE and sum(c1.values()) == 0:
        im2 = _preprocess_for_fallback(im)
        c2, a2, d2, ov2 = _infer_once(im2, CONF_FALLBACK, IOU_FALLBACK, IMG_SIZE_FALLBACK)
        if sum(c2.values()) > 0:
            s1, s2 = sum(c1.values()), sum(c2.values())
            area1, area2 = sum(a1.values()), sum(a2.values())
            if s2 > s1 or (s2 == s1 and area2 > area1):
                use_counts, use_areas, use_dets, use_overlay = c2, a2, d2, ov2
                used_pass = "fallback"

    # 3차: 아직도 비면 2×2 타일링 백업
    if sum(use_counts.values()) == 0:
        c3, a3, d3, ov3 = _infer_tiled(im, grid=2, conf=0.22, iou=0.45, imgsz=1280)
        if sum(c3.values()) > 0:
            use_counts, use_areas, use_dets, use_overlay = c3, a3, d3, ov3
            used_pass = "tiled"

    # 검출 0개면 고정값
    if sum(use_counts.values()) == 0 and all(v == 0.0 for v in use_areas.values()):
        out = {
            "risk_percent": float(RISK_EMPTY),
            "explanations": [],
            "detections": [],
            "used_pass": used_pass,
            "latency_ms": int((time.time()-t0)*1000),
        }
        if return_overlay:
            out["preview_overlay_png"] = _encode_overlay_png(use_overlay)
        return out

    # 위험도 계산
    score = RISK_BIAS
    explanations = []
    for c in W.keys():  # 이름 기반
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

# =========================
# 집계 유틸 & 배치 엔드포인트
# =========================
def _percentile(values: List[float], p: float = 90) -> float:
    if not values: return 0.0
    vals = sorted(values)
    p = max(0.0, min(100.0, float(p)))
    # nearest-rank
    k = max(0, min(len(vals)-1, int(math.ceil(p/100.0 * len(vals))) - 1))
    return float(vals[k])

def _aggregate_risks(risks: List[float], mode: str = "max") -> float:
    mode = (mode or "max").lower()
    if not risks: return 0.0
    if mode == "mean":
        return float(sum(risks) / len(risks))
    if mode in ("p90", "p95"):
        return _percentile(risks, p=90 if mode=="p90" else 95)
    return float(max(risks))  # 기본: 안전 우선

# =========================
# 엔드포인트
# =========================
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model": os.path.basename(MODEL_PATH),
        "model_names": [MODEL_NAMES[i] for i in sorted(MODEL_NAMES.keys())],
        "class_keys": CLASS_KEYS,
        "params": {
            "fast": {"conf": CONF_THRES, "iou": IOU_THRES, "imgsz": IMG_SIZE, "max_det": MAX_DET},
            "fallback": {
                "enabled": bool(FALLBACK_ENABLE),
                "conf": CONF_FALLBACK, "iou": IOU_FALLBACK, "imgsz": IMG_SIZE_FALLBACK,
                "preproc": PREPROC_FALLBACK
            },
            "risk": {"scale": RISK_SCALE, "bias": RISK_BIAS, "empty": RISK_EMPTY, "min_area_ratio": MIN_AREA_RATIO},
            "overlay": {"max_w": OVERLAY_MAX_W},
            "batch": {"agg_default": AGG_DEFAULT},
            
        }
    }

@app.post("/infer", dependencies=[Depends(_check_key)])
async def infer(
    image: UploadFile = File(...),
    text: Optional[str] = Form(None),            # 백에서 함께 넘어와도 무시(로그용)
    conf: Optional[float] = Query(None),
    iou: Optional[float] = Query(None),
    imgsz: Optional[int] = Query(None),
    overlay: int = Query(0),
    lite: int = Query(0)
):
    raw = await image.read()
    out = analyze_image(
        raw,
        conf_thres = float(conf) if conf is not None else CONF_THRES,
        iou_thres  = float(iou)  if iou  is not None else IOU_THRES,
        img_size   = int(imgsz)  if imgsz is not None else IMG_SIZE,
        return_overlay = bool(overlay),
    )
    if lite:
        return JSONResponse({"risk_percent": out.get("risk_percent", 0.0)})
    return JSONResponse(out)

@app.post("/infer_batch", dependencies=[Depends(_check_key)])
async def infer_batch(
    images: List[UploadFile] = File(...),          # 최대 3장
    text: Optional[str] = Form(None),
    conf: Optional[float] = Query(None),
    iou: Optional[float] = Query(None),
    imgsz: Optional[int] = Query(None),
    overlay: int = Query(0),                       # 1이면 각 이미지 오버레이 포함
    lite: int = Query(0),                          # 1이면 종합 위험도만
    agg: str = Query(AGG_DEFAULT)                       # max | mean | p90
):
    if not images:
        raise HTTPException(status_code=400, detail="no images")
    if len(images) > 3:
        raise HTTPException(status_code=400, detail="up to 3 images allowed")

    conf_v = float(conf) if conf is not None else CONF_THRES
    iou_v  = float(iou)  if iou  is not None else IOU_THRES
    img_v  = int(imgsz)  if imgsz is not None else IMG_SIZE

    per_results = []
    risks = []

    for uf in images:
        raw = await uf.read()
        out = analyze_image(
            raw,
            conf_thres = conf_v,
            iou_thres  = iou_v,
            img_size   = img_v,
            return_overlay = bool(overlay),
        )
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

    return JSONResponse({
        "combined_risk_percent": combined,
        "aggregate_mode": agg,
        "items": per_results
    })
