import os, json, csv, math
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GEOJSON_PATH = os.path.join(DATA_DIR, "seoul_gu.geojson")
INCIDENTS_PATH = os.path.join(DATA_DIR, "incidents.json")
STATIONS_PATH = os.path.join(DATA_DIR, "stations.json")       
CONS_PATH = os.path.join(DATA_DIR, "construction.json")      
MANUAL_CSV = os.path.join(DATA_DIR, "manual_grades.csv")      
OUT_CSV = os.path.join(DATA_DIR, "district_scores.csv")

# ------------------------------
# Geometry utils (dependencies 없이)
# ------------------------------
def point_in_ring(x, y, ring):
    """ray casting for one ring (no holes)"""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        # edges: (x1,y1)->(x2,y2)
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if xin > x:
                inside = not inside
    return inside

def point_in_polygon(x, y, polygon):
    """polygon: [outer, hole1, hole2, ...] each a ring of [x,y]"""
    if not polygon:
        return False
    outer = polygon[0]
    if not point_in_ring(x, y, outer):
        return False
    # holes: if in any hole → outside
    for hole in polygon[1:]:
        if point_in_ring(x, y, hole):
            return False
    return True

def point_in_multipolygon(x, y, multipoly):
    """multipoly: list of polygons"""
    for poly in multipoly:
        if point_in_polygon(x, y, poly):
            return True
    return False

# ------------------------------
# Load GeoJSON (구 경계)
# ------------------------------
def load_district_geoms(path):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    geoms = {}
    names = {}

    for ft in feats:
        props = ft.get("properties", {}) or {}
        tags = props.get("tags", {}) or {}
        name = (
            props.get("name:ko")
            or tags.get("name:ko")
            or props.get("name")
            or tags.get("name")
        )
        if not name:
            continue

        geom = ft.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        # 통일된 내부 표현: MultiPolygon -> [ [ [ring], [hole]... ], ... ]
        mp = []
        if gtype == "Polygon":
            # Polygon = [ [ring], [hole], ... ]
            mp = [coords]
        elif gtype == "MultiPolygon":
            # MultiPolygon = [ [ [ring],[hole]... ], ... ]
            mp = coords
        else:
            continue

        # GeoJSON은 [lon, lat]
        # 내부 표현: polygon: 각 ring은 [(x,y), ...] = [(lon,lat), ...]
        norm_mp = []
        for poly in mp:
            norm_poly = []
            for ring in poly:
                norm_ring = [(float(x), float(y)) for x, y in ring]
                norm_poly.append(norm_ring)
            norm_mp.append(norm_poly)

        geoms[name] = norm_mp
        names[name] = name

    return geoms, sorted(names.keys())

# ------------------------------
# JSON 로더 (위도/경도 키 자동 인식)
# ------------------------------
def load_points(json_path):
    if not os.path.exists(json_path):
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts = []
    for row in data:
        lat = row.get("lat")
        lon = row.get("lng") or row.get("lon")
        if lat is None or lon is None:
            lat = row.get("위도")
            lon = row.get("경도")
        if lat is None or lon is None:
            continue
        try:
            pts.append((float(lon), float(lat)))  # (x=lon, y=lat)
        except:
            continue
    return pts

# ------------------------------
# 집계: 포인트를 구로 매핑
# ------------------------------
def count_by_district(points, geoms):
    counts = defaultdict(int)
    for (x, y) in points:
        for gu, mp in geoms.items():
            if point_in_multipolygon(x, y, mp):
                counts[gu] += 1
                break
    return counts

# ------------------------------
# 등급화 (1~5) - 상위일수록 위험 5
# ------------------------------
def to_grades(counts, all_gus):
    vals = [counts.get(gu, 0) for gu in all_gus]
    if not any(vals):
        return {gu: 1 for gu in all_gus}
    # 절단값: 20/40/60/80 분위
    sorted_vals = sorted(vals)
    def q(p):
        idx = max(0, min(len(sorted_vals)-1, int(math.ceil(p/100.0*len(sorted_vals)))-1))
        return sorted_vals[idx]
    t20, t40, t60, t80 = q(20), q(40), q(60), q(80)

    grades = {}
    for gu in all_gus:
        v = counts.get(gu, 0)
        if v <= t20: g = 1
        elif v <= t40: g = 2
        elif v <= t60: g = 3
        elif v <= t80: g = 4
        else: g = 5
        grades[gu] = g
    return grades

# ------------------------------
# 메인
# ------------------------------
def main():
    geoms, all_gus = load_district_geoms(GEOJSON_PATH)

    incidents = load_points(INCIDENTS_PATH)
    stations  = load_points(STATIONS_PATH) if os.path.exists(STATIONS_PATH) else []
    cons      = load_points(CONS_PATH) if os.path.exists(CONS_PATH) else []

    inc_cnt = count_by_district(incidents, geoms)
    st_cnt  = count_by_district(stations, geoms) if stations else defaultdict(int)
    cons_cnt= count_by_district(cons, geoms)     if cons else defaultdict(int)

    inc_grade = to_grades(inc_cnt, all_gus)      # 싱크홀 사고 이력 (가중↑)
    sub_grade = to_grades(st_cnt, all_gus)       # 지하철 분포 (역 밀도)
    con_grade = to_grades(cons_cnt, all_gus)     # 공사 지점 (선택 지표)

    # 수기 입력 템플릿 없으면 생성
    if not os.path.exists(MANUAL_CSV):
        with open(MANUAL_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["district","groundwater","ground","old"])
            for gu in all_gus:
                w.writerow([gu, "", "", ""])
        print(f"[i] 수기 입력 템플릿 생성: {MANUAL_CSV} (groundwater/ground/old 1~5 등급 입력)")

    # 수기 등급 로드
    manual = {}
    if os.path.exists(MANUAL_CSV):
        with open(MANUAL_CSV, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                gu = r.get("district")
                if not gu: continue
                def iv(s):
                    try:
                        v = int(s)
                        return min(5, max(1, v))
                    except:
                        return None
                manual[gu] = {
                    "groundwater": iv(r.get("groundwater")),
                    "ground":      iv(r.get("ground")),
                    "old":         iv(r.get("old")),
                }

    # CSV 출력
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "district",
            "incident_count","incident_grade",
            "station_count","subway_grade",
            "construction_count","construction_grade",
            "groundwater","ground","old",
            "final_grade_simple"
        ])
        for gu in all_gus:
            gw = manual.get(gu, {}).get("groundwater")
            gr = manual.get(gu, {}).get("ground")
            od = manual.get(gu, {}).get("old")

            # 간단 최종: 5개 지표 중 존재하는 것만 평균 (incident/subway + 수기 3개)
            comps = []
            comps.append(inc_grade.get(gu,1))
            comps.append(sub_grade.get(gu,1))
            if gw: comps.append(gw)
            if gr: comps.append(gr)
            if od: comps.append(od)
            final = round(sum(comps)/len(comps), 2) if comps else ""

            w.writerow([
                gu,
                inc_cnt.get(gu,0), inc_grade.get(gu,1),
                st_cnt.get(gu,0),  sub_grade.get(gu,1),
                cons_cnt.get(gu,0),con_grade.get(gu,1),
                gw or "", gr or "", od or "",
                final
            ])

    print(f"[✓] 저장: {OUT_CSV}")
    print("    - 필요하면 data/manual_grades.csv 를 채운 뒤 다시 실행하세요.")

if __name__ == "__main__":
    main()

# scripts/build_district_scores.py (추가/변경)

DONG_MANUAL_CSV = os.path.join(DATA_DIR, "manual_grades_dong.csv")  # NEW

def safe_int_1_5(s):
    try:
        v = int(str(s).strip())
        return min(5, max(1, v))
    except:
        return None

def load_manual_gu_csv(path):
    """기존 구 단위 수기 CSV 로더 (그대로)"""
    manual = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                gu = (r.get("district") or "").strip()
                if not gu:
                    continue
                manual[gu] = {
                    "groundwater": safe_int_1_5(r.get("groundwater")),
                    "ground":      safe_int_1_5(r.get("ground")),
                    "old":         safe_int_1_5(r.get("old")),
                }
    return manual

def load_manual_dong_csv(path):
    """동 단위 수기 CSV 로드 → 구 단위로 가중 평균 집계"""
    if not os.path.exists(path):
        return {}

    by_gu = {}
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            gu   = (r.get("gu")   or "").strip()
            dong = (r.get("dong") or "").strip()
            if not gu or not dong:
                continue

            gw = safe_int_1_5(r.get("groundwater"))
            gr = safe_int_1_5(r.get("ground"))
            od = safe_int_1_5(r.get("old"))

            # 가중치 (없으면 1)
            try:
                w = float(r.get("weight") or 1.0)
                if w <= 0: w = 1.0
            except:
                w = 1.0

            slot = by_gu.setdefault(gu, {"gw_sum":0.0,"gw_w":0.0,
                                         "gr_sum":0.0,"gr_w":0.0,
                                         "od_sum":0.0,"od_w":0.0})
            if gw is not None:
                slot["gw_sum"] += gw * w
                slot["gw_w"]   += w
            if gr is not None:
                slot["gr_sum"] += gr * w
                slot["gr_w"]   += w
            if od is not None:
                slot["od_sum"] += od * w
                slot["od_w"]   += w

    # 가중 평균 → 정수 1~5로 클리핑
    agg = {}
    for gu, s in by_gu.items():
        def avg(sumv, w):
            if w <= 0: return None
            v = round(sumv / w)   # 반올림
            return min(5, max(1, int(v)))
        agg[gu] = {
            "groundwater": avg(s["gw_sum"], s["gw_w"]),
            "ground":      avg(s["gr_sum"], s["gr_w"]),
            "old":         avg(s["od_sum"], s["od_w"]),
        }
    return agg

def ensure_dong_template(path, all_gus):
    """동 수기 템플릿이 없으면 생성: 비어 있는 샘플(gu만)"""
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dong","gu","groundwater","ground","old","weight"])
        # 동 목록을 모를 때는 빈 줄 + 안내용 머리만 생성
        # 만약 동 목록 CSV가 있으면 여기서 채워 넣을 수도 있음.
    print(f"[i] 동 수기 입력 템플릿 생성: {path} (dong,gu,groundwater,ground,old[,weight])")

def main():
    geoms, all_gus = load_district_geoms(GEOJSON_PATH)

    incidents = load_points(INCIDENTS_PATH)
    stations  = load_points(STATIONS_PATH) if os.path.exists(STATIONS_PATH) else []
    cons      = load_points(CONS_PATH) if os.path.exists(CONS_PATH) else []

    inc_cnt = count_by_district(incidents, geoms)
    st_cnt  = count_by_district(stations, geoms) if stations else defaultdict(int)
    cons_cnt= count_by_district(cons, geoms)     if cons else defaultdict(int)

    inc_grade = to_grades(inc_cnt, all_gus)
    sub_grade = to_grades(st_cnt,  all_gus)
    con_grade = to_grades(cons_cnt,all_gus)

    # (A) 구 단위 수기 템플릿 보장
    if not os.path.exists(MANUAL_CSV):
        with open(MANUAL_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["district","groundwater","ground","old"])
            for gu in all_gus:
                w.writerow([gu, "", "", ""])
        print(f"[i] 수기 입력 템플릿 생성: {MANUAL_CSV} (groundwater/ground/old 1~5 등급 입력)")

    # (B) 동 단위 수기 템플릿도 보장
    ensure_dong_template(DONG_MANUAL_CSV, all_gus)

    # (C) 수기 로드
    manual_gu   = load_manual_gu_csv(MANUAL_CSV)           # 구 단위(기존)
    manual_dong = load_manual_dong_csv(DONG_MANUAL_CSV)    # 동 → 구 집계(신규)

    # (D) 최종 수기값 우선순위: "구 단위 직접 입력" > "동 집계"
    def merged_manual_for_gu(gu):
        g1 = manual_gu.get(gu, {}) if manual_gu else {}
        g2 = manual_dong.get(gu, {}) if manual_dong else {}
        return {
            "groundwater": g1.get("groundwater") or g2.get("groundwater"),
            "ground":      g1.get("ground")      or g2.get("ground"),
            "old":         g1.get("old")         or g2.get("old"),
        }

    # CSV 출력
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "district",
            "incident_count","incident_grade",
            "station_count","subway_grade",
            "construction_count","construction_grade",
            "groundwater","ground","old",
            "final_grade_simple"
        ])
        for gu in all_gus:
            mm = merged_manual_for_gu(gu)
            gw, gr, od = mm.get("groundwater"), mm.get("ground"), mm.get("old")

            comps = [inc_grade.get(gu,1), sub_grade.get(gu,1)]
            if gw: comps.append(gw)
            if gr: comps.append(gr)
            if od: comps.append(od)
            final = round(sum(comps)/len(comps), 2) if comps else ""

            w.writerow([
                gu,
                inc_cnt.get(gu,0), inc_grade.get(gu,1),
                st_cnt.get(gu,0),  sub_grade.get(gu,1),
                cons_cnt.get(gu,0),con_grade.get(gu,1),
                gw or "", gr or "", od or "",
                final
            ])

    print(f"[✓] 저장: {OUT_CSV}")
    print("    - 필요하면 data/manual_grades.csv 또는 data/manual_grades_dong.csv 를 채운 뒤 다시 실행하세요.")
