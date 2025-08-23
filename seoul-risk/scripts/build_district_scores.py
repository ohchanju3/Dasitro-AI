# scripts/build_district_scores.py
import os, json, csv, math
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GEOJSON_PATH = os.path.join(DATA_DIR, "seoul_gu.geojson")
INCIDENTS_PATH = os.path.join(DATA_DIR, "incidents.json")
STATIONS_PATH = os.path.join(DATA_DIR, "stations.json")       # optional
CONS_PATH = os.path.join(DATA_DIR, "construction.json")       # optional
MANUAL_CSV = os.path.join(DATA_DIR, "manual_grades.csv")      # optional (없으면 생성)
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
