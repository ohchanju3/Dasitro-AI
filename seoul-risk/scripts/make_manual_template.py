import csv
from pathlib import Path

SRC = Path("data/district_scores.csv")
DST = Path("data/manual_grades.csv")

def main():
    rows = []
    with SRC.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r.get("district") or r.get("gu") or "").strip()
            if not d:
                continue
            rows.append({"district": d, "groundwater": "", "ground": "", "old": ""})

    # 중복 제거(혹시 대비)
    seen = set()
    uniq = []
    for r in rows:
        k = r["district"]
        if k not in seen:
            seen.add(k)
            uniq.append(r)

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["district", "groundwater", "ground", "old"])
        w.writeheader()
        w.writerows(uniq)

    print(f"✨ Wrote {DST} ({len(uniq)} rows). 이제 1~5 등급으로 채워 넣으세요.")

if __name__ == "__main__":
    main()
