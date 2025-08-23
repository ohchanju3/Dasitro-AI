import csv, sys, json, argparse
import requests

def main():
    p = argparse.ArgumentParser(description="Bulk upload district baseline grades to AI server")
    p.add_argument("--csv", required=True, help="CSV file path (utf-8)")
    p.add_argument("--url", required=True, help="AI server base URL, e.g. http://52.78.104.121:8001")
    p.add_argument("--key", required=True, help="X-AI-Key")
    args = p.parse_args()

    endpoint = args.url.rstrip("/") + "/districts/baseline/upsert"
    headers = {"X-AI-Key": args.key, "Content-Type": "application/json"}

    ok, fail = 0, 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            payload = {"district": row.get("district")}
            for k in ["groundwater","ground","subway","incident","old"]:
                v = row.get(k, "")
                if v.strip():
                    payload[k] = float(v)
            if not payload.get("district"):
                print(f"[{i}] SKIP (no district)")
                fail += 1
                continue
            try:
                r = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=20)
                if r.ok:
                    print(f"[{i}] OK {payload['district']}")
                    ok += 1
                else:
                    print(f"[{i}] FAIL {payload['district']} -> {r.status_code} {r.text}")
                    fail += 1
            except Exception as e:
                print(f"[{i}] ERROR {payload.get('district')} -> {e}")
                fail += 1

    print(f"\nDone. OK={ok}, FAIL={fail}")

if __name__ == "__main__":
    main()
