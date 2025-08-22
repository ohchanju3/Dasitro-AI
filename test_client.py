# test_client.py
import requests
import json

BASE = "http://localhost:8001"

def test_health():
    r = requests.get(f"{BASE}/healthz", timeout=10)
    print("healthz:", r.status_code, r.json())

def test_infer(img_path):
    with open(img_path, "rb") as f:
        files = {"image": (img_path, f, "image/jpeg")}
        r = requests.post(f"{BASE}/infer", files=files, timeout=60)
    print("infer:", r.status_code)
    js = r.json()
    print("risk_percent:", js.get("risk_percent"))
    print("explanations:", js.get("explanations")[:3])
    # overlay_b64는 길어서 생략

def test_ingest(img_path, lat=37.5636, lng=126.9978, district="중구"):
    with open(img_path, "rb") as f:
        files = {"image": (img_path, f, "image/jpeg")}
        data = {"lat": str(lat), "lng": str(lng), "district": district}
        r = requests.post(f"{BASE}/ingest", files=files, data=data, timeout=60)
    print("ingest:", r.status_code, r.json().get("risk_percent"))

def test_district(district="중구", days=30):
    r = requests.get(f"{BASE}/districts/{district}/risk", params={"period_days": days}, timeout=30)
    print("district risk:", r.status_code, r.json())
    r2 = requests.get(f"{BASE}/districts/{district}/risk_mixed", params={"period_days": days}, timeout=30)
    print("district risk(mixed):", r2.status_code, r2.json())

if __name__ == "__main__":
    test_health()
    # 이미지 경로 바꿔서 테스트
    sample = "sample.jpg"
    test_infer(sample)
    test_ingest(sample, district="중구")
    test_district("중구", 30)
