"""Test STAC endpoints for connectivity"""
import urllib.request, json, sys

endpoints = [
    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a",
    "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a",
]

for url in endpoints:
    name = url.split("/")[2]
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        print(f"[OK] {name}: {data.get('id', data.get('description','?'))}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")