"""Test STAC endpoints for sentinel-2-l2a data availability"""
import cubo
import json
import urllib.request

# Check what collections exist on Element84  
url = 'https://earth-search.aws.element84.com/v1/collections'
resp = urllib.request.urlopen(url, timeout=10)
data = json.loads(resp.read())
for c in data['collections']:
    cid = c['id']
    if 'sentinel' in cid.lower():
        desc = c.get('description', '')[:80]
        print(f'  {cid}: {desc}')

print()

# Try Element84 with sentinel-2-l2a-1 (yearly) or sentinel-2-l1c
for col in ['sentinel-2-l2a', 'sentinel-2-l1c', 'sentinel-2-l2a-1']:
    try:
        da = cubo.create(
            lat=-5.788973, lon=-45.092558,
            collection=col,
            bands=['B04'],
            start_date='2025-01-01', end_date='2025-03-01',
            edge_size=128, resolution=10,
            stac='https://earth-search.aws.element84.com/v1',
            max_items=2
        )
        print(f'{col}: shape={da.shape}, time={len(da.time)}')
        if len(da.time) > 0:
            print(f'  Times: {da.time.values}')
    except Exception as e:
        print(f'{col}: ERROR - {str(e)[:100]}')