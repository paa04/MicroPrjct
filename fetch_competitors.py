"""Fetch fitness-centre candidates within 3 km of Maryino's actual boundary."""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from spatial_model import ROOT, metric, inside


ENDPOINT = 'https://maps.mail.ru/osm/tools/overpass/api/interpreter'
QUERY = '[out:json][timeout:45];nwr["leisure"="fitness_centre"](55.612,37.645,55.696,37.836);out center tags;'
RAW = ROOT / 'data/raw_osm/fitness_buffer.json'
OUT = ROOT / 'data/maryino/fitness_buffer_candidates.json'


def distance_to_segments(point, ring):
    a=ring[:-1];b=ring[1:]
    v=b-a
    denom=(v*v).sum(axis=1)
    t=np.clip(((point-a)*v).sum(axis=1)/np.maximum(denom,1e-12),0,1)
    return float(np.sqrt(((point-(a+t[:,None]*v))**2).sum(axis=1)).min())


def main():
    request=urllib.request.Request(ENDPOINT,data=urllib.parse.urlencode({'data':QUERY}).encode(),
        headers={'User-Agent':'Maryino-spatial-research/0.1 (single research query)'})
    with urllib.request.urlopen(request,timeout=70) as response:
        data=json.load(response)
    if not isinstance(data.get('elements'),list): raise ValueError('Unexpected Overpass response')
    RAW.parent.mkdir(parents=True,exist_ok=True)
    RAW.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    boundary=json.loads((ROOT/'data/maryino/boundary.geojson').read_text())
    coords=boundary['features'][0]['geometry']['coordinates'][0]
    ring=metric(coords)
    records=[]
    for e in data['elements']:
        loc=e if e['type']=='node' else e.get('center',{})
        if 'lon' not in loc or 'lat' not in loc: continue
        xy=metric([(loc['lon'],loc['lat'])])[0]
        is_inside=inside(xy,ring)
        distance=0. if is_inside else distance_to_segments(xy,ring)
        if distance>3000: continue
        records.append(dict(osm_type=e['type'],osm_id=e['id'],name=e.get('tags',{}).get('name',''),
            brand=e.get('tags',{}).get('brand',''),lon=loc['lon'],lat=loc['lat'],
            scope='inside' if is_inside else 'buffer',distance_to_boundary_m=round(distance),
            tags=e.get('tags',{}),source=f"https://www.openstreetmap.org/{e['type']}/{e['id']}"))
    records.sort(key=lambda r:(r['scope'],r['osm_type'],r['osm_id']))
    OUT.write_text(json.dumps(dict(source=ENDPOINT,osm_timestamp=data.get('osm3s',{}).get('timestamp_osm_base'),
        boundary_relation=455222,buffer_m=3000,features=records),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Found {len(records)} fitness-centre candidates: "
          f"{sum(r['scope']=='inside' for r in records)} inside, "
          f"{sum(r['scope']=='buffer' for r in records)} within 3 km outside")


if __name__=='__main__': main()
