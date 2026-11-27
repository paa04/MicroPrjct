"""Refresh the compact Moscow OSM layers and official chain directories."""
import argparse
import ast
import csv
from datetime import date
import json
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile

DATA = Path(__file__).resolve().parent / 'data/economy'
DDX = 'https://cdn-ps.ddxfitness.ru/clubs/'
SPIRIT = 'https://spiritfit.ru/clubs/'
BBOX = (55.53, 37.32, 55.94, 37.91)
FILTERS = {
    'metro': 'node["railway"="subway_entrance"]',
    'shopping': 'nwr["shop"="mall"]',
    'retail': 'way["building"="retail"]',
    'stores': 'nwr["shop"="supermarket"]',
    'housing': 'way["building"="apartments"]',
    'office': 'way["building"="office"]',
    'bus': 'node["highway"="bus_stop"]',
}


def download(url, *args):
    return subprocess.check_output(['curl', '-fsSL', '--retry', '2', '--max-time', '120', url, *args])


def write_table(path, fields, rows):
    with NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    Path(f.name).replace(path)


def ddx_rows(html, as_of):
    for block in re.findall(r'self\.__next_f\.push\((.*?)\)</script>', html, re.S):
        payload = json.loads(block)[-1]
        if isinstance(payload, str) and '"clubs":[' in payload:
            clubs = json.JSONDecoder().raw_decode(payload.split('"clubs":', 1)[1])[0]
            for club in clubs:
                if club['city']['name'] != 'Москва' or club['is_hidden']:
                    continue
                opened = (club['open_date'] or '').removeprefix('$D')[:10]
                if not opened or date.fromisoformat(opened) > as_of:
                    continue
                yield dict(network='DDX', id=club['id'], name=club['name'], opened=opened,
                           lon=club['longitude'], lat=club['latitude'],
                           address=f"{club['street']}, {club['building_name']}", url=DDX+club['symbolic_code']+'/')
            return
    raise ValueError('DDX directory schema changed')


def spirit_rows(html):
    literal = re.search(r'window\.clubs\s*=\s*(\[.*?\]);', html, re.S).group(1)
    for club in ast.literal_eval(re.sub(r'(?<=:)(true|false)(?=[,}])', lambda m: m[0].title(), literal)):
        if club['city_code'] != 'msk' or club['club_not_open'] != 'N' or club['club_soon_open'] != 'N':
            continue
        lat, lon = club['coords']
        yield dict(network='Spirit', id=club['id'], name='Spirit. '+club['name'], opened='',
                   lon=lon, lat=lat, address=' '.join(club['address'].split()), url='https://spiritfit.ru'+club['page'])


def refresh_catalog(ddx_html, spirit_html, as_of):
    rows = list(ddx_rows(ddx_html, as_of)) + list(spirit_rows(spirit_html))
    if len({r['network'] for r in rows}) != 2:
        raise ValueError('Missing chain in official data')
    write_table(DATA/'catalog.csv', rows[0].keys(), rows)
    (DATA/'catalog_date.txt').write_text(as_of.isoformat()+'\n')


def venue_query():
    bbox = ','.join(map(str, BBOX))
    return (f'[out:json][timeout:100][bbox:{bbox}];(way["shop"="mall"];way["building"="retail"];'
            'way["amenity"="cinema"];way["amenity"="marketplace"];);out geom;')


def save_venues(obj):
    from spatial_model import area_centroid, geographic

    if obj.get('remark') or not obj['elements']:
        raise ValueError('Incomplete venue geometry')
    rows = []
    for elem in obj['elements']:
        shape = [(p['lon'], p['lat']) for p in elem.get('geometry', [])]
        if len(shape) < 4 or shape[0] != shape[-1]:
            continue
        center, area = area_centroid(shape)
        if area < 50:
            continue
        lon, lat = geographic(center[None, :])[0]
        rows.append(dict(osm_id=elem['id'], lon=round(lon, 7), lat=round(lat, 7),
                         footprint_m2=round(area), name=elem['tags'].get('name', '')))
    write_table(DATA/'venues.csv', rows[0].keys(), rows)


def refresh_layers():
    bbox = '('+','.join(map(str, BBOX))+')'
    for name, expression in FILTERS.items():
        query = f'[out:json][timeout:100];{expression}{bbox};out center tags;'
        obj = json.loads(download('https://maps.mail.ru/osm/tools/overpass/api/interpreter',
                                  '-G', '--data-urlencode', 'data='+query))
        if obj.get('remark') or not obj['elements']:
            raise ValueError(f'Incomplete response for {name}')
        rows = []
        for elem in obj['elements']:
            point = elem.get('center', elem)
            levels = elem['tags'].get('building:levels', '') if name in ('housing', 'office') else ''
            rows.append(dict(osm_id=f"{elem['type']}/{elem['id']}", lon=point['lon'], lat=point['lat'], value=levels))
        write_table(DATA/(name+'.csv'), rows[0].keys(), rows)
        print(name, len(rows), obj['osm3s']['timestamp_osm_base'])
    save_venues(json.loads(download('https://maps.mail.ru/osm/tools/overpass/api/interpreter',
                                    '-G', '--data-urlencode', 'data='+venue_query())))


def district_grids():
    from spatial_model import FEATURES, build_features, grid, load_osm

    ring, nodes, ways = load_osm()
    for step in (100, 200):
        cells = grid(ring, step)
        baseline, allowed, _ = build_features(cells, nodes, ways)
        rows = [dict(x_m=x, y_m=y, allowed=int(free), **dict(zip(FEATURES, values)))
                for (x, y), free, values in zip(cells, allowed, baseline)]
        write_table(DATA/f'district_{step}.csv', rows[0].keys(), rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--catalog-only', action='store_true')
    group.add_argument('--district-only', action='store_true')
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True, parents=True)
    if args.district_only:
        district_grids()
    else:
        refresh_catalog(download(DDX).decode(), download(SPIRIT).decode(), date.today())
        if not args.catalog_only:
            refresh_layers()
