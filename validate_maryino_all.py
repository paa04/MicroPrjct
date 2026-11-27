"""Evaluate the walking nearest-club model against all reviewed Maryino gyms."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from compare_walking import WalkingGraph, score_market
from economy_model import DATA, MIN_DISTANCE, N, Features, Model, coordinates, records, zones
from spatial_model import ROOT, geographic, metric, save_csv

BASE = ROOT / 'results/final_unfiltered'
OUTPUT = ROOT / 'results/maryino_all_clubs'


def main():
    catalog = records(BASE / 'competition.csv')
    reviewed = [r for r in records(ROOT / 'data/clubs_reviewed.csv') if r['eligible'] == 'yes']
    for row in reviewed:
        if row['network'] not in ('DDX', 'Spirit', 'Spirit Fitness') and row['club_id'] not in {c['id'] for c in catalog}:
            catalog.append(dict(row, id=row['club_id']))
    clubs = coordinates(catalog)
    network = np.array([c['network'] for c in catalog])
    layer = json.loads((BASE / 'density.geojson').read_text())['features']
    lonlat = np.array([f['geometry']['coordinates'] for f in layer])
    mass = np.array([f['properties']['demand_share'] for f in layer])
    allowed = np.array([f['properties']['allowed'] for f in layer])
    sites = metric(lonlat[allowed])
    model = json.loads((BASE / 'report.json').read_text())
    model = Model(*(np.array(model[k]) for k in ('means', 'scales', 'coefficients')))
    site_log = model.terms(Features().measure(sites))[:, N:].sum(axis=1)
    with ZipFile(ROOT / 'marino_project.zip') as archive:
        routes, _, _ = WalkingGraph(archive).routes(np.vstack([geographic(clubs), lonlat[allowed]]), lonlat)
    club_routes, site_routes = np.split(routes, [len(clubs)])
    results = []
    for j, club in enumerate(catalog):
        if club['scope'] != 'inside':
            continue
        keep = np.arange(len(clubs)) != j
        free = cKDTree(clubs[keep]).query(sites)[0] >= MIN_DISTANCE
        score, _, _, _, _ = score_market(club_routes[keep], site_routes, network[keep] == network[j],
                                          mass, site_log, free)
        selected = zones(sites, score)
        distances = np.linalg.norm(sites[selected] - clubs[j], axis=1)
        predicted = geographic(sites[selected])
        results.append(dict(club=club['name'], network=club['network'],
                            top1_m=float(distances[0]), top3_m=float(distances.min()),
                            hit3_300=int(distances.min() <= 300),
                            top1_lat=predicted[0, 1], top1_lon=predicted[0, 0],
                            nearest_allowed_m=float(cKDTree(sites[free]).query(clubs[j])[0])))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUTPUT / 'historical_validation.csv', results)
    summary = dict(clubs=len(results), competitors=len(catalog), mean_top1_m=float(np.mean([r['top1_m'] for r in results])),
                   median_top1_m=float(np.median([r['top1_m'] for r in results])),
                   hit3_300=sum(r['hit3_300'] for r in results))
    (OUTPUT / 'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
