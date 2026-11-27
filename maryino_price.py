"""Compare walking and price-sensitive club choice for Maryino residents."""
import json
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from compare_walking import WalkingGraph, score_market
from economy_model import (DATA, MIN_DISTANCE, N, Features, Model, coordinates,
                           label_sites, records, zones)
from spatial_model import ROOT, geographic, metric, save_csv

BASE = ROOT / 'results/final_unfiltered'
OUTPUT = ROOT / 'results/maryino_price'
DDX_PRICE = 1900
METRES_PER_MINUTE = 80
DECAY_MINUTES = 10
OUTSIDE_MINUTES = 20


def choice(club_minutes, site_minutes, prices, candidate_price, own, restricted,
           candidate_restricted, mass, sensitivity):
    weights = np.exp(-(club_minutes + sensitivity * (prices[:, None] - DDX_PRICE) / 1000) / DECAY_MINUTES)
    entrant = np.exp(-(site_minutes + sensitivity * (candidate_price - DDX_PRICE) / 1000) / DECAY_MINUTES)
    outside = np.exp(-OUTSIDE_MINUTES / DECAY_MINUTES)
    flows = np.zeros((3, len(site_minutes)))
    # Scenario: half the population can choose women-only gyms.
    for eligible in (True, False):
        if candidate_restricted and not eligible:
            continue
        active = np.ones(len(weights), bool) if eligible else ~restricted
        total = outside + weights[active].sum(axis=0)
        share = entrant / (entrant + total)
        before = np.stack([weights[active & own].sum(axis=0),
                           weights[active & ~own].sum(axis=0), np.full_like(total, outside)]) / total
        flows += ((before * mass) @ share.T) / 2
    return flows  # transfers from own network, rivals, and outside option


def run():
    OUTPUT.mkdir(exist_ok=True)
    catalog = records(BASE / 'competition.csv')
    local_prices = {r['club_id']: r for r in records(DATA / 'local_prices.csv')}
    reviewed = [r for r in records(ROOT / 'data/clubs_reviewed.csv') if r['eligible'] == 'yes']
    for r in reviewed:
        if r['network'] not in ('DDX', 'Spirit', 'Spirit Fitness') and r['club_id'] not in {c['id'] for c in catalog}:
            catalog.append(dict(r, id=r['club_id'], url=r['verification_source']))
    price_rows = []
    for c in catalog:
        p = local_prices.get(c['id'], {})
        value = {'DDX': DDX_PRICE, 'Spirit': 2600}.get(c['network'], p.get('regular_month_rub') or None)
        price_rows.append(dict(name=c['name'], scope=c['scope'], regular_rub=value,
                               source=p.get('source', c['url']), basis='chain' if c['network'] in ('DDX','Spirit')
                               else ('published' if value else 'scenario')))
    save_csv(OUTPUT / 'prices.csv', price_rows)
    clubs = coordinates(catalog)
    network = np.array([c['network'] for c in catalog])
    restricted = np.array(['мисс фитнес' in c['name'].lower() for c in catalog])
    layer = json.loads((BASE / 'density.geojson').read_text())['features']
    ll = np.array([f['geometry']['coordinates'] for f in layer])
    mass = np.array([f['properties']['demand_share'] for f in layer])
    allowed = np.array([f['properties']['allowed'] for f in layer])
    sites = metric(ll[allowed])
    model = json.loads((BASE / 'report.json').read_text())
    terms = Model(*(np.array(model[k]) for k in ('means','scales','coefficients'))).terms(Features().measure(sites))
    site_log = terms[:, N:].sum(axis=1)
    with ZipFile(ROOT / 'marino_project.zip') as archive:
        graph = WalkingGraph(archive)
        routes, offsets, _ = graph.routes(np.vstack([geographic(clubs), ll[allowed]]), ll)
    club_m, site_m = np.split(routes, [len(clubs)])
    labels = label_sites(sites)
    free = cKDTree(clubs).query(sites)[0] >= MIN_DISTANCE
    rankings, historical, fixed = [], [], []
    control, control_net, _, _, _ = score_market(club_m, site_m, network == 'DDX', mass, site_log, free)
    winner = np.argmax(control)
    print('nearest, all clubs:', labels[winner][0], 100*control_net[winner])
    anchors = [r for r in records(ROOT / 'results/maryino_local/scenarios.csv') if r['market']=='base']
    for sensitivity, unknown in [(0, 3000)] + [(k, p) for p in (1900, 3000, 5000) for k in (3, 6, 10)]:
        prices = np.array([float(r['regular_rub'] or unknown) for r in price_rows])
        for j in [None] + [i for i,c in enumerate(catalog) if c['scope']=='inside']:
            keep = np.ones(len(clubs), bool)
            if j is not None:
                keep[j] = False
            own = network[keep] == ('DDX' if j is None else network[j])
            feasible = free if j is None else cKDTree(clubs[keep]).query(sites)[0] >= MIN_DISTANCE
            transfer, rival, new = choice(club_m[keep]/METRES_PER_MINUTE, site_m/METRES_PER_MINUTE,
                                          prices[keep], DDX_PRICE if j is None else prices[j], own,
                                          restricted[keep], False if j is None else restricted[j], mass, sensitivity)
            net = rival + new
            score = np.where(feasible, site_log + np.log(net), -np.inf)
            selected = zones(sites, score)
            if j is not None:
                distances = np.linalg.norm(sites[selected] - clubs[j], axis=1)
                historical.append(dict(k=sensitivity, unknown_rub=unknown, club=catalog[j]['name'],
                                       network=network[j], top1_m=distances[0], top3_m=min(distances)))
                continue
            for anchor in anchors:
                xy = metric([(float(anchor['lon']), float(anchor['lat']))])[0]
                i = np.argmin(np.linalg.norm(sites-xy, axis=1))
                fixed.append(dict(k=sensitivity, unknown_rub=unknown, landmark=anchor['landmark'],
                                  net_pp=100*net[i], score=score[i], feasible=bool(feasible[i])))
            for criterion, indices in [('site_and_gain', selected), ('gain', zones(sites, np.where(feasible, net, -np.inf)))]:
                for rank,i in enumerate(indices, 1):
                    rankings.append(dict(k=sensitivity, unknown_rub=unknown, criterion=criterion, rank=rank,
                                         landmark=labels[i][0], lon=ll[allowed][i,0], lat=ll[allowed][i,1],
                                         net_pp=100*net[i], rivals_pp=100*rival[i], outside_pp=100*new[i],
                                         own_transfer_pp=100*transfer[i], gross_pp=100*(net[i]+transfer[i])))
    save_csv(OUTPUT / 'opening.csv', rankings)
    save_csv(OUTPUT / 'fixed_sites.csv', fixed)
    save_csv(OUTPUT / 'historical.csv', historical)
    print('club count', len(catalog), 'max club access (m)', max(offsets[:len(clubs)]))


if __name__ == '__main__':
    run()
