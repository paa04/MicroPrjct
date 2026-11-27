"""Fit a conditional location model for DDX and evaluate a segment-specific entry."""
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from fetch_economy import BBOX, DATA
from spatial_model import (ROOT, FEATURES, PRIOR, geographic, inside,
                           joining_share, metric, nearest, opening_gains, save_csv)

CUT = '2025-01-01'
TILE = 3600
STEP = 100
GUARD = 3000
SPACING = 600
MIN_DISTANCE = 150
DENSITY = ('housing_floors', 'office_floors', 'supermarkets', 'bus_stops', 'metro_access')
SITE = ('mall_proximity', 'retail_building_proximity', 'metro_entrance_proximity', 'large_venue_proximity')
N = len(DENSITY)


def records(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def coordinates(rows):
    return metric([(r['lon'], r['lat']) for r in rows])


class Features:
    def __init__(self):
        self.trees = {}
        self.levels = {}
        for name in ('housing', 'office', 'stores', 'bus', 'metro', 'shopping', 'retail'):
            rows = records(DATA/(name+'.csv'))
            self.trees[name] = cKDTree(coordinates(rows))
            if name in ('housing', 'office'):
                floors = []
                for row in rows:
                    try:
                        floors.append(float(row['value'].replace(',', '.')))
                    except ValueError:
                        floors.append(np.nan)
                floors = np.array(floors)
                floors[(floors < 1) | (floors > 85)] = np.nan
                self.levels[name] = np.nan_to_num(floors, nan=np.nanmedian(floors))

        venues = records(DATA/'venues.csv')
        self.venues = cKDTree(coordinates(venues))
        self.areas = np.array([float(r['footprint_m2']) for r in venues])
        self.large_venues = cKDTree(self.venues.data[self.areas >= 1000])

    def measure(self, xy):
        columns = []
        for key, radius, k in [('housing', 500, 96), ('office', 500, 48), ('stores', 400, 32),
                               ('bus', 400, 32), ('metro', 600, 1)]:
            dist, idx = self.trees[key].query(xy, k=list(range(1, k+1)))
            weights = np.exp(-.5*(dist/radius)**2)
            if key in self.levels:
                weights *= self.levels[key][idx]
            columns.append(np.log1p(weights.sum(axis=1)))
        for key, radius in [('shopping', 180), ('retail', 125), ('metro', 250)]:
            dist = self.trees[key].query(xy)[0]
            columns.append(np.exp(-dist/radius))
        dist, idx = self.venues.query(xy, k=8)
        capacity = np.log1p(np.minimum(self.areas[idx], 30000)/2000)
        columns.append((capacity*np.exp(-dist/150)).max(axis=1))
        return np.column_stack(columns)


@dataclass
class Event:
    club: dict
    tile: tuple
    xy: np.ndarray
    x: np.ndarray
    target: int
    location: np.ndarray


def make_events(catalog, ring, features):
    polygon = metric(ring)
    south, west, north, east = BBOX
    low, high = metric([(west, south), (east, north)])
    cache, events, excluded = {}, [], {}
    for club in catalog:
        if club['network'] != 'DDX':
            continue
        loc = coordinates([club])[0]
        tile = tuple(np.floor(loc/TILE).astype(int))
        corner = np.array(tile)*TILE
        # Exclude whole tiles near the target and avoid clipped OSM catchments.
        if np.all(corner <= polygon.max(axis=0)+GUARD) and np.all(corner+TILE >= polygon.min(axis=0)-GUARD):
            excluded[club['id']] = 'district_holdout'
            continue
        if np.any(corner < low+800) or np.any(corner+TILE > high-800):
            excluded[club['id']] = 'coverage_edge'
            continue
        if tile not in cache:
            axis = np.arange(STEP/2, TILE, STEP)
            xy = corner+np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
            cache[tile] = xy, features.measure(xy)
        xy, x = cache[tile]
        events.append(Event(club, tile, xy, x, int(np.argmin(((xy-loc)**2).sum(axis=1))), loc))
    return events, excluded


@dataclass
class Model:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def terms(self, x):
        return (x-self.mean)/self.scale*self.coef

    def scores(self, x):
        return self.terms(x).sum(axis=1)


def fit(events, ridge, dimensions=None):
    # Each opening chooses one of the same 100-m cells in its fixed Moscow tile.
    x = np.stack([e.x for e in events])
    mean = x.mean(axis=(0, 1))
    scale = np.maximum(x.std(axis=(0, 1)), .01)
    x = (x-mean)/scale
    chosen = x[np.arange(len(events)), [e.target for e in events]].mean(axis=0)

    def objective(coef):
        scores = x @ coef
        normalizer = logsumexp(scores, axis=1)
        prob = np.exp(scores-normalizer[:, None])
        loss = normalizer.mean() - chosen @ coef + .5*ridge*(coef @ coef)
        grad = np.einsum('ij,ijk->k', prob, x)/len(events) - chosen + ridge*coef
        return loss, grad

    bounds = [(0, 4) if dimensions is None or k in dimensions else (0, 0) for k in range(x.shape[-1])]
    result = minimize(objective, np.zeros(len(bounds)), jac=True, bounds=bounds, method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(result.message)
    return Model(mean, scale, result.x)


def select_ridge(events):
    # Fold membership depends only on the fixed tile. No tile appears on both sides.
    tiles = sorted({e.tile for e in events})
    order = np.random.default_rng(42).permutation(len(tiles))
    folds = {tiles[k]: j % 4 for j, k in enumerate(order)}
    losses = {}
    for ridge in (.02, .1, .5):
        loss = []
        for fold in range(4):
            train = [e for e in events if folds[e.tile] != fold]
            model = fit(train, ridge)
            for e in events:
                if folds[e.tile] == fold:
                    score = model.scores(e.x)
                    loss.append(logsumexp(score)-score[e.target])
        losses[ridge] = float(np.mean(loss))
    return min(losses, key=losses.get), losses


def zones(xy, scores, n=3):
    selected = []
    for i in np.argsort(-scores, kind='stable'):
        if not np.isfinite(scores[i]):
            break
        if all(np.linalg.norm(xy[i]-xy[j]) >= SPACING for j in selected):
            selected.append(i)
            if len(selected) == n:
                break
    return selected


def evaluate(name, e, score):
    available = score[np.isfinite(score)]
    if not len(available):
        raise ValueError('No available alternatives for '+e.club['name'])
    target_score = score[e.target]
    selected = zones(e.xy, score)
    distances = np.linalg.norm(e.xy[selected]-e.location, axis=1)
    top_lon, top_lat = geographic(e.xy[selected[:1]])[0]
    rank_fraction = (sum(available > target_score)+.5*sum(available == target_score))/len(available)
    return dict(name=e.club['name'], date=e.club['opened'], model=name, alternatives=len(available),
                top_fraction=float(rank_fraction), in_top_decile=int(rank_fraction <= .10),
                top1_lon=float(top_lon), top1_lat=float(top_lat),
                top1_m=float(distances[0]), top3_m=float(distances.min()), hit3_300=int(distances.min() <= 300),
                log_loss=float(logsumexp(available)-target_score) if np.isfinite(target_score) else None)


def summary(rows):
    result = {}
    for name in dict.fromkeys(r['model'] for r in rows):
        subset = [r for r in rows if r['model']==name]
        losses = [r['log_loss'] for r in subset if r['log_loss'] is not None]
        result[name] = dict(n=len(subset), top_decile=sum(r['in_top_decile'] for r in subset),
                            hit3_300=sum(r['hit3_300'] for r in subset),
                            median_top1_m=round(float(np.median([r['top1_m'] for r in subset]))),
                            median_top3_m=round(float(np.median([r['top3_m'] for r in subset]))),
                            median_top_fraction=round(float(np.median([r['top_fraction'] for r in subset])), 4),
                            log_loss_n=len(losses), mean_log_loss=round(float(np.mean(losses)), 3) if losses else None)
    return result


def distance_to_polygon(point, poly):
    if inside(point, poly):
        return 0.
    starts, vectors = poly[:-1], np.diff(poly, axis=0)
    k = np.clip(((point-starts)*vectors).sum(axis=1)/np.maximum((vectors**2).sum(axis=1), 1e-9), 0, 1)
    return np.linalg.norm(point-starts-k[:, None]*vectors, axis=1).min()


def local_market(catalog, ring, radius, uncertain=False):
    reviewed = records(ROOT/'data/clubs_reviewed.csv')
    prices = records(DATA/'local_prices.csv')
    market = [dict(r) for r in catalog]
    for price in prices:
        if price['include'] == 'yes' and not (0 < float(price['regular_month_rub']) <= 3000):
            raise ValueError('Price outside the DDX comparison segment for '+price['club_id'])
        if price['include'] == 'yes' or (uncertain and price['include'] == 'uncertain'):
            row = next(r for r in reviewed if r['club_id'] == price['club_id'] and r['eligible']=='yes')
            market.append(dict(name=row['name'], network=row['network'], id=row['club_id'], lon=row['lon'],
                               lat=row['lat'], address=row['address'], opened='', url=price['source']))
    # The chain map pin for Spirit Maryino is ~200 m outside its published building.
    # Use the audited entrance at the same address as the chain and the mall.
    for correction in records(DATA/'coordinate_review.csv'):
        for club in market:
            if club['network']==correction['network'] and club['id']==correction['id']:
                club.update(lon=correction['lon'], lat=correction['lat'])
    result = []
    poly = metric(ring)
    for r, xy in zip(market, coordinates(market)):
        distance = distance_to_polygon(xy, poly)
        if distance <= radius:
            result.append(dict(r, scope='inside' if distance==0 else 'buffer', boundary_m=round(distance)))
    return result


def entry_scores(cells, rho, sites, site_log, clubs, network):
    xy = coordinates(clubs)
    chains = np.array([r['network'] for r in clubs])
    # First column is incremental network demand, second is gross new-club demand.
    net = opening_gains(cells, rho[:, None], xy, chains, network, sites)[:, 0]
    gross = opening_gains(cells, rho[:, None], xy, chains, '__entrant__', sites)[:, 0]
    occupied = cKDTree(xy).query(sites)[0] < MIN_DISTANCE
    score = site_log + np.log(np.maximum(net, 1e-15))
    score[occupied | (net <= 1e-14)] = -np.inf
    return score, net, gross


def label_sites(sites):
    poi = [r for r in records(ROOT/'data/maryino/poi.csv') if r['name'] and r['address'] and
           r['category'] in ('shopping_mall', 'retail_building', 'large_store', 'commercial')]
    distance, idx = cKDTree(coordinates(poi)).query(sites)
    return [(poi[i]['name'], poi[i]['address'], round(d)) for d, i in zip(distance, idx)]


def historical_entry(events, catalog, model):
    rows, skipped = [], []
    for e in events:
        # Residents of the tile, with a 3-km competitor buffer, like the district calculation.
        rivals = [r for r in catalog if r['network'] == 'Spirit' or r['opened'] < e.club['opened']]
        corner = np.array(e.tile)*TILE
        xy = coordinates(rivals)
        dist = np.linalg.norm(np.maximum(np.maximum(corner-xy, xy-(corner+TILE)), 0), axis=1)
        rivals = [r for r, d in zip(rivals, dist) if d <= GUARD]
        if not any(r['network']=='Spirit' for r in rivals):
            skipped.append(dict(name=e.club['name'], reason='no_rival_within_buffer'))
            continue
        terms = model.terms(e.x)
        broad = terms[:, :N].sum(axis=1)
        mass = np.exp(broad-logsumexp(broad))
        score, net, _ = entry_scores(e.xy, mass, e.xy, terms[:, N:].sum(axis=1), rivals, 'DDX')
        if not np.isfinite(score[e.target]):
            skipped.append(dict(name=e.club['name'], reason='occupied_or_zero_capture_at_target'))
            continue
        available = np.isfinite(score)
        for name, values in [('entry', score), ('capture_only', np.log(np.maximum(net, 1e-15))),
                             ('mall', e.x[:, N]), ('joint', model.scores(e.x))]:
            rows.append(evaluate(name, e, np.where(available, values, -np.inf)))
    return rows, skipped


def stability(train, ridge, x, allowed, cells, market, sites, shortlist, unavailable):
    owner, best, ties = nearest(cells, coordinates(market))
    non_ddx = np.array([r['network'] != 'DDX' for r in market])
    eligible = owner[:, non_ddx].sum(axis=1)
    wins = joining_share(((sites[:, None]-cells)**2).sum(axis=2), best, ties)
    captured = wins*eligible
    blocks = sorted({e.tile for e in train})
    groups = [[e for e in train if e.tile == tile] for tile in blocks]
    rng = np.random.default_rng(2026)
    net_samples, winners = [], []
    for _ in range(40):
        sample = [e for g in rng.integers(len(groups), size=len(groups)) for e in groups[g]]
        terms = fit(sample, ridge).terms(x)
        broad = terms[:, :N].sum(axis=1)
        net = captured @ np.exp(broad-logsumexp(broad))
        scores = terms[allowed, N:].sum(axis=1)+np.log(np.maximum(net, 1e-15))
        scores[unavailable | (net <= 1e-14)] = -np.inf
        winners.append(sites[np.argmax(scores)])
        net_samples.append(100*net[shortlist])
    q10, q90 = np.quantile(net_samples, [.1, .9], axis=0)
    frequencies = [float(np.mean(np.linalg.norm(winners-sites[j], axis=1) <= 300)) for j in shortlist]
    return list(zip(q10, q90, frequencies))


def run(output, step=100, radius=3000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = records(DATA/'catalog.csv')
    boundary = json.loads((ROOT/'data/maryino/boundary.geojson').read_text())
    ring = boundary['features'][0]['geometry']['coordinates'][0]
    district = records(DATA/f'district_{step}.csv')
    cells = np.array([(r['x_m'], r['y_m']) for r in district], dtype=float)
    legacy = np.array([[r[k] for k in FEATURES] for r in district], dtype=float)
    allowed = np.array([r['allowed']=='1' for r in district])
    sites = cells[allowed]
    features = Features()
    events, excluded = make_events(catalog, ring, features)
    train = [e for e in events if e.club['opened'] < CUT]
    test = [e for e in events if e.club['opened'] >= CUT]
    if len(train) < 8 or not test:
        raise ValueError('Insufficient dated DDX locations within the covered area')
    ridge, cv = select_ridge(train)
    model = fit(train, ridge)
    density_only = fit(train, ridge, range(N))
    site_only = fit(train, ridge, range(N, N+len(SITE)))
    temporal, commercial, uncovered = [], [], []
    for e in test:
        for name, score in [('housing', e.x[:, 0]), ('mall', e.x[:, N]), ('site_only', site_only.scores(e.x)),
                            ('density_only', density_only.scores(e.x)), ('joint', model.scores(e.x))]:
            temporal.append(evaluate(name, e, score))
        viable = features.large_venues.query(e.xy)[0] <= 150
        if viable[e.target]:
            for name, scores in [('mall', e.x[:, N]), ('site_only', site_only.scores(e.x)),
                                 ('joint', model.scores(e.x))]:
                commercial.append(evaluate(name, e, np.where(viable, scores, -np.inf)))
        else:
            uncovered.append(e.club['name'])
    retrospective, skipped = historical_entry(test, catalog, model)
    market = local_market(catalog, ring, radius)
    wide_market = local_market(catalog, ring, radius, uncertain=True)
    # Avoid suggesting premises occupied by a reviewed gym of another format.
    reviewed = [r for r in records(ROOT/'data/clubs_reviewed.csv') if r['eligible']=='yes']
    market_tree = cKDTree(coordinates(market))
    other = [r for r, dist in zip(reviewed, market_tree.query(coordinates(reviewed))[0]) if dist > MIN_DISTANCE]
    other_occupied = cKDTree(coordinates(other)).query(sites)[0] < MIN_DISTANCE if other else np.zeros(len(sites), bool)
    save_csv(output/'competition.csv', market)
    x = features.measure(cells)
    terms = model.terms(x)
    density_log = terms[:, :N].sum(axis=1)
    rho = np.exp(density_log-logsumexp(density_log))
    site_log = terms[allowed, N:].sum(axis=1)
    legacy_rho = legacy @ PRIOR
    local = []
    regret = []
    for i, club in enumerate(market):
        if club['scope'] != 'inside':
            continue
        rest = market[:i]+market[i+1:]
        loc = coordinates([club])[0]
        free = (cKDTree(coordinates(rest)).query(sites)[0] >= MIN_DISTANCE) & ~other_occupied
        valid = np.flatnonzero(free)
        j = valid[np.argmin(np.linalg.norm(sites[free]-loc, axis=1))]
        event = Event(club, (), sites, x[allowed], j, loc)
        score, gain, _ = entry_scores(cells, rho, sites, site_log, rest, club['network'])
        old = opening_gains(cells, legacy_rho[:, None], coordinates(rest),
                            np.array([r['network'] for r in rest]), club['network'], sites)[:, 0]
        for name, scores in [('legacy_capture', np.log(np.maximum(old, 1e-15))),
                             ('learned_capture', np.log(np.maximum(gain, 1e-15))),
                             ('legacy_capture_plus_site', site_log+np.log(np.maximum(old, 1e-15))),
                             ('mall', x[allowed, N]), ('joint', model.scores(x[allowed])),
                             ('entry', score)]:
            scores = scores.copy()
            scores[~free | ~np.isfinite(score)] = -np.inf
            local.append(evaluate(name, event, scores))
        actual = opening_gains(cells, rho[:, None], coordinates(rest), np.array([r['network'] for r in rest]),
                               club['network'], loc[None, :])[0, 0]
        regret.append(dict(name=club['name'], network=club['network'], max_regret_pp=100*max(0., gain[free].max()-actual)))
    score, net, gross = entry_scores(cells, rho, sites, site_log, market, 'DDX')
    score[other_occupied] = -np.inf
    shortlist = zones(sites, score)
    if not shortlist:
        raise ValueError('No eligible search zone with positive incremental demand')
    bands = stability(train, ridge, x, allowed, cells, market, sites, shortlist, ~np.isfinite(score))
    wide_score, wide_net, _ = entry_scores(cells, rho, sites, site_log, wide_market, 'DDX')
    wide_score[other_occupied] = -np.inf
    wide_best = int(np.argmax(wide_score))
    labels = label_sites(sites)
    geo = geographic(sites)
    ranked = [dict(lon=round(geo[i, 0], 6), lat=round(geo[i, 1], 6), landmark=labels[i][0],
                   address=labels[i][1], landmark_m=labels[i][2], net_gain_pp=round(100*net[i], 3), gross_pp=round(100*gross[i], 3),
                   own_transfer_pp=round(100*(gross[i]-net[i]), 3), score=round(score[i], 5),
                   shortlist=shortlist.index(i)+1 if i in shortlist else '') for i in np.argsort(-score)
              if np.isfinite(score[i])]
    for row in ranked:
        if row['shortlist']:
            rank = row['shortlist']-1
            lo, hi, frequency = bands[rank]
            row.update(bootstrap_gain_p10_pp=round(lo, 3), bootstrap_gain_p90_pp=round(hi, 3),
                       bootstrap_first_within_300m=frequency, wider_market_gain_pp=round(100*wide_net[shortlist[rank]], 3))
        else:
            row.update(bootstrap_gain_p10_pp='', bootstrap_gain_p90_pp='',
                       bootstrap_first_within_300m='', wider_market_gain_pp='')
    save_csv(output/'opening_candidates.csv', ranked)
    save_csv(output/'relocation_regret.csv', regret)
    equilibrium = []
    for chain in sorted({r['network'] for r in regret}):
        worst = max((r for r in regret if r['network']==chain), key=lambda r: r['max_regret_pp'])
        equilibrium.append(dict(network=chain, max_unilateral_gain_pp=worst['max_regret_pp'],
                                club=worst['name']))
    save_csv(output/'equilibrium_networks.csv', equilibrium)
    for group, subset in [('temporal', temporal), ('commercial', commercial), ('temporal_current_rivals', retrospective), ('district', local)]:
        for row in subset:
            row['evaluation'] = group
    save_csv(output/'historical_validation.csv', temporal+commercial+retrospective+local)
    spatial = [dict(type='Feature', geometry=dict(type='Point', coordinates=ll.tolist()),
                    properties=dict(demand_share=float(p), allowed=bool(a)))
               for ll, p, a in zip(geographic(cells), rho, allowed)]
    (output/'density.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=spatial)))
    audit = []
    for r in catalog:
        if r['network']=='DDX':
            split = excluded.get(r['id'], 'train' if r['opened'] < CUT else 'time_test')
            audit.append(dict(id=r['id'], name=r['name'], opened=r['opened'], split=split, url=r['url']))
    save_csv(output/'training_split.csv', audit)
    report = dict(snapshot=(DATA/'catalog_date.txt').read_text().strip(), cutoff=CUT, train=len(train),
                  train_tiles=len({e.tile for e in train}), test_tiles=len({e.tile for e in test}), grid_m=step,
                  radius_m=radius, spatial_cv_log_loss=cv, ridge=ridge, features=list(DENSITY+SITE),
                  means=model.mean.tolist(), scales=model.scale.tolist(), coefficients=model.coef.tolist(),
                  temporal=summary(temporal), entry_with_current_spirit=summary(retrospective),
                  commercial=summary(commercial), commercial_outside_support=uncovered,
                  density_top_decile_mass=float(np.sort(rho)[-int(np.ceil(len(rho)/10)):].sum()),
                  historical_entry_skipped=skipped, maryino=summary(local),
                  wider_market=dict(clubs=len(wide_market), winner=labels[wide_best],
                                    distance_to_main_m=round(np.linalg.norm(sites[wide_best]-sites[shortlist[0]]))),
                  bootstrap_replicates=40, recommendations=[r for r in ranked if r['shortlist']],
                  local_competitors=len(market), local_targets=sum(c['scope']=='inside' for c in market))
    (output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return report
