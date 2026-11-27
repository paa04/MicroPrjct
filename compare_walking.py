"""Paired DDX entry comparison using the archived Maryino pedestrian graph."""
import argparse
import csv
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from economy_model import (DATA, MIN_DISTANCE, N, Event, Features, Model, coordinates,
                           evaluate, label_sites, local_market, other_gym_mask, records, zones)
from spatial_model import ROOT, geographic, metric, save_csv


class WalkingGraph:
    def __init__(self, archive):
        from pyproj import Transformer

        node_index, points, edges, keys = {}, [], {}, {}
        ns = '{http://graphml.graphdrawing.org/xmlns}'
        with archive.open('marino_project/cache/walk_3000.0.graphml') as stream:
            for _, item in ET.iterparse(stream, events=('end',)):
                if item.tag == ns+'key':
                    keys[item.get('id')] = item.get('attr.name')
                elif item.tag == ns+'data' and keys.get(item.get('key')) == 'crs':
                    self.project = Transformer.from_crs('EPSG:4326', item.text, always_xy=True)
                elif item.tag == ns+'node':
                    values = {keys[v.get('key')]: v.text for v in item}
                    node_index[item.get('id')] = len(points)
                    points.append((float(values['x']), float(values['y'])))
                    item.clear()
                elif item.tag == ns+'edge':
                    a, b = node_index[item.get('source')], node_index[item.get('target')]
                    if a != b:
                        pair = min(a, b), max(a, b)
                        length = float(next(v.text for v in item if keys[v.get('key')] == 'length'))
                        edges[pair] = min(edges.get(pair, np.inf), length)
                    item.clear()
        self.tree = cKDTree(points)
        i, j = np.array(list(edges)).T
        upper = coo_matrix((list(edges.values()), (i, j)), shape=(len(points), len(points))).tocsr()
        self.graph = upper + upper.T

    def routes(self, starts, ends):
        """Metres, including the straight access links at both ends of each route."""
        a = np.column_stack(self.project.transform(*starts.T))
        b = np.column_stack(self.project.transform(*ends.T))
        access_a, ids = self.tree.query(a)
        access_b, targets = self.tree.query(b)
        nodes, inverse = np.unique(ids, return_inverse=True)
        lengths = np.empty((len(nodes), len(b)))
        for k in range(0, len(nodes), 24):
            distances = dijkstra(self.graph, indices=nodes[k:k+24])
            lengths[k:k+24] = distances[:, targets]
        result = lengths[inverse]+access_a[:, None]+access_b
        # A club and demand at the identical coordinate require no walk to the graph.
        result[cdist(a, b) < 1e-6] = 0
        if not np.isfinite(result).all():
            raise ValueError('Not all demand is reachable within the archived walking graph')
        return result, access_a, access_b


def verify_archive(archive, graph):
    def read(name):
        return archive.read('marino_project/results/'+name)
    clubs = list(csv.DictReader(io.StringIO(read('club_demand.csv').decode('utf-8-sig'))))
    starts = np.array([(r['lon'], r['lat']) for r in clubs], float)
    ends = np.array([f['geometry']['coordinates'] for f in json.loads(read('demand.geojson'))['features']])
    distances, _, _ = graph.routes(starts, ends)
    with np.load(io.BytesIO(read('distance_matrices.npz')), allow_pickle=False) as arrays:
        error = float(np.max(np.abs(distances-arrays['Dc'])))
    if error > .01:
        raise ValueError(f'Walking distances disagree with archive by {error:.2f} m')
    return error


def score_market(club_dist, site_dist, own, mass, site_log, free):
    best = club_dist.min(axis=0)
    tied = np.abs(club_dist-best) <= 1e-6
    counts = tied.sum(axis=0)
    owners = tied/counts
    own_frac = owners[own].sum(axis=0)
    wins = np.where(site_dist < best-1e-6, 1.,
                    np.where(np.abs(site_dist-best) <= 1e-6, 1./(1+counts), 0.))
    gross = wins @ mass
    net = wins @ (mass*(1-own_frac))
    score = np.where(free & (net > 1e-14), site_log+np.log(np.maximum(net, 1e-15)), -np.inf)
    return score, net, gross, owners, wins


def specs(base, archive_path):
    config = json.loads((base/'report.json').read_text())
    features = json.loads((base/'density.geojson').read_text())['features']
    lonlat = np.array([r['geometry']['coordinates'] for r in features])
    dist_grid = records(DATA/f"district_{config['grid_m']}.csv")
    cells = np.array([(r['x_m'], r['y_m']) for r in dist_grid], float)
    np.testing.assert_allclose(cells, metric(lonlat), atol=1e-6)
    allowed = np.array([r['properties']['allowed'] for r in features])
    mass = np.array([r['properties']['demand_share'] for r in features])
    model = Model(*(np.array(config[k]) for k in ('means', 'scales', 'coefficients')))
    terms = model.terms(Features().measure(cells))
    broad = np.exp(terms[:, :N].sum(axis=1))
    np.testing.assert_allclose(mass, broad/broad.sum(), rtol=1e-8)
    site_log = terms[allowed, N:].sum(axis=1)
    clubs = records(base/'competition.csv')
    ring = json.loads((ROOT/'data/maryino/boundary.geojson').read_text())['features'][0]['geometry']['coordinates'][0]
    wide = local_market(records(DATA/'catalog.csv'), ring, config['radius_m'], uncertain=True)
    keys = {(c['network'], c['id']) for c in clubs}
    all_clubs = clubs+[c for c in wide if (c['network'], c['id']) not in keys]
    club_ll = np.array([(r['lon'], r['lat']) for r in all_clubs], float)
    with ZipFile(archive_path) as archive:
        graph = WalkingGraph(archive)
        error = verify_archive(archive, graph)
    walking, offsets, demand_offsets = graph.routes(np.vstack([club_ll, lonlat[allowed]]), lonlat)
    n = len(all_clubs)
    straight = (cdist(coordinates(all_clubs), cells), cdist(cells[allowed], cells))
    diagnostics = dict(max_archive_route_error_m=error, max_club_snap_m=float(offsets[:len(clubs)].max()),
                       max_site_snap_m=float(offsets[n:].max()), max_demand_snap_m=float(demand_offsets.max()),
                       demand_mass_snap_over_100m=float(mass[demand_offsets>100].sum()))
    return (cells, allowed, mass, site_log, all_clubs, len(clubs), offsets[n:], diagnostics,
            {'straight': straight, 'walk': (walking[:n], walking[n:])})


def comparison_map(base, output, cells, sites, market, after_maps, scenarios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from economy_figures import District, INK, NEW, PAPER, tile_layer

    district = District.load(base)
    np.testing.assert_allclose(cells, district.cells, atol=1e-6)
    ring = district.ring
    active = np.flatnonzero(sum(fractions.sum(axis=1) for _, fractions in after_maps.values()) > 0)
    palette = ['#a4c9bb', '#a8c8e5', '#cfb5d7', '#f1b7ab', '#bdcb91', '#e4cbb2',
               '#a6d0d5', '#cad2e4']
    colors = np.tile(to_rgb(NEW), (len(market)+1, 1))
    for i, color in zip(active[active<len(market)], palette):
        colors[i] = to_rgb(color)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), facecolor=PAPER)
    fig.subplots_adjust(left=.02, right=.98, bottom=.24, top=.78, wspace=.04)
    fig.suptitle('Марьино · зоны клубов после открытия нового DDX', y=.96, fontsize=24,
                 weight='bold', color=INK)
    fig.text(.5, .86, 'Звезда — лучшая площадка модели. Цвет — ближайший клуб для жителей ячейки.',
             ha='center', fontsize=12, color=INK)
    titles = {'straight':'Расстояние по прямой', 'walk':'Пешеходные маршруты'}
    for ax, (method, (k, fractions)) in zip(axes, after_maps.items()):
        sc = next(r for r in scenarios if r['method']==method and r['market']=='base')
        ax.set_title(titles[method]+'\n'+sc['landmark']+'   ·   +'+f"{sc['net_gain_pp']:.2f}".replace('.', ',')+' п.п. сети',
                     fontsize=14, color=INK)
        ax.set_aspect('equal')
        ax.set_xlim(ring[:, 0].min()-100, ring[:, 0].max()+100)
        ax.set_ylim(ring[:, 1].min()-175, ring[:, 1].max()+125)
        ax.set_axis_off()
        ax.fill(*ring.T, color='white', zorder=0)
        tile_layer(ax, district, colors=fractions.T @ colors)
        ax.plot(*ring.T, color=INK, lw=1.2, zorder=3)
        ax.scatter(*coordinates(market).T, c=INK, s=26, edgecolor='white', lw=.8, zorder=4)
        ax.scatter(*sites[k], marker='*', c=NEW, edgecolor=INK, s=210, zorder=5)
        left, bottom = ring.min(axis=0)
        ax.plot([left,left+1000],[bottom-80,bottom-80],lw=2,color=INK)
        ax.text(left+500,bottom-40,'1 км',ha='center',fontsize=8)
    labels = [c['name'].replace('Братиславская Люблю Молл','Люблю Молл') for c in market]+['Новый DDX']
    handles = [Patch(facecolor=colors[i], label=labels[i]) for i in active]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.08), ncol=3, frameon=False)
    fig.text(.04,.032, f"Сетка {district.report['grid_m']} м  ·  {district.report['snapshot']}  ·  © OpenStreetMap contributors (ODbL)",
             fontsize=9, color=INK)
    fig.savefig(output/'catchment_comparison.png', dpi=170)
    plt.close(fig)


def run(base, archive_path, output):
    cells, allowed, mass, site_log, market, n, snaps, diagnostics, distances = specs(base, archive_path)
    sites = cells[allowed]
    groups = np.array([c['network'] for c in market])
    club_xy = coordinates(market)
    labels = label_sites(sites)
    ll = geographic(sites)
    base_other = other_gym_mask(market[:n], sites)
    rankings, checks, share_rows, scenarios, regrets = [], [], [], [], []
    after_maps = {}
    for method, (club_d, site_d) in distances.items():
        for case, count in [('base', n), ('wider_market', len(market))]:
            loc = club_xy[:count]
            own = groups[:count]=='DDX'
            free = ~base_other & (cKDTree(loc).query(sites)[0] >= MIN_DISTANCE)
            score, net, gross, owners, wins = score_market(club_d[:count], site_d, own, mass, site_log, free)
            selected = zones(sites, score)
            if not selected:
                raise ValueError(f'No viable sites under {method}/{case}')
            k = selected[0]
            before = owners @ mass
            after = owners @ (mass*(1-wins[k]))
            np.testing.assert_allclose(before.sum(), 1., atol=1e-10)
            np.testing.assert_allclose(after.sum()+gross[k], 1., atol=1e-10)
            np.testing.assert_allclose(after[own].sum()+gross[k]-before[own].sum(), net[k], atol=1e-10)
            scenarios.append(dict(method=method, market=case, clubs=count, feasible_sites=int(free.sum()),
                                  positive_sites=int(np.isfinite(score).sum()), landmark=labels[k][0], address=labels[k][1],
                                  net_gain_pp=100*net[k], gross_pp=100*gross[k], transfer_pp=100*(gross[k]-net[k]),
                                  ddx_before_pct=100*before[own].sum(), ddx_after_pct=100*(after[own].sum()+gross[k]),
                                  lon=ll[k, 0], lat=ll[k, 1], snap_m=snaps[k]))
            if case != 'base':
                continue
            for i in np.flatnonzero(free)[np.argsort(-score[free], kind='stable')]:
                rankings.append(dict(method=method, lon=round(ll[i, 0], 6), lat=round(ll[i, 1], 6),
                                     landmark=labels[i][0], address=labels[i][1], net_gain_pp=100*net[i],
                                     gross_pp=100*gross[i], transfer_pp=100*(gross[i]-net[i]),
                                     score=float(score[i]) if np.isfinite(score[i]) else None,
                                     shortlist=selected.index(i)+1 if i in selected else '', snap_m=snaps[i]))
            for j, club in enumerate(market[:n]):
                share_rows.append(dict(method=method, club=club['name'], before_pct=100*before[j],
                                       after_pct=100*after[j], transferred_pp=100*(before[j]-after[j])))
            after_maps[method] = (k, np.vstack((owners*(1-wins[k]), wins[k])))
        for j, club in enumerate(market[:n]):
            if club['scope'] != 'inside':
                continue
            keep = np.arange(n)!=j
            free = ~base_other & (cKDTree(club_xy[:n][keep]).query(sites)[0] >= MIN_DISTANCE)
            valid = np.flatnonzero(free)
            target = valid[np.argmin(np.linalg.norm(sites[valid]-club_xy[j], axis=1))]
            event = Event(club, (), sites, None, target, club_xy[j])
            own_remaining = groups[:n][keep]==club['network']
            score, gains, *_ = score_market(club_d[:n][keep], site_d, own_remaining, mass, site_log, free)
            checks.append(evaluate(method, event, score))
            # Incumbent's value at its actual coordinates relative to the same reduced market.
            _, actual, *_ = score_market(club_d[:n][keep], club_d[j:j+1], own_remaining,
                                         mass, np.zeros(1), np.ones(1, bool))
            regrets.append(dict(method=method, club=club['name'], network=club['network'],
                                max_regret_pp=100*max(0., gains[free].max()-actual[0])))

    # Anchor the paired comparison to the released Euclidean results on the entire ranked surface.
    reference = records(base/'opening_candidates.csv')
    straight = [r for r in rankings if r['method']=='straight' and r['score'] is not None]
    indexed = {(r['lon'],r['lat']): r for r in straight}
    if len(indexed)!=len(reference):
        raise ValueError('Straight-line candidate set differs from the baseline report')
    for r in reference:
        s = indexed[(float(r['lon']), float(r['lat']))]
        np.testing.assert_allclose([s['net_gain_pp'], s['gross_pp'], s['score']],
                                   [float(r['net_gain_pp']),float(r['gross_pp']),float(r['score'])], atol=.00051)
    historic = {r['name']:r for r in records(base/'historical_validation.csv')
                if r['evaluation']=='district' and r['model']=='entry'}
    for r in checks:
        if r['model']=='straight':
            np.testing.assert_allclose(r['top1_m'], float(historic[r['name']]['top1_m']), atol=.001)
    diagnostics['candidate_count_snap_over_100m'] = sum(1 for r in rankings if r['method']=='walk' and r['snap_m']>100)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('opening_candidates',rankings),('historical_validation',checks),
                       ('club_transfers',share_rows),('relocation_regret',regrets),('scenarios',scenarios)]:
        save_csv(output/(name+'.csv'), rows)
    errors = {method:dict(mean_top1_m=float(np.mean([r['top1_m'] for r in checks if r['model']==method])),
                          mean_top3_m=float(np.mean([r['top3_m'] for r in checks if r['model']==method])),
                          hit3_300=sum(r['hit3_300'] for r in checks if r['model']==method))
              for method in distances}
    report = dict(diagnostics=diagnostics, scenarios=scenarios, local_validation=errors)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return report, (cells, sites, market[:n], after_maps, scenarios)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'marino_project.zip')
    parser.add_argument('--baseline', type=Path, default=ROOT/'results/economy')
    parser.add_argument('--output', type=Path, default=ROOT/'results/combined')
    parser.add_argument('--map', action='store_true', help='Also render a side-by-side catchment map')
    args = parser.parse_args()
    report, map_data = run(args.baseline, args.archive, args.output)
    if args.map:
        comparison_map(args.baseline, args.output, *map_data)
    print(json.dumps(report, ensure_ascii=False, indent=2))
