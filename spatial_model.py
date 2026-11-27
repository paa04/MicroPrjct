"""Spatial fitness market model for Maryino. Run: python3 spatial_model.py run."""

import argparse
import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from build_maryino_poi import read_osm


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
RAW = ROOT / 'data/raw_osm'
LON0, LAT0 = 37.74, 55.654
MX = 111320 * np.cos(np.deg2rad(LAT0))
MY = 111320
FEATURES = ('housing', 'metro', 'retail', 'office')
PRIOR = np.array([.60, .15, .15, .10])


def metric(lonlat):
    a = np.asarray(lonlat, dtype=float)
    return np.column_stack(((a[:, 0] - LON0) * MX, (a[:, 1] - LAT0) * MY))


def geographic(xy):
    a = np.asarray(xy, dtype=float)
    return np.column_stack((a[:, 0] / MX + LON0, a[:, 1] / MY + LAT0))


def inside(point, ring):
    x, y = point
    hit = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1)+x1:
            hit = not hit
    return hit


def area_centroid(coords):
    p = metric(coords)
    if len(p) < 3:
        return p.mean(axis=0), 0.
    if not np.allclose(p[0], p[-1]):
        p = np.vstack((p, p[0]))
    cross = p[:-1, 0]*p[1:, 1]-p[1:, 0]*p[:-1, 1]
    area2 = cross.sum()
    if abs(area2) < 1e-8:
        return p[:-1].mean(axis=0), 0.
    center = np.array((((p[:-1, 0]+p[1:, 0])*cross).sum(),
                       ((p[:-1, 1]+p[1:, 1])*cross).sum()))/(3*area2)
    return center, abs(area2)/2


def load_osm():
    bn, bw, br = read_osm(RAW/'boundary.osm')
    members = br[455222][0]
    rings = []
    for typ, ref, role in members:
        if typ == 'way' and role == 'outer':
            refs = bw[ref][0]
            rings.append([bn[n][:2] for n in refs])
    # OSM relation 455222 has one outer ring made from multiple ways.
    segments = [list(r) for r in rings]
    ring = segments.pop(0)
    while ring[0] != ring[-1]:
        for k, s in enumerate(segments):
            if ring[-1] == s[0]: ring += s[1:]
            elif ring[-1] == s[-1]: ring += list(reversed(s[:-1]))
            else: continue
            segments.pop(k)
            break
        else: raise ValueError('Incomplete Maryino boundary')
    nodes, ways = {}, {}
    for path in sorted(RAW.glob('tile_*.osm')):
        n, w, _ = read_osm(path)
        nodes.update(n); ways.update(w)
    if not nodes: raise FileNotFoundError('Run fetch_maryino_osm.py first')
    return ring, nodes, ways


def grid(ring, step=100):
    p = metric(ring)
    xs = np.arange(np.floor(p[:, 0].min()/step)*step+step/2, p[:, 0].max(), step)
    ys = np.arange(np.floor(p[:, 1].min()/step)*step+step/2, p[:, 1].max(), step)
    return np.array([(x, y) for x in xs for y in ys if inside((x, y), p)])


def gaussian(grid_xy, points, weights, sigma):
    out = np.zeros(len(grid_xy))
    if not len(points): return out
    points = np.asarray(points)
    weights = np.asarray(weights)
    for start in range(0, len(points), 100):
        q = points[start:start+100]
        d2 = ((grid_xy[:, None, :]-q[None, :, :])**2).sum(axis=2)
        out += (np.exp(-d2/(2*sigma*sigma))*weights[start:start+100]).sum(axis=1)
    return out


def build_features(grid_xy, nodes, ways):
    points = {k: [] for k in FEATURES}
    weights = {k: [] for k in FEATURES}
    excluded = []
    transport_segments = []
    for way_id, (refs, tags) in ways.items():
        if not refs or any(ref not in nodes for ref in refs): continue
        coords = [nodes[ref][:2] for ref in refs]
        center, area = area_centroid(coords)
        building = tags.get('building', '')
        if area > 5 and building in {'apartments','residential','dormitory','house','detached'}:
            level = tags.get('building:levels','')
            try: levels = max(1., min(40., float(level)))
            except ValueError: levels = 1.
            points['housing'].append(center); weights['housing'].append(area*levels)
        if area > 5 and building == 'office':
            points['office'].append(center); weights['office'].append(area)
        # MARi and Ekran have verified retail addresses but no shop=mall tag.
        if area > 5 and (tags.get('shop') == 'mall' or way_id in {679606869, 27411001}):
            points['retail'].append(center); weights['retail'].append(area)
        if tags.get('natural') == 'water' or tags.get('waterway') == 'riverbank':
            excluded.append(metric(coords))
        if tags.get('highway') in {'motorway','trunk','primary','secondary'} or tags.get('railway') == 'rail':
            transport_segments.extend(zip(metric(coords)[:-1],metric(coords)[1:]))
    for lon, lat, tags in nodes.values():
        if tags.get('railway') == 'station' and tags.get('station') == 'subway':
            points['metro'].append(metric([(lon,lat)])[0]); weights['metro'].append(1.)
    raw = np.column_stack([gaussian(grid_xy, points[k], weights[k], 250 if k != 'metro' else 500)
                           for k in FEATURES])
    # Small uniform floor prevents unsupported cells from having exactly zero demand.
    raw += np.maximum(raw.mean(axis=0), 1e-12)*.01
    raw /= raw.sum(axis=0)
    allowed = np.ones(len(grid_xy), dtype=bool)
    for polygon in excluded:
        if len(polygon) > 3 and np.allclose(polygon[0], polygon[-1]):
            allowed &= np.array([not inside(q, polygon) for q in grid_xy])
    for p1,p2 in transport_segments:
        delta=p2-p1
        denom=delta@delta
        if denom < 1e-8: continue
        nearby=np.where(allowed & (grid_xy[:,0]>=min(p1[0],p2[0])-18) &
                        (grid_xy[:,0]<=max(p1[0],p2[0])+18) &
                        (grid_xy[:,1]>=min(p1[1],p2[1])-18) &
                        (grid_xy[:,1]<=max(p1[1],p2[1])+18))[0]
        if len(nearby):
            q=grid_xy[nearby]
            t=np.clip(((q-p1)@delta)/denom,0,1)
            allowed[nearby[((q-(p1+t[:,None]*delta))**2).sum(axis=1)<18**2]]=False
    return raw, allowed, {k: len(points[k]) for k in FEATURES}


def load_clubs(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    needed = {'club_id','network','name','lon','lat','verified'}
    if not rows or not needed.issubset(rows[0]): raise ValueError('clubs.csv needs '+','.join(sorted(needed)))
    rows = [r for r in rows if r.get('eligible','yes').lower() not in ('no','false','0')]
    if not rows:
        raise ValueError('The club catalog contains no eligible clubs')
    ids = [r['club_id'] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate club_id')
    for r in rows:
        if not r['network']: raise ValueError('Empty network')
        float(r['lon']); float(r['lat'])
    return rows, metric([[float(r['lon']),float(r['lat'])] for r in rows])


def seed_clubs(path):
    old = {}
    if path.exists():
        old = {r['club_id']:r for r in csv.DictReader(open(path,encoding='utf-8-sig'))}
    buffered=ROOT/'data/maryino/fitness_buffer_candidates.json'
    if buffered.exists():
        raw=json.loads(buffered.read_text(encoding='utf-8'))['features']
        rows=[dict(osm_type=r['osm_type'],osm_id=r['osm_id'],name=r['name'],
                   lon=r['lon'],lat=r['lat'],scope=r['scope'],
                   distance_m=r['distance_to_boundary_m'],osm_url=r['source']) for r in raw]
    else:
        src=ROOT/'data/maryino/poi.csv'
        rows=[dict(r,scope='inside',distance_m=0) for r in csv.DictReader(open(src,encoding='utf-8-sig'))
              if r['category']=='fitness_club']
    # An unnamed outline and a named node at the same venue are one club.
    named_nodes=[r for r in rows if r['osm_type']=='node' and r['name'].strip()]
    def redundant(r):
        if r['osm_type']!='way' or r['name'].strip(): return False
        p=metric([(float(r['lon']),float(r['lat']))])[0]
        return any(np.linalg.norm(p-metric([(float(n['lon']),float(n['lat']))])[0])<25
                   for n in named_nodes)
    rows=[r for r in rows if not redundant(r)]
    def network_name(name,club_id):
        low=name.casefold()
        for token,canonical in [('ddx','DDX'),('spirit','Spirit Fitness'),('x-fit','X-Fit'),
                                ('xfit','X-Fit'),('зебра','Zebra'),('zebra','Zebra'),
                                ('территория фитнеса','Территория Фитнеса'),
                                ('мисс фитнес','Мисс Фитнес')]:
            if token in low: return canonical
        return name or f'unknown:{club_id}'
    fields=['club_id','network','name','lon','lat','verified','eligible','scope','distance_m','source']
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields)
        w.writeheader()
        for r in rows:
            name=r['name'].strip()
            club_id=f"{r['osm_type']}/{r['osm_id']}"
            prior=old.get(club_id,{})
            w.writerow(dict(club_id=club_id,network=prior.get('network') or network_name(name,club_id),
                            name=name,lon=r['lon'],lat=r['lat'],verified=prior.get('verified','no'),
                            eligible=prior.get('eligible','no' if 'йоги' in name.casefold() else 'yes'),
                            scope=r['scope'],distance_m=r['distance_m'],source=r['osm_url']))
    print(f'Seeded {len(rows)} club candidates in {path}; existing review fields preserved')


def nearest(grid_xy, club_xy):
    d2=((grid_xy[:,None,:]-club_xy[None,:,:])**2).sum(axis=2)
    best = d2.min(axis=1)
    tied = np.abs(d2 - best[:, None]) <= 1e-8
    count = tied.sum(axis=1)
    return tied / count[:, None], best, count


def joining_share(distance_squared, best, count):
    return np.where(distance_squared < best - 1e-8, 1.,
                    np.where(np.abs(distance_squared - best) <= 1e-8, 1 / (count + 1), 0.))


def gains(grid_xy, components, club_xy, networks, moving, candidates):
    """Candidate x feature gain in network share from moving one club."""
    keep = np.array([i for i in range(len(club_xy)) if i != moving])
    own = networks[moving]
    before, _, _ = nearest(grid_xy, club_xy)
    remaining, _, _ = nearest(grid_xy, club_xy[keep])
    loss = (remaining[:, networks[keep] == own].sum(axis=1)
            - before[:, networks == own].sum(axis=1)) @ components
    return loss + opening_gains(grid_xy, components, club_xy[keep], networks[keep],
                                own, candidates)


def opening_gains(grid_xy, components, club_xy, networks, own, candidates):
    owner, best, count = nearest(grid_xy, club_xy)
    eligible = owner[:, networks != own].sum(axis=1)
    result=np.empty((len(candidates),components.shape[1]))
    for start in range(0,len(candidates),128):
        q=candidates[start:start+128]
        wins = joining_share(((q[:,None,:]-grid_xy[None,:,:])**2).sum(axis=2), best, count)
        result[start:start+len(q)] = (wins*eligible[None,:]) @ components
    return result


def fit_weights(regrets, lam):
    """Minimize mean maximum regret plus lambda L1 distance from prior."""
    n=len(regrets); p=4
    # variables: alpha(4), per-club regret(n), prior deviations(4)
    c=np.r_[np.zeros(p),np.ones(n)/n,np.ones(p)*lam]
    rows=[]; bounds=[]
    for i, matrix in enumerate(regrets):
        for v in matrix:
            row=np.zeros(p+n+p); row[:p]=v; row[p+i]=-1
            rows.append(row); bounds.append(0.)
    for k in range(p):
        row=np.zeros(p+n+p);row[k]=1;row[p+n+k]=-1
        rows.append(row);bounds.append(PRIOR[k])
        row=np.zeros(p+n+p);row[k]=-1;row[p+n+k]=-1
        rows.append(row);bounds.append(-PRIOR[k])
    eq=np.zeros((1,p+n+p));eq[0,:p]=1
    sol=linprog(c,A_ub=np.array(rows),b_ub=np.array(bounds),A_eq=eq,b_eq=[1],
                bounds=[(0,1)]*p+[(0,None)]*(n+p),method='highs')
    if not sol.success: raise RuntimeError(sol.message)
    return sol.x[:p]


def relocation_regret(matrix, weights):
    return max(0., float((matrix @ weights).max()))


def calibrate(regrets, groups):
    """Fit a candidate and evaluate it in nested leave-one-network-out folds."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return PRIOR.copy(), None, [], dict(status='scenario_fallback',
            reasons=['at least three verified networks are needed for nested validation'])

    settings = (.1, .02, .005, 0.)
    all_indices = np.arange(len(regrets))
    @lru_cache(maxsize=None)
    def fit_cached(indices, penalty):
        return fit_weights([regrets[i] for i in indices], penalty)

    def fit(indices, penalty):
        return fit_cached(tuple(indices), penalty)

    def choose_penalty(indices):
        splits = [(indices[groups[indices] != network], indices[groups[indices] == network])
                  for network in np.unique(groups[indices])]
        def score(penalty):
            losses = []
            for train, test in splits:
                weights = fit(train, penalty)
                losses.extend(relocation_regret(regrets[i], weights) for i in test)
            return float(np.mean(losses))
        return min(settings, key=score)

    penalty = choose_penalty(all_indices)
    fitted = fit(all_indices, penalty)
    unregularized = fit(all_indices, 0.)
    checks = []
    for network in unique_groups:
        test = all_indices[groups == network]
        train = all_indices[groups != network]
        fold_penalty = choose_penalty(train)
        weights = fit(train, fold_penalty)
        unrestricted_weights = fit(train, 0.)
        for i in test:
            checks.append(dict(index=int(i), network=str(network), lambda_l1=fold_penalty,
                baseline=relocation_regret(regrets[i], PRIOR),
                held_out=relocation_regret(regrets[i], weights),
                unregularized_held_out=relocation_regret(regrets[i], unrestricted_weights),
                fitted=relocation_regret(regrets[i], fitted), weights=weights))

    baseline = float(np.mean([r['baseline'] for r in checks]))
    score = float(np.mean([r['held_out'] for r in checks]))
    fold_weights = np.array([r['weights'] for r in checks])
    reasons = []
    if len(regrets) < 12:
        reasons.append('fewer than 12 verified clubs inside Maryino')
    if baseline <= 1e-10 or score > baseline * .95:
        reasons.append('less than 5% improvement in nested validation')
    diagnostics = dict(status='scenario_fallback' if reasons else 'indirectly_calibrated',
        reasons=reasons, observations=len(regrets), networks=len(unique_groups),
        validation='nested leave-one-network-out; competitors remain fixed in each relocation',
        lambda_l1=penalty, baseline_regret=baseline, cv_regret=score,
        fitted_regret=float(np.mean([r['fitted'] for r in checks])),
        unregularized_weights=dict(zip(FEATURES, unregularized.tolist())),
        unregularized_fitted_regret=float(np.mean([relocation_regret(r, unregularized) for r in regrets])),
        unregularized_cv_regret=float(np.mean([r['unregularized_held_out'] for r in checks])),
        relative_cv_improvement=(baseline-score)/baseline if baseline > 1e-10 else None,
        held_out_clubs_improved=sum(r['held_out'] < r['baseline'] - 1e-10 for r in checks),
        cv_weight_min=dict(zip(FEATURES, fold_weights.min(axis=0).tolist())),
        cv_weight_max=dict(zip(FEATURES, fold_weights.max(axis=0).tolist())))
    return PRIOR.copy() if reasons else fitted, fitted, checks, diagnostics


def save_csv(path,rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['seed-clubs','run','economy','legacy'])
    ap.add_argument('--clubs')
    ap.add_argument('--output')
    ap.add_argument('--grid-step', type=int, choices=(100, 200), default=100)
    ap.add_argument('--buffer-m', type=int, choices=(1000, 3000), default=3000)
    ap.add_argument('--network',default=None)
    ap.add_argument('--allow-unverified',action='store_true')
    args=ap.parse_args()
    if args.command == 'economy' or (args.command == 'run' and args.network in (None, 'DDX') and not args.clubs and not args.allow_unverified):
        if args.clubs or args.allow_unverified or args.network not in (None, 'DDX'):
            ap.error('economy uses the verified DDX market and its official catalogs')
        from economy_model import run
        report = run(args.output or OUT/'economy', args.grid_step, args.buffer_m)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    default_clubs = 'data/clubs.csv'
    if args.command in ('run', 'legacy') and (ROOT/'data/clubs_reviewed.csv').exists():
        default_clubs = 'data/clubs_reviewed.csv'
    club_path = ROOT / (args.clubs or default_clubs)
    output = ROOT / (args.output or OUT)
    if args.command=='seed-clubs':
        club_path.parent.mkdir(parents=True,exist_ok=True);seed_clubs(club_path);return
    clubs,club_xy=load_clubs(club_path)
    if args.buffer_m < 3000:
        if any(r.get('scope') == 'buffer' and not r.get('distance_m') for r in clubs):
            raise ValueError('Buffer sensitivity requires distance_m for external clubs')
        indices = [i for i, r in enumerate(clubs) if r.get('scope') != 'buffer'
                   or float(r['distance_m']) <= args.buffer_m]
        clubs = [clubs[i] for i in indices]
        club_xy = club_xy[indices]
    unverified=[r['club_id'] for r in clubs if r['verified'].lower() not in ('yes','true','1')]
    if unverified and not args.allow_unverified:
        raise ValueError(f'{len(unverified)} unverified clubs; use the reviewed catalog or --allow-unverified')
    if len(clubs) < 2:
        raise ValueError('At least two eligible clubs are required')
    ring,nodes,ways=load_osm()
    cells=grid(ring, args.grid_step)
    components,allowed,counts=build_features(cells,nodes,ways)
    candidates=cells[allowed]
    # For each relocation, preserve actual position in the action set.
    networks=np.array([r['network'] for r in clubs])
    candidate_sets=[]; regrets=[]
    inside_indices=[i for i,r in enumerate(clubs) if r.get('scope','inside')=='inside']
    if not inside_indices:
        raise ValueError('At least one eligible club inside Maryino is required')
    for i in inside_indices:
        choices=np.vstack((candidates,club_xy[i]))
        matrix=gains(cells,components,club_xy,networks,i,choices)
        candidate_sets.append(choices)
        regrets.append(matrix-matrix[-1])
    calibration_indices = [k for k, i in enumerate(inside_indices)
                           if clubs[i]['club_id'] not in unverified
                           and clubs[i].get('calibration_eligible', 'yes') == 'yes']
    weights, fitted, checks, diagnostics = calibrate(
        [regrets[k] for k in calibration_indices],
        [networks[inside_indices[k]] for k in calibration_indices])
    if unverified or fitted is None:
        weights = PRIOR.copy()
        if unverified:
            diagnostics['status'] = 'scenario_fallback'
            diagnostics['reasons'].append(f'{len(unverified)} unverified competitors in market geometry')
    selected=args.network or Counter(networks[inside_indices]).most_common(1)[0][0]
    if selected not in networks: raise ValueError(f'Network {selected!r} absent')
    opening=opening_gains(cells,components,club_xy,networks,selected,candidates)
    increments=opening@weights
    best_idx=int(np.argmax(increments))
    owner, best_d2, count = nearest(cells, club_xy)
    winning = joining_share(((cells-candidates[best_idx])**2).sum(axis=1), best_d2, count)
    density=components@weights
    outside_demand_share = float(density @ owner[:, [r.get('scope') == 'buffer' for r in clubs]].sum(axis=1))
    gross_new = float(density @ winning)
    from_own = float((density * winning) @ owner[:, networks == selected].sum(axis=1))
    from_competitors = gross_new - from_own
    lonlat=geographic(candidates)
    ordering=np.argsort(-increments)
    opening_rows=[]
    for i in ordering:
        opening_rows.append(dict(lon=round(lonlat[i,0],7),lat=round(lonlat[i,1],7),
            incremental_share=round(float(increments[i]),8),
            net_at_cost_0_5pct=round(float(increments[i]-.005),8),
            net_at_cost_1pct=round(float(increments[i]-.01),8),
            net_at_cost_2pct=round(float(increments[i]-.02),8)))
    relocation=[]
    for k,i in enumerate(inside_indices):
        r=clubs[i]
        v=regrets[k]@weights;j=int(np.argmax(v));q=geographic(candidate_sets[k][j:j+1])[0]
        relocation.append(dict(club_id=r['club_id'],network=r['network'],name=r['name'],
           regret_share=round(max(0.,float(v[j])),8),best_lon=round(q[0],7),best_lat=round(q[1],7)))
    network_regret=[]
    for net in sorted({r['network'] for r in relocation}):
        group=[r for r in relocation if r['network']==net]
        win=max(group,key=lambda r:r['regret_share'])
        network_regret.append(dict(network=net,regret_share=win['regret_share'],
                                   movable_club_id=win['club_id'],best_lon=win['best_lon'],best_lat=win['best_lat']))
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output/'opening_candidates.csv',opening_rows)
    save_csv(output/'relocation_clubs.csv',relocation)
    save_csv(output/'equilibrium_networks.csv',network_regret)
    scenario_rows=[]
    scenarios = [('housing_heavy',np.array([.75,.10,.10,.05])),
                    ('baseline',PRIOR),('metro_heavy',np.array([.40,.35,.15,.10])),
                    ('retail_office_heavy',np.array([.40,.10,.25,.25]))]
    if fitted is not None:
        scenarios.append(('fitted', fitted))
    for label,a in scenarios:
        ix=int(np.argmax(opening@a));ll=lonlat[ix]
        scenario_rows.append(dict(scenario=label,lon=round(ll[0],7),lat=round(ll[1],7),
                                  incremental_share=round(float(opening[ix]@a),8)))
    save_csv(output/'scenarios.csv',scenario_rows)
    validation = []
    for check in checks:
        k = calibration_indices[check['index']]
        i = inside_indices[k]
        # The exact observed site is only the reference for regret. Exclude it from location prediction.
        predicted = candidates[np.argmax(regrets[k][:-1] @ check['weights'])]
        baseline = candidates[np.argmax(regrets[k][:-1] @ PRIOR)]
        validation.append(dict(club_id=clubs[i]['club_id'], network=check['network'],
            baseline_regret=check['baseline'], held_out_regret=check['held_out'],
            fitted_regret=check['fitted'], unregularized_held_out_regret=check['unregularized_held_out'],
            lambda_l1=check['lambda_l1'],
            distance_m=float(np.linalg.norm(predicted-club_xy[i])),
            baseline_distance_m=float(np.linalg.norm(baseline-club_xy[i])),
            **dict(zip(FEATURES, check['weights']))))
    if validation:
        save_csv(output/'calibration_validation.csv', validation)
        diagnostics['median_held_out_distance_m'] = float(np.median([r['distance_m'] for r in validation]))
        diagnostics['median_baseline_distance_m'] = float(np.median([r['baseline_distance_m'] for r in validation]))
        diagnostics['fitted_weights'] = dict(zip(FEATURES, fitted.tolist()))
    else:
        (output/'calibration_validation.csv').unlink(missing_ok=True)
    fitted_density = components @ fitted if fitted is not None else None
    with open(output/'density.geojson','w',encoding='utf-8') as f:
        xy=geographic(cells)
        json.dump(dict(type='FeatureCollection',features=[dict(type='Feature',geometry=dict(type='Point',coordinates=[float(x),float(y)]),
            properties=dict(demand_share=float(v), **({'fitted_share': float(fitted_density[i])}
                if fitted is not None else {}))) for i, ((x,y),v) in enumerate(zip(xy,density))]),f,ensure_ascii=False)
    report=dict(club_catalog=str(club_path.relative_to(ROOT)) if club_path.is_relative_to(ROOT) else str(club_path),
                grid_step_m=args.grid_step, buffer_m=args.buffer_m,
                clubs=len(clubs),inside_clubs=len(inside_indices),buffer_clubs=len(clubs)-len(inside_indices),
                unverified=len(unverified),cells=len(cells),candidates=len(candidates),
                feature_sources=counts,weights=dict(zip(FEATURES,map(float,weights))),calibration=diagnostics,
                target_network=selected,best_opening=opening_rows[0],
                new_club_gross_share=gross_new,from_competitors_share=from_competitors,
                cannibalized_own_share=from_own,
                baseline_share_served_by_buffer_clubs=outside_demand_share,
                break_even_F_over_mM=float(increments[best_idx]),
                max_relocation_regret_share=max(x['regret_share'] for x in network_regret),
                approximate_relocation_equilibrium=(max(x['regret_share'] for x in network_regret)<=.01)
                    if not unverified else None,
                note='Normalized market M=1, m=1. Confirmed competitors in the selected buffer are included, '
                     'but demand is modeled only inside Maryino. Opening location is a search cell, not a confirmed premise. '
                     'Relocation regret covers inside clubs only and assumes fixed club count and costs that cancel.')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
