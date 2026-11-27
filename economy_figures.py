"""District-scale figures for the DDX study. All geography comes from the saved run."""
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib import patheffects
import numpy as np

from economy_model import coordinates, records
from spatial_model import ROOT, joining_share, metric, nearest

INK = '#173333'
MUTED = '#4f6465'
DDX = '#53367b'
RIVAL = '#8f4425'
NEW = '#ffbb39'
GAIN = '#267c70'
OWN = '#a7cebd'
PAPER = '#fbfaf6'


@dataclass
class District:
    folder: Path
    report: dict
    cells: np.ndarray
    mass: np.ndarray
    ring: np.ndarray
    clubs: list

    @classmethod
    def load(cls, folder):
        report = json.loads((folder/'report.json').read_text())
        features = json.loads((folder/'density.geojson').read_text())['features']
        ring = json.loads((ROOT/'data/maryino/boundary.geojson').read_text())['features'][0]['geometry']
        return cls(folder, report, metric([f['geometry']['coordinates'] for f in features]),
                   np.array([f['properties']['demand_share'] for f in features]),
                   metric(ring['coordinates'][0]), records(folder/'competition.csv'))

    @property
    def candidate(self):
        return min(self.report['recommendations'], key=lambda r: r['shortlist'])

    @property
    def site(self):
        # The report rounds WGS84 for display. Use the original cell centre for market shares.
        target = coordinates([self.candidate])[0]
        return self.cells[np.argmin(np.linalg.norm(self.cells-target, axis=1))]


def ru(value, digits=1):
    return f'{value:.{digits}f}'.replace('.', ',')


def page(d, title, subtitle):
    fig = plt.figure(figsize=(12.8, 10.1), facecolor=PAPER)
    fig.text(.045, .945, title, fontsize=24, weight='bold', color=INK)
    fig.text(.046, .908, subtitle, fontsize=12, color=MUTED)
    ax = fig.add_axes([.025, .264, .95, .628])
    panel = fig.add_axes([.046, .079, .918, .161])
    panel.set_axis_off()
    ring = d.ring
    ax.set_xlim(ring[:, 0].min()-100, ring[:, 0].max()+100)
    ax.set_ylim(ring[:, 1].min()-165, ring[:, 1].max()+120)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.fill(*ring.T, color='#e7edeb', zorder=0)
    ax.plot(*ring.T, color='#304c53', lw=1.55, zorder=2)
    boundary_key = Line2D([], [], color='#304c53', lw=1.55, label='Граница района Марьино')
    ax.legend(handles=[boundary_key], loc='upper left', frameon=False, fontsize=10.5, labelcolor=INK)
    left, bottom = ax.get_xlim()[0], ax.get_ylim()[0]
    x, y = left+180, bottom+85
    ax.plot([x, x+1000], [y, y], lw=2, color=INK)
    ax.text(x+500, y+55, '1 км', ha='center', color=INK, fontsize=9)
    ax.annotate('С', xy=(ax.get_xlim()[1]-120, ax.get_ylim()[1]-110), xytext=(ax.get_xlim()[1]-120, ax.get_ylim()[1]-290),
                ha='center', va='top', fontsize=9, color=INK,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1))
    fig.text(.046, .024, f"Марьино, Москва  ·  {d.report['snapshot']}  ·  © OpenStreetMap contributors, ODbL",
             fontsize=9.5, color=MUTED)
    return fig, ax, panel


def tile_layer(ax, d, colors=None, values=None):
    step = d.report['grid_m']
    tiles = [Rectangle((x-step/2, y-step/2), step, step) for x, y in d.cells]
    pixels = PatchCollection(tiles, linewidths=0, antialiased=False, facecolors=colors, cmap='YlGnBu')
    if values is not None:
        pixels.set_array(values)
        pixels.set_clim(0, values.max())
    pixels.set_clip_path(ax.patches[0])
    ax.add_collection(pixels)
    return pixels


def point(ax, xy, marker, color, size, edge='white', z=5):
    ax.scatter(*np.asarray(xy).reshape(-1, 2).T, s=size, marker=marker, c=color,
               edgecolors=edge, linewidths=1.2, zorder=z)


def club_marks(ax, clubs):
    for own, marker, color in [(True, 'o', DDX), (False, '^', RIVAL)]:
        rows = [r for r in clubs if (r['network']=='DDX')==own]
        if rows:
            point(ax, coordinates(rows), marker, color, 68)


def label(ax, xy, delta, text):
    ax.annotate(text, xy, xytext=xy+np.asarray(delta), ha='left' if delta[0]>=0 else 'right',
                va='center', color=INK, fontsize=10.3, weight='medium', zorder=7,
                bbox=dict(boxstyle='round,pad=.20', fc=PAPER, ec='none', alpha=.93),
                arrowprops=dict(arrowstyle='-', lw=.85, color=INK, shrinkA=0, shrinkB=5))


def text(panel, x, y, value, size=12, weight='normal', color=INK):
    panel.text(x, y, value, fontsize=size, weight=weight, color=color, transform=panel.transAxes,
               ha='left', va='top', linespacing=1.4)


def symbol(marker, color, title):
    return Line2D([], [], marker=marker, markerfacecolor=color, markeredgecolor='white',
                  lw=0, markersize=10, label=title)


def legend(panel, handles, x, y, columns=1):
    panel.legend(handles=handles, ncol=columns, columnspacing=1.1, bbox_to_anchor=(x, y), loc='upper left', borderaxespad=0,
                 frameon=False, fontsize=10.7, labelspacing=.45, handlelength=1.25, labelcolor=INK)


def store(fig, folder, name):
    fig.savefig(folder/name, dpi=175, facecolor=fig.get_facecolor())
    plt.close(fig)


def density(d):
    fig, ax, panel = page(d, 'Где в Марьино сосредоточен спрос на фитнес',
                          'Расчётная плотность спроса на клубы эконом-сегмента и городские ориентиры')
    values = 100*d.mass/(d.report['grid_m']/1000)**2
    layer = tile_layer(ax, d, values=values)
    poi = records(ROOT/'data/maryino/poi.csv')
    malls = [r for r in poi if r['category']=='shopping_mall' or r['osm_id'] in ('27411001', '679606869')]
    offices = [r for r in poi if r['category']=='office_building']
    metros = [r for r in poi if r['category']=='metro_station']
    point(ax, coordinates(offices), 'P', '#667a7b', 40)
    point(ax, coordinates(malls), 's', '#fffdf7', 50, edge=INK)
    club_marks(ax, d.clubs)
    point(ax, coordinates(metros), 'D', '#d35b48', 62)
    # Callouts are displaced in metres to keep nearby venues and metro stations legible.
    callouts = {
        'Марьино': ((580, 140), 'м. Марьино'),
        'Братиславская': ((460, 350), 'м. Братиславская'),
        'Марьинский пассаж': ((-420, 220), 'Марьинский пассаж'),
        'Люблю Молл': ((-300, 520), 'Люблю Молл'),
        'Братиславский рынок': ((470, 50), 'Братиславский рынок'),
        'МариЭль': ((520, -310), 'МариЭль'),
        'Место встречи Экран': ((-270, 330), 'РЦ Экран'),
        'Торговый корпус "Евромаг"': ((-220, -240), 'Евромаг'),
    }
    for r in malls+metros:
        if r['name'] == d.candidate['landmark']:
            continue
        if r['osm_id']=='679606869':
            label(ax, coordinates([r])[0], (220, 140), 'ТРК MARi')
        elif r['name'] in callouts:
            label(ax, coordinates([r])[0], *callouts[r['name']])
    point(ax, d.site, '*', NEW, 350, edge=INK, z=8)
    label(ax, d.site, (-360, 190), f"Новый DDX\nу «{d.candidate['landmark']}»")
    text(panel, 0, .99, 'Плотность спроса', 14, 'bold')
    text(panel, 0, .74, 'Доля спроса района на 1 км², %', 11, color=MUTED)
    bar = fig.colorbar(layer, cax=panel.inset_axes([0, .39, .32, .085]), orientation='horizontal',
                       format=FuncFormatter(lambda x, _: f'{x:g}'.replace('.', ',')))
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=9, colors=MUTED, length=2)
    text(panel, .40, .99, 'Ориентиры', 14, 'bold')
    legend(panel, [symbol('D', '#d35b48', 'Станция метро'),
                   Line2D([], [], marker='s', lw=0, markerfacecolor='white', markeredgecolor=INK,
                          markersize=8, label='ТЦ или районный центр'),
                   symbol('P', '#667a7b', 'Офисное здание'),
                   symbol('o', DDX, 'Действующий DDX'), symbol('^', RIVAL, 'Конкурент эконом-сегмента'),
                   symbol('*', NEW, 'Предлагаемый DDX')], .395, .74, 2)
    text(panel, 0, .10, f"Сетка {d.report['grid_m']} м.",
         10.5, color=MUTED)
    fig.text(.046, .050, 'В торговых центрах с действующим клубом поверх квадрата показан знак клуба.',
             fontsize=10.5, color=MUTED)
    store(fig, d.folder, 'density_poi.png')


def entry(d):
    fig, ax, panel = page(d, 'Как изменится доля DDX после открытия',
                          f"Сценарий одного нового клуба у ориентира «{d.candidate['landmark']}»  ·  {d.candidate['address']}")
    owners, best, ties = nearest(d.cells, coordinates(d.clubs))
    ddx = np.array([c['network']=='DDX' for c in d.clubs])
    own_before = owners[:, ddx].sum(axis=1)
    won = joining_share(((d.cells-d.site)**2).sum(axis=1), best, ties)
    gain = 100*np.dot(d.mass, won*(1-own_before))
    moved = 100*np.dot(d.mass, won*own_before)
    before = 100*np.dot(d.mass, own_before)
    if not np.allclose([gain, moved], [d.candidate['net_gain_pp'], d.candidate['own_transfer_pp']], atol=.002):
        raise ValueError('The scenario and saved recommendation have inconsistent market shares')
    # On tied cells blend the three source classes in proportion to their allocation.
    weights = np.column_stack([1-won, won*(1-own_before), won*own_before])
    tiles = weights @ np.array([to_rgb(c) for c in ['#e3e9e6', GAIN, OWN]])
    tile_layer(ax, d, colors=tiles)
    club_marks(ax, d.clubs)
    point(ax, d.site, '*', NEW, 350, edge=INK, z=8)
    label(ax, d.site, (-600, 370), 'Новый DDX')
    for r in d.clubs:
        if r['scope']!='inside':
            continue
        xy = coordinates([r])[0]
        label(ax, xy, (340, 370) if r['network']=='DDX' and xy[1]>0 else
              ((-350, -260) if r['network']=='DDX' else (340, -240)), r['name'].replace(' Братиславская Люблю Молл', ' Люблю Молл'))
    text(panel, 0, .99, 'Доля сети в районе', 14, 'bold')
    text(panel, 0, .68, f'{ru(before)}%  →  {ru(before+gain)}%', 23, 'bold')
    text(panel, 0, .25, f'+{ru(gain, 2)} п.п. к доле DDX', 14, 'bold', GAIN)
    text(panel, .34, .99, 'Откуда придёт спрос', 14, 'bold')
    legend(panel, [Patch(fc=GAIN, label=f'{ru(gain, 2)} п.п. — от конкурентов'),
                   Patch(fc=OWN, label=f'{ru(moved, 2)} п.п. — от других DDX'),
                   Patch(fc='#e3e9e6', label='Остаётся у прежних клубов')], .335, .73)
    text(panel, .70, .99, f'Всего у нового клуба\n{ru(gain+moved, 2)}% спроса района.', 13.5, 'bold')
    text(panel, .70, .42, f"В расчёте все {len(d.clubs)} действующих клубов,\nвключая клубы за границей района.",
         10.5, color=MUTED)
    fig.text(.046, .050, 'Звезда — предлагаемое место. Цвет ячейки показывает, откуда новый клуб получит спрос по модели ближайшего клуба.',
             fontsize=10.5, color=MUTED)
    store(fig, d.folder, 'opening_impact.png')


def historical(d):
    validations = records(d.folder/'historical_validation.csv')
    local = {r['name']: r for r in validations if r['evaluation']=='district' and r['model']=='entry'}
    clubs = [r for r in d.clubs if r['scope']=='inside']
    fig, ax, panel = page(d, 'Как модель восстанавливает известные адреса',
                          'Каждый клуб по очереди скрыт. Линия соединяет реальный адрес с первой рекомендацией модели.')
    text(panel, 0, .99, 'Ошибка по прямой', 14, 'bold')
    errors = []
    for index, club in enumerate(clubs, 1):
        row = local[club['name']]
        if 'top1_lon' not in row:
            raise ValueError('Для координат исторического прогноза повторите python3 spatial_model.py economy')
        actual = coordinates([club])[0]
        predicted = metric([[row['top1_lon'], row['top1_lat']]])[0]
        distance = np.linalg.norm(actual-predicted)
        if not np.isclose(distance, float(row['top1_m']), atol=.01):
            raise ValueError('Inconsistent historical location for '+club['name'])
        errors.append(distance)
        offset = (235, 260) if actual[1]>0 else ((-240, -280) if club['network']=='DDX' else (340, -200))
        point(ax, actual, 'o', DDX if club['network']=='DDX' else RIVAL, 100)
        ax.plot(*np.stack([actual, predicted]).T, color=INK, lw=1.7, zorder=4)
        cross = ax.scatter(*predicted, marker='x', s=110, c=INK, linewidths=2.4, zorder=7)
        cross.set_path_effects([patheffects.Stroke(linewidth=4.5, foreground='white'), patheffects.Normal()])
        label(ax, actual, offset, club['name'].replace(' Братиславская Люблю Молл', ' Люблю Молл'))
        x = (index-1)*.22
        text(panel, x, .66, club['name'].replace(' Братиславская Люблю Молл', ' Люблю Молл'), 12)
        text(panel, x, .43, f'{distance:.0f} м', 15, 'bold')
    text(panel, 0, .11, f'Средняя ошибка  {np.mean(errors):.0f} м', 15, 'bold')
    legend(panel, [symbol('o', DDX, 'Реальный адрес DDX'), symbol('o', RIVAL, 'Реальный адрес конкурента'),
                   Line2D([], [], marker='x', lw=0, markersize=9, markeredgewidth=2, color=INK,
                          label='Прогноз для скрытого клуба')], .68, .99)
    fig.text(.046, .050, 'Марьино исключено из обучения. Проверка использует расположение остальных клубов и объектов на дату снимка.',
             fontsize=10.5, color=MUTED)
    store(fig, d.folder, 'historical_locations.png')


def voronoi_cell(sites, index, bounds):
    """Intersect the map rectangle with the nearer half-plane for every other site."""
    west, east, south, north = bounds
    vertices = np.array([(west, south), (east, south), (east, north), (west, north)])
    site = sites[index]
    for other in sites:
        direction = other-site
        if not np.any(direction):
            continue
        offsets = (vertices-(site+other)/2) @ direction
        clipped = []
        for i, end in enumerate(vertices):
            start = vertices[i-1]
            a, b = offsets[i-1], offsets[i]
            if (a <= 0) != (b <= 0):
                clipped.append(start + (end-start)*(a/(a-b)))
            if b <= 0:
                clipped.append(end)
        if not clipped:
            return np.empty((0, 2))
        vertices = np.asarray(clipped)
    return vertices


def zone_panel(ax, d, sites, owners, colors, numbers, proposed):
    xmin, ymin = d.ring.min(axis=0)-[95, 150]
    xmax, ymax = d.ring.max(axis=0)+[95, 110]
    bounds = (xmin, xmax, ymin, ymax)
    ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect='equal')
    ax.set_axis_off()
    border = Polygon(d.ring, closed=True, facecolor='none', edgecolor=INK, lw=1.45, zorder=4)
    ax.add_patch(border)
    for i in range(len(sites)):
        vertices = voronoi_cell(sites, i, bounds)
        if len(vertices) < 3:
            continue
        new = proposed and i==len(sites)-1
        area = Polygon(vertices, facecolor=colors[i], edgecolor='#713800' if new else '#56666a',
                       lw=2.8 if new else 1., zorder=3 if new else 2)
        ax.add_patch(area)
        area.set_clip_path(border)
        cells = d.cells[owners[:, i]>.5]
        if len(cells) and not new:
            centre = cells[np.argmin(np.linalg.norm(cells-cells.mean(axis=0), axis=1))]
            # Separate the zone number from the dot marking the actual club.
            ax.annotate(str(numbers[i]), centre, ha='center', va='center', weight='bold',
                        color=INK, fontsize=12, zorder=6,
                        bbox=dict(boxstyle='round,pad=.16', fc='white', alpha=.82, ec='none'))
    point(ax, sites[:-1] if proposed else sites, 'o', INK, 42)
    if proposed:
        point(ax, sites[-1], '*', '#fff3be', 250, edge='#572900', z=8)
    x, y = xmin+145, ymin+55
    ax.plot([x, x+1000], [y, y], c=INK, lw=2)
    ax.text(x+500, y+55, '1 км', ha='center', color=INK, fontsize=9)


def voronoi_maps(d):
    original = coordinates(d.clubs)
    expanded = np.vstack([original, d.site])
    before = nearest(d.cells, original)[0]
    after = nearest(d.cells, expanded)[0]
    before_pct = np.append(d.mass @ before, 0.)*100
    after_pct = d.mass @ after*100
    visible = np.flatnonzero(before_pct+after_pct > 0)
    colors = ['#d2dbdd']*len(expanded)
    palette = ['#aed5cc', '#a9c9e8', '#d4b9db', '#f0bcad', '#c1cd95', '#e1ce95']
    numbers = {}
    for j, i in enumerate(visible):
        colors[i] = palette[j % len(palette)]
        numbers[i] = j+1
    colors[-1] = NEW
    if not np.isclose(after_pct[-1], d.candidate['gross_pp'], atol=.001):
        raise ValueError('Voronoi demand disagrees with the saved entry scenario')
    for compare, filename in [(False, 'voronoi_after.png'), (True, 'voronoi_comparison.png')]:
        fig = plt.figure(figsize=(16, 7.5) if compare else (12.8, 10.1), facecolor=PAPER)
        title = 'Как меняются зоны клубов при открытии DDX' if compare else 'Зона нового DDX в Марьино'
        fig.text(.04, .94, title, size=24, weight='bold', color=INK)
        fig.text(.04, .898, f"Новый клуб у «{d.candidate['landmark']}»  ·  {d.candidate['address']}", size=13, color=MUTED)
        if compare:
            for k, (sites, owners, head) in enumerate([(original, before, 'До открытия'), (expanded, after, 'После открытия')]):
                left = .025 + .485*k
                fig.text(left+.015, .808, head, size=15, weight='bold', color=INK)
                zone_panel(fig.add_axes([left, .335, .465, .46]), d, sites, owners, colors, numbers, bool(k))
        else:
            zone_panel(fig.add_axes([.025, .256, .95, .625]), d, expanded, after, colors, numbers, True)
        fig.text(.04, .285 if compare else .215, 'Доля модельного спроса жителей Марьино, до → после открытия',
                 weight='bold', size=13, color=INK)
        names = [r['name'].replace('Братиславская Люблю Молл', 'Люблю Молл') for r in d.clubs]+['Новый DDX']
        cols = 3
        for j, i in enumerate(visible):
            x = .04+.32*(j%cols)
            y = (.210 if compare else .16)-(.07 if compare else .055)*(j//cols)
            code = '★' if i==len(expanded)-1 else str(numbers[i])
            fig.text(x, y, code, color=INK, weight='bold', fontsize=11,
                     bbox=dict(boxstyle='round,pad=.26', fc=colors[i], ec='none'))
            fig.text(x+.022, y, f'{names[i]}   {ru(before_pct[i])} → {ru(after_pct[i])}%', color=INK, size=11)
        fig.text(.04, .054, f"Точки — действующие клубы. Звезда — новый DDX. Границы проходят на равном расстоянии "
                 f"от клубов. Доли рассчитаны на сетке {d.report['grid_m']} м.", fontsize=10, color=MUTED)
        fig.text(.04, .023, f"Учитываются клубы в буфере {d.report['radius_m']/1000:g} км. "
                 f"Марьино, Москва  ·  {d.report['snapshot']}  ·  © OpenStreetMap contributors, ODbL", fontsize=9, color=MUTED)
        store(fig, d.folder, filename)


def draw_figures(folder):
    district = District.load(folder)
    density(district)
    entry(district)
    historical(district)
    voronoi_maps(district)
