"""Render the district demand and shortlisted zones from a completed DDX run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np

from economy_model import coordinates, records
from spatial_model import OUT, ROOT, metric


def draw(folder):
    report = json.loads((folder/'report.json').read_text())
    features = json.loads((folder/'density.geojson').read_text())['features']
    cells = metric([r['geometry']['coordinates'] for r in features])/1000
    step = report['grid_m']/1000
    mass = np.array([r['properties']['demand_share'] for r in features])*100/step**2
    boundary = json.loads((ROOT/'data/maryino/boundary.geojson').read_text())['features'][0]['geometry']
    ring = metric(boundary['coordinates'][0])/1000
    clubs = records(folder/'competition.csv')

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    tiles = [Rectangle(c-step/2, step, step) for c in cells]
    heat = PatchCollection(tiles, cmap='YlGnBu', norm=Normalize(0, mass.max()), edgecolor='face', linewidth=.15)
    heat.set_array(mass)
    ax.add_collection(heat)
    ax.plot(*ring.T, color='#375961', linewidth=.8)
    for own, marker, color, label in [(True, 'o', '#55206b', 'Действующий DDX'),
                                      (False, '^', '#ae3824', 'Конкурент эконом-сегмента')]:
        xy = coordinates([r for r in clubs if (r['network']=='DDX')==own])/1000
        ax.scatter(*xy.T, s=58, c=color, marker=marker, edgecolor='white', linewidth=1, label=label)
    for i, row in enumerate(report['recommendations']):
        xy = coordinates([row])[0]/1000
        ax.annotate(str(i+1), xy=xy, xytext=(0, 0), textcoords='offset points', ha='center', va='center',
                    weight='bold', color='black', bbox=dict(boxstyle='circle,pad=.22', fc='#ffc950', ec='#43371e'))
    ax.set_xlim(ring[:, 0].min()-.35, ring[:, 0].max()+.35)
    ax.set_ylim(ring[:, 1].min()-.48, ring[:, 1].max()+.33)
    ax.set_aspect('equal')
    ax.set_xlabel('км с запада на восток')
    ax.set_ylabel('км с юга на север')
    ax.legend(loc='lower left', fontsize=8, framealpha=.95)
    ax.set_title('Марьино · расчётная плотность спроса и три зоны для DDX', loc='left', weight='bold')
    bar = fig.colorbar(heat, ax=ax, shrink=.78, pad=.015)
    bar.set_label('Доля модельного спроса района, % на км²')
    caption = '   ·   '.join(f"{r['shortlist']}. {r['landmark']}" for r in report['recommendations'])
    fig.supxlabel(caption+'\nГеоданные © OpenStreetMap contributors (ODbL). Показаны клубы в пределах карты.', fontsize=10)
    fig.savefig(folder/'map.png', dpi=170, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=OUT/'economy')
    from economy_figures import draw_figures
    results = parser.parse_args().results
    draw(results)
    draw_figures(results)
