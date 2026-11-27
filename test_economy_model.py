import unittest
import numpy as np

from economy_model import (DATA, DENSITY, SITE, Event, coordinates, evaluate, fit,
                           local_market, records, zones)
from spatial_model import geographic


class EconomyTests(unittest.TestCase):
    def test_constant_site_features_do_not_cancel_density_learning(self):
        matrix = np.zeros((3, len(DENSITY)+len(SITE)))
        matrix[:, 0] = [0, 1, 2]
        xy = np.array([[0, 0], [500, 0], [1000, 0]])
        event = Event({'name': 'Example', 'opened': '2019-01-01'}, (0, 0), xy, matrix, 2, xy[2])
        model = fit([event]*3, .1)
        self.assertGreater(model.coef[0], 0)
        np.testing.assert_array_equal(model.coef[1:], 0)
        self.assertEqual(evaluate('joint', event, model.scores(matrix))['top1_m'], 0)
        self.assertEqual(zones(xy, model.scores(matrix)), [2, 0])

    def test_catalog_snapshot_and_segment(self):
        catalog = records(DATA/'catalog.csv')
        keys = [(c['network'], c['id']) for c in catalog]
        self.assertEqual(len(keys), len(set(keys)))
        as_of = (DATA/'catalog_date.txt').read_text().strip()
        self.assertTrue(all('2010' < c['opened'] <= as_of for c in catalog if c['network']=='DDX'))
        ring = geographic([[-6000, -6000], [6000, -6000], [6000, 6000], [-6000, 6000], [-6000, -6000]])
        market = local_market(catalog, ring, 3000)
        self.assertFalse(any(c['network']=='Kometa.fit' for c in market))
        self.assertTrue(any(c['name']=='Fitnation Домодедовская' for c in market))
        wide = local_market(catalog, ring, 3000, uncertain=True)
        self.assertGreater(len(wide), len(market))
        spirit = next(r for r in market if r['name']=='Spirit. Марьино')
        building = [{'lon':37.7449989, 'lat':55.6490758}]
        self.assertLess(np.linalg.norm(coordinates([spirit])-coordinates(building)), 30)


if __name__ == '__main__':
    unittest.main()
