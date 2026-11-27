import unittest

import numpy as np

from compare_walking import score_market


class WalkingEntryTests(unittest.TestCase):
    def test_tied_clubs_split_gross_demand_and_only_rival_transfer_is_a_gain(self):
        # One demand origin is shared by two incumbents. A third at the same walking
        # distance gets 1/3 of its demand, half transferred within the DDX network.
        score, net, gross, owners, wins = score_market(
            np.array([[250., 10.], [250., 20.]]), np.array([[250., 12.], [250., 0.]]),
            np.array([True, False]), np.array([.6, .4]), np.zeros(2), np.array([True, False]))
        np.testing.assert_allclose(owners.sum(axis=0), [1., 1.])
        np.testing.assert_allclose(wins[0], [1/3, 0])
        np.testing.assert_allclose(net, [.1, .1])
        np.testing.assert_allclose(gross, [.2, .6])
        self.assertTrue(np.isfinite(score[0]))
        self.assertEqual(score[1], -np.inf)
