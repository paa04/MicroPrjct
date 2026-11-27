import unittest

import numpy as np

from spatial_model import PRIOR, calibrate, fit_weights, gains, nearest, opening_gains


class MarketTests(unittest.TestCase):
    def setUp(self):
        self.cells=np.array([[0.,0.],[12.,0.],[20.,0.]])
        self.components=np.tile(np.array([1/3]*3)[:,None],(1,4))

    def test_opening_counts_competitor_capture_only(self):
        clubs=np.array([[0.,0.],[20.,0.]])
        networks=np.array(['A','B'])
        result=opening_gains(self.cells,self.components,clubs,networks,'A',np.array([[11.,0.]]))
        np.testing.assert_allclose(result[0],np.array([1/3]*4))

    def test_cannibalization_does_not_raise_network_demand(self):
        clubs=np.array([[0.,0.],[20.,0.]])
        networks=np.array(['A','A'])
        result=opening_gains(self.cells,self.components,clubs,networks,'A',np.array([[11.,0.]]))
        np.testing.assert_allclose(result,0)

    def test_relocation_at_original_point_has_zero_gain(self):
        clubs=np.array([[0.,0.],[20.,0.]])
        networks=np.array(['A','B'])
        result=gains(self.cells,self.components,clubs,networks,0,clubs[:1])
        np.testing.assert_allclose(result,0)

    def test_fit_simplex_and_prior_regularization(self):
        regrets=[np.array([[.1,-.1,-.1,-.1]]) for _ in range(3)]
        result=fit_weights(regrets,10.)
        np.testing.assert_allclose(result,PRIOR)
        self.assertAlmostEqual(result.sum(),1)

    def test_tied_candidates_split_demand_equally(self):
        cell = np.array([[10., 0.]])
        clubs = np.array([[0., 0.], [20., 0.]])
        networks = np.array(['A', 'B'])
        shares, _, _ = nearest(cell, clubs)
        np.testing.assert_allclose(shares, [[.5, .5]])
        np.testing.assert_allclose(
            opening_gains(cell, np.ones((1, 4)), clubs, networks, 'A', clubs[:1]),
            [[1/6]*4],
        )
        np.testing.assert_allclose(gains(cell, np.ones((1, 4)), clubs, networks, 0, clubs[:1]), 0)

    def test_small_verified_sample_retains_fitted_candidate_and_baseline(self):
        matrices = [np.array([[.3, -.1, -.1, -.1], [0., 0., 0., 0.]])] * 4
        weights, fitted, checks, diagnostics = calibrate(matrices, ['A', 'A', 'B', 'C'])
        self.assertEqual(diagnostics['status'], 'scenario_fallback')
        self.assertEqual(len(checks), 4)
        np.testing.assert_allclose(weights, PRIOR)
        self.assertAlmostEqual(fitted.sum(), 1)
        self.assertLess(diagnostics['cv_regret'], diagnostics['baseline_regret'])


if __name__=='__main__': unittest.main()
