import unittest

import numpy as np

from maryino_price import choice


class PriceChoiceTests(unittest.TestCase):
    def test_transfers_match_direct_recalculation(self):
        # One incumbent DDX, one rival, outside utility equal to a 20-minute walk.
        args = (np.array([[10.], [15.]]), np.array([[12.]]), np.array([1900., 2900.]))
        own = np.array([True, False])
        unrestricted = np.zeros(2, bool)
        for k in (0, 6):
            weights = np.exp(-np.array([10., 15.+k, 20., 12.])/10)
            before = weights[:3]/weights[:3].sum()
            after = weights/weights.sum()
            flows = choice(*args, 1900, own, unrestricted, False, np.ones(1), k)
            np.testing.assert_allclose(flows[:, 0], before-after[:3])
            self.assertAlmostEqual(flows.sum(), after[3])
            costly = choice(*args, 2900, own, unrestricted, False, np.ones(1), k)
            if k == 0:
                self.assertAlmostEqual(costly.sum(), flows.sum())
            else:
                self.assertLess(costly.sum(), flows.sum())
