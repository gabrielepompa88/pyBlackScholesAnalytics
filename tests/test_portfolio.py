import unittest
import copy
import numpy as np
import pandas as pd
import pandas.testing as pd_test

from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.options.options import PlainVanillaOption
from pyblackscholesanalytics.portfolio.portfolio import Portfolio


class TestPortfolio(unittest.TestCase):
    """Class to test public methods of Portfolio class"""

    def setUp(self) -> None:

        # common market environment
        mkt_env = MarketEnvironment(t="01-06-2020")

        # underlying values to test
        S_vector = [60, 90, 120]

        # options maturities
        T_call = "31-12-2020"
        T_put = "30-06-2021"

        # time parameter
        t_range = pd.date_range(start=mkt_env.get_t(), end=T_call, periods=3)

        # pricing parameters
        self.params = {"S": S_vector, "t": t_range, "np_output": False}

        # options strikes
        K_put = 80
        K_call = 110

        # portfolio options positions
        self.call_pos = 2
        self.put_pos = -5

        # empty portfolio initialized
        self.ptf = Portfolio()

        # adding 2 long plain-vanilla call contracts
        self.call_opt = PlainVanillaOption(mkt_env, K=K_call, T=T_call)

        # adding 5 short plain-vanilla put contracts
        self.put_opt = PlainVanillaOption(mkt_env, option_type="put", K=K_put, T=T_put)

        # adding contracts to portfolio
        self.ptf.add_instrument(self.call_opt, self.call_pos)
        self.ptf.add_instrument(self.put_opt, self.put_pos)

    def test_portfolio_price(self):
        """Test that portfolio metrics matches algebraic sum of individual instruments' metrics.
        Metrics: price"""

        metrics = "price"

        test = getattr(self.ptf, metrics)(**self.params)
        expected = self.call_pos*getattr(self.call_opt, metrics)(**self.params) + \
                   self.put_pos*getattr(self.put_opt, metrics)(**self.params)

        pd_test.assert_frame_equal(test, expected)

    def test_portfolio_PnL(self):
        """Test that portfolio metrics matches algebraic sum of individual instruments' metrics.
        Metrics: PnL"""

        metrics = "PnL"

        test = getattr(self.ptf, metrics)(**self.params)
        expected = self.call_pos * getattr(self.call_opt, metrics)(**self.params) + \
                   self.put_pos * getattr(self.put_opt, metrics)(**self.params)

        pd_test.assert_frame_equal(test, expected)

    def test_portfolio_delta(self):
        """Test that portfolio metrics matches algebraic sum of individual instruments' metrics.
        Metrics: delta"""

        metrics = "delta"

        test = getattr(self.ptf, metrics)(**self.params)
        expected = self.call_pos * getattr(self.call_opt, metrics)(**self.params) + \
                   self.put_pos * getattr(self.put_opt, metrics)(**self.params)

        pd_test.assert_frame_equal(test, expected)

    def test_portfolio_gamma(self):
        """Test that portfolio metrics matches algebraic sum of individual instruments' metrics.
        Metrics: gamma"""

        metrics = "gamma"

        test = getattr(self.ptf, metrics)(**self.params)
        expected = self.call_pos * getattr(self.call_opt, metrics)(**self.params) + \
                   self.put_pos * getattr(self.put_opt, metrics)(**self.params)

        pd_test.assert_frame_equal(test, expected)

    def test_portfolio_vega(self):
        """Test that portfolio metrics matches algebraic sum of individual instruments' metrics.
        Metrics: vega"""

        metrics = "vega"

        test = getattr(self.ptf, metrics)(**self.params)
        expected = self.call_pos * getattr(self.call_opt, metrics)(**self.params) + \
                   self.put_pos * getattr(self.put_opt, metrics)(**self.params)

        pd_test.assert_frame_equal(test, expected)

    def test_portfolio_theta(self):
        """Test that portfolio metrics matches algebraic sum of individual instruments' metrics.
        Metrics: theta"""

        metrics = "theta"

        test = getattr(self.ptf, metrics)(**self.params)
        expected = self.call_pos * getattr(self.call_opt, metrics)(**self.params) + \
                   self.put_pos * getattr(self.put_opt, metrics)(**self.params)

        pd_test.assert_frame_equal(test, expected)

    def test_portfolio_rho(self):
        """Test that portfolio metrics matches algebraic sum of individual instruments' metrics.
        Metrics: rho"""

        metrics = "rho"

        test = getattr(self.ptf, metrics)(**self.params)
        expected = self.call_pos * getattr(self.call_opt, metrics)(**self.params) + \
                   self.put_pos * getattr(self.put_opt, metrics)(**self.params)

        pd_test.assert_frame_equal(test, expected)


if __name__ == '__main__':
    unittest.main()
