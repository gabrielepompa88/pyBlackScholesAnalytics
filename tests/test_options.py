import unittest
import copy
import numpy as np
import numpy.testing as np_test
import pandas as pd
import pandas.testing as pd_test

from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.options.options import PlainVanillaOption, DigitalOption
from pyblackscholesanalytics.utils.utils import scalarize


# TODO: improve test coverage
class TestPlainVanillaOption(unittest.TestCase):

    def setUp(self) -> None:
        # common market environment
        mkt_env = MarketEnvironment()

        # option objects
        self.call_opt = PlainVanillaOption(mkt_env)
        self.put_opt = PlainVanillaOption(mkt_env, option_type="put")

        # pricing parameters
        S_scalar = 100
        S_vector = [90, 100, 110]
        t_scalar_string = "01-06-2020"
        t_date_range = pd.date_range(start="2020-04-19", end="2020-12-21", periods=5)
        self.vector_params = {"S": S_vector, "t": t_date_range, "np_output": True}
        self.scalar_params = {"S": S_scalar, "t": t_scalar_string, "np_output": True}

    def test_price_scalar(self):
        """Test price - scalar case"""

        # call
        test_call = scalarize(self.call_opt.price(**self.scalar_params))
        expected_call = 7.548381716811839
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.price(**self.scalar_params))
        expected_put = 4.672730506407959
        self.assertEqual(test_put, expected_put)

    def test_price_vector_np(self):
        """Test price - np.ndarray output case"""

        # call
        test_call = self.call_opt.price(**self.vector_params)
        expected_call = np.array([[3.48740247e+00, 8.42523213e+00, 1.55968082e+01],
                                  [2.53045128e+00, 7.14167587e+00, 1.43217796e+01],
                                  [1.56095778e+00, 5.72684668e+00, 1.29736886e+01],
                                  [5.89165298e-01, 4.00605304e+00, 1.14939139e+01],
                                  [7.21585753e-04, 1.38927959e+00, 1.01386434e+01]])
        np_test.assert_allclose(test_call, expected_call)

        # put
        test_put = self.put_opt.price(**self.vector_params)
        expected_put = np.array([[1.00413306e+01, 4.97916024e+00, 2.15073633e+00],
                                 [9.90791873e+00, 4.51914332e+00, 1.69924708e+00],
                                 [9.75553655e+00, 3.92142545e+00, 1.16826738e+00],
                                 [9.62127704e+00, 3.03816479e+00, 5.26025639e-01],
                                 [9.86382907e+00, 1.25238707e+00, 1.75090342e-03]])
        np_test.assert_allclose(test_put, expected_put)

    def test_price_vector_df(self):
        """Test price - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.price(**self.vector_params)
        expected_call = pd.DataFrame(data=[[3.48740247e+00, 8.42523213e+00, 1.55968082e+01],
                                           [2.53045128e+00, 7.14167587e+00, 1.43217796e+01],
                                           [1.56095778e+00, 5.72684668e+00, 1.29736886e+01],
                                           [5.89165298e-01, 4.00605304e+00, 1.14939139e+01],
                                           [7.21585753e-04, 1.38927959e+00, 1.01386434e+01]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.price(**self.vector_params)
        expected_put = pd.DataFrame(data=[[1.00413306e+01, 4.97916024e+00, 2.15073633e+00],
                                          [9.90791873e+00, 4.51914332e+00, 1.69924708e+00],
                                          [9.75553655e+00, 3.92142545e+00, 1.16826738e+00],
                                          [9.62127704e+00, 3.03816479e+00, 5.26025639e-01],
                                          [9.86382907e+00, 1.25238707e+00, 1.75090342e-03]],
                                    index=self.vector_params["t"],
                                    columns=self.vector_params["S"])

        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put)

    def test_delta_scalar(self):
        """Test Delta - scalar case"""

        # call
        test_call = scalarize(self.call_opt.delta(**self.scalar_params))
        expected_call = 0.6054075531684143
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.delta(**self.scalar_params))
        expected_put = -0.3945924468315857
        self.assertEqual(test_put, expected_put)

    def test_delta_vector_np(self):
        """Test Delta - np.ndarray output case"""

        # call
        test_call = self.call_opt.delta(**self.vector_params)
        expected_call = np.array([[3.68466757e-01, 6.15283790e-01, 8.05697003e-01],
                                  [3.20097309e-01, 6.00702480e-01, 8.18280131e-01],
                                  [2.54167521e-01, 5.83663527e-01, 8.41522350e-01],
                                  [1.49152172e-01, 5.61339299e-01, 8.91560577e-01],
                                  [8.89758553e-04, 5.23098767e-01, 9.98343116e-01]])
        np_test.assert_allclose(test_call, expected_call)

        # put
        test_put = self.put_opt.delta(**self.vector_params)
        expected_put = np.array([[-0.63153324, -0.38471621, -0.194303],
                                 [-0.67990269, -0.39929752, -0.18171987],
                                 [-0.74583248, -0.41633647, -0.15847765],
                                 [-0.85084783, -0.4386607,  -0.10843942],
                                 [-0.99911024, -0.47690123, -0.00165688]])
        np_test.assert_allclose(test_put, expected_put, rtol=5e-6)

    def test_delta_vector_df(self):
        """Test Delta - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.delta(**self.vector_params)
        expected_call = pd.DataFrame(data=[[3.68466757e-01, 6.15283790e-01, 8.05697003e-01],
                                           [3.20097309e-01, 6.00702480e-01, 8.18280131e-01],
                                           [2.54167521e-01, 5.83663527e-01, 8.41522350e-01],
                                           [1.49152172e-01, 5.61339299e-01, 8.91560577e-01],
                                           [8.89758553e-04, 5.23098767e-01, 9.98343116e-01]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.delta(**self.vector_params)
        expected_put = pd.DataFrame(data=[[-0.63153324, -0.38471621, -0.194303],
                                          [-0.67990269, -0.39929752, -0.18171987],
                                          [-0.74583248, -0.41633647, -0.15847765],
                                          [-0.85084783, -0.4386607,  -0.10843942],
                                          [-0.99911024, -0.47690123, -0.00165688]],
                                    index=self.vector_params["t"],
                                    columns=self.vector_params["S"])

        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put)

    def test_gamma_scalar(self):
        """Test Gamma - scalar case"""

        # call
        test_call = scalarize(self.call_opt.gamma(**self.scalar_params))
        expected_call = 0.025194958512498786
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.gamma(**self.scalar_params))
        expected_put = copy.deepcopy(expected_call)
        self.assertEqual(test_put, expected_put)

        # assert call and put gamma coincide
        self.assertEqual(test_call, test_put)

    def test_gamma_vector_np(self):
        """Test Gamma - np.ndarray output case"""

        # call
        test_call = self.call_opt.gamma(**self.vector_params)
        expected_call = np.array([[0.02501273, 0.02281654, 0.01493167],
                                  [0.02725456, 0.02648423, 0.01645793],
                                  [0.02950243, 0.03231528, 0.01820714],
                                  [0.02925862, 0.0446913, 0.01918121],
                                  [0.00101516, 0.12030889, 0.00146722]])
        np_test.assert_allclose(test_call, expected_call, rtol=5e-6)

        # put
        test_put = self.put_opt.gamma(**self.vector_params)
        expected_put = copy.deepcopy(expected_call)
        np_test.assert_allclose(test_put, expected_put, rtol=5e-6)

        # assert call and put gamma coincide
        np_test.assert_allclose(test_call, test_put)

    def test_gamma_vector_df(self):
        """Test Gamma - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.gamma(**self.vector_params)
        expected_call = pd.DataFrame(data=[[0.02501273, 0.02281654, 0.01493167],
                                           [0.02725456, 0.02648423, 0.01645793],
                                           [0.02950243, 0.03231528, 0.01820714],
                                           [0.02925862, 0.0446913, 0.01918121],
                                           [0.00101516, 0.12030889, 0.00146722]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.gamma(**self.vector_params)
        expected_put = copy.deepcopy(expected_call)
        pd_test.assert_frame_equal(test_put, expected_put)

        # assert call and put gamma coincide
        pd_test.assert_frame_equal(test_call, test_put)

    def test_vega_scalar(self):
        """Test vega - scalar case"""

        # call
        test_call = scalarize(self.call_opt.vega(**self.scalar_params))
        expected_call = 0.29405622811847903
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.vega(**self.scalar_params))
        expected_put = copy.deepcopy(expected_call)
        self.assertEqual(test_put, expected_put)

        # assert call and put vega coincide
        self.assertEqual(test_call, test_put)

    def test_vega_vector_np(self):
        """Test Vega - np.ndarray output case"""

        # call
        test_call = self.call_opt.vega(**self.vector_params)
        expected_call = np.array([[0.28419942, 0.32005661, 0.2534375],
                                  [0.23467293, 0.28153094, 0.21168961],
                                  [0.17415326, 0.23550311, 0.16055207],
                                  [0.09220072, 0.17386752, 0.09029355],
                                  [0.00045056, 0.06592268, 0.00097279]])
        np_test.assert_allclose(test_call, expected_call, rtol=1e-5)

        # put
        test_put = self.put_opt.vega(**self.vector_params)
        expected_put = copy.deepcopy(expected_call)
        np_test.assert_allclose(test_put, expected_put, rtol=1e-5)

        # assert call and put vega coincide
        np_test.assert_allclose(test_call, test_put)

    def test_vega_vector_df(self):
        """Test Vega - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.vega(**self.vector_params)
        expected_call = pd.DataFrame(data=[[0.28419942, 0.32005661, 0.2534375],
                                           [0.23467293, 0.28153094, 0.21168961],
                                           [0.17415326, 0.23550311, 0.16055207],
                                           [0.09220072, 0.17386752, 0.09029355],
                                           [0.00045056, 0.06592268, 0.00097279]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)

        # put
        test_put = self.put_opt.vega(**self.vector_params)
        expected_put = copy.deepcopy(expected_call)
        pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)

        # assert call and put vega coincide
        pd_test.assert_frame_equal(test_call, test_put)


if __name__ == '__main__':
    unittest.main()
