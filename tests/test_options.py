import unittest
import copy
import numpy as np
import numpy.testing as np_test
import pandas as pd
import pandas.testing as pd_test
import warnings

from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.options.options import PlainVanillaOption, DigitalOption
from pyblackscholesanalytics.utils.utils import scalarize


class TestPlainVanillaOption(unittest.TestCase):

    def setUp(self) -> None:

        warnings.filterwarnings("ignore")

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

        # common pricing parameter setup
        common_params = {"np_output": True, "minimization_method": "Least-Squares"}

        # scalar parameters setup
        self.scalar_params = copy.deepcopy(common_params)
        self.scalar_params["S"] = S_scalar
        self.scalar_params["t"] = t_scalar_string

        # vector parameters setup
        self.vector_params = copy.deepcopy(common_params)
        self.vector_params["S"] = S_vector
        self.vector_params["t"] = t_date_range

        # complex pricing parameter setup
        # (S scalar, K and t vector, sigma distributed as Kxt grid, r distributed as Kxt grid)
        K_vector = [75, 85, 90, 95]
        mK = len(K_vector)
        n = 3
        sigma_grid_K = np.array([0.1 * (1 + i) for i in range(mK * n)]).reshape(n, mK)
        r_grid_K = np.array([0.01 * (1 + i) for i in range(mK * n)]).reshape(n, mK)
        self.complex_params = {"S": S_vector[0],
                               "K": K_vector,
                               "t": pd.date_range(start="2020-04-19", end="2020-12-21", periods=n),
                               "sigma": sigma_grid_K,
                               "r": r_grid_K,
                               "np_output": False}

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

    def test_PnL_scalar(self):
        """Test P&L - scalar case"""

        # call
        test_call = scalarize(self.call_opt.PnL(**self.scalar_params))
        expected_call = 4.060979245868182
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.PnL(**self.scalar_params))
        expected_put = -5.368600081057167
        self.assertEqual(test_put, expected_put)

    def test_PnL_vector_np(self):
        """Test P&L - np.ndarray output case"""

        # call
        test_call = self.call_opt.PnL(**self.vector_params)
        expected_call = np.array([[0.,           4.93782966, 12.10940574],
                                  [-0.95695119,  3.6542734,  10.83437716],
                                  [-1.92644469,  2.2394442,   9.48628613],
                                  [-2.89823717,  0.51865057,  8.00651142],
                                  [-3.48668089, -2.09812288,  6.65124095]])
        np_test.assert_allclose(test_call, expected_call)

        # put
        test_put = self.put_opt.PnL(**self.vector_params)
        expected_put = np.array([[0.,           -5.06217034,  -7.89059426],
                                 [-0.13341186,  -5.52218727,  -8.3420835 ],
                                 [-0.28579403,  -6.11990513,  -8.87306321],
                                 [-0.42005355,  -7.0031658,   -9.51530495],
                                 [-0.17750152,  -8.78894351, -10.03957968]])
        np_test.assert_allclose(test_put, expected_put)

    def test_PnL_vector_df(self):
        """Test P&L - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.PnL(**self.vector_params)
        expected_call = pd.DataFrame(data=[[0.,           4.93782966, 12.10940574],
                                           [-0.95695119,  3.6542734,  10.83437716],
                                           [-1.92644469,  2.2394442,   9.48628613],
                                           [-2.89823717,  0.51865057,  8.00651142],
                                           [-3.48668089, -2.09812288,  6.65124095]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.PnL(**self.vector_params)
        expected_put = pd.DataFrame(data=[[0.,           -5.06217034,  -7.89059426],
                                          [-0.13341186,  -5.52218727,  -8.3420835],
                                          [-0.28579403,  -6.11990513,  -8.87306321],
                                          [-0.42005355,  -7.0031658,   -9.51530495],
                                          [-0.17750152,  -8.78894351,  -10.03957968]],
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
        """Test Vega - scalar case"""

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

    def test_theta_scalar(self):
        """Test Theta - scalar case"""

        # call
        test_call = scalarize(self.call_opt.theta(**self.scalar_params))
        expected_call = -0.021064685979455443
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.theta(**self.scalar_params))
        expected_put = -0.007759980665812141
        self.assertEqual(test_put, expected_put)

    def test_theta_vector_np(self):
        """Test Theta - np.ndarray output case"""

        # call
        test_call = self.call_opt.theta(**self.vector_params)
        expected_call = np.array([[-0.01516655, -0.01977662, -0.01990399],
                                  [-0.01569631, -0.02176239, -0.0212802],
                                  [-0.01601397, -0.02491789, -0.02297484],
                                  [-0.01474417, -0.03162919, -0.02457737],
                                  [-0.00046144, -0.0728981, -0.01462746]])
        np_test.assert_allclose(test_call, expected_call, rtol=5e-4)

        # put
        test_put = self.put_opt.theta(**self.vector_params)
        expected_put = np.array([[-0.00193999, -0.00655005, -0.00667743],
                                 [-0.00235693, -0.00842301, -0.00794082],
                                 [-0.00256266, -0.01146658, -0.00952353],
                                 [-0.00117813, -0.01806315, -0.01101133],
                                 [0.01321844, -0.05921823, -0.00094758]])
        np_test.assert_allclose(test_put, expected_put, rtol=1e-5)

    def test_theta_vector_df(self):
        """Test Theta - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.theta(**self.vector_params)
        expected_call = pd.DataFrame(data=[[-0.01516655, -0.01977662, -0.01990399],
                                           [-0.01569631, -0.02176239, -0.0212802],
                                           [-0.01601397, -0.02491789, -0.02297484],
                                           [-0.01474417, -0.03162919, -0.02457737],
                                           [-0.00046144, -0.0728981, -0.01462746]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)

        # put
        test_put = self.put_opt.theta(**self.vector_params)
        expected_put = pd.DataFrame(data=[[-0.00193999, -0.00655005, -0.00667743],
                                          [-0.00235693, -0.00842301, -0.00794082],
                                          [-0.00256266, -0.01146658, -0.00952353],
                                          [-0.00117813, -0.01806315, -0.01101133],
                                          [0.01321844, -0.05921823, -0.00094758]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)

    def test_rho_scalar(self):
        """Test Rho - scalar case"""

        # call
        test_call = scalarize(self.call_opt.rho(**self.scalar_params))
        expected_call = 0.309243166487844
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.rho(**self.scalar_params))
        expected_put = -0.2575372798733608
        self.assertEqual(test_put, expected_put)

    def test_rho_vector_np(self):
        """Test Rho - np.ndarray output case"""

        # call
        test_call = self.call_opt.rho(**self.vector_params)
        expected_call = np.array([[2.08128741e-01, 3.72449469e-01, 5.12209444e-01],
                                  [1.39670999e-01, 2.81318986e-01, 4.02292404e-01],
                                  [7.76651463e-02, 1.91809707e-01, 2.90026614e-01],
                                  [2.49657984e-02, 1.01399432e-01, 1.68411513e-01],
                                  [2.17415573e-05, 1.39508485e-02, 2.73093423e-02]])
        np_test.assert_allclose(test_call, expected_call, rtol=1e-5)

        # put
        test_put = self.put_opt.rho(**self.vector_params)
        expected_put = np.array([[-4.69071412e-01, -3.04750685e-01, -1.64990710e-01],
                                 [-3.77896910e-01, -2.36248923e-01, -1.15275505e-01],
                                 [-2.80139757e-01, -1.65995197e-01, -6.77782897e-02],
                                 [-1.67672008e-01, -9.12383748e-02, -2.42262934e-02],
                                 [-2.73380139e-02, -1.34089069e-02, -5.04131783e-05]])
        np_test.assert_allclose(test_put, expected_put, rtol=1e-5)

    def test_rho_vector_df(self):
        """Test Theta - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.rho(**self.vector_params)
        expected_call = pd.DataFrame(data=[[2.08128741e-01, 3.72449469e-01, 5.12209444e-01],
                                           [1.39670999e-01, 2.81318986e-01, 4.02292404e-01],
                                           [7.76651463e-02, 1.91809707e-01, 2.90026614e-01],
                                           [2.49657984e-02, 1.01399432e-01, 1.68411513e-01],
                                           [2.17415573e-05, 1.39508485e-02, 2.73093423e-02]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)

        # put
        test_put = self.put_opt.rho(**self.vector_params)
        expected_put = pd.DataFrame(data=[[-4.69071412e-01, -3.04750685e-01, -1.64990710e-01],
                                          [-3.77896910e-01, -2.36248923e-01, -1.15275505e-01],
                                          [-2.80139757e-01, -1.65995197e-01, -6.77782897e-02],
                                          [-1.67672008e-01, -9.12383748e-02, -2.42262934e-02],
                                          [-2.73380139e-02, -1.34089069e-02, -5.04131783e-05]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)

    def test_Implied_Vol_scalar(self):
        """Test Implied Volatility - scalar case"""

        # call
        test_call = scalarize(self.call_opt.implied_volatility(**self.scalar_params))
        expected_call = 0.2
        self.assertAlmostEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.implied_volatility(**self.scalar_params))
        expected_put = 0.2
        self.assertAlmostEqual(test_put, expected_put)

    def test_Implied_Vol_vector_np(self):
        """Test Implied Volatility - np.ndarray output case"""

        # call
        test_call = self.call_opt.implied_volatility(**self.vector_params)
        expected_call = 0.2 + np.zeros_like(test_call)
        np_test.assert_allclose(test_call, expected_call, rtol=1e-5)

        # put
        test_put = self.put_opt.implied_volatility(**self.vector_params)
        expected_put = 0.2 + np.zeros_like(test_put)
        np_test.assert_allclose(test_put, expected_put, rtol=1e-5)

    def test_Implied_Vol_vector_df(self):
        """Test Implied Volatility - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.implied_volatility(**self.vector_params)
        expected_call = pd.DataFrame(data=0.2 + np.zeros_like(test_call),
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)

        # put
        test_put = self.put_opt.implied_volatility(**self.vector_params)
        expected_put = pd.DataFrame(data=0.2 + np.zeros_like(test_put),
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)

    def test_complex_parameters_setup(self):
        """
        Test complex parameter setup:
        (S scalar, K and t vector, sigma distributed as Kxt grid, r distributed as Kxt grid)
        """

        # call
        test_call_price = self.call_opt.price(**self.complex_params)
        test_call_PnL = self.call_opt.PnL(**self.complex_params)
        test_call_delta = self.call_opt.delta(**self.complex_params)
        test_call_gamma = self.call_opt.gamma(**self.complex_params)
        test_call_vega = self.call_opt.vega(**self.complex_params)
        test_call_theta = self.call_opt.theta(**self.complex_params)
        test_call_rho = self.call_opt.rho(**self.complex_params)
        test_call_iv = self.call_opt.implied_volatility(**self.complex_params)

        expected_call_price = pd.DataFrame(data=[[15.55231058,  9.40714796,  9.87150919, 10.97983523],
                                                 [20.05777231, 16.15277891, 16.02977848, 16.27588191],
                                                 [15.81433361,  8.75227505,  6.65476799,  5.19785143]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_price.rename_axis("K", axis='columns', inplace=True)
        expected_call_price.rename_axis("t", axis='rows', inplace=True)

        expected_call_PnL = pd.DataFrame(data=[[12.06490811,  5.91974549,  6.38410672,  7.49243276],
                                               [16.57036984, 12.66537644, 12.54237601, 12.78847944],
                                               [12.32693114,  5.26487258,  3.16736552,  1.71044896]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_PnL.rename_axis("K", axis='columns', inplace=True)
        expected_call_PnL.rename_axis("t", axis='rows', inplace=True)

        expected_call_delta = pd.DataFrame(data=[[0.98935079, 0.69453583, 0.58292013, 0.53579465],
                                                 [0.79256302, 0.65515368, 0.60705014, 0.57529078],
                                                 [0.90573251, 0.6717088, 0.54283905, 0.43788167]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_delta.rename_axis("K", axis='columns', inplace=True)
        expected_call_delta.rename_axis("t", axis='rows', inplace=True)

        expected_call_gamma = pd.DataFrame(data=[[0.00373538, 0.02325203, 0.01726052, 0.01317896],
                                                 [0.01053321, 0.01130107, 0.01011038, 0.0090151 ],
                                                 [0.01253481, 0.0242596,  0.02420515, 0.02204576]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_gamma.rename_axis("K", axis='columns', inplace=True)
        expected_call_gamma.rename_axis("t", axis='rows', inplace=True)

        expected_call_vega = pd.DataFrame(data=[[0.02122104, 0.26419398, 0.29417607, 0.29948378],
                                                 [0.15544424, 0.20013116, 0.20888592, 0.2128651],
                                                 [0.02503527, 0.05383637, 0.05908709, 0.05870816]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_vega.rename_axis("K", axis='columns', inplace=True)
        expected_call_vega.rename_axis("t", axis='rows', inplace=True)

        expected_call_theta = pd.DataFrame(data=[[-0.00242788, -0.01322973, -0.02073753, -0.02747845],
                                                 [-0.03624253, -0.0521798, -0.06237363, -0.07180046],
                                                 [-0.12885912, -0.28334665, -0.33769702, -0.36349655]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_theta.rename_axis("K", axis='columns', inplace=True)
        expected_call_theta.rename_axis("t", axis='rows', inplace=True)

        expected_call_rho = pd.DataFrame(data=[[0.51543152, 0.37243495, 0.29872256, 0.26120194],
                                               [0.18683002, 0.15599644, 0.14066931, 0.12935721],
                                               [0.01800044, 0.0141648,  0.01156185, 0.00937301]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_rho.rename_axis("K", axis='columns', inplace=True)
        expected_call_rho.rename_axis("t", axis='rows', inplace=True)

        expected_call_iv = pd.DataFrame(data=[[0.1,  0.2, 0.3, 0.4],
                                              [0.5,  0.6, 0.7, 0.8],
                                              [np.nan, 1., 1.1, 1.2]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_iv.rename_axis("K", axis='columns', inplace=True)
        expected_call_iv.rename_axis("t", axis='rows', inplace=True)

        pd_test.assert_frame_equal(test_call_price, expected_call_price)
        pd_test.assert_frame_equal(test_call_PnL, expected_call_PnL)
        pd_test.assert_frame_equal(test_call_delta, expected_call_delta)
        pd_test.assert_frame_equal(test_call_gamma, expected_call_gamma)
        pd_test.assert_frame_equal(test_call_vega, expected_call_vega)
        pd_test.assert_frame_equal(test_call_theta, expected_call_theta)
        pd_test.assert_frame_equal(test_call_rho, expected_call_rho)
        pd_test.assert_frame_equal(test_call_iv, expected_call_iv)

        # put
        test_put_price = self.put_opt.price(**self.complex_params)
        test_put_PnL = self.put_opt.PnL(**self.complex_params)
        test_put_delta = self.put_opt.delta(**self.complex_params)
        test_put_gamma = self.put_opt.gamma(**self.complex_params)
        test_put_vega = self.put_opt.vega(**self.complex_params)
        test_put_theta = self.put_opt.theta(**self.complex_params)
        test_put_rho = self.put_opt.rho(**self.complex_params)
        test_put_iv = self.put_opt.implied_volatility(**self.complex_params)

        expected_put_price = pd.DataFrame(data=[[0.02812357, 3.22314287,  7.9975943,  13.35166847],
                                                [3.70370639, 9.31459014, 13.76319167, 18.54654119],
                                                [0.62962992, 3.51971706,  6.38394341,  9.88603552]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_put_price.rename_axis("K", axis='columns', inplace=True)
        expected_put_price.rename_axis("t", axis='rows', inplace=True)

        expected_put_PnL = pd.DataFrame(data=[[-10.01320701,  -6.81818772,  -2.04373628,   3.31033788],
                                              [-6.3376242,    -0.72674045,   3.72186108,   8.5052106 ],
                                              [-9.41170067,   -6.52161353,  -3.65738717,  -0.15529507]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_put_PnL.rename_axis("K", axis='columns', inplace=True)
        expected_put_PnL.rename_axis("t", axis='rows', inplace=True)

        expected_put_delta = pd.DataFrame(data=[[-0.01064921, -0.30546417, -0.41707987, -0.46420535],
                                                [-0.20743698, -0.34484632, -0.39294986, -0.42470922],
                                                [-0.09426749, -0.3282912, -0.45716095, -0.56211833]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_put_delta.rename_axis("K", axis='columns', inplace=True)
        expected_put_delta.rename_axis("t", axis='rows', inplace=True)

        expected_put_gamma = copy.deepcopy(expected_call_gamma)

        expected_put_vega = copy.deepcopy(expected_call_vega)

        expected_put_theta = pd.DataFrame(data=[[-0.00038744, -0.00863707, -0.01349429, -0.01735551],
                                                [-0.02615404, -0.03850937, -0.04554804, -0.05157676],
                                                [-0.11041151, -0.26012269, -0.31065535, -0.33236619]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_put_theta.rename_axis("K", axis='columns', inplace=True)
        expected_put_theta.rename_axis("t", axis='rows', inplace=True)

        expected_put_rho = pd.DataFrame(data=[[-0.00691938, -0.21542518, -0.31936724, -0.38666626],
                                              [-0.08152366, -0.14703153, -0.17901683, -0.2068619],
                                              [-0.00249691, -0.00905916, -0.01302149, -0.01656895]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_put_rho.rename_axis("K", axis='columns', inplace=True)
        expected_put_rho.rename_axis("t", axis='rows', inplace=True)

        expected_put_iv = pd.DataFrame(data=[[0.1,  0.2, 0.3, 0.4],
                                              [0.5,  0.6, 0.7, 0.8],
                                              [np.nan, 1., 1.1, 1.2]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_put_iv.rename_axis("K", axis='columns', inplace=True)
        expected_put_iv.rename_axis("t", axis='rows', inplace=True)

        pd_test.assert_frame_equal(test_put_price, expected_put_price)
        pd_test.assert_frame_equal(test_put_PnL, expected_put_PnL)
        pd_test.assert_frame_equal(test_put_delta, expected_put_delta)
        pd_test.assert_frame_equal(test_put_gamma, expected_put_gamma)
        pd_test.assert_frame_equal(test_put_vega, expected_put_vega)
        pd_test.assert_frame_equal(test_put_theta, expected_put_theta, check_less_precise=True)
        pd_test.assert_frame_equal(test_put_rho, expected_put_rho)
        pd_test.assert_frame_equal(test_put_iv, expected_put_iv)

        # test gamma and vega consistency
        pd_test.assert_frame_equal(test_call_gamma, test_put_gamma)
        pd_test.assert_frame_equal(test_call_vega, test_put_vega)


# class TestDigitalOption(unittest.TestCase):
#
#     def setUp(self) -> None:
#
#         warnings.filterwarnings("ignore")
#
#         # common market environment
#         mkt_env = MarketEnvironment()
#
#         # option objects
#         self.call_opt = DigitalOption(mkt_env)
#         self.put_opt = DigitalOption(mkt_env, option_type="put")
#
#         # pricing parameters
#         S_scalar = 100
#         S_vector = [90, 100, 110]
#         t_scalar_string = "01-06-2020"
#         t_date_range = pd.date_range(start="2020-04-19", end="2020-12-21", periods=5)
#
#         # common pricing parameter setup
#         common_params = {"np_output": True, "minimization_method": "Least-Squares"}
#
#         # scalar parameters setup
#         self.scalar_params = copy.deepcopy(common_params)
#         self.scalar_params["S"] = S_scalar
#         self.scalar_params["t"] = t_scalar_string
#
#         # vector parameters setup
#         self.vector_params = copy.deepcopy(common_params)
#         self.vector_params["S"] = S_vector
#         self.vector_params["t"] = t_date_range
#
#         # complex pricing parameter setup
#         # (S scalar, K and t vector, sigma distributed as Kxt grid, r distributed as Kxt grid)
#         K_vector = [75, 85, 90, 95]
#         mK = len(K_vector)
#         n = 3
#         sigma_grid_K = np.array([0.1 * (1 + i) for i in range(mK * n)]).reshape(n, mK)
#         r_grid_K = np.array([0.01 * (1 + i) for i in range(mK * n)]).reshape(n, mK)
#         self.complex_params = {"S": S_vector[0],
#                                "K": K_vector,
#                                "t": pd.date_range(start="2020-04-19", end="2020-12-21", periods=n),
#                                "sigma": sigma_grid_K,
#                                "r": r_grid_K,
#                                "np_output": False}
#
#     def test_price_scalar(self):
#         """Test price - scalar case"""
#
#         # call
#         test_call = scalarize(self.call_opt.price(**self.scalar_params))
#         expected_call = 7.548381716811839
#         self.assertEqual(test_call, expected_call)
#
#         # put
#         test_put = scalarize(self.put_opt.price(**self.scalar_params))
#         expected_put = 4.672730506407959
#         self.assertEqual(test_put, expected_put)
#
#     def test_price_vector_np(self):
#         """Test price - np.ndarray output case"""
#
#         # call
#         test_call = self.call_opt.price(**self.vector_params)
#         expected_call = np.array([[3.48740247e+00, 8.42523213e+00, 1.55968082e+01],
#                                   [2.53045128e+00, 7.14167587e+00, 1.43217796e+01],
#                                   [1.56095778e+00, 5.72684668e+00, 1.29736886e+01],
#                                   [5.89165298e-01, 4.00605304e+00, 1.14939139e+01],
#                                   [7.21585753e-04, 1.38927959e+00, 1.01386434e+01]])
#         np_test.assert_allclose(test_call, expected_call)
#
#         # put
#         test_put = self.put_opt.price(**self.vector_params)
#         expected_put = np.array([[1.00413306e+01, 4.97916024e+00, 2.15073633e+00],
#                                  [9.90791873e+00, 4.51914332e+00, 1.69924708e+00],
#                                  [9.75553655e+00, 3.92142545e+00, 1.16826738e+00],
#                                  [9.62127704e+00, 3.03816479e+00, 5.26025639e-01],
#                                  [9.86382907e+00, 1.25238707e+00, 1.75090342e-03]])
#         np_test.assert_allclose(test_put, expected_put)
#
#     def test_price_vector_df(self):
#         """Test price - pd.DataFrame output case"""
#
#         # request Pandas DataFrame as output format
#         self.vector_params["np_output"] = False
#
#         # call
#         test_call = self.call_opt.price(**self.vector_params)
#         expected_call = pd.DataFrame(data=[[3.48740247e+00, 8.42523213e+00, 1.55968082e+01],
#                                            [2.53045128e+00, 7.14167587e+00, 1.43217796e+01],
#                                            [1.56095778e+00, 5.72684668e+00, 1.29736886e+01],
#                                            [5.89165298e-01, 4.00605304e+00, 1.14939139e+01],
#                                            [7.21585753e-04, 1.38927959e+00, 1.01386434e+01]],
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_call.rename_axis("S", axis='columns', inplace=True)
#         expected_call.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_call, expected_call)
#
#         # put
#         test_put = self.put_opt.price(**self.vector_params)
#         expected_put = pd.DataFrame(data=[[1.00413306e+01, 4.97916024e+00, 2.15073633e+00],
#                                           [9.90791873e+00, 4.51914332e+00, 1.69924708e+00],
#                                           [9.75553655e+00, 3.92142545e+00, 1.16826738e+00],
#                                           [9.62127704e+00, 3.03816479e+00, 5.26025639e-01],
#                                           [9.86382907e+00, 1.25238707e+00, 1.75090342e-03]],
#                                     index=self.vector_params["t"],
#                                     columns=self.vector_params["S"])
#
#         expected_put.rename_axis("S", axis='columns', inplace=True)
#         expected_put.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_put, expected_put)
#
#     def test_PnL_scalar(self):
#         """Test P&L - scalar case"""
#
#         # call
#         test_call = scalarize(self.call_opt.PnL(**self.scalar_params))
#         expected_call = 4.060979245868182
#         self.assertEqual(test_call, expected_call)
#
#         # put
#         test_put = scalarize(self.put_opt.PnL(**self.scalar_params))
#         expected_put = -5.368600081057167
#         self.assertEqual(test_put, expected_put)
#
#     def test_PnL_vector_np(self):
#         """Test P&L - np.ndarray output case"""
#
#         # call
#         test_call = self.call_opt.PnL(**self.vector_params)
#         expected_call = np.array([[0.,           4.93782966, 12.10940574],
#                                   [-0.95695119,  3.6542734,  10.83437716],
#                                   [-1.92644469,  2.2394442,   9.48628613],
#                                   [-2.89823717,  0.51865057,  8.00651142],
#                                   [-3.48668089, -2.09812288,  6.65124095]])
#         np_test.assert_allclose(test_call, expected_call)
#
#         # put
#         test_put = self.put_opt.PnL(**self.vector_params)
#         expected_put = np.array([[0.,           -5.06217034,  -7.89059426],
#                                  [-0.13341186,  -5.52218727,  -8.3420835 ],
#                                  [-0.28579403,  -6.11990513,  -8.87306321],
#                                  [-0.42005355,  -7.0031658,   -9.51530495],
#                                  [-0.17750152,  -8.78894351, -10.03957968]])
#         np_test.assert_allclose(test_put, expected_put)
#
#     def test_PnL_vector_df(self):
#         """Test P&L - pd.DataFrame output case"""
#
#         # request Pandas DataFrame as output format
#         self.vector_params["np_output"] = False
#
#         # call
#         test_call = self.call_opt.PnL(**self.vector_params)
#         expected_call = pd.DataFrame(data=[[0.,           4.93782966, 12.10940574],
#                                            [-0.95695119,  3.6542734,  10.83437716],
#                                            [-1.92644469,  2.2394442,   9.48628613],
#                                            [-2.89823717,  0.51865057,  8.00651142],
#                                            [-3.48668089, -2.09812288,  6.65124095]],
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_call.rename_axis("S", axis='columns', inplace=True)
#         expected_call.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_call, expected_call)
#
#         # put
#         test_put = self.put_opt.PnL(**self.vector_params)
#         expected_put = pd.DataFrame(data=[[0.,           -5.06217034,  -7.89059426],
#                                           [-0.13341186,  -5.52218727,  -8.3420835],
#                                           [-0.28579403,  -6.11990513,  -8.87306321],
#                                           [-0.42005355,  -7.0031658,   -9.51530495],
#                                           [-0.17750152,  -8.78894351,  -10.03957968]],
#                                     index=self.vector_params["t"],
#                                     columns=self.vector_params["S"])
#
#         expected_put.rename_axis("S", axis='columns', inplace=True)
#         expected_put.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_put, expected_put)
#
#     def test_delta_scalar(self):
#         """Test Delta - scalar case"""
#
#         # call
#         test_call = scalarize(self.call_opt.delta(**self.scalar_params))
#         expected_call = 0.6054075531684143
#         self.assertEqual(test_call, expected_call)
#
#         # put
#         test_put = scalarize(self.put_opt.delta(**self.scalar_params))
#         expected_put = -0.3945924468315857
#         self.assertEqual(test_put, expected_put)
#
#     def test_delta_vector_np(self):
#         """Test Delta - np.ndarray output case"""
#
#         # call
#         test_call = self.call_opt.delta(**self.vector_params)
#         expected_call = np.array([[3.68466757e-01, 6.15283790e-01, 8.05697003e-01],
#                                   [3.20097309e-01, 6.00702480e-01, 8.18280131e-01],
#                                   [2.54167521e-01, 5.83663527e-01, 8.41522350e-01],
#                                   [1.49152172e-01, 5.61339299e-01, 8.91560577e-01],
#                                   [8.89758553e-04, 5.23098767e-01, 9.98343116e-01]])
#         np_test.assert_allclose(test_call, expected_call)
#
#         # put
#         test_put = self.put_opt.delta(**self.vector_params)
#         expected_put = np.array([[-0.63153324, -0.38471621, -0.194303],
#                                  [-0.67990269, -0.39929752, -0.18171987],
#                                  [-0.74583248, -0.41633647, -0.15847765],
#                                  [-0.85084783, -0.4386607,  -0.10843942],
#                                  [-0.99911024, -0.47690123, -0.00165688]])
#         np_test.assert_allclose(test_put, expected_put)#, rtol=5e-6)
#
#     def test_delta_vector_df(self):
#         """Test Delta - pd.DataFrame output case"""
#
#         # request Pandas DataFrame as output format
#         self.vector_params["np_output"] = False
#
#         # call
#         test_call = self.call_opt.delta(**self.vector_params)
#         expected_call = pd.DataFrame(data=[[3.68466757e-01, 6.15283790e-01, 8.05697003e-01],
#                                            [3.20097309e-01, 6.00702480e-01, 8.18280131e-01],
#                                            [2.54167521e-01, 5.83663527e-01, 8.41522350e-01],
#                                            [1.49152172e-01, 5.61339299e-01, 8.91560577e-01],
#                                            [8.89758553e-04, 5.23098767e-01, 9.98343116e-01]],
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_call.rename_axis("S", axis='columns', inplace=True)
#         expected_call.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_call, expected_call)
#
#         # put
#         test_put = self.put_opt.delta(**self.vector_params)
#         expected_put = pd.DataFrame(data=[[-0.63153324, -0.38471621, -0.194303],
#                                           [-0.67990269, -0.39929752, -0.18171987],
#                                           [-0.74583248, -0.41633647, -0.15847765],
#                                           [-0.85084783, -0.4386607,  -0.10843942],
#                                           [-0.99911024, -0.47690123, -0.00165688]],
#                                     index=self.vector_params["t"],
#                                     columns=self.vector_params["S"])
#
#         expected_put.rename_axis("S", axis='columns', inplace=True)
#         expected_put.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_put, expected_put)
#
#     def test_gamma_scalar(self):
#         """Test Gamma - scalar case"""
#
#         # call
#         test_call = scalarize(self.call_opt.gamma(**self.scalar_params))
#         expected_call = 0.025194958512498786
#         self.assertEqual(test_call, expected_call)
#
#         # put
#         test_put = scalarize(self.put_opt.gamma(**self.scalar_params))
#         expected_put = copy.deepcopy(expected_call)
#         self.assertEqual(test_put, expected_put)
#
#         # assert call and put gamma coincide
#         self.assertEqual(test_call, test_put)
#
#     def test_gamma_vector_np(self):
#         """Test Gamma - np.ndarray output case"""
#
#         # call
#         test_call = self.call_opt.gamma(**self.vector_params)
#         expected_call = np.array([[0.02501273, 0.02281654, 0.01493167],
#                                   [0.02725456, 0.02648423, 0.01645793],
#                                   [0.02950243, 0.03231528, 0.01820714],
#                                   [0.02925862, 0.0446913, 0.01918121],
#                                   [0.00101516, 0.12030889, 0.00146722]])
#         np_test.assert_allclose(test_call, expected_call)#, rtol=5e-6)
#
#         # put
#         test_put = self.put_opt.gamma(**self.vector_params)
#         expected_put = copy.deepcopy(expected_call)
#         np_test.assert_allclose(test_put, expected_put)#, rtol=5e-6)
#
#         # assert call and put gamma coincide
#         np_test.assert_allclose(test_call, test_put)
#
#     def test_gamma_vector_df(self):
#         """Test Gamma - pd.DataFrame output case"""
#
#         # request Pandas DataFrame as output format
#         self.vector_params["np_output"] = False
#
#         # call
#         test_call = self.call_opt.gamma(**self.vector_params)
#         expected_call = pd.DataFrame(data=[[0.02501273, 0.02281654, 0.01493167],
#                                            [0.02725456, 0.02648423, 0.01645793],
#                                            [0.02950243, 0.03231528, 0.01820714],
#                                            [0.02925862, 0.0446913, 0.01918121],
#                                            [0.00101516, 0.12030889, 0.00146722]],
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_call.rename_axis("S", axis='columns', inplace=True)
#         expected_call.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_call, expected_call)
#
#         # put
#         test_put = self.put_opt.gamma(**self.vector_params)
#         expected_put = copy.deepcopy(expected_call)
#         pd_test.assert_frame_equal(test_put, expected_put)
#
#         # assert call and put gamma coincide
#         pd_test.assert_frame_equal(test_call, test_put)
#
#     def test_vega_scalar(self):
#         """Test Vega - scalar case"""
#
#         # call
#         test_call = scalarize(self.call_opt.vega(**self.scalar_params))
#         expected_call = 0.29405622811847903
#         self.assertEqual(test_call, expected_call)
#
#         # put
#         test_put = scalarize(self.put_opt.vega(**self.scalar_params))
#         expected_put = copy.deepcopy(expected_call)
#         self.assertEqual(test_put, expected_put)
#
#         # assert call and put vega coincide
#         self.assertEqual(test_call, test_put)
#
#     def test_vega_vector_np(self):
#         """Test Vega - np.ndarray output case"""
#
#         # call
#         test_call = self.call_opt.vega(**self.vector_params)
#         expected_call = np.array([[0.28419942, 0.32005661, 0.2534375],
#                                   [0.23467293, 0.28153094, 0.21168961],
#                                   [0.17415326, 0.23550311, 0.16055207],
#                                   [0.09220072, 0.17386752, 0.09029355],
#                                   [0.00045056, 0.06592268, 0.00097279]])
#         np_test.assert_allclose(test_call, expected_call)#, rtol=1e-5)
#
#         # put
#         test_put = self.put_opt.vega(**self.vector_params)
#         expected_put = copy.deepcopy(expected_call)
#         np_test.assert_allclose(test_put, expected_put)#, rtol=1e-5)
#
#         # assert call and put vega coincide
#         np_test.assert_allclose(test_call, test_put)
#
#     def test_vega_vector_df(self):
#         """Test Vega - pd.DataFrame output case"""
#
#         # request Pandas DataFrame as output format
#         self.vector_params["np_output"] = False
#
#         # call
#         test_call = self.call_opt.vega(**self.vector_params)
#         expected_call = pd.DataFrame(data=[[0.28419942, 0.32005661, 0.2534375],
#                                            [0.23467293, 0.28153094, 0.21168961],
#                                            [0.17415326, 0.23550311, 0.16055207],
#                                            [0.09220072, 0.17386752, 0.09029355],
#                                            [0.00045056, 0.06592268, 0.00097279]],
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_call.rename_axis("S", axis='columns', inplace=True)
#         expected_call.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)
#
#         # put
#         test_put = self.put_opt.vega(**self.vector_params)
#         expected_put = copy.deepcopy(expected_call)
#         pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)
#
#         # assert call and put vega coincide
#         pd_test.assert_frame_equal(test_call, test_put)
#
#     def test_theta_scalar(self):
#         """Test Theta - scalar case"""
#
#         # call
#         test_call = scalarize(self.call_opt.theta(**self.scalar_params))
#         expected_call = -0.021064685979455443
#         self.assertEqual(test_call, expected_call)
#
#         # put
#         test_put = scalarize(self.put_opt.theta(**self.scalar_params))
#         expected_put = -0.007759980665812141
#         self.assertEqual(test_put, expected_put)
#
#     def test_theta_vector_np(self):
#         """Test Theta - np.ndarray output case"""
#
#         # call
#         test_call = self.call_opt.theta(**self.vector_params)
#         expected_call = np.array([[-0.01516655, -0.01977662, -0.01990399],
#                                   [-0.01569631, -0.02176239, -0.0212802],
#                                   [-0.01601397, -0.02491789, -0.02297484],
#                                   [-0.01474417, -0.03162919, -0.02457737],
#                                   [-0.00046144, -0.0728981, -0.01462746]])
#         np_test.assert_allclose(test_call, expected_call)#, rtol=5e-4)
#
#         # put
#         test_put = self.put_opt.theta(**self.vector_params)
#         expected_put = np.array([[-0.00193999, -0.00655005, -0.00667743],
#                                  [-0.00235693, -0.00842301, -0.00794082],
#                                  [-0.00256266, -0.01146658, -0.00952353],
#                                  [-0.00117813, -0.01806315, -0.01101133],
#                                  [0.01321844, -0.05921823, -0.00094758]])
#         np_test.assert_allclose(test_put, expected_put)#, rtol=1e-5)
#
#     def test_theta_vector_df(self):
#         """Test Theta - pd.DataFrame output case"""
#
#         # request Pandas DataFrame as output format
#         self.vector_params["np_output"] = False
#
#         # call
#         test_call = self.call_opt.theta(**self.vector_params)
#         expected_call = pd.DataFrame(data=[[-0.01516655, -0.01977662, -0.01990399],
#                                            [-0.01569631, -0.02176239, -0.0212802],
#                                            [-0.01601397, -0.02491789, -0.02297484],
#                                            [-0.01474417, -0.03162919, -0.02457737],
#                                            [-0.00046144, -0.0728981, -0.01462746]],
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_call.rename_axis("S", axis='columns', inplace=True)
#         expected_call.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)
#
#         # put
#         test_put = self.put_opt.theta(**self.vector_params)
#         expected_put = pd.DataFrame(data=[[-0.00193999, -0.00655005, -0.00667743],
#                                           [-0.00235693, -0.00842301, -0.00794082],
#                                           [-0.00256266, -0.01146658, -0.00952353],
#                                           [-0.00117813, -0.01806315, -0.01101133],
#                                           [0.01321844, -0.05921823, -0.00094758]],
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_put.rename_axis("S", axis='columns', inplace=True)
#         expected_put.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)
#
#     def test_rho_scalar(self):
#         """Test Rho - scalar case"""
#
#         # call
#         test_call = scalarize(self.call_opt.rho(**self.scalar_params))
#         expected_call = 0.309243166487844
#         self.assertEqual(test_call, expected_call)
#
#         # put
#         test_put = scalarize(self.put_opt.rho(**self.scalar_params))
#         expected_put = -0.2575372798733608
#         self.assertEqual(test_put, expected_put)
#
#     def test_rho_vector_np(self):
#         """Test Rho - np.ndarray output case"""
#
#         # call
#         test_call = self.call_opt.rho(**self.vector_params)
#         expected_call = np.array([[2.08128741e-01, 3.72449469e-01, 5.12209444e-01],
#                                   [1.39670999e-01, 2.81318986e-01, 4.02292404e-01],
#                                   [7.76651463e-02, 1.91809707e-01, 2.90026614e-01],
#                                   [2.49657984e-02, 1.01399432e-01, 1.68411513e-01],
#                                   [2.17415573e-05, 1.39508485e-02, 2.73093423e-02]])
#         np_test.assert_allclose(test_call, expected_call)#, rtol=1e-5)
#
#         # put
#         test_put = self.put_opt.rho(**self.vector_params)
#         expected_put = np.array([[-4.69071412e-01, -3.04750685e-01, -1.64990710e-01],
#                                  [-3.77896910e-01, -2.36248923e-01, -1.15275505e-01],
#                                  [-2.80139757e-01, -1.65995197e-01, -6.77782897e-02],
#                                  [-1.67672008e-01, -9.12383748e-02, -2.42262934e-02],
#                                  [-2.73380139e-02, -1.34089069e-02, -5.04131783e-05]])
#         np_test.assert_allclose(test_put, expected_put)#, rtol=1e-5)
#
#     def test_rho_vector_df(self):
#         """Test Theta - pd.DataFrame output case"""
#
#         # request Pandas DataFrame as output format
#         self.vector_params["np_output"] = False
#
#         # call
#         test_call = self.call_opt.rho(**self.vector_params)
#         expected_call = pd.DataFrame(data=[[2.08128741e-01, 3.72449469e-01, 5.12209444e-01],
#                                            [1.39670999e-01, 2.81318986e-01, 4.02292404e-01],
#                                            [7.76651463e-02, 1.91809707e-01, 2.90026614e-01],
#                                            [2.49657984e-02, 1.01399432e-01, 1.68411513e-01],
#                                            [2.17415573e-05, 1.39508485e-02, 2.73093423e-02]],
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_call.rename_axis("S", axis='columns', inplace=True)
#         expected_call.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)
#
#         # put
#         test_put = self.put_opt.rho(**self.vector_params)
#         expected_put = pd.DataFrame(data=[[-4.69071412e-01, -3.04750685e-01, -1.64990710e-01],
#                                           [-3.77896910e-01, -2.36248923e-01, -1.15275505e-01],
#                                           [-2.80139757e-01, -1.65995197e-01, -6.77782897e-02],
#                                           [-1.67672008e-01, -9.12383748e-02, -2.42262934e-02],
#                                           [-2.73380139e-02, -1.34089069e-02, -5.04131783e-05]],
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_put.rename_axis("S", axis='columns', inplace=True)
#         expected_put.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)
#
#     def test_Implied_Vol_scalar(self):
#         """Test Implied Volatility - scalar case"""
#
#         # call
#         test_call = scalarize(self.call_opt.implied_volatility(**self.scalar_params))
#         expected_call = 0.2
#         self.assertAlmostEqual(test_call, expected_call)
#
#         # put
#         test_put = scalarize(self.put_opt.implied_volatility(**self.scalar_params))
#         expected_put = 0.2
#         self.assertAlmostEqual(test_put, expected_put)
#
#     def test_Implied_Vol_vector_np(self):
#         """Test Implied Volatility - np.ndarray output case"""
#
#         # call
#         test_call = self.call_opt.implied_volatility(**self.vector_params)
#         expected_call = 0.2 + np.zeros_like(test_call)
#         np_test.assert_allclose(test_call, expected_call)#, rtol=1e-5)
#
#         # put
#         test_put = self.put_opt.implied_volatility(**self.vector_params)
#         expected_put = 0.2 + np.zeros_like(test_put)
#         np_test.assert_allclose(test_put, expected_put)#, rtol=1e-5)
#
#     def test_Implied_Vol_vector_df(self):
#         """Test Implied Volatility - pd.DataFrame output case"""
#
#         # request Pandas DataFrame as output format
#         self.vector_params["np_output"] = False
#
#         # call
#         test_call = self.call_opt.implied_volatility(**self.vector_params)
#         expected_call = pd.DataFrame(data=0.2 + np.zeros_like(test_call),
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_call.rename_axis("S", axis='columns', inplace=True)
#         expected_call.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)
#
#         # put
#         test_put = self.put_opt.implied_volatility(**self.vector_params)
#         expected_put = pd.DataFrame(data=0.2 + np.zeros_like(test_put),
#                                      index=self.vector_params["t"],
#                                      columns=self.vector_params["S"])
#         expected_put.rename_axis("S", axis='columns', inplace=True)
#         expected_put.rename_axis("t", axis='rows', inplace=True)
#         pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)
#
#     def test_complex_parameters_setup(self):
#         """
#         Test complex parameter setup:
#         (S scalar, K and t vector, sigma distributed as Kxt grid, r distributed as Kxt grid)
#         """
#
#         # call
#         test_call_price = self.call_opt.price(**self.complex_params)
#         test_call_PnL = self.call_opt.PnL(**self.complex_params)
#         test_call_delta = self.call_opt.delta(**self.complex_params)
#         test_call_gamma = self.call_opt.gamma(**self.complex_params)
#         test_call_vega = self.call_opt.vega(**self.complex_params)
#         test_call_theta = self.call_opt.theta(**self.complex_params)
#         test_call_rho = self.call_opt.rho(**self.complex_params)
#         test_call_iv = self.call_opt.implied_volatility(**self.complex_params)
#
#         expected_call_price = pd.DataFrame(data=[[15.55231058,  9.40714796,  9.87150919, 10.97983523],
#                                                  [20.05777231, 16.15277891, 16.02977848, 16.27588191],
#                                                  [15.81433361,  8.75227505,  6.65476799,  5.19785143]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_call_price.rename_axis("K", axis='columns', inplace=True)
#         expected_call_price.rename_axis("t", axis='rows', inplace=True)
#
#         expected_call_PnL = pd.DataFrame(data=[[12.06490811,  5.91974549,  6.38410672,  7.49243276],
#                                                [16.57036984, 12.66537644, 12.54237601, 12.78847944],
#                                                [12.32693114,  5.26487258,  3.16736552,  1.71044896]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_call_PnL.rename_axis("K", axis='columns', inplace=True)
#         expected_call_PnL.rename_axis("t", axis='rows', inplace=True)
#
#         expected_call_delta = pd.DataFrame(data=[[0.98935079, 0.69453583, 0.58292013, 0.53579465],
#                                                  [0.79256302, 0.65515368, 0.60705014, 0.57529078],
#                                                  [0.90573251, 0.6717088, 0.54283905, 0.43788167]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_call_delta.rename_axis("K", axis='columns', inplace=True)
#         expected_call_delta.rename_axis("t", axis='rows', inplace=True)
#
#         expected_call_gamma = pd.DataFrame(data=[[0.00373538, 0.02325203, 0.01726052, 0.01317896],
#                                                  [0.01053321, 0.01130107, 0.01011038, 0.0090151 ],
#                                                  [0.01253481, 0.0242596,  0.02420515, 0.02204576]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_call_gamma.rename_axis("K", axis='columns', inplace=True)
#         expected_call_gamma.rename_axis("t", axis='rows', inplace=True)
#
#         expected_call_vega = pd.DataFrame(data=[[0.02122104, 0.26419398, 0.29417607, 0.29948378],
#                                                  [0.15544424, 0.20013116, 0.20888592, 0.2128651],
#                                                  [0.02503527, 0.05383637, 0.05908709, 0.05870816]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_call_vega.rename_axis("K", axis='columns', inplace=True)
#         expected_call_vega.rename_axis("t", axis='rows', inplace=True)
#
#         expected_call_theta = pd.DataFrame(data=[[-0.00242788, -0.01322973, -0.02073753, -0.02747845],
#                                                  [-0.03624253, -0.0521798, -0.06237363, -0.07180046],
#                                                  [-0.12885912, -0.28334665, -0.33769702, -0.36349655]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_call_theta.rename_axis("K", axis='columns', inplace=True)
#         expected_call_theta.rename_axis("t", axis='rows', inplace=True)
#
#         expected_call_rho = pd.DataFrame(data=[[0.51543152, 0.37243495, 0.29872256, 0.26120194],
#                                                [0.18683002, 0.15599644, 0.14066931, 0.12935721],
#                                                [0.01800044, 0.0141648,  0.01156185, 0.00937301]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_call_rho.rename_axis("K", axis='columns', inplace=True)
#         expected_call_rho.rename_axis("t", axis='rows', inplace=True)
#
#         expected_call_iv = pd.DataFrame(data=[[0.1,  0.2, 0.3, 0.4],
#                                               [0.5,  0.6, 0.7, 0.8],
#                                               [np.nan, 1., 1.1, 1.2]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_call_iv.rename_axis("K", axis='columns', inplace=True)
#         expected_call_iv.rename_axis("t", axis='rows', inplace=True)
#
#         pd_test.assert_frame_equal(test_call_price, expected_call_price)
#         pd_test.assert_frame_equal(test_call_PnL, expected_call_PnL)
#         pd_test.assert_frame_equal(test_call_delta, expected_call_delta)
#         pd_test.assert_frame_equal(test_call_gamma, expected_call_gamma)
#         pd_test.assert_frame_equal(test_call_vega, expected_call_vega)
#         pd_test.assert_frame_equal(test_call_theta, expected_call_theta)
#         pd_test.assert_frame_equal(test_call_rho, expected_call_rho)
#         pd_test.assert_frame_equal(test_call_iv, expected_call_iv)
#
#         # put
#         test_put_price = self.put_opt.price(**self.complex_params)
#         test_put_PnL = self.put_opt.PnL(**self.complex_params)
#         test_put_delta = self.put_opt.delta(**self.complex_params)
#         test_put_gamma = self.put_opt.gamma(**self.complex_params)
#         test_put_vega = self.put_opt.vega(**self.complex_params)
#         test_put_theta = self.put_opt.theta(**self.complex_params)
#         test_put_rho = self.put_opt.rho(**self.complex_params)
#         test_put_iv = self.put_opt.implied_volatility(**self.complex_params)
#
#         expected_put_price = pd.DataFrame(data=[[0.02812357, 3.22314287,  7.9975943,  13.35166847],
#                                                 [3.70370639, 9.31459014, 13.76319167, 18.54654119],
#                                                 [0.62962992, 3.51971706,  6.38394341,  9.88603552]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_put_price.rename_axis("K", axis='columns', inplace=True)
#         expected_put_price.rename_axis("t", axis='rows', inplace=True)
#
#         expected_put_PnL = pd.DataFrame(data=[[-10.01320701,  -6.81818772,  -2.04373628,   3.31033788],
#                                               [-6.3376242,    -0.72674045,   3.72186108,   8.5052106 ],
#                                               [-9.41170067,   -6.52161353,  -3.65738717,  -0.15529507]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_put_PnL.rename_axis("K", axis='columns', inplace=True)
#         expected_put_PnL.rename_axis("t", axis='rows', inplace=True)
#
#         expected_put_delta = pd.DataFrame(data=[[-0.01064921, -0.30546417, -0.41707987, -0.46420535],
#                                                 [-0.20743698, -0.34484632, -0.39294986, -0.42470922],
#                                                 [-0.09426749, -0.3282912, -0.45716095, -0.56211833]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_put_delta.rename_axis("K", axis='columns', inplace=True)
#         expected_put_delta.rename_axis("t", axis='rows', inplace=True)
#
#         expected_put_gamma = copy.deepcopy(expected_call_gamma)
#
#         expected_put_vega = copy.deepcopy(expected_call_vega)
#
#         expected_put_theta = pd.DataFrame(data=[[-0.00038744, -0.00863707, -0.01349429, -0.01735551],
#                                                 [-0.02615404, -0.03850937, -0.04554804, -0.05157676],
#                                                 [-0.11041151, -0.26012269, -0.31065535, -0.33236619]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_put_theta.rename_axis("K", axis='columns', inplace=True)
#         expected_put_theta.rename_axis("t", axis='rows', inplace=True)
#
#         expected_put_rho = pd.DataFrame(data=[[-0.00691938, -0.21542518, -0.31936724, -0.38666626],
#                                               [-0.08152366, -0.14703153, -0.17901683, -0.2068619],
#                                               [-0.00249691, -0.00905916, -0.01302149, -0.01656895]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_put_rho.rename_axis("K", axis='columns', inplace=True)
#         expected_put_rho.rename_axis("t", axis='rows', inplace=True)
#
#         expected_put_iv = pd.DataFrame(data=[[0.1,  0.2, 0.3, 0.4],
#                                               [0.5,  0.6, 0.7, 0.8],
#                                               [np.nan, 1., 1.1, 1.2]],
#                                            index=self.complex_params["t"],
#                                            columns=self.complex_params["K"])
#         expected_put_iv.rename_axis("K", axis='columns', inplace=True)
#         expected_put_iv.rename_axis("t", axis='rows', inplace=True)
#
#         pd_test.assert_frame_equal(test_put_price, expected_put_price)
#         pd_test.assert_frame_equal(test_put_PnL, expected_put_PnL)
#         pd_test.assert_frame_equal(test_put_delta, expected_put_delta)
#         pd_test.assert_frame_equal(test_put_gamma, expected_put_gamma)
#         pd_test.assert_frame_equal(test_put_vega, expected_put_vega)
#         pd_test.assert_frame_equal(test_put_theta, expected_put_theta, check_less_precise=True)
#         pd_test.assert_frame_equal(test_put_rho, expected_put_rho)
#         pd_test.assert_frame_equal(test_put_iv, expected_put_iv)
#
#         # test gamma and vega consistency
#         pd_test.assert_frame_equal(test_call_gamma, test_put_gamma)
#         pd_test.assert_frame_equal(test_call_vega, test_put_vega)


if __name__ == '__main__':
    unittest.main()
