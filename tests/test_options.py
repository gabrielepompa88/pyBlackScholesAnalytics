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
    """Class to test public methods of PlainVanillaOption class"""

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
                               "np_output": False,
                               "minimization_method": "Least-Squares"}

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
        expected_call = np.array([[0., 4.93782966, 12.10940574],
                                  [-0.95695119, 3.6542734, 10.83437716],
                                  [-1.92644469, 2.2394442, 9.48628613],
                                  [-2.89823717, 0.51865057, 8.00651142],
                                  [-3.48668089, -2.09812288, 6.65124095]])
        np_test.assert_allclose(test_call, expected_call)

        # put
        test_put = self.put_opt.PnL(**self.vector_params)
        expected_put = np.array([[0., -5.06217034, -7.89059426],
                                 [-0.13341186, -5.52218727, -8.3420835],
                                 [-0.28579403, -6.11990513, -8.87306321],
                                 [-0.42005355, -7.0031658, -9.51530495],
                                 [-0.17750152, -8.78894351, -10.03957968]])
        np_test.assert_allclose(test_put, expected_put)

    def test_PnL_vector_df(self):
        """Test P&L - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.PnL(**self.vector_params)
        expected_call = pd.DataFrame(data=[[0., 4.93782966, 12.10940574],
                                           [-0.95695119, 3.6542734, 10.83437716],
                                           [-1.92644469, 2.2394442, 9.48628613],
                                           [-2.89823717, 0.51865057, 8.00651142],
                                           [-3.48668089, -2.09812288, 6.65124095]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.PnL(**self.vector_params)
        expected_put = pd.DataFrame(data=[[0., -5.06217034, -7.89059426],
                                          [-0.13341186, -5.52218727, -8.3420835],
                                          [-0.28579403, -6.11990513, -8.87306321],
                                          [-0.42005355, -7.0031658, -9.51530495],
                                          [-0.17750152, -8.78894351, -10.03957968]],
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
                                 [-0.85084783, -0.4386607, -0.10843942],
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
                                          [-0.85084783, -0.4386607, -0.10843942],
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

        expected_call_price = pd.DataFrame(data=[[15.55231058, 9.40714796, 9.87150919, 10.97983523],
                                                 [20.05777231, 16.15277891, 16.02977848, 16.27588191],
                                                 [15.81433361, 8.75227505, 6.65476799, 5.19785143]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_price.rename_axis("K", axis='columns', inplace=True)
        expected_call_price.rename_axis("t", axis='rows', inplace=True)

        expected_call_PnL = pd.DataFrame(data=[[12.06490811, 5.91974549, 6.38410672, 7.49243276],
                                               [16.57036984, 12.66537644, 12.54237601, 12.78847944],
                                               [12.32693114, 5.26487258, 3.16736552, 1.71044896]],
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
                                                 [0.01053321, 0.01130107, 0.01011038, 0.0090151],
                                                 [0.01253481, 0.0242596, 0.02420515, 0.02204576]],
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
                                               [0.01800044, 0.0141648, 0.01156185, 0.00937301]],
                                         index=self.complex_params["t"],
                                         columns=self.complex_params["K"])
        expected_call_rho.rename_axis("K", axis='columns', inplace=True)
        expected_call_rho.rename_axis("t", axis='rows', inplace=True)

        expected_call_iv = pd.DataFrame(data=self.complex_params["sigma"],
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

        expected_put_price = pd.DataFrame(data=[[0.02812357, 3.22314287, 7.9975943, 13.35166847],
                                                [3.70370639, 9.31459014, 13.76319167, 18.54654119],
                                                [0.62962992, 3.51971706, 6.38394341, 9.88603552]],
                                          index=self.complex_params["t"],
                                          columns=self.complex_params["K"])
        expected_put_price.rename_axis("K", axis='columns', inplace=True)
        expected_put_price.rename_axis("t", axis='rows', inplace=True)

        expected_put_PnL = pd.DataFrame(data=[[-10.01320701, -6.81818772, -2.04373628, 3.31033788],
                                              [-6.3376242, -0.72674045, 3.72186108, 8.5052106],
                                              [-9.41170067, -6.52161353, -3.65738717, -0.15529507]],
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

        expected_put_iv = pd.DataFrame(data=self.complex_params["sigma"],
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


class TestDigitalOption(unittest.TestCase):
    """Class to test public methods of DigitalOption class"""

    def setUp(self) -> None:
        warnings.filterwarnings("ignore")

        # common market environment
        mkt_env = MarketEnvironment()

        # option objects
        self.call_opt = DigitalOption(mkt_env)
        self.put_opt = DigitalOption(mkt_env, option_type="put")

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
                               "np_output": False,
                               "minimization_method": "Least-Squares"}

    def test_price_scalar(self):
        """Test price - scalar case"""

        # call
        test_call = scalarize(self.call_opt.price(**self.scalar_params))
        expected_call = 0.529923736000296
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.price(**self.scalar_params))
        expected_put = 0.4413197518956652
        self.assertEqual(test_put, expected_put)

    def test_price_vector_np(self):
        """Test price - np.ndarray output case"""

        # call
        test_call = self.call_opt.price(**self.vector_params)
        expected_call = np.array([[2.96746057e-01, 5.31031469e-01, 7.30298621e-01],
                                  [2.62783065e-01, 5.29285722e-01, 7.56890348e-01],
                                  [2.13141191e-01, 5.26395060e-01, 7.95937699e-01],
                                  [1.28345302e-01, 5.21278768e-01, 8.65777496e-01],
                                  [7.93566840e-04, 5.09205971e-01, 9.96790994e-01]])
        np_test.assert_allclose(test_call, expected_call)

        # put
        test_put = self.put_opt.price(**self.vector_params)
        expected_put = np.array([[0.66879322, 0.43450781, 0.23524066],
                                 [0.71099161, 0.44448895, 0.21688433],
                                 [0.7688046, 0.45555073, 0.18600809],
                                 [0.86197582, 0.46904235, 0.12454362],
                                 [0.99783751, 0.4894251, 0.00184008]])
        np_test.assert_allclose(test_put, expected_put, rtol=1e-6)

    def test_price_vector_df(self):
        """Test price - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.price(**self.vector_params)
        expected_call = pd.DataFrame(data=[[2.96746057e-01, 5.31031469e-01, 7.30298621e-01],
                                           [2.62783065e-01, 5.29285722e-01, 7.56890348e-01],
                                           [2.13141191e-01, 5.26395060e-01, 7.95937699e-01],
                                           [1.28345302e-01, 5.21278768e-01, 8.65777496e-01],
                                           [7.93566840e-04, 5.09205971e-01, 9.96790994e-01]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.price(**self.vector_params)
        expected_put = pd.DataFrame(data=[[0.66879322, 0.43450781, 0.23524066],
                                          [0.71099161, 0.44448895, 0.21688433],
                                          [0.7688046, 0.45555073, 0.18600809],
                                          [0.86197582, 0.46904235, 0.12454362],
                                          [0.99783751, 0.4894251, 0.00184008]],
                                    index=self.vector_params["t"],
                                    columns=self.vector_params["S"])

        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put)

    def test_PnL_scalar(self):
        """Test P&L - scalar case"""

        # call
        test_call = scalarize(self.call_opt.PnL(**self.scalar_params))
        expected_call = 0.23317767915072352
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.PnL(**self.scalar_params))
        expected_put = -0.22747347241997717
        self.assertEqual(test_put, expected_put)

    def test_PnL_vector_np(self):
        """Test P&L - np.ndarray output case"""

        # call
        test_call = self.call_opt.PnL(**self.vector_params)
        expected_call = np.array([[0., 0.23428541, 0.43355256],
                                  [-0.03396299, 0.23253966, 0.46014429],
                                  [-0.08360487, 0.229649, 0.49919164],
                                  [-0.16840076, 0.22453271, 0.56903144],
                                  [-0.29595249, 0.21245991, 0.70004494]])
        np_test.assert_allclose(test_call, expected_call)

        # put
        test_put = self.put_opt.PnL(**self.vector_params)
        expected_put = np.array([[0., -0.23428541, -0.43355256],
                                 [0.04219839, -0.22430427, -0.4519089],
                                 [0.10001137, -0.2132425, -0.48278514],
                                 [0.19318259, -0.19975088, -0.5442496],
                                 [0.32904428, -0.17936812, -0.66695314]])
        np_test.assert_allclose(test_put, expected_put, rtol=1e-6)

    def test_PnL_vector_df(self):
        """Test P&L - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.PnL(**self.vector_params)
        expected_call = pd.DataFrame(data=[[0., 0.23428541, 0.43355256],
                                           [-0.03396299, 0.23253966, 0.46014429],
                                           [-0.08360487, 0.229649, 0.49919164],
                                           [-0.16840076, 0.22453271, 0.56903144],
                                           [-0.29595249, 0.21245991, 0.70004494]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.PnL(**self.vector_params)
        expected_put = pd.DataFrame(data=[[0., -0.23428541, -0.43355256],
                                          [0.04219839, -0.22430427, -0.4519089],
                                          [0.10001137, -0.2132425, -0.48278514],
                                          [0.19318259, -0.19975088, -0.5442496],
                                          [0.32904428, -0.17936812, -0.66695314]],
                                    index=self.vector_params["t"],
                                    columns=self.vector_params["S"])

        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put)

    def test_delta_scalar(self):
        """Test Delta - scalar case"""

        # call
        test_call = scalarize(self.call_opt.delta(**self.scalar_params))
        expected_call = 0.025194958512498786
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.delta(**self.scalar_params))
        expected_put = copy.deepcopy(-expected_call)
        self.assertEqual(test_put, expected_put)

        # assert call and put delta consistency
        self.assertEqual(test_call, -test_put)

    def test_delta_vector_np(self):
        """Test Delta - np.ndarray output case"""

        # call
        test_call = self.call_opt.delta(**self.vector_params)
        expected_call = np.array([[0.02251146, 0.02281654, 0.01642484],
                                  [0.0245291, 0.02648423, 0.01810373],
                                  [0.02655219, 0.03231528, 0.02002786],
                                  [0.02633276, 0.0446913, 0.02109933],
                                  [0.00091364, 0.12030889, 0.00161394]])
        np_test.assert_allclose(test_call, expected_call, rtol=5e-6)

        # put
        test_put = self.put_opt.delta(**self.vector_params)
        expected_put = copy.deepcopy(-expected_call)
        np_test.assert_allclose(test_put, expected_put, rtol=5e-6)

        # assert call and put delta consistency
        np_test.assert_allclose(test_call, -test_put)

    def test_delta_vector_df(self):
        """Test Delta - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.delta(**self.vector_params)
        expected_call = pd.DataFrame(data=[[0.02251146, 0.02281654, 0.01642484],
                                           [0.0245291, 0.02648423, 0.01810373],
                                           [0.02655219, 0.03231528, 0.02002786],
                                           [0.02633276, 0.0446913, 0.02109933],
                                           [0.00091364, 0.12030889, 0.00161394]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.delta(**self.vector_params)
        expected_put = copy.deepcopy(-expected_call)
        pd_test.assert_frame_equal(test_put, expected_put)

        # assert call and put delta consistency
        pd_test.assert_frame_equal(test_call, -test_put)

    def test_gamma_scalar(self):
        """Test Gamma - scalar case"""

        # call
        test_call = scalarize(self.call_opt.gamma(**self.scalar_params))
        expected_call = -0.0004409117739687288
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.gamma(**self.scalar_params))
        expected_put = copy.deepcopy(-expected_call)
        self.assertEqual(test_put, expected_put)

        # assert call and put gamma coincide
        self.assertEqual(test_call, -test_put)

    def test_gamma_vector_np(self):
        """Test Gamma - np.ndarray output case"""

        # call
        test_call = self.call_opt.gamma(**self.vector_params)
        expected_call = np.array([[0.00050164, -0.00039929, -0.00076858],
                                  [0.00087371, -0.00046347, -0.00102583],
                                  [0.00161634, -0.00056552, -0.00150922],
                                  [0.0034499, -0.0007821, -0.00268525],
                                  [0.00095822, -0.00210541, -0.00130173]])
        np_test.assert_allclose(test_call, expected_call, rtol=1e-5)

        # put
        test_put = self.put_opt.gamma(**self.vector_params)
        expected_put = copy.deepcopy(-expected_call)
        np_test.assert_allclose(test_put, expected_put, rtol=1e-5)

        # assert call and put gamma coincide
        np_test.assert_allclose(test_call, -test_put)

    def test_gamma_vector_df(self):
        """Test Gamma - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.gamma(**self.vector_params)
        expected_call = pd.DataFrame(data=[[0.00050164, -0.00039929, -0.00076858],
                                           [0.00087371, -0.00046347, -0.00102583],
                                           [0.00161634, -0.00056552, -0.00150922],
                                           [0.0034499, -0.0007821, -0.00268525],
                                           [0.00095822, -0.00210541, -0.00130173]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)

        # put
        test_put = self.put_opt.gamma(**self.vector_params)
        expected_put = copy.deepcopy(-expected_call)
        pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)

        # assert call and put gamma coincide
        pd_test.assert_frame_equal(test_call, -test_put)

    def test_vega_scalar(self):
        """Test Vega - scalar case"""

        # call
        test_call = scalarize(self.call_opt.vega(**self.scalar_params))
        expected_call = -0.005145983992073383
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.vega(**self.scalar_params))
        expected_put = copy.deepcopy(-expected_call)
        self.assertEqual(test_put, expected_put)

        # assert call and put vega coincide
        self.assertEqual(test_call, -test_put)

    def test_vega_vector_np(self):
        """Test Vega - np.ndarray output case"""

        # call
        test_call = self.call_opt.vega(**self.vector_params)
        expected_call = np.array([[0.00569969, -0.00560099, -0.01304515],
                                  [0.00752302, -0.00492679, -0.01319465],
                                  [0.0095413, -0.0041213, -0.01330838],
                                  [0.01087143, -0.00304268, -0.01264053],
                                  [0.00042529, -0.00115365, -0.00086306]])
        np_test.assert_allclose(test_call, expected_call, rtol=1e-5)

        # put
        test_put = self.put_opt.vega(**self.vector_params)
        expected_put = copy.deepcopy(-expected_call)
        np_test.assert_allclose(test_put, expected_put, rtol=5e-5)

        # assert call and put vega coincide
        np_test.assert_allclose(test_call, -test_put)

    def test_vega_vector_df(self):
        """Test Vega - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.vega(**self.vector_params)
        expected_call = pd.DataFrame(data=[[0.00569969, -0.00560099, -0.01304515],
                                           [0.00752302, -0.00492679, -0.01319465],
                                           [0.0095413, -0.0041213, -0.01330838],
                                           [0.01087143, -0.00304268, -0.01264053],
                                           [0.00042529, -0.00115365, -0.00086306]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)

        # put
        test_put = self.put_opt.vega(**self.vector_params)
        expected_put = copy.deepcopy(-expected_call)
        pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)

        # assert call and put vega coincide
        pd_test.assert_frame_equal(test_call, -test_put)

    def test_theta_scalar(self):
        """Test Theta - scalar case"""

        # call
        test_call = scalarize(self.call_opt.theta(**self.scalar_params))
        expected_call = -3.094863279105034e-05
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.theta(**self.scalar_params))
        expected_put = 0.00016399568592748338
        self.assertEqual(test_put, expected_put)

    def test_theta_vector_np(self):
        """Test Theta - np.ndarray output case"""

        # call
        test_call = self.call_opt.theta(**self.vector_params)
        expected_call = np.array([[-4.59532646e-04, -2.10225482e-05, 3.62119705e-04],
                                  [-6.54200322e-04, -3.63343877e-05, 5.11024346e-04],
                                  [-1.01554953e-03, -6.06935961e-05, 8.07873242e-04],
                                  [-1.83825681e-03, -1.12254822e-04, 1.58102099e-03],
                                  [-4.36449645e-04, -4.24665854e-04, 9.75289889e-04]])
        np_test.assert_allclose(test_call, expected_call)

        # put
        test_put = self.put_opt.theta(**self.vector_params)
        expected_put = np.array([[0.0005918, 0.00015329, -0.00022985],
                                 [0.00078759, 0.00016973, -0.00037763],
                                 [0.00115006, 0.00019521, -0.00067336],
                                 [0.00197392, 0.00024792, -0.00144536],
                                 [0.00057325, 0.00056146, -0.00083849]])
        np_test.assert_allclose(test_put, expected_put, rtol=5e-5)

    def test_theta_vector_df(self):
        """Test Theta - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.theta(**self.vector_params)
        expected_call = pd.DataFrame(data=[[-4.59532646e-04, -2.10225482e-05, 3.62119705e-04],
                                           [-6.54200322e-04, -3.63343877e-05, 5.11024346e-04],
                                           [-1.01554953e-03, -6.06935961e-05, 8.07873242e-04],
                                           [-1.83825681e-03, -1.12254822e-04, 1.58102099e-03],
                                           [-4.36449645e-04, -4.24665854e-04, 9.75289889e-04]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call, check_less_precise=True)

        # put
        test_put = self.put_opt.theta(**self.vector_params)
        expected_put = pd.DataFrame(data=[[0.0005918, 0.00015329, -0.00022985],
                                          [0.00078759, 0.00016973, -0.00037763],
                                          [0.00115006, 0.00019521, -0.00067336],
                                          [0.00197392, 0.00024792, -0.00144536],
                                          [0.00057325, 0.00056146, -0.00083849]],
                                    index=self.vector_params["t"],
                                    columns=self.vector_params["S"])
        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put, check_less_precise=True)

    def test_rho_scalar(self):
        """Test Rho - scalar case"""

        # call
        test_call = scalarize(self.call_opt.rho(**self.scalar_params))
        expected_call = 0.011610379741045512
        self.assertEqual(test_call, expected_call)

        # put
        test_put = scalarize(self.put_opt.rho(**self.scalar_params))
        expected_put = -0.01727818420465756
        self.assertEqual(test_put, expected_put)

    def test_rho_vector_np(self):
        """Test Rho - np.ndarray output case"""

        # call
        test_call = self.call_opt.rho(**self.vector_params)
        expected_call = np.array([[1.21286837e-02, 1.22783358e-02, 7.54978064e-03],
                                  [1.03369366e-02, 1.12633572e-02, 6.56155667e-03],
                                  [7.93101136e-03, 9.85705868e-03, 5.12733711e-03],
                                  [4.36037779e-03, 7.67938163e-03, 2.83056260e-03],
                                  [2.23107982e-05, 3.15662549e-03, -2.24454016e-04]])
        np_test.assert_allclose(test_call, expected_call)

        # put
        test_put = self.put_opt.rho(**self.vector_params)
        expected_put = np.array([[-1.89006853e-02, -1.90503374e-02, -1.43217822e-02],
                                 [-1.55126157e-02, -1.64390362e-02, -1.17372358e-02],
                                 [-1.15090604e-02, -1.34351077e-02, -8.70538615e-03],
                                 [-6.28675585e-03, -9.60575970e-03, -4.75694066e-03],
                                 [-2.95908353e-04, -3.43022305e-03, -4.91435388e-05]])
        np_test.assert_allclose(test_put, expected_put)

    def test_rho_vector_df(self):
        """Test Theta - pd.DataFrame output case"""

        # request Pandas DataFrame as output format
        self.vector_params["np_output"] = False

        # call
        test_call = self.call_opt.rho(**self.vector_params)
        expected_call = pd.DataFrame(data=[[1.21286837e-02, 1.22783358e-02, 7.54978064e-03],
                                           [1.03369366e-02, 1.12633572e-02, 6.56155667e-03],
                                           [7.93101136e-03, 9.85705868e-03, 5.12733711e-03],
                                           [4.36037779e-03, 7.67938163e-03, 2.83056260e-03],
                                           [2.23107982e-05, 3.15662549e-03, -2.24454016e-04]],
                                     index=self.vector_params["t"],
                                     columns=self.vector_params["S"])
        expected_call.rename_axis("S", axis='columns', inplace=True)
        expected_call.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.rho(**self.vector_params)
        expected_put = pd.DataFrame(data=[[-1.89006853e-02, -1.90503374e-02, -1.43217822e-02],
                                          [-1.55126157e-02, -1.64390362e-02, -1.17372358e-02],
                                          [-1.15090604e-02, -1.34351077e-02, -8.70538615e-03],
                                          [-6.28675585e-03, -9.60575970e-03, -4.75694066e-03],
                                          [-2.95908353e-04, -3.43022305e-03, -4.91435388e-05]],
                                    index=self.vector_params["t"],
                                    columns=self.vector_params["S"])
        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put)

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
        np_test.assert_allclose(test_call, expected_call, rtol=5e-7)

        # put
        test_put = self.put_opt.implied_volatility(**self.vector_params)
        expected_put = 0.2 + np.zeros_like(test_put)
        np_test.assert_allclose(test_put, expected_put, rtol=5e-7)

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
        pd_test.assert_frame_equal(test_call, expected_call)

        # put
        test_put = self.put_opt.implied_volatility(**self.vector_params)
        expected_put = pd.DataFrame(data=0.2 + np.zeros_like(test_put),
                                    index=self.vector_params["t"],
                                    columns=self.vector_params["S"])
        expected_put.rename_axis("S", axis='columns', inplace=True)
        expected_put.rename_axis("t", axis='rows', inplace=True)
        pd_test.assert_frame_equal(test_put, expected_put)

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

        expected_call_price = pd.DataFrame(data=[[0.9798568, 0.62471855, 0.47323669, 0.39201772],
                                                 [0.68363866, 0.50365944, 0.42894149, 0.37368724],
                                                 [0.87602123, 0.60825314, 0.46889718, 0.36012104]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_price.rename_axis("K", axis='columns', inplace=True)
        expected_call_price.rename_axis("t", axis='rows', inplace=True)

        expected_call_PnL = pd.DataFrame(data=[[0.68311074, 0.3279725, 0.17649063, 0.09527166],
                                               [0.3868926, 0.20691339, 0.13219544, 0.07694118],
                                               [0.57927518, 0.31150708, 0.17215112, 0.06337498]],
                                         index=self.complex_params["t"],
                                         columns=self.complex_params["K"])
        expected_call_PnL.rename_axis("K", axis='columns', inplace=True)
        expected_call_PnL.rename_axis("t", axis='rows', inplace=True)

        expected_call_delta = pd.DataFrame(data=[[0.00448245, 0.02461979, 0.01726052, 0.01248533],
                                                 [0.01263986, 0.01196584, 0.01011038, 0.00854062],
                                                 [0.01504177, 0.02568663, 0.02420515, 0.02088546]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_delta.rename_axis("K", axis='columns', inplace=True)
        expected_call_delta.rename_axis("t", axis='rows', inplace=True)

        expected_call_gamma = pd.DataFrame(data=[[-1.36939276e-03, -8.30886510e-04, -1.59819664e-04, -3.72062630e-05],
                                                 [-3.79395726e-04, -1.46568014e-04, -7.22169767e-05, -3.73088881e-05],
                                                 [-1.47523572e-03, -7.66683834e-04, -1.58922692e-04, 1.82659675e-04]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_gamma.rename_axis("K", axis='columns', inplace=True)
        expected_call_gamma.rename_axis("t", axis='rows', inplace=True)

        expected_call_vega = pd.DataFrame(data=[[-0.00777965, -0.00944069, -0.00272385, -0.00084549],
                                                [-0.00559895, -0.00259558, -0.00149204, -0.00088094],
                                                [-0.00294643, -0.00170141, -0.00038795, 0.00048643]],
                                          index=self.complex_params["t"],
                                          columns=self.complex_params["K"])
        expected_call_vega.rename_axis("K", axis='columns', inplace=True)
        expected_call_vega.rename_axis("t", axis='rows', inplace=True)

        expected_call_theta = pd.DataFrame(data=[[1.67739087e-04, 2.81595503e-04, 7.08163129e-05, -1.41282958e-05],
                                                 [9.90248652e-04, 4.91233384e-04, 3.00397593e-04, 1.78375694e-04],
                                                 [1.31411352e-02, 8.04031544e-03, 1.61848869e-03, -3.41813594e-03]],
                                           index=self.complex_params["t"],
                                           columns=self.complex_params["K"])
        expected_call_theta.rename_axis("K", axis='columns', inplace=True)
        expected_call_theta.rename_axis("t", axis='rows', inplace=True)

        expected_call_rho = pd.DataFrame(data=[[-0.00404295, 0.01115923, 0.00757627, 0.00513166],
                                               [0.00165411, 0.00208889, 0.00175266, 0.0014392],
                                               [0.00013089, 0.00046672, 0.00046837, 0.00041632]],
                                         index=self.complex_params["t"],
                                         columns=self.complex_params["K"])
        expected_call_rho.rename_axis("K", axis='columns', inplace=True)
        expected_call_rho.rename_axis("t", axis='rows', inplace=True)

        expected_call_iv = pd.DataFrame(data=self.complex_params["sigma"],
                                        index=self.complex_params["t"],
                                        columns=self.complex_params["K"])
        expected_call_iv.rename_axis("K", axis='columns', inplace=True)
        expected_call_iv.rename_axis("t", axis='rows', inplace=True)

        pd_test.assert_frame_equal(test_call_price, expected_call_price)
        pd_test.assert_frame_equal(test_call_PnL, expected_call_PnL)
        pd_test.assert_frame_equal(test_call_delta, expected_call_delta)
        pd_test.assert_frame_equal(test_call_gamma, expected_call_gamma)
        pd_test.assert_frame_equal(test_call_vega, expected_call_vega, check_less_precise=True)
        pd_test.assert_frame_equal(test_call_theta, expected_call_theta)
        pd_test.assert_frame_equal(test_call_rho, expected_call_rho, check_less_precise=True)
        pd_test.assert_frame_equal(test_call_iv.iloc[:, :-1], expected_call_iv.iloc[:, :-1])
        self.assertAlmostEqual(test_call_iv.iloc[-1, -1], expected_call_iv.iloc[-1, -1], places=5)

        # put
        test_put_price = self.put_opt.price(**self.complex_params)
        test_put_PnL = self.put_opt.PnL(**self.complex_params)
        test_put_delta = self.put_opt.delta(**self.complex_params)
        test_put_gamma = self.put_opt.gamma(**self.complex_params)
        test_put_vega = self.put_opt.vega(**self.complex_params)
        test_put_theta = self.put_opt.theta(**self.complex_params)
        test_put_rho = self.put_opt.rho(**self.complex_params)
        test_put_iv = self.put_opt.implied_volatility(**self.complex_params)

        expected_put_price = pd.DataFrame(data=[[0.01315404, 0.36135198, 0.50594203, 0.58031737],
                                                [0.29830713, 0.47471481, 0.54587421, 0.59758286],
                                                [0.12151605, 0.38901089, 0.52809366, 0.63659669]],
                                          index=self.complex_params["t"],
                                          columns=self.complex_params["K"])
        expected_put_price.rename_axis("K", axis='columns', inplace=True)
        expected_put_price.rename_axis("t", axis='rows', inplace=True)

        expected_put_PnL = pd.DataFrame(data=[[-0.65563919, -0.30744125, -0.16285119, -0.08847586],
                                              [-0.37048609, -0.19407842, -0.12291901, -0.07121037],
                                              [-0.54727717, -0.27978234, -0.14069957, -0.03219654]],
                                        index=self.complex_params["t"],
                                        columns=self.complex_params["K"])
        expected_put_PnL.rename_axis("K", axis='columns', inplace=True)
        expected_put_PnL.rename_axis("t", axis='rows', inplace=True)

        expected_put_delta = copy.deepcopy(-expected_call_delta)

        expected_put_gamma = copy.deepcopy(-expected_call_gamma)

        expected_put_vega = copy.deepcopy(-expected_call_vega)

        expected_put_theta = pd.DataFrame(data=[[-1.40533310e-04, -2.27564242e-04, 9.66413009e-06, 1.20685566e-04],
                                                [-8.55735531e-04, -3.30404740e-04, -1.13446636e-04, 3.45054237e-05],
                                                [-1.28951671e-02, -7.76709242e-03, -1.31802569e-03, 3.74582396e-03]],
                                          index=self.complex_params["t"],
                                          columns=self.complex_params["K"])
        expected_put_theta.rename_axis("K", axis='columns', inplace=True)
        expected_put_theta.rename_axis("t", axis='rows', inplace=True)

        expected_put_rho = pd.DataFrame(data=[[-0.00292173, -0.01807524, -0.01444393, -0.01195132],
                                              [-0.00523216, -0.00565392, -0.00530473, -0.00497835],
                                              [-0.00040418, -0.00073995, -0.00074152, -0.00068939]],
                                        index=self.complex_params["t"],
                                        columns=self.complex_params["K"])
        expected_put_rho.rename_axis("K", axis='columns', inplace=True)
        expected_put_rho.rename_axis("t", axis='rows', inplace=True)

        expected_put_iv = pd.DataFrame(data=self.complex_params["sigma"],
                                       index=self.complex_params["t"],
                                       columns=self.complex_params["K"])
        expected_put_iv.rename_axis("K", axis='columns', inplace=True)
        expected_put_iv.rename_axis("t", axis='rows', inplace=True)

        pd_test.assert_frame_equal(test_put_price, expected_put_price)
        pd_test.assert_frame_equal(test_put_PnL, expected_put_PnL)
        pd_test.assert_frame_equal(test_put_delta, expected_put_delta)
        pd_test.assert_frame_equal(test_put_gamma, expected_put_gamma)
        pd_test.assert_frame_equal(test_put_vega, expected_put_vega, check_less_precise=True)
        pd_test.assert_frame_equal(test_put_theta, expected_put_theta)
        pd_test.assert_frame_equal(test_put_rho, expected_put_rho, check_less_precise=True)
        pd_test.assert_frame_equal(test_put_iv.iloc[:, :-1], expected_put_iv.iloc[:, :-1])
        self.assertAlmostEqual(test_put_iv.iloc[-1, -1], expected_put_iv.iloc[-1, -1], places=5)

        # test gamma and vega consistency
        pd_test.assert_frame_equal(test_call_delta, -test_put_delta)
        pd_test.assert_frame_equal(test_call_gamma, -test_put_gamma)
        pd_test.assert_frame_equal(test_call_vega, -test_put_vega)


if __name__ == '__main__':
    unittest.main()
