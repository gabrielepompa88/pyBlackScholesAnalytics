"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: example_options_other_params.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script shows usage of PlainVanillaOption and DigitalOption classes.
Instantiation examples are provided involving combinations of the underlying
level (S), strike-price (K), time parameter (t/tau), as well as underlying 
volatility (sigma) and short-rate (r) parameters. Price, P&L, first-order 
greeks as well as Black-Scholes implied-volatilies are computed for 
plain-vanilla and digital option contracts.
"""

import numpy as np
import pandas as pd
import warnings

from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.options.options import PlainVanillaOption, DigitalOption

warnings.filterwarnings("ignore")

def option_factory(mkt_env, plain_or_digital, option_type):

    option_dispatcher = {
            "plain_vanilla": {"call": PlainVanillaOption(mkt_env),
                              "put":  PlainVanillaOption(mkt_env, option_type="put")
                             },
            "digital": {"call": DigitalOption(mkt_env),
                        "put":  DigitalOption(mkt_env, option_type="put")
                       }
    }
    
    return option_dispatcher[plain_or_digital][option_type]

def get_param_dict(option, np_output, case):
    
    # S
    S_vector = [90, 100, 110]
    mS = len(S_vector)
    
    # K
    K_vector = [75, 85, 90, 95, 105, 115]
    mK = len(K_vector)

    # tau: a date-range of 5 valuation dates between t and T-10d
    n = 5
    valuation_date = option.get_t()
    expiration_date = option.get_T()
    t_vector = pd.date_range(start=valuation_date, 
                             end=expiration_date-pd.Timedelta(days=10), 
                             periods=n)    
    # sigma
    sigma_axis = np.array([0.1*(1 + i) for i in range(3)])
    sigma_grid_S = np.array([0.1*(1 + i) for i in range(mS*n)]).reshape(n,mS)
    sigma_grid_K = np.array([0.1*(1 + i) for i in range(mK*n)]).reshape(n,mK)
    
    # r
    r_axis = np.array([0.01*(1 + i) for i in range(3)])
    r_grid_S = np.array([0.01*(1 + i) for i in range(mS*n)]).reshape(n,mS)
    r_grid_K = np.array([0.01*(1 + i) for i in range(mK*n)]).reshape(n,mK)

    cases_dict = {
            "All_scalar": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector[0],
                                         "t": t_vector[0],
                                         "sigma": 0.1,
                                         "r": 0.01,
                                         "np_output": np_output},
                  "info": "Case 0 - all scalar parameters"
                  },
            "S": {"parameters": 
                                        {"S": S_vector,
                                         "K": K_vector[0],
                                         "t": t_vector[0],
                                         "sigma": 0.1,
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case S - (S vector, other scalar)"
                  },
            "S.sigma_distributed": {"parameters": 
                                        {"S": S_vector,
                                         "K": K_vector[0],
                                         "t": t_vector[0],
                                         "sigma": [0.1*(1 + i) for i in range(mS)],
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case S.sigma_distributed - (S vector, K scalar, t scalar, sigma distributed along S, r scalar)"
                  },
            "S.r_distributed": {"parameters": 
                                        {"S": S_vector,
                                         "K": K_vector[0],
                                         "t": t_vector[0],
                                         "sigma": 0.1,
                                         "r": [0.01*(1 + i) for i in range(mS)],
                                         "np_output": np_output},
                    "info": "Case S.r_distributed - (S vector, K scalar, t scalar, sigma scalar, r distributed along S)"
                  },
            "S.sigma_and_r_distributed": {"parameters": 
                                        {"S": S_vector,
                                         "K": K_vector[0],
                                         "t": t_vector[0],
                                         "sigma": [0.1*(1 + i) for i in range(mS)],
                                         "r": [0.01*(1 + i) for i in range(mS)],
                                         "np_output": np_output},
                    "info": "Case S.sigma_and_r_distributed - (S vector, K scalar, t scalar, sigma distributed along S, r distributed along S)"
                  },
            "K": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector,
                                         "t": t_vector[0],
                                         "sigma": 0.1,
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case K - (K vector, other scalar)"
                  },
            "K.sigma_distributed": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector,
                                         "t": t_vector[0],
                                         "sigma": [0.1*(1 + i) for i in range(mK)],
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case K.sigma_distributed - (S scalar, K vector, t scalar, sigma distributed along K, r scalar)"
                  },
            "K.r_distributed": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector,
                                         "t": t_vector[0],
                                         "sigma": 0.1,
                                         "r": [0.01*(1 + i) for i in range(mK)],
                                         "np_output": np_output},
                    "info": "Case S.r_distributed - (S scalar, K vector, t scalar, sigma scalar, r distributed along K)"
                  },
            "K.sigma_and_r_distributed": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector,
                                         "t": t_vector[0],
                                         "sigma": [0.1*(1 + i) for i in range(mK)],
                                         "r": [0.01*(1 + i) for i in range(mK)],
                                         "np_output": np_output},
                    "info": "Case K.sigma_and_r_distributed - (S scalar, K vector, t scalar, sigma distributed along K, r distributed along K)"
                  },
            "t": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": 0.1,
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case t - (t vector, other scalar)"
                  },
            "t.sigma_distributed": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": [0.1*(1 + i) for i in range(n)],
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case t.sigma_distributed - (S scalar, K scalar, t vector, sigma distributed along t, r scalar)"
                  },
            "t.r_distributed": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": 0.1,
                                         "r": [0.01*(1 + i) for i in range(n)],
                                         "np_output": np_output},
                    "info": "Case t.r_distributed - (S scalar, K scalar, t vector, sigma scalar, r distributed along t)"
                  },
            "t.sigma_and_r_distributed": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": [0.1*(1 + i) for i in range(n)],
                                         "r": [0.01*(1 + i) for i in range(n)],
                                         "np_output": np_output},
                    "info": "Case t.sigma_and_r_distributed - (S scalar, K scalar, t vector, sigma distributed along t, r distributed along t)"
                  },
            "S.t": {"parameters": 
                                        {"S": S_vector,
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": 0.1,
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case S.t - (S and t vector, other scalar)"
                  },
            "S.t.sigma_distributed_as_Sxt_grid": {"parameters": 
                                        {"S": S_vector,
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": sigma_grid_S,
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case S.t.sigma_distributed_as_Sxt_grid - (S and t vector, K scalar, sigma distributed as Sxt grid, r scalar)"
                  },
            "S.t.r_distributed_as_Sxt_grid": {"parameters": 
                                        {"S": S_vector,
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": 0.1,
                                         "r": r_grid_S,
                                         "np_output": np_output},
                    "info": "Case S.t.r_distributed_as_Sxt_grid - (S and t vector, K scalar, sigma scalar, r distributed as Sxt grid)"
                  },
            "S.t.sigma_and_r_distributed_as_Sxt_grid": {"parameters": 
                                        {"S": S_vector,
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": sigma_grid_S,
                                         "r": r_grid_S,
                                         "np_output": np_output},
                    "info": "Case S.t.sigma_and_r_distributed_as_Sxt_grid - (S and t vector, K scalar, sigma distributed as Sxt grid, r distributed as Sxt grid)"
                  },
            "K.t": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector,
                                         "t": t_vector,
                                         "sigma": 0.1,
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case K.t - (K and t vector, other scalar)"
                  },
            "K.t.sigma_distributed_as_Kxt_grid": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector,
                                         "t": t_vector,
                                         "sigma": sigma_grid_K,
                                         "r": 0.01,
                                         "np_output": np_output},
                    "info": "Case K.t.sigma_distributed_as_Kxt_grid - (S scalar, K and t vector, sigma distributed as Kxt grid, r scalar)"
                  },
            "K.t.r_distributed_as_Kxt_grid": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector,
                                         "t": t_vector,
                                         "sigma": 0.1,
                                         "r": r_grid_K,
                                         "np_output": np_output},
                    "info": "Case K.t.r_distributed_as_Kxt_grid - (S scalar, K and t vector, sigma scalar, r distributed as Kxt grid)"
                  },
            "K.t.sigma_and_r_distributed_as_Kxt_grid": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector,
                                         "t": t_vector,
                                         "sigma": sigma_grid_K,
                                         "r": r_grid_K,
                                         "np_output": np_output},
                    "info": "Case K.t.sigma_and_r_distributed_as_Kxt_grid - (S scalar, K and t vector, sigma distributed as Kxt grid, r distributed as Kxt grid)"
                  },
            # if we want to have the x-axis spanned by sigma or r, we have to explicitly
            # ask for it, using "sigma_axis" or "r_axis" flags. Otherwise, sigma and r
            # parameters are interpreted as parameters to be distributed along the 
            # other(s) axis (and require length/shape match)
            "t.sigma_axis": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": sigma_axis,
                                         "r": 0.01,
                                         "np_output": np_output,
                                         "sigma_axis": True},
                    "info": "Case t.sigma_axis - (S scalar, K scalar, t vector, sigma vector axis, r scalar)"
                  },
            "t.r_axis": {"parameters": 
                                        {"S": S_vector[0],
                                         "K": K_vector[0],
                                         "t": t_vector,
                                         "sigma": 0.1,
                                         "r": r_axis,
                                         "np_output": np_output,
                                         "r_axis": True},
                    "info": "Case t.r_axis - (S scalar, K scalar, t vector, sigma scalar, r vector axis)"
                  }
    }
    
    return cases_dict[case]["parameters"], cases_dict[case]["info"]

def main():

    # if np_output is True, the output will be np.ndarray, otherwise pd.DataFrame    
    np_output = False # True
    
    # default market environment
    market_env = MarketEnvironment()
    print(market_env)
    
    # define option style and type
    opt_style = "plain_vanilla" # "digital"
    opt_type = "call" # "put"   
    option = option_factory(market_env, opt_style, opt_type)
    print(option)
    
    for case in ['All_scalar', \
                 'S', 'S.sigma_distributed', 'S.r_distributed', 'S.sigma_and_r_distributed', \
                 'K', 'K.sigma_distributed', 'K.r_distributed', 'K.sigma_and_r_distributed', \
                 't', 't.sigma_distributed', 't.r_distributed', 't.sigma_and_r_distributed', \
                 'S.t', 'S.t.sigma_distributed_as_Sxt_grid', 'S.t.r_distributed_as_Sxt_grid', 'S.t.sigma_and_r_distributed_as_Sxt_grid', \
                 'K.t', 'K.t.sigma_distributed_as_Kxt_grid', 'K.t.r_distributed_as_Kxt_grid', 'K.t.sigma_and_r_distributed_as_Kxt_grid', \
                 't.sigma_axis', 't.r_axis']:
    
        # get parameters dictionary for case considered
        param_dict, case_info = get_param_dict(option, np_output, case)
    
        print("\n--------------------------------------------\n")
        print("\n" + case_info + "\n")
        
        print("Parameters:")
        print("S: {}".format(param_dict["S"]))
        print("K: {}".format(param_dict["K"]))
        print("t: {}".format(param_dict["t"]))
        print("sigma: {}".format(param_dict["sigma"]))
        print("r: {}\n".format(param_dict["r"]))
        
        print("Metrics:")
        print("Payoff:\n", option.payoff(**param_dict))
        print("\nPrice upper limit:\n", option.price_upper_limit(**param_dict))
        print("\nPrice lower limit:\n", option.price_lower_limit(**param_dict))
        print("\nPrice:\n", option.price(**param_dict))
        print("\nP&L:\n", option.PnL(**param_dict))
        print("\nDelta:\n", option.delta(**param_dict))
        print("\nTheta:\n", option.theta(**param_dict))
        print("\nGamma:\n", option.gamma(**param_dict))
        print("\nVega:\n", option.vega(**param_dict))
        print("\nRho:\n", option.rho(**param_dict))

        # Implied volatility calculation is not implemented for x-axis 
        # (columns) spanned by sigma
        if ('sigma_axis' not in param_dict) or (param_dict['sigma_axis'] == False):
            
            print("\nExpected Implied Volatility: \n{}\n".format(param_dict["sigma"]))

            print("\nImplied Volatility - Newton method:\n{}\n"\
                  .format(option.implied_volatility(**param_dict)))
            
            param_dict["minimization_method"] = "Least-Squares"
            print("\nImplied Volatility - Least-Squares constrained method:\n{}\n"\
                  .format(option.implied_volatility(**param_dict)))

#----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    
    main()    