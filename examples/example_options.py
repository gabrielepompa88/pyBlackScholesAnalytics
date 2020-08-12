"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: example_options.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script shows basic usage of PlainVanillaOption and DigitalOption classes.
Basic instantiation examples are provided with combinations of the underlying
level (S) and time parameter (t/tau). Price, P&L, first-order greeks as well as 
Black-Scholes implied-volatilies are computed for plain-vanilla and digital 
option contracts.
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
    S_scalar = 100
    S_vector = [90, 100, 110]
    
    # tau: several possibilities
    t_scalar_dt = option.get_t()
    t_scalar_string = "01-06-2020"
    tau_scalar = 0.5
    valuation_date = option.get_t()
    expiration_date = option.get_T()
    t_range = pd.date_range(start=valuation_date, 
                            end=expiration_date-pd.Timedelta(days=10),
                            periods=5)
    t_list = ["10-07-2020", "11-09-2020", "06-08-2020", "15-10-2020", "01-06-2020"] # order doesn't matter
    tau_list = [0.3, 0.4, 0.5, 0.6, 0.7]

    # sigma
    sigma_axis = np.array([0.1*(1 + i) for i in range(3)])

    # r
    r_axis = np.array([0.01*(1 + i) for i in range(3)])

    cases_dict = {
            "S_scalar_default.t_scalar_default": {"parameters": 
                                        {"np_output": np_output},
                  "info": "Case S_scalar_default.t_scalar_default: (S scalar, t scalar) default values"
                  },
            "S_scalar.t_scalar_dt": {"parameters": 
                                        {"S": S_scalar, 
                                         "t": t_scalar_dt,
                                         "np_output": np_output},
                  "info": "Case S_scalar.t_scalar_dt: (S scalar, t scalar as dt obj)"
                  },
            "S_scalar.t_scalar_str": {"parameters": 
                                        {"S": S_scalar, 
                                         "t": t_scalar_string,
                                         "np_output": np_output},
                  "info": "Case S_scalar.t_scalar_str: (S scalar, t scalar as string)"
                  },
            "S_scalar.tau_scalar": {"parameters": 
                                        {"S": S_scalar, 
                                         "tau": tau_scalar,
                                         "np_output": np_output},
                  "info": "Case S_scalar.tau_scalar: (S scalar, t scalar as time-to-maturity)"
                  },
            "S_vector.t_scalar_default": {"parameters": 
                                        {"S": S_vector, 
                                         "np_output": np_output},
                  "info": "Case S_vector.t_scalar_default: (S vector, t left default)"
                  },
            "S_scalar_default.t_date_range": {"parameters": 
                                        {"t": t_range, 
                                         "np_output": np_output},
                  "info": "Case S_scalar_default.t_date_range: (S left default, t vector as pd.date_range)"
                  },
            "S_vector.t_date_range": {"parameters": 
                                        {"S": S_vector,
                                         "t": t_range, 
                                         "np_output": np_output},
                  "info": "Case S_vector.t_date_range: (S vector, t vector as pd.date_range)"
                  },
            "S_vector.t_str_list": {"parameters": 
                                        {"S": S_vector,
                                         "t": t_list, 
                                         "np_output": np_output},
                  "info": "Case S_vector.t_str_list: (S vector, t vector as list of strings)"
                  },
            "S_vector.tau_list": {"parameters": 
                                        {"S": S_vector,
                                         "tau": tau_list, 
                                         "np_output": np_output},
                  "info": "Case S_vector.tau_list: (S vector, t vector as list of times-to-maturity)"
                  },
            "S_scalar_default.t_scalar_default.sigma_axis": {"parameters": 
                                        {"sigma": sigma_axis,
                                         "sigma_axis": True,
                                         "np_output": np_output},
                  "info": "Case S_scalar_default.t_scalar_default.sigma_axis: (S scalar, t scalar as time-to-maturity, sigma vector axis)"
                  },
            "S_scalar_default.t_scalar_default.r_axis": {"parameters": 
                                        {"r": r_axis,
                                         "r_axis": True,
                                         "np_output": np_output},
                  "info": "Case S_scalar_default.t_scalar_default.sigma_axis: (S scalar, t scalar as time-to-maturity, r vector axis)"
                  },
    }

    return cases_dict[case]["parameters"], cases_dict[case]["info"]

def main():

    # if np_output is True, the output will be np.ndarray, otherwise pd.DataFrame    
    np_output = False # True
    
    # default market environment
    market_env = MarketEnvironment()
    print(market_env)
    
    # define option style and type
    opt_style = "plain_vanilla" # "digital" # "plain_vanilla"
    opt_type = "call" # "put"   
    option = option_factory(market_env, opt_style, opt_type)
    print(option)
    
    # loop over different cases:
    for case in ["S_scalar_default.t_scalar_default", \
                 "S_scalar.t_scalar_dt", \
                 "S_scalar.t_scalar_str", \
                 "S_scalar.tau_scalar", \
                 "S_vector.t_scalar_default", \
                 "S_scalar_default.t_date_range", \
                 "S_vector.t_date_range", \
                 "S_vector.t_str_list", \
                 "S_vector.tau_list", \
                 "S_scalar_default.t_scalar_default.sigma_axis", \
                 "S_scalar_default.t_scalar_default.r_axis"]:    

        # get parameters dictionary for case considered
        param_dict, case_info = get_param_dict(option, np_output, case)
    
        not_sigma_axis = ('sigma_axis' not in param_dict) or (param_dict['sigma_axis'] == False)
        not_r_axis = ('r_axis' not in param_dict) or (param_dict['r_axis'] == False)

        print("\n--------------------------------------------\n")
        print("\n" + case_info + "\n")
        
        print("Parameters:")
        print("S: {}".format(param_dict["S"] if "S" in param_dict else str(option.get_S()) + " (default)"))
        print("K: {}".format(param_dict["K"] if "K" in param_dict else str(option.get_K()) + " (default)"))
        print("t: {}".format(param_dict["t"] if "t" in param_dict else str(option.get_tau()) + " (default)"))
        print("sigma: {}".format(param_dict["sigma"] if "sigma" in param_dict else str(option.get_sigma()) + " (default)"))
        print("r: {}\n".format(param_dict["r"] if "r" in param_dict else str(option.get_r()) + " (default)"))

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
        
        # Implied volatility calculation is not implemented for x-axis (columns) 
        # spanned by parameters different from S or K (like sigma or r)
        if not_sigma_axis and not_r_axis: 
            print("\nExpected Implied Volatility: \n{}\n"\
                  .format(option.get_sigma()))
            print("\nImplied Volatility - Newton method:\n{}\n"\
                  .format(option.implied_volatility(**param_dict)))
            param_dict["minimization_method"] = "Least-Squares"
            print("\nImplied Volatility - Least-Squares constrained method:\n{}\n"\
                  .format(option.implied_volatility(**param_dict)))
  
#----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    
    main()    

