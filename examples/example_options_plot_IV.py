"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: example_options_plot_IV.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script shows integration of PlainVanillaOption and DigitalOption classes
with Plotter class (in particular .plot_IV() method). Focus is on the 
visualization of Black-Scholes implied volatilities for plain-vanilla and 
digital option contracts both as a line plot and surface-plot.
"""

import numpy as np
import pandas as pd
import warnings

from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.options.options import PlainVanillaOption, DigitalOption
from pyblackscholesanalytics.plotter.plotter import OptionPlotter

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

def main():
    
    # output format: pd.DataFrame    
    np_output = False 
    
    # choose whether to plot expected IV or reconstructed one (takes longer)
    plot_expected_iv = True # False
    
    # default market environment
    market_env = MarketEnvironment()
    print(market_env)
    
    # define option style and type
    opt_style = "plain_vanilla" # "digital"
    opt_type = "call" # "call" # "put"   
    option = option_factory(market_env, opt_style, opt_type)
    print(option)
    
    # K
    K_vector = np.linspace(50,150,100)

    # tau: a date-range of 5 valuation dates between t and T-10d
    n = 6
    valuation_date = option.get_t()
    expiration_date = option.get_T()
    t_vector = pd.date_range(start=valuation_date, 
                             end=expiration_date-pd.Timedelta(days=25), 
                             periods=n)    
    
    # sigma (qualitatively reproducing the smile)
    k, tau = np.meshgrid(K_vector, option.time_to_maturity(t=t_vector))
    sigma_grid_K = 0.01 + ((k - 100)**2)/(100*k)/tau

    # pricing parameters
    param_dict = {"S": 100,
                  "K": K_vector,
                  "t": t_vector,
                  "sigma": sigma_grid_K,
                  "r": 0.01,
                  "np_output": np_output}

    print("Parameters:")
    print("S: {}".format(param_dict["S"]))
    print("K: {}".format(param_dict["K"]))
    print("t: {}".format(param_dict["t"]))
    print("sigma: \n{}".format(param_dict["sigma"]))
    print("r: {}\n".format(param_dict["r"]))
    
    # expected implied volatility: is the 'sigma' parameter with which the 
    # target price has been generated
    expected_IV = pd.DataFrame(data=param_dict["sigma"],
                               columns=K_vector,
                               index=t_vector)
    expected_IV.rename_axis('K', axis = 'columns', inplace=True)
    expected_IV.rename_axis('t', axis = 'rows', inplace=True)
    print("\nExpected Kxt Implied volatiltiy Surface: \n", expected_IV)
    
    if plot_expected_iv:
        
        IV_plot = expected_IV

    else:
    
        # compute target price
        target_price = option.price(**param_dict)
        print("\nTarget Price in input: \n", target_price)
    
        # Add target_price to parameters dictionary:
        param_dict['target_price'] = target_price
            
        # Least=Squares method
        param_dict["minimization_method"] = "Least-Squares"
        ls_IV = option.implied_volatility(**param_dict)
        RMSE_ls = np.sqrt(np.nanmean((ls_IV - expected_IV)**2))
        RMSRE_ls = np.sqrt(np.nanmean(((ls_IV - expected_IV)/expected_IV)**2))
    
        print("\nImplied Volatility - Least-Squares constrained method - Metrics (NaN excluded): RMSE={:.1E}, RMSRE={:.1E}:\n"\
              .format(RMSE_ls, RMSRE_ls), ls_IV)
        
        IV_plot = ls_IV
        
    # option plotter instance
    plotter = OptionPlotter(option)
    plotter.plot(plot_metrics="implied_volatility", IV=IV_plot)


#----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    
    main()    