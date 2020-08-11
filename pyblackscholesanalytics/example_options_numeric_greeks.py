"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: example_options_numeric_greeks.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script provides an example of first-order numeric greeks implemented in 
the NumericGreeks class using finite-difference methods for plain-vanilla 
and digital option contracts.
"""

import numpy as np

from market.market import MarketEnvironment
from options.options import PlainVanillaOption, DigitalOption
from utils.numeric_routines import NumericGreeks
from utils.utils import plot, homogenize

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

def greeks_factory(ObjWithGreeksMethod, greek_type):
    
    greeks_dispatcher = {"delta": ObjWithGreeksMethod.delta,
                         "theta": ObjWithGreeksMethod.theta,
                         "gamma": ObjWithGreeksMethod.gamma,
                         "vega":  ObjWithGreeksMethod.vega,
                         "rho":   ObjWithGreeksMethod.rho
    }
    
    return greeks_dispatcher[greek_type]

def greeks_label_factory(greek_type, opt_type, kind, underlying="S"):

    labels_dispatcher = {"delta": r"$\Delta^{" + kind + "}_{" + opt_type + "}(" + underlying + ")$",
                         "theta": r"$\Theta^{" + kind + "}_{" + opt_type + "}(" + underlying + ")$",
                         "gamma": r"$\Gamma^{" + kind + "}_{" + opt_type + "}(" + underlying + ")$",
                         "vega":  r"$Vega^{" + kind + "}_{" + opt_type + "}(" + underlying + ")$",
                         "rho":  r"$\rho^{" + kind + "}_{" + opt_type + "}(" + underlying + ")$",
    }
    
    return labels_dispatcher[greek_type]

def greeks_title_factory(ObjWithGreeksMethod, greek_type, underlying="S"):

    # plot common title
    common_title = ObjWithGreeksMethod.get_info() + "\n" + "Market at emission: " + ObjWithGreeksMethod.get_mkt_info()

    title_dispatcher = {"delta": r"$\Delta(" + underlying + ")$",
                         "theta": r"$\Theta(" + underlying + ")$",
                         "gamma": r"$\Gamma(" + underlying + ")$",
                         "vega":  r"$Vega(" + underlying + ")$",
                         "rho":  r"$\rho(" + underlying + ")$",
    }

    # plot title
    plot_title = r"Numeric " + greek_type + " " + title_dispatcher[greek_type] + " Vs $" + underlying + "$ for a \n" + common_title
    
    return plot_title

def main():

    # numeric greeks example
            
    # default market environment
    market_env = MarketEnvironment()
    print(market_env)

    # define option style and type
    opt_style = "plain_vanilla" # "digital"
    opt_type = "call" # "call"  
    option = option_factory(market_env, opt_style, opt_type)
    print(option)

    # numeric greeks instance
    NumGreeks = NumericGreeks(option)
    
    # underlying range at which compute greeks
    S_range = np.linspace(50, 150, 2000)

    # time-to-maturity range at which compute greeks
    tau_range = np.linspace(1e-4,1.0,1000)
    tau_range = homogenize(tau_range, reverse_order=True)

    # select greek
    for greek_type in ["delta", "theta", "gamma", "vega", "rho"]:
                
        #
        # greek Vs Underlying level S
        # 
            
        # numeric greek calculation
        greek_numeric_Vs_S = greeks_factory(NumGreeks, greek_type)(S=S_range)
        
        # labels
        label_numeric_S = greeks_label_factory(greek_type, opt_type, kind="num")
            
        # plot title
        plot_title_S = greeks_title_factory(option, greek_type)
        
        # plot
        plot(x=S_range, f=greek_numeric_Vs_S, x_label=r"$S$", 
             f_label=label_numeric_S, title=plot_title_S)  

        #
        # greek Vs residual time to maturity tau
        # 
                    
        # numeric greek calculation
        greek_numeric_Vs_tau = greeks_factory(NumGreeks, greek_type)(tau=tau_range)
    
        # labels
        label_numeric_tau = greeks_label_factory(greek_type, opt_type, 
                                                 kind="num", underlying=r"\tau")
            
        # plot title
        plot_title_tau = greeks_title_factory(option, greek_type, underlying=r"\tau")

        # plot
        plot(x=tau_range, f=greek_numeric_Vs_tau, x_label=r"$\tau$", 
             f_label=label_numeric_tau, title=plot_title_tau)  
        
#----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    
    main()    

