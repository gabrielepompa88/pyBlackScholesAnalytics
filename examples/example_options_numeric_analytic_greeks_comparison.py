"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: example_options_numeric_analytic_greeks_comparison.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script provides a comparison of first-order greeks for plain-vanilla and 
digital option contracts implemented either through finite-difference methods 
in NumericGreeks class or using their analytic expression implemented in 
PlainVanillaOption and DigitalOption classes.
"""

import numpy as np

from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.options.options import PlainVanillaOption, DigitalOption
from pyblackscholesanalytics.utils.numeric_routines import NumericGreeks
from pyblackscholesanalytics.utils.utils import plot_compare, homogenize


def option_factory(mkt_env, plain_or_digital, option_type, **kwargs):
    option_dispatcher = {
        "plain_vanilla": {"call": PlainVanillaOption(mkt_env, **kwargs),
                          "put": PlainVanillaOption(mkt_env, option_type="put", **kwargs)
                          },
        "digital": {"call": DigitalOption(mkt_env, **kwargs),
                    "put": DigitalOption(mkt_env, option_type="put", **kwargs)
                    }
    }

    return option_dispatcher[plain_or_digital][option_type]


def greeks_factory(ObjWithGreeksMethod, greek_type):
    greeks_dispatcher = {"delta": ObjWithGreeksMethod.delta,
                         "theta": ObjWithGreeksMethod.theta,
                         "gamma": ObjWithGreeksMethod.gamma,
                         "vega": ObjWithGreeksMethod.vega,
                         "rho": ObjWithGreeksMethod.rho
                         }

    return greeks_dispatcher[greek_type]


def greeks_label_factory(greek_type, opt_type, kind, underlying="S"):
    labels_dispatcher = {"delta": r"\Delta^{" + kind + "}_{" + opt_type + "}(" + underlying + ")",
                         "theta": r"\Theta^{" + kind + "}_{" + opt_type + "}(" + underlying + ")",
                         "gamma": r"\Gamma^{" + kind + "}_{" + opt_type + "}(" + underlying + ")",
                         "vega": r"Vega^{" + kind + "}_{" + opt_type + "}(" + underlying + ")",
                         "rho": r"\rho^{" + kind + "}_{" + opt_type + "}(" + underlying + ")",
                         }

    return labels_dispatcher[greek_type]


def greeks_title_factory(ObjWithGreeksMethod, greek_type, underlying="S"):
    # plot common title
    common_title = ObjWithGreeksMethod.get_info() + "\n" + "Market at emission: " + ObjWithGreeksMethod.get_mkt_info()

    title_dispatcher = {"delta": r"$\Delta(" + underlying + ")$",
                        "theta": r"$\Theta(" + underlying + ")$",
                        "gamma": r"$\Gamma(" + underlying + ")$",
                        "vega": r"$Vega(" + underlying + ")$",
                        "rho": r"$\rho(" + underlying + ")$",
                        }

    # plot title
    plot_title = r"Comparison of Numeric and Exact " + greek_type + " " + title_dispatcher[
        greek_type] + " Vs $" + underlying + "$ for a \n" + common_title

    return plot_title


def main():
    # numeric Vs analytic greeks example

    # default market environment
    market_env = MarketEnvironment()
    print(market_env)

    # define option style and type
    opt_style = "plain_vanilla"  # "digital"
    opt_type = "call"  # "put"
    option = option_factory(market_env, opt_style, opt_type)
    print(option)

    # select greek
    for greek_type in ["delta", "theta", "gamma", "vega", "rho"]:
        # numeric greeks instance
        NumGreeks = NumericGreeks(option)

        #
        # greek Vs Underlying level S
        # 

        # underlying range at which compute greeks
        S_range = np.linspace(50, 150, 2000)

        # analytic greek
        greek_analytic_Vs_S = greeks_factory(option, greek_type)(S=S_range)

        # numeric greek
        greek_numeric_Vs_S = greeks_factory(NumGreeks, greek_type)(S=S_range)

        # labels
        label_numeric_S = greeks_label_factory(greek_type, opt_type, kind="num")
        label_analytic_S = greeks_label_factory(greek_type, opt_type, kind="exact")

        # plot title
        plot_title_S = greeks_title_factory(option, greek_type)

        # comparison
        plot_compare(S_range, f=greek_numeric_Vs_S, f_ref=greek_analytic_Vs_S,
                     f_label=label_numeric_S, f_ref_label=label_analytic_S,
                     x_label="S", f_test_name="Numeric " + greek_type,
                     f_ref_name="Exact " + greek_type, title=plot_title_S)

        #
        # greek Vs residual time to maturity tau
        # 

        # time-to-maturity range at which compute greeks
        tau_range = np.linspace(1e-4, 1.0, 1000)
        tau_range = homogenize(tau_range, reverse_order=True)

        # analytic greek
        greek_analytic_Vs_tau = greeks_factory(option, greek_type)(tau=tau_range)

        # numeric greek
        greek_numeric_Vs_tau = greeks_factory(NumGreeks, greek_type)(tau=tau_range)

        # labels
        label_numeric_tau = greeks_label_factory(greek_type, opt_type,
                                                 kind="num", underlying=r"\tau")
        label_analytic_tau = greeks_label_factory(greek_type, opt_type,
                                                  kind="exact", underlying=r"\tau")

        # plot title
        plot_title_tau = greeks_title_factory(option, greek_type, underlying=r"\tau")

        # comparison
        plot_compare(tau_range, f=greek_numeric_Vs_tau, f_ref=greek_analytic_Vs_tau,
                     f_label=label_numeric_tau, f_ref_label=label_analytic_tau,
                     x_label=r"\tau", f_test_name="Numeric " + greek_type,
                     f_ref_name="Exact " + greek_type, title=plot_title_tau)


# ----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    main()
