"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: bull_spread_other_params.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script shows usage of Portfolio class to create a Bull-Spread option strategy. Instantiation examples are provided
with combinations of the underlying level (S) and time parameter (t/tau) as well as underlying volatility (sigma) and
short-rate (r) parameters. Price, P&L, first-order greeks are computed and plotted using the Plotter class as line plots
and surfaces-plots Vs time parameter.
"""

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.portfolio.portfolio import Portfolio
from pyblackscholesanalytics.options.options import PlainVanillaOption
from pyblackscholesanalytics.plotter.plotter import PortfolioPlotter


def options_x_axis_parameters_factory(option, parameter_name):
    param_dict = {"S": option.get_S(),
                  "K": option.get_K(),
                  "sigma": option.get_sigma(),
                  "r": option.get_r()}

    # if we want to have the x-axis spanned by sigma or r, we have to explicitly
    # ask for it, setting "sigma_axis" or "r_axis" flags to True. 
    # Otherwise, sigma and r parameters are interpreted as parameters to be 
    # distributed along the other(s) axis (and require length/shape match)
    if parameter_name in ["sigma", "r"]:
        return {parameter_name: param_dict[parameter_name],
                parameter_name + "_axis": True}
    else:
        return {parameter_name: param_dict[parameter_name]}


def get_azimut_angle(parameter_name):
    angles_dict = {"S": {"x-axis side": 180,
                         "Date side": -90},
                   "K": {"x-axis side": -90,
                         "Date side": 180},
                   "sigma": {"x-axis side": -90,
                             "Date side": 180},
                   "r": {"x-axis side": -90,
                         "Date side": 180}}

    return angles_dict[parameter_name]


def get_time_parameter(option, kind='date'):
    # date time-parameter
    if kind == 'date':

        # valuation date of the option
        emission_date = option.get_t()

        # emission/expiration date of the option
        expiration_date = option.get_T()

        # time-parameter as a date-range of 5 valuation dates between t and T-10d
        time_parameter = pd.date_range(start=emission_date,
                                       end=expiration_date - pd.Timedelta(days=20),
                                       periods=5)

    # time-to-maturity time parameter    
    else:

        # time-parameter as a list of times-to-maturity
        time_parameter = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    return time_parameter


def main():
    # Bull-Spread implementation example

    # default market environment
    market_env = MarketEnvironment()
    print(market_env)

    # options strikes
    K_long = 80
    K_short = 110

    # bull-spread portfolio initialized (as empty portfolio)   
    bull_spread_ptf = Portfolio(name="Bull Spread Strategy")
    print(bull_spread_ptf)

    # 80-call
    Vanilla_Call_long = PlainVanillaOption(market_env, K=K_long, T='31-12-2021')
    print(Vanilla_Call_long)

    # 110-call
    Vanilla_Call_short = PlainVanillaOption(market_env, K=K_short, T='31-12-2021')
    print(Vanilla_Call_short)

    # creation of bull-spread portfolio strategy   
    bull_spread_ptf.add_instrument(Vanilla_Call_long, 1)
    bull_spread_ptf.add_instrument(Vanilla_Call_short, -1)
    print(bull_spread_ptf)

    # portfolio plotter instance
    bull_spread_ptf_plotter = PortfolioPlotter(bull_spread_ptf)

    # select dependency to plot as x-axis of the plot 
    # (strike 'K' is skipped because a bull-spread is a multi-strike portfolio)
    for dependency_type in ["S", "sigma", "r"]:

        # keyboard parameter and corresponding range to test
        x_axis_dict = options_x_axis_parameters_factory(bull_spread_ptf, dependency_type)

        # appropriate azimut angle for best viewing
        azimut_angle = get_azimut_angle(dependency_type)

        # select metrics to plot
        for plot_metrics in ["price", "PnL", "delta", "theta", "gamma", "vega", "rho"]:

            plot_details_flag = True if plot_metrics == "price" else False

            # Bull-Spread price plot
            bull_spread_ptf_plotter.plot(**x_axis_dict, t='01-06-2020', plot_metrics=plot_metrics,
                                         plot_details=plot_details_flag)

            for time_kind in ['date', 'tau']:
                # set time-parameter to plot
                multiple_valuation_dates = get_time_parameter(bull_spread_ptf, kind=time_kind)
                print(multiple_valuation_dates)

                # Plot at multiple dates
                bull_spread_ptf_plotter.plot(**x_axis_dict, t=multiple_valuation_dates,
                                             plot_metrics=plot_metrics)

                # Surface plot
                bull_spread_ptf_plotter.plot(**x_axis_dict, t=multiple_valuation_dates,
                                             plot_metrics=plot_metrics, surf_plot=True)

                # Surface plot (rotate) - x-axis side
                bull_spread_ptf_plotter.plot(**x_axis_dict, t=multiple_valuation_dates,
                                             plot_metrics=plot_metrics, surf_plot=True,
                                             view=(0, azimut_angle["x-axis side"]))

                # Price surface plot (rotate) - Date side
                bull_spread_ptf_plotter.plot(**x_axis_dict, t=multiple_valuation_dates,
                                             plot_metrics=plot_metrics, surf_plot=True,
                                             view=(0, azimut_angle["Date side"]))


# ----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    main()
