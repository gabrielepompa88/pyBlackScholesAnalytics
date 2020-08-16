"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: example_calendar_spread_other_params.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script shows usage of Portfolio class to create a Calendar-Spread option 
strategy. Instantiation examples are provided with combinations of the underlying
level (S) and time parameter (t) as well as underlying volatility (sigma) 
and short-rate (r) parameters. Price, P&L, first-order greeks are computed and 
plotted using the Plotter class as line plots and surfaces-plots Vs time 
parameter.
"""

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.portfolio.portfolio import Portfolio
from pyblackscholesanalytics.options.options import PlainVanillaOption
from pyblackscholesanalytics.plotter.plotter import PortfolioPlotter
from pyblackscholesanalytics.utils.utils import date_string_to_datetime_obj


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
    angles_dict = {"S": {"x-axis side": -90,
                         "Date side": 180},
                   "K": {"x-axis side": -90,
                         "Date side": 180},
                   "sigma": {"x-axis side": -90,
                             "Date side": 180},
                   "r": {"x-axis side": -90,
                         "Date side": 180}}

    return angles_dict[parameter_name]


def main():
    # Calendar-Spread implementation example

    # default market environment
    market_env = MarketEnvironment()
    print(market_env)

    # options expirations
    T_short = "31-05-2020"
    T_long = "30-08-2020"

    # current underlying level
    S_t = market_env.get_S()

    # calendar-spread portfolio initialized (as empty portfolio)   
    calendar_spread_ptf = Portfolio(name="Calendar Spread Strategy")
    print(calendar_spread_ptf)

    # T_long-call
    Vanilla_Call_long = PlainVanillaOption(market_env, T=T_long, K=S_t)
    print(Vanilla_Call_long)

    # T_short-call
    Vanilla_Call_short = PlainVanillaOption(market_env, T=T_short, K=S_t)
    print(Vanilla_Call_short)

    # creation of Calendar-Spread portfolio strategy   
    calendar_spread_ptf.add_instrument(Vanilla_Call_long, 1)
    calendar_spread_ptf.add_instrument(Vanilla_Call_short, -1)
    print(calendar_spread_ptf)

    # portfolio plotter instance
    calendar_spread_ptf_plotter = PortfolioPlotter(calendar_spread_ptf)

    # valuation date of the portfolio
    valuation_date = calendar_spread_ptf.get_t()
    print(valuation_date)

    # select dependency to plot as x-axis of the plot 
    for dependency_type in ["S", "K", "sigma", "r"]:

        # keyboard parameter and corresponding range to test
        x_axis_dict = options_x_axis_parameters_factory(calendar_spread_ptf, dependency_type)

        # appropriate azimut angle for best viewing
        azimut_angle = get_azimut_angle(dependency_type)

        # select metrics to plot
        for plot_metrics in ["price", "PnL", "delta", "theta", "gamma", "vega", "rho"]:
            plot_details_flag = True if plot_metrics == "price" else False

            # time-parameter as a date-range of 5 valuation dates between t and T_short
            # being the Calendar-Spread a multi-horizon portfolio, time-to-maturity
            # time parameters are not allowed.
            last_date = T_short if plot_metrics in ["price", "PnL"] else date_string_to_datetime_obj(
                T_short) - pd.Timedelta(days=1)
            multiple_valuation_dates = pd.date_range(start=valuation_date,
                                                     end=last_date,
                                                     periods=5)
            print(multiple_valuation_dates)

            # Bull-Spread price plot
            calendar_spread_ptf_plotter.plot(**x_axis_dict, t=last_date, plot_metrics=plot_metrics,
                                             plot_details=plot_details_flag)

            # Plot at multiple dates
            calendar_spread_ptf_plotter.plot(**x_axis_dict, t=multiple_valuation_dates, plot_metrics=plot_metrics)

            # Surface plot
            calendar_spread_ptf_plotter.plot(**x_axis_dict, t=multiple_valuation_dates,
                                             plot_metrics=plot_metrics, surf_plot=True)

            # Surface plot (rotate) - Underlying value side
            calendar_spread_ptf_plotter.plot(**x_axis_dict, t=multiple_valuation_dates,
                                             plot_metrics=plot_metrics, surf_plot=True,
                                             view=(0, azimut_angle["x-axis side"]))

            # Price surface plot (rotate) - Date side
            calendar_spread_ptf_plotter.plot(**x_axis_dict, t=multiple_valuation_dates,
                                             plot_metrics=plot_metrics, surf_plot=True,
                                             view=(0, azimut_angle["Date side"]))


# ----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    main()
