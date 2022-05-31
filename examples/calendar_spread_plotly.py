"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: calendar_spread.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script shows usage of Portfolio class to create a Calendar-Spread option strategy. Basic instantiation examples are
provided with combinations of the underlying level (S) and time parameter (t). Price, P&L, first-order greeks are
computed and plotted using the Plotter class as line plots and surface-plots Vs time parameter.
"""

import pandas as pd
import warnings
import logging
from pyblackscholesanalytics.utils.loggingConfig import logger

warnings.filterwarnings("ignore")

logger.setLevel(logging.DEBUG)
logger = logging.getLogger('pyBlackScholesAnalytics')


from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.portfolio.portfolio import Portfolio
from pyblackscholesanalytics.options.options import PlainVanillaOption
from pyblackscholesanalytics.utils.utils import date_string_to_datetime_obj
from pyblackscholesanalytics.plotter.plotly import PlotlyPortfolioPlotter

def main():
    # Calendar-Spread implementation example

    # default market environment
    market_env = MarketEnvironment()
    market_env.set_S(23.94) # spot price
    market_env.set_r(0.1175) # risk-free tax
    market_env.set_t("29-04-2022") # actual date
    market_env.set_sigma(0.2382) # volatility

    logger.info(market_env)

    # options expirations
    T_short = "20-05-2022"
    T_long = "17-06-2022"

    # current underlying level
    S_t = market_env.get_S()

    # calendar-spread portfolio initialized (as empty portfolio)   
    calendar_spread_ptf = Portfolio(name="Calendar Spread Strategy")
    logger.info(calendar_spread_ptf)

    # T_long-call
    Vanilla_Call_long = PlainVanillaOption(market_env, T=T_long, K=25.40)
    logger.info(Vanilla_Call_long)

    # T_short-call
    Vanilla_Call_short = PlainVanillaOption(market_env, T=T_short, K=25.40)
    logger.info(Vanilla_Call_short)

    # creation of Calendar-Spread portfolio strategy   
    calendar_spread_ptf.add_instrument(Vanilla_Call_long, 1)
    calendar_spread_ptf.add_instrument(Vanilla_Call_short, -1)
    logger.info(calendar_spread_ptf)

    calendar_spread_ptf_plotter = PlotlyPortfolioPlotter(calendar_spread_ptf)

    # valuation date of the portfolio
    valuation_date = calendar_spread_ptf.get_t()
    logger.info(valuation_date)

    # select metrics to plot
    #for plot_metrics in ["price", "PnL", "delta", "theta", "gamma", "vega", "rho"]:
    for plot_metrics in ["price", "PnL"]:
        plot_details_flag = True if plot_metrics == "price" else False

        # time-parameter as a date-range of 5 valuation dates between t and T_short
        # being the Calendar-Spread a multi-horizon portfolio, time-to-maturity
        # time parameters are not allowed.
        last_date = T_short if plot_metrics in ["price", "PnL"] else date_string_to_datetime_obj(T_short) - \
                                                                     pd.Timedelta(days=1)
        multiple_valuation_dates = pd.date_range(start=valuation_date,
                                                 end=last_date,
                                                 periods=5)
        logger.info(multiple_valuation_dates)

        # Bull-Spread price plot
        calendar_spread_ptf_plotter.plot(t=last_date, plot_metrics=plot_metrics,
                                         plot_details=plot_details_flag)

        # Plot at multiple dates
        calendar_spread_ptf_plotter.plot(t=multiple_valuation_dates, plot_metrics=plot_metrics)

        # Surface plot
        calendar_spread_ptf_plotter.plot(t=multiple_valuation_dates, plot_metrics=plot_metrics,
                                         surf_plot=True)

        # Surface plot (rotate) - Underlying value side
        calendar_spread_ptf_plotter.plot(t=multiple_valuation_dates, plot_metrics=plot_metrics,
                                         surf_plot=True, view=(0, 180))

        # Price surface plot (rotate) - Date side
        calendar_spread_ptf_plotter.plot(t=multiple_valuation_dates, plot_metrics=plot_metrics,
                                         surf_plot=True, view=(0, -90))


# ----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    main()
