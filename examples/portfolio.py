"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: portfolio.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script shows basic usage of Portfolio class to construct a derivative portfolio of plain-vanilla and digital option
contracts. Basic instantiation examples are provided with combinations of the underlying level (S) and time parameter
(t/tau). Price, P&L, first-order greeks are computed for constructed portfolio and benchmarked with the corresponding
metrics calculated combining constituent options metrics.
"""

import numpy as np
import pandas as pd

from pyblackscholesanalytics.utils.utils import date_string_to_datetime_obj
from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.options.options import PlainVanillaOption
from pyblackscholesanalytics.portfolio.portfolio import Portfolio


def get_time_parameter(mkt_env, end_date, periods, kind='date', multi_horizon_ptf=True):
    if kind == 'date':

        # a date-range of 5 valuation dates between t and the nearest maturity
        t = pd.date_range(start=mkt_env.get_t(), end=end_date, periods=periods)
        print("t ([t...T] pd.date_range): {}\n".format(t))

    else:

        if multi_horizon_ptf:
            raise TypeError("No time-to-maturity time parameter allowed for multi-horizon portfolio")
        else:
            t = np.array([0.1 * (1 + i) for i in range(periods)])
            print("t (list of times-to-maturity): {}\n".format(t))

    return t


def main():
    #
    # portfolio instantiation example
    #

    # if np_output is True, the output will be np.ndarray, otherwise pd.DataFrame    
    np_output = False  # True

    # default market environment
    market_env = MarketEnvironment(t="01-06-2020")
    print(market_env)

    # underlying values to test
    S_vector = [60, 90, 120]
    print("S_vector: {}\n".format(S_vector))

    # options maturities
    T_call = "31-12-2020"
    T_put = "30-06-2021"  # T_call

    # choose the kind of time-parameter to use: either a date ('date') or a 
    # time-to-maturity ('ttm'). Time-to-maturity time parameter is not allowed 
    # for multi-horizon portfolios.
    time_parameter = 'date'  # 'ttm'

    # get time parameter
    t_range = get_time_parameter(market_env,
                                 end_date=min(T_call, T_put, key=date_string_to_datetime_obj),
                                 periods=5,
                                 kind=time_parameter,
                                 multi_horizon_ptf=T_call != T_put)

    # options strikes
    K_put = 80
    K_call = 110

    # portfolio options positions
    call_pos = 2
    put_pos = -5

    #
    # Step 0: empty portfolio initialized
    #

    ptf = Portfolio()
    print(ptf)

    #
    # Step 1: adding 2 long plain-vanilla call contracts
    #

    # plain-vanilla call option
    call = PlainVanillaOption(market_env, K=K_call, T=T_call)
    print(call)

    # adding contract to portfolio  
    ptf.add_instrument(call, call_pos)
    print(ptf)

    # metrics to compare
    for metrics in ["price", "PnL", "delta", "theta", "gamma", "vega", "rho"]:
        # portfolio metrics
        ptf_metrics = getattr(ptf, metrics)(S=S_vector, t=t_range, np_output=np_output)
        print("\nPortfolio {}:\n{}".format(metrics, ptf_metrics))

        # verification with benchmark metrics
        call_metrics = getattr(call, metrics)(S=S_vector, t=t_range, np_output=np_output)
        benchmark_metrics = call_pos * call_metrics
        print("\nBenchmark {}:\n{}".format(metrics, benchmark_metrics))

        # check effective match
        diff = (ptf_metrics - benchmark_metrics).astype('float')
        num_nonzero_diff = np.count_nonzero(diff) - np.isnan(diff).sum().sum()
        exact_match = True if num_nonzero_diff == 0 else False
        print("\nIs replication exact (NaN excluded)? {}\n".format(exact_match))

    #
    # Step 2: adding 5 short plain-vanilla put contracts
    #

    # plain-vanilla put option
    put = PlainVanillaOption(market_env, option_type="put", K=K_put, T=T_put)
    print(put)

    # adding contract to portfolio  
    ptf.add_instrument(put, put_pos)
    print(ptf)

    # metrics to compare
    for metrics in ["price", "PnL", "delta", "theta", "gamma", "vega", "rho"]:
        # portfolio metrics
        ptf_metrics = getattr(ptf, metrics)(S=S_vector, t=t_range, np_output=np_output)
        print("\nPortfolio {}:\n{}".format(metrics, ptf_metrics))

        # verification with benchmark metrics
        call_metrics = getattr(call, metrics)(S=S_vector, t=t_range, np_output=np_output)
        put_metrics = getattr(put, metrics)(S=S_vector, t=t_range, np_output=np_output)
        benchmark_metrics = call_pos * call_metrics + put_pos * put_metrics
        print("\nBenchmark {}:\n{}".format(metrics, benchmark_metrics))

        # check effective match
        diff = (ptf_metrics - benchmark_metrics).astype('float')
        num_nonzero_diff = np.count_nonzero(diff) - np.isnan(diff).sum().sum()
        exact_match = True if num_nonzero_diff == 0 else False
        print("\nIs replication exact (NaN excluded)? {}\n".format(exact_match))


# ----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    main()
