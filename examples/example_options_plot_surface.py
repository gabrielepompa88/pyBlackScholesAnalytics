"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: example_options_plot_surface.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This script shows integration of PlainVanillaOption and DigitalOption classes
with Plotter class. Price, P&L and first-order greeks plots are shown for 
plain-vanilla and digital option contracts as underlying level (S), 
strike-price (K), volatility (sigma) and short-rate (r) surface-plots Vs time 
parameter.
"""

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from pyblackscholesanalytics.market.market import MarketEnvironment
from pyblackscholesanalytics.options.options import PlainVanillaOption, DigitalOption
from pyblackscholesanalytics.plotter.plotter import OptionPlotter

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
    
    # vanilla call implementation example
            
    # default market environment
    market_env = MarketEnvironment()
    print(market_env)
    
    # define option style and type
    opt_style = "plain_vanilla" # "digital"
    opt_type = "call" # "put"   
    option = option_factory(market_env, opt_style, opt_type)
    print(option)
        
    # option plotter instance
    plotter = OptionPlotter(option)
        
    # valuation date of the option
    emission_date = option.get_t()
            
    # select dependency to plot as x-axis of the plot
    for dependency_type in ["S", "K", "sigma", "r"]:

        # keyboard parameter and corresponding range to test
        x_axis_dict = options_x_axis_parameters_factory(option, dependency_type)    

        # appropriate azimut angle for best viewing
        azimut_angle = get_azimut_angle(dependency_type)

        # select metrics to plot
        for plot_metrics in ["price", "PnL", "delta", "theta", "gamma", "vega", "rho"]:
            
            for time_kind in ['date', 'tau']:
    
                # set time-parameter to plot
                multiple_valuation_dates = get_time_parameter(option, kind=time_kind)
                print(multiple_valuation_dates)
                                        
                # Surface plot
                plotter.plot(**x_axis_dict, t=multiple_valuation_dates, 
                             plot_metrics=plot_metrics, surf_plot=True)
                            
                # Surface plot (rotate) - x-axis side
                plotter.plot(**x_axis_dict, t=multiple_valuation_dates, 
                             plot_metrics=plot_metrics, surf_plot=True, 
                             view=(0,azimut_angle["x-axis side"]))
            
                # Price surface plot (rotate) - Date side
                plotter.plot(**x_axis_dict, t=multiple_valuation_dates, 
                             plot_metrics=plot_metrics, surf_plot=True, 
                             view=(0,azimut_angle["Date side"]))
            
#----------------------------- usage example ---------------------------------#
if __name__ == "__main__":
    
    main()    

