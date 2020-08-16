"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: portfolio.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This file contains the definition of Portfolio class.
"""

# ----------------------- standard imports ---------------------------------- #
# for NumPy arrays
import numpy as np

# ----------------------- sub-modules imports ------------------------------- #

from ..utils.utils import *


# -----------------------------------------------------------------------------#

class Portfolio:
    """
    Portfolio class modeling a portfolio of options. It implements an aggregation of EuropeanOption sub-classes 
    defining the .add_instrument() composition method which takes in input a FinancialInstrument (either
    PlainVanillaOption or DigitalOption) to be added to the portfolio.
    
    Attributes:
    -----------
    
        composition (List of Dicts): List of Dicts, each describing a single constituent FinancialInstrument, together
                                     with the position the portfolio is holding on it.
        info and mkt_info (Strings): information labels on portfolio and constituent instruments.
        S (Float):                   underlying value when the portfolio is formed.
        K (np.ndarray):              Strikes of constituent options.
        tau (np.ndarray):            Time(s) to maturity of constituent options, when the portfolio is formed.
        is_multi_horizon (Bool):     True if constituent options have different expiration dates.
        
    Public Methods:
    --------
    
        getters for all attributes
        
        setters for common attributes, not belonging to mkt_env
        
        payoff: float
            Computes the payoff of the portfolio.

        price: float
            Computes the Black-Scholes value of the portfolio.

        PnL: float
            Computes the P&L of the portfolio.
            
        delta: float
            Computes the Black-Scholes delta of the portfolio.

        theta: float
            Computes the Black-Scholes theta of the portfolio.

        gamma: float
            Computes the Black-Scholes gamma of the portfolio.

        vega: float
            Computes the Black-Scholes vega of the portfolio.

        rho: float
            Computes the Black-Scholes rho of the portfolio.

    Instantiation and Usage examples: 
    --------   
        
        - portfolio.py
        - portfolio_single_strike.py
        - example_portfolio_multi_strike.py
        - example_portfolio_bull_spread.py
        - example_portfolio_bull_spread_other_params.py
        - example_portfolio_calendar_spread.py
        - example_portfolio_calendar_spread_other_params.py

    """

    def __init__(self, name="Dummy"):

        # initialize an empty portfolio
        self.__composition = []

        # initialize empty info strings
        self.__info = "{} Portfolio: \n".format(name)
        self.__mkt_info = None

        # initialize valuation date, underlying value, strikes and times-to-maturity attributes
        self.__t = None
        self.__T = np.array([])
        self.__S = None
        self.__sigma = None
        self.__r = None
        self.__K = np.array([])
        self.__tau = np.array([])
        self.is_multi_horizon = False
        self.is_multi_strike = False
        self.is_empty = True

    def __repr__(self):
        return self.get_info()

    # 
    # getters
    #

    def get_info(self):
        return self.__info

    def get_mkt_info(self):
        return self.__mkt_info

    def get_t(self):
        return self.__t

    def get_T(self):
        return scalarize(self.__T)

    def get_K(self):
        return self.__K

    def get_S(self):
        return self.__S

    def get_sigma(self):
        return self.__sigma

    def get_r(self):
        return self.__r

    def get_tau(self):
        return self.__tau

    def get_composition(self):
        return self.__composition

    #
    # setters
    #

    def set_t(self, t):
        self.__t = t

    def set_S(self, S):
        self.__S = S

    def set_sigma(self, sigma):
        self.__sigma = sigma

    def set_r(self, r):
        self.__r = r

    #
    # Composition method
    #

    def add_instrument(self, FinancialInstrument, position):

        if self.is_empty:
            self.is_empty = False

        long_short = 'Long' if position > 0 else 'Short'
        instrument_info = long_short + " {} ".format(abs(position)) + FinancialInstrument.get_info()

        self.__composition.append({"instrument": FinancialInstrument,
                                   "position": position,
                                   "info": instrument_info})

        # update portfolio info strings
        self.__update_info(FinancialInstrument)

        # update portfolio attributes
        self.__update_t(FinancialInstrument)
        self.__update_T(FinancialInstrument)
        self.__update_S(FinancialInstrument)
        self.__update_sigma(FinancialInstrument)
        self.__update_r(FinancialInstrument)
        self.__update_K(FinancialInstrument)
        self.__update_tau(FinancialInstrument)

    # 
    # Private methods
    #

    def __update_info(self, fin_inst):
        self.__info += self.__composition[-1]["info"] + "\n"
        if self.__mkt_info is None:
            self.__mkt_info = fin_inst.get_mkt_info()

    def __update_t(self, fin_inst):
        if self.get_t() is None:
            self.set_t(fin_inst.get_t())
        else:
            if self.get_t() != fin_inst.get_t():
                raise ValueError(
                    "No multiple valuation dates in input allowed: \n\n current: {}, \n\n other given input: {}"
                    .format(self, self.get_t(), fin_inst.get_t()))

    def __update_T(self, fin_inst):
        expiration_dates = np.append(self.get_T(), fin_inst.get_T())
        # filter only distinct strikes
        self.__T = iterable_to_numpy_array(np.unique(expiration_dates), sort_func=date_string_to_datetime_obj)
        # check if the portfolio is a multi-horizon portfolio
        if len(self.__T) > 1:
            self.is_multi_horizon = True

    def __update_S(self, fin_inst):
        if self.get_S() is None:
            self.set_S(fin_inst.get_S())

    def __update_sigma(self, fin_inst):
        if self.get_sigma() is None:
            self.set_sigma(fin_inst.get_sigma())

    def __update_r(self, fin_inst):
        if self.get_r() is None:
            self.set_r(fin_inst.get_r())

    def __update_K(self, fin_inst):
        # append new instrument strike
        strikes = np.append(self.get_K(), fin_inst.get_K())
        # filter only distinct strikes
        self.__K = np.unique(strikes)
        # check if the portfolio is a multi-strike portfolio
        if len(self.__K) > 1:
            self.is_multi_strike = True
        else:
            self.__K = self.__K[0]

    def __update_tau(self, fin_inst):
        # append new instrument tau
        times_to_maturity = np.append(self.get_tau(), fin_inst.get_tau())
        # filter only distinct times-to-maturity
        self.__tau = np.unique(times_to_maturity)
        # consistency check
        if (len(self.__tau) > 1) and (not self.is_multi_horizon):
            raise AttributeError("Multi-horizon portfolio not properly handled: \n \tau = {}"
                                 .format(self.__tau))

    def check_parameters(self, *args, **kwargs):
        """Check both x-axis and time dimensional parameters."""

        # check x-axis
        self.__check_x_axis(**kwargs)

        # check time parameter
        self.__check_time_parameter(*args, **kwargs)

    def __check_x_axis(self, **kwargs):
        """Check that multi-strike portfolio do not get strike as input x-axis parameter,
        which is not well defined."""

        # x-axis parameter:
        strike = kwargs['K'] if 'K' in kwargs else None

        if self.is_multi_strike and strike is not None:
            raise NotImplementedError("No 'strike' x-axis parameter allowed for multi-strike portfolio.")

    def __check_time_parameter(self, *args, **kwargs):
        """Check that multi-horizon portfolio do not get time(s)-to-maturity as input time parameter,
        which is not well defined."""

        # time parameter:
        time_param = args[1] if len(args) > 1 \
            else kwargs['tau'] if 'tau' in kwargs \
            else (kwargs['t'] if 't' in kwargs else None)

        # Case of no time parameter in input allowed: sigma x r grid case
        if time_param is not None:

            # check that time parameter is not a time-to-maturity if the portfolio is multi-horizon:
            if self.is_multi_horizon and is_numeric(time_param):
                raise TypeError(
                    "No time-to-maturity time parameter allowed for multi-horizon portfolio: \n\n tau={} given in input"
                    .format(time_param))

                #

    # Public methods
    #

    def time_to_maturity(self, *args, **kwargs):
        """
        Utility method to compute time-to-maturity of the portfolio, 
        if possible (that is, if the portfolio is not multi-horizon and if it's
        not empty. Method taken from an instrument.
        """

        if self.is_empty:
            raise NotImplementedError("No time-to-maturity defined for empty portfolio")
        elif self.is_multi_horizon:
            raise NotImplementedError("No time-to-maturity defined for multi-horizon portfolio")
        else:
            return self.get_composition()[0]["instrument"].time_to_maturity(*args, **kwargs)

    def payoff(self, *args, **kwargs):
        """
        Returns the portfolio payoff as the scalar product (i.e. sum of elementwise products) 
        between single instrument payoffs and positions.
        
        Can be called with the same signature of the .payoff() public method of
        constituent options.
        """

        # check parameters
        self.check_parameters(*args, **kwargs)

        # portfolio payoff is the sum position * instrument_payoff
        return sum([inst["position"] * inst["instrument"].payoff(*args, **kwargs) for inst in self.get_composition()])

    def price(self, *args, **kwargs):
        """
        Returns the portfolio value as the scalar product (i.e. sum of elementwise products) 
        between single instrument prices and positions.
        
        Can be called with the same signature of the .price() public method of
        constituent options.
        """

        # check parameters
        self.check_parameters(*args, **kwargs)

        # portfolio price is the sum position * instrument_price
        return sum([inst["position"] * inst["instrument"].price(*args, **kwargs) for inst in self.get_composition()])

    def PnL(self, *args, **kwargs):
        """
        Returns the portfolio Profit & Loss as the scalar product (i.e. sum of elementwise products) 
        between single instrument P&Ls and positions.
        
        Can be called with the same signature of the .PnL() public method of
        constituent options.
        """

        # check parameters
        self.check_parameters(*args, **kwargs)

        # portfolio P&L is the sum position * instrument_payoff
        return sum([inst["position"] * inst["instrument"].PnL(*args, **kwargs) for inst in self.get_composition()])

    def delta(self, *args, **kwargs):
        """
        Returns the portfolio Delta as the scalar product (i.e. sum of elementwise products) 
        between single instrument Deltas and positions.
        
        Can be called with the same signature of the .delta() public method of
        constituent options.
        """

        # check parameters
        self.check_parameters(*args, **kwargs)

        # portfolio delta is the sum position * instrument_payoff
        return sum([inst["position"] * inst["instrument"].delta(*args, **kwargs) for inst in self.get_composition()])

    def theta(self, *args, **kwargs):
        """
        Returns the portfolio Theta as the scalar product (i.e. sum of elementwise products) 
        between single instrument Thetas and positions.
        
        Can be called with the same signature of the .theta() public method of
        constituent options.
        """

        # check parameters
        self.check_parameters(*args, **kwargs)

        # portfolio theta is the sum position * instrument_payoff
        return sum([inst["position"] * inst["instrument"].theta(*args, **kwargs) for inst in self.get_composition()])

    def gamma(self, *args, **kwargs):
        """
        Returns the portfolio Gamma as the scalar product (i.e. sum of elementwise products) 
        between single instrument Gammas and positions.
        
        Can be called with the same signature of the .gamma() public method of
        constituent options.
        """

        # check parameters
        self.check_parameters(*args, **kwargs)

        # portfolio gamma is the sum position * instrument_payoff
        return sum([inst["position"] * inst["instrument"].gamma(*args, **kwargs) for inst in self.get_composition()])

    def vega(self, *args, **kwargs):
        """
        Returns the portfolio Vega as the scalar product (i.e. sum of elementwise products) 
        between single instrument Vegas and positions.
        
        Can be called with the same signature of the .vega() public method of
        constituent options.
        """

        # check parameters
        self.check_parameters(*args, **kwargs)

        # portfolio vega is the sum position * instrument_payoff
        return sum([inst["position"] * inst["instrument"].vega(*args, **kwargs) for inst in self.get_composition()])

    def rho(self, *args, **kwargs):
        """
        Returns the portfolio Rho as the scalar product (i.e. sum of elementwise products) 
        between single instrument Rhos and positions.
        
        Can be called with the same signature of the .rho() public method of
        constituent options.
        """

        # check parameters
        self.check_parameters(*args, **kwargs)

        # portfolio rho is the sum position * instrument_payoff
        return sum([inst["position"] * inst["instrument"].rho(*args, **kwargs) for inst in self.get_composition()])
