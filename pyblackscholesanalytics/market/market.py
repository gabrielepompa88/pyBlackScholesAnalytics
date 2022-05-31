"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: market.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This file contains the definition of MarketEnvironment class.
"""

from ..utils.loggingConfig import logger
# ----------------------- sub-modules imports ------------------------------- #

from ..utils.utils import date_string_to_datetime_obj


# -----------------------------------------------------------------------------#

class MarketEnvironment:
    """
    MarketEnvironment class: setting the market environment: date, underlying level, volatility level and short-rate.
    
    Attributes:
    -----------
        t (str; dt.datetime):     valuation date. Can be either a "dd-mm-YYYY" String or a dt.datetime object
        r (float):                continuously compounded short-rate;
        S_t (float):              spot price of the underlying asset at the valuation date 't';
        sigma (float):            volatility of underlying asset;
    
    Public Methods:
    --------   
    
        getters and setters for all attributes
        
    Usage examples: 
    --------   
    
        See docstrings for EuropeanOption and Portfolio classes.

    Instantiation: 
    --------   
        - default: MarketEnvironment() is equivalent to 
                   MarketEnvironment(t="19-04-2020", r=0.05, S_t=90.0, sigma=0.2)

        - general: MarketEnvironment(t="DD-MM-YYYY" String, r=Float, S_t=Float, sigma=Float)
    """

    def __init__(self, t="19-04-2020", r=0.05, S_t=90.0, sigma=0.2):
        logger.debug("Initializing the MarketEnvironment!")

        self.__t = date_string_to_datetime_obj(t)
        self.__r = r
        self.__S = S_t
        self.__sigma = sigma

    def __repr__(self):
        return r"MarketEnvironment(t={}, r={:.1f}%, S_t={:.1f}, sigma={:.1f}%)". \
            format(self.get_t().strftime("%d-%m-%Y"), self.get_r() * 100, self.get_S(), self.get_sigma() * 100)

    # getters
    def get_t(self):
        return self.__t

    def get_r(self):
        return self.__r

    def get_S(self):
        return self.__S

    def get_sigma(self):
        return self.__sigma

    # setters 
    def set_t(self, t):
        self.__t = date_string_to_datetime_obj(t)

    def set_r(self, r):
        self.__r = r

    def set_S(self, S):
        self.__S = S

    def set_sigma(self, sigma):
        self.__sigma = sigma
