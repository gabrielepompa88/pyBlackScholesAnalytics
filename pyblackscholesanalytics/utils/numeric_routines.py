"""
Author: Gabriele Pompa (gabriele.pompa@gmail.com)

Date: 21-May-2020
File name: numerical_routines.py
"""

# ----------------------- sub-modules imports ------------------------------- #

from .utils import *


# -----------------------------------------------------------------------------#

class NumericGreeks:
    """
    NumericGreeks class: a class implementing finite-differences numerical methods
    to compute main greeks for option contracts.

    Attributes:
    -----------
        FinancialObject (EuropeanOption sub-class or Portfolio):      Instance of an EuropeanOption sub-class 
                                                                      (PlainVanillaOption or DigitalOption) or Portfolio
                                                                      class.
       epsilon (float):                                               tolerance for finite-differences                                                                      

    Public Methods:
    --------   
    
        getter and setter for epsilon attribute
        
        delta: float
            Computes the numeric delta of the FinancialObject.

        theta: float
            Computes the numeric theta of the FinancialObject.

        gamma: float
            Computes the numeric gamma of the FinancialObject.

        vega: float
            Computes the numeric vega of the FinancialObject.

        rho: float
            Computes the numeric rho of the FinancialObject.
        
    Instantiation and Usage examples: 
    --------   
        
        - example_options_numeric_greeks.py
        - example_options_numeric_analytic_greeks_comparison.py
    """

    def __init__(self, FinancialObject, epsilon=1e-4):
        self.opt = FinancialObject
        self.f = FinancialObject.price
        self.__eps = epsilon

    def get_epsilon(self):
        return self.__eps

    def set_epsilon(self, eps=1e-4):
        self.__eps = eps

    def delta(self, **kwargs):
        """
        Numeric derivative df/dS.
        
        Can be called as self.opt.price() method.
        """
        # process parameter
        S0 = kwargs["S"] if "S" in kwargs else self.opt.get_S()
        S0 = homogenize(S0)

        # delete "S" from kwargs if it exists
        kwargs.pop("S", None)

        return (self.f(S=S0 + self.get_epsilon(), **kwargs) - self.f(S=S0 - self.get_epsilon(), **kwargs)) / (
                    2 * self.get_epsilon())

    def gamma(self, **kwargs):
        """
        Numeric derivative d^2f/dS^2.
        
        Can be called as self.opt.price() method.
        """
        # process parameter
        S0 = kwargs["S"] if "S" in kwargs else self.opt.get_S()
        S0 = homogenize(S0)

        # delete "S" from kwargs if it exists
        kwargs.pop("S", None)

        return (self.f(S=S0 - self.get_epsilon(), **kwargs) - 2.0 * self.f(S=S0, **kwargs) + self.f(
            S=S0 + self.get_epsilon(), **kwargs)) / (self.get_epsilon() * self.get_epsilon())

    def vega(self, **kwargs):
        """
        Numeric derivative df/dsigma. By default, it is scaled to consider 
        variation of +1% of sigma (not +100%).
        
        Can be called as self.opt.price() method.
        """
        # process parameter
        sigma0 = kwargs["sigma"] if "sigma" in kwargs else self.opt.get_sigma()
        sigma0 = homogenize(sigma0)

        # delete "sigma" from kwargs if it exists
        kwargs.pop("sigma", None)

        # rescaling factor
        rescaling_factor = kwargs["factor"] if "factor" in kwargs else 0.01

        return (self.f(sigma=sigma0 + self.get_epsilon(), **kwargs) - self.f(sigma=sigma0 - self.get_epsilon(),
                                                                             **kwargs)) / (
                           2 * self.get_epsilon()) * rescaling_factor

    def theta(self, **kwargs):
        """
        Numeric derivative df/dt = -df/dtau. By default, it is scaled to consider 
        variation of +1 calendar day of t (not +1 year).
        
        Can be called as self.opt.price() method.
        """
        # process parameter
        tau0 = kwargs["tau"] if "tau" in kwargs else self.opt.get_tau()
        tau0 = homogenize(tau0, reverse_order=True)

        # delete "tau" from kwargs if it exists
        kwargs.pop("tau", None)

        # rescaling factor
        rescaling_factor = kwargs["factor"] if "factor" in kwargs else 1.0 / 365.0

        return -((self.f(tau=tau0 + self.get_epsilon(), **kwargs) - self.f(tau=tau0 - self.get_epsilon(), **kwargs)) / (
                    2 * self.get_epsilon())) * rescaling_factor

    def rho(self, **kwargs):
        """
        Numeric derivative df/dr. By default, it is scaled to consider 
        variation of +1% of r (not +100%).
        
        Can be called as self.opt.price() method.
        """
        # process parameter
        r0 = kwargs["r"] if "r" in kwargs else self.opt.get_r()
        r0 = homogenize(r0)

        # delete "r" from kwargs if it exists
        kwargs.pop("r", None)

        # rescaling factor
        rescaling_factor = kwargs["factor"] if "factor" in kwargs else 0.01

        return ((self.f(r=r0 + self.get_epsilon(), **kwargs) - self.f(r=r0 - self.get_epsilon(), **kwargs)) / (
                    2 * self.get_epsilon())) * rescaling_factor
