"""
Created by: Gabriele Pompa (gabriele.pompa@gmail.com)

File: options.py

Created on Tue Jul 14 2020 - Version: 1.0

Description: 
    
This file contains definitions for EuropeanOption abstract base-class as well 
as PlainVanillaOption and DigitalOption derived classes.
"""

# ----------------------- standard imports ---------------------------------- #
# for NumPy arrays
import numpy as np

# for Pandas Series and DataFrame
import pandas as pd

# for statistical functions
from scipy import stats

# for optimization routines
import scipy.optimize as sc_opt

# for some mathematical functions
import math

# for date management
import datetime as dt

# for warning messages
import warnings

# ----------------------- sub-modules imports ------------------------------- #

from pyblackscholesanalytics.utils.utils import *

#-----------------------------------------------------------------------------#

class EuropeanOption:
    """
    EuropeanOption abstract class: an interface setting the template for any option with european-style exercise.
    It uses a MarketEnvironment object to define the current market conditions under which the option is modeled. 
    This class is not meant to be instantiated.
    
    Attributes:
    -----------
        mkt_env (MarketEnvironment): Instance of MarketEnvironment class
        type (str):                  Optional. Type of the option. Can be either 'call' or 'put';
        S_t (float):                 'S' attribute of mkt_env.
        K (float):                   Optional. Strike price;
        t (str; dt.datetime):        't' attribute of mkt_env.
        T (str; dt.datetime):        Optional. Expiration date. Can be either a "dd-mm-YYYY" String or a dt.datetime object
        tau (float):                 time to maturity in years, computed as tau=T-t by time_to_maturity() method
        r (float):                   'r' attribute of mkt_env.
        sigma (float):               'sigma' attribute of mkt_env.

    Public Methods:
    --------
    
        time_to_maturity: float
            Computes the time-to-maturity of the option.
        
        process_pricing_parameters: float
            Parses underlying, strike-price, time, volatility and short-rate parameters, 
            discriminating between time-to-maturity and valuation date
            time parameter and coordinating pricing parameters together.
    
        d1_and_d2: flaot, float
            Computes the d1 and d2 terms of Black-Scholes pricing formula

        payoff: float
            Computes the payoff of the option.
            
        price: float
            Computes the Black-Scholes price of the option.
            
        PnL: float
            Computes the P&L of the option.

        implied_volatility: float
            Computes the Black-Scholes implied-volatility of the option.

        delta: float
            Computes the Black-Scholes delta of the option.

        theta: float
            Computes the Black-Scholes theta of the option.

        gamma: float
            Computes the Black-Scholes gamma of the option.

        vega: float
            Computes the Black-Scholes vega of the option.

        rho: float
            Computes the Black-Scholes rho of the option.

    Template Methods:
    --------   
    
        getters for all common private attributes
        
        setters for common private attributes, not belonging to mkt_env
        
        price_upper_limit: float 
            Template method for upper limit. Raises NotImplementedError if called.

        price_lower_limit: float 
            Template method for lower limit. Raises NotImplementedError if called.
            
    Usage examples: 
    --------   
        
        - example_options.py
        - example_options_other_params.py
        - example_options_IV.py
        - example_options_numeric_analytic_greeks_comparison.py

    """

    def __init__(self, mkt_env, option_type='call', K=100.0, T="31-12-2020"):
        
        print("Initializing the EuropeanOption!")

        # option type check
        if option_type not in ['call', 'put']:
            raise NotImplementedError("Option Type: '{}' does not exist!".format(option_type))
        
        self.__type  = option_type
        self.__S     = mkt_env.get_S()
        self.__K     = K
        self.__t     = mkt_env.get_t()
        self.__T     = date_string_to_datetime_obj(T)
        self.__tau   = self.time_to_maturity()
        self.__r     = mkt_env.get_r()
        self.__sigma = mkt_env.get_sigma()
        
        # empty initial price of the option
        self.__initial_price = None
               
        # empty informations dictionary
        self.__docstring_dict = {}        
        
    # string representation method template
    def __repr__(self):
        raise NotImplementedError()
    
    #
    # getters
    #
    
    def get_type(self):
        return self.__type

    def get_S(self):
        return self.__S
    
    def get_K(self):
        return self.__K
    
    def get_t(self):
        return self.__t

    def get_T(self):
        return self.__T

    def get_tau(self):
        return self.__tau

    def get_r(self):
        return self.__r
    
    def get_sigma(self):
        return self.__sigma
        
    def get_initial_price(self):
        return NotImplementedError()
    
    # doctring getter template
    def get_docstring(self, label):
        raise NotImplementedError()

    #
    # setters
    #
    
    def set_type(self, option_type):
        self.__type = option_type
        
        # option type check
        if option_type not in ['call', 'put']:
            raise NotImplementedError("Option Type: '{}' does not exist!".format(option_type))
            
    def set_K(self, K):
        self.__K = K
    
    def set_T(self, T):
        self.__T = date_string_to_datetime_obj(T)
        # update time to maturity, given changed T, to keep internal consistency
        self.__update_tau() 
    
    def set_tau(self, tau):
        self.__tau = tau
        # update expiration date, given changed tau, to keep internal consistency
        self.__update_T()
        
    #
    # update methods (private)
    #
    
    def __update_tau(self):
        self.__tau = self.time_to_maturity()

    def __update_T(self):
        self.__T = self.__t + dt.timedelta(days=math.ceil(self.__tau*365))

    #
    # utility methods
    #
    
    def time_to_maturity(self, *args, **kwargs):
        """
        Utility method to compute time-to-maturity
        """
        
        # parsing optional parameters
        t = args[0] if len(args) > 0 else kwargs['t'] if 't' in kwargs else self.get_t()
        T = args[1] if len(args) > 1 else kwargs['T'] if 'T' in kwargs else self.get_T()
        
        # convert to dt.datetime objects, if needed
        t = date_string_to_datetime_obj(t)
        T = date_string_to_datetime_obj(T)
        
        # compute and return time to maturity (in years)
        return homogenize((T-t).days / 365.0, sort=False)
      
    def process_pricing_parameters(self, *args, **kwargs):
        """
        Utility method to parse underlying, strike-price, time, volatility and 
        short-rate parameters
        """
        
        # 
        # Parsing input parameters 
        # 
        
        # underlying value 
        S = args[0] if len(args) > 0 else kwargs['S'] if 'S' in kwargs else self.get_S()
        
        # strike price
        K = kwargs['K'] if 'K' in kwargs else self.get_K()

        # time parameter:
        time_param = args[1] if len(args) > 1 \
                     else kwargs['tau'] if 'tau' in kwargs \
                        else (kwargs['t'] if 't' in kwargs else None)

        # underlying volatility 
        sigma = kwargs['sigma'] if 'sigma' in kwargs else self.get_sigma()
        
        # span the x-axis with volatility values if True, otherwise distribute its values
        sigma_axis = kwargs['sigma_axis'] if 'sigma_axis' in kwargs else False

        # short rate
        r = kwargs['r'] if 'r' in kwargs else self.get_r()

        # span the x-axis with short-rate values if True, otherwise distribute its values
        r_axis = kwargs['r_axis'] if 'r_axis' in kwargs else False

        # squeeze output flag
        np_output = kwargs['np_output'] if 'np_output' in kwargs else True

        #
        # Iterable parameters check
        #
        
        # counter for iterable parameters in input 
        iterable_parameters = 0
        iterable_S = False
        iterable_K = False
        iterable_tau = False
        iterable_sigma = False
        iterable_r = False

        if is_iterable(S):
            iterable_S = True
            iterable_parameters += 1
            
        if is_iterable(K):
            iterable_K = True
            iterable_parameters += 1
        
        if is_iterable_not_string(time_param):
            iterable_tau = True
            iterable_parameters += 1
            
        if is_iterable(sigma):
            iterable_sigma = True
            iterable_parameters += 1
            
        if is_iterable(r):
            iterable_r = True
            iterable_parameters += 1
            
        #
        # Checking that only one of S or K is iterable
        #
        
        if iterable_S and iterable_K:
            raise NotImplementedError("Just one between 'S' and 'K' parameters allowed to be iterable."\
                                      " Both iterable given in input:\nS={}\nK={}".format(S,K))
        
        # flag for iterability of S or K
        iterable_S_or_K = iterable_S or iterable_K
        
        # flag for iterability of S only
        iterable_S_not_K = iterable_S and (not iterable_K)
        
        # 
        # Checking consistency between iterable_sigma and sigma_axis
        #
        
        if not iterable_sigma and sigma_axis:
            raise ValueError("Non-iterable sigma cannot span the x-axis.")
            
        # 
        # Checking consistency between iterable_r and r_axis
        #
        
        if not iterable_r and r_axis:
            raise ValueError("Non-iterable r cannot span the x-axis.")
            
        #
        # Checking that sigma_axis and r_axis are not simultaneously True
        #
        
        if sigma_axis and r_axis:
            raise NotImplementedError("x-axis cannot be spanned simultaneously by sigma and r")

        # 
        # Checking that if S/K are iterables and sigma is vector, then sigma_axis == False
        #
        
        if iterable_S_or_K and iterable_sigma:
            if sigma_axis:
                raise NotImplementedError("x-axis already spanned by S/K, cannot be spanned by sigma.")
                
        # 
        # Checking that if S/K are iterables and r is vector, then r_axis == False
        #
        
        if iterable_S_or_K and iterable_r:
            if r_axis:
                raise NotImplementedError("x-axis already spanned by S/K, cannot be spanned by r.")

        #
        # Homogenizing and checking each parameters
        #
            
        # 
        # 1) Underlying value
        #
        
        # homogenize underlying in input
        S = homogenize(S)
 
        # checking whether any value in S is smaller than zero. Works if S is scalar too.
        if np.any(S < 0):
            warnings.warn("Warning: S = {} < 0 value encountered".format(S))
                   
        # 
        # 2) Strike price
        #
        
        # homogenize strike in input
        K = homogenize(K)
 
        # checking whether any value in K is smaller than zero. Works if K is scalar too.
        if np.any(K <= 0):
            warnings.warn("Warning: K = {} <= 0 value encountered".format(K))

        # 
        # 3) Time parameter
        #
        
        time_name = "tau"
                                
        # time parameter interpretation (and homogenization) according to its type        
        # case 1: no time-parameter in input
        if time_param is None:
            tau = time_param = self.get_tau()
        # case 2: valid time-to-maturity in input
        elif is_numeric(time_param):
            time_param = homogenize(time_param, reverse_order=True)
            tau = time_param
        # case 3: valuation date in input, to be converted into time-to-maturity
        elif is_date(time_param):
            time_name = "t"
            time_param = homogenize(time_param, sort_func=date_string_to_datetime_obj)
            tau = self.time_to_maturity(t=time_param)
        # error case: the time parameter in input has a data-type that is not recognized
        else: 
            raise TypeError("Type {} of input time parameter not recognized".format(type(time_param)))
              
        # checking whether any value in tau is smaller than zero. Works if tau is scalar too.
        if np.any(tau < 0):
            warnings.warn("Warning: tau = {} < 0 value encountered".format(tau))

        # 
        # 4) Underlying volatility
        #
        
        # homogenize underlying volatility in input
        sigma = homogenize(sigma, sort=False)
 
        # We allow for deterministic dynamics (sigma==0), but we raise a warning anyway
        # if any value of sigma is smaller-or-equal than zero. Works if sigma is scalar too.
        if np.any(sigma <= 0):
            warnings.warn("Warning: sigma = {} <= 0 value encountered".format(sigma))
        
        # 
        # 5) Short-rate
        #
        
        # homogenize short-rate in input
        r = homogenize(r, sort=False)
 
        # We allow for negative short rate, but we raise a warning anyway 
        # if any value in r is smaller than zero. Works if r is scalar too.
        if np.any(r < 0):
            warnings.warn("Warning: r = {} < 0 value encountered".format(r))

        #
        # Coordinate parameters
        #
        
        # Case 0: all scalar parameters
        #
        # make the 4 parameters coordinated together as 1-dim np.ndarray
        # or pd.DataFrame
        if iterable_parameters == 0:
            
            coord_params = coordinate(x=S, y=tau, 
                                      x_name="S", y_name=time_name,
                                      others_scalar={"K": K, "sigma": sigma, "r": r}, 
                                      np_output=np_output, 
                                      col_labels=S, ind_labels=time_param) 
            
        # Case 1: S (or K) and/or tau iterable parameters
        #
        # Make x-axis spanned by S, K, sigma or r, creating a (x-axis,time) 
        # grid if both are iterable. Parameters sigma and r are either 
        # distributed along the other(s) axes (shape-match is required) or can 
        # span be used to span the x-axis too (sigma_axis or r_axis flags must 
        # be set to True)
        elif iterable_S_or_K or iterable_tau:
            
            scalar_params = {}
            vector_params = {}
            
            # x-axis default setup            
            x=S
            x_name="S"
            x_col=S
            scalar_params["K"] = K

            if iterable_sigma:
                if sigma_axis:
                    x=sigma
                    x_name="sigma"
                    x_col=sigma
                    scalar_params["S"] = S
                else:
                    vector_params["sigma"] = sigma
            else:
                scalar_params["sigma"] = sigma
                
            if iterable_r:
                if r_axis:
                    x=r
                    x_name="r"
                    x_col=r
                    scalar_params["S"] = S
                else:
                    vector_params["r"] = r
            else:
                scalar_params["r"] = r

            if iterable_K:
                x=K
                x_name="K"
                x_col=K
                del scalar_params["K"]
                scalar_params["S"] = S

            coord_params = coordinate(x=x, y=tau, 
                                      x_name=x_name, y_name=time_name,
                                      others_scalar=scalar_params, 
                                      others_vector=vector_params,
                                      np_output=np_output, 
                                      col_labels=x_col, ind_labels=time_param)   
                        
        # Case 2: sigma and/or r are iterable 1-dim vectors 
        #         and S, K and tau are both scalar
        elif iterable_sigma or iterable_r:
            
            # case 2.1: sigma and r are iterable 1-dim vectors
            #
            # make sigma and r coordinated np.ndarray or pd.DataFrames
            # creating a (sigma, r) grid and S, K and tau coordinated accordingly
            if iterable_sigma and iterable_r:
                coord_params = coordinate(x=sigma, y=r, 
                                          x_name="sigma", y_name="r",
                                          others_scalar={"S": S, "K": K, time_name: tau}, 
                                          np_output=np_output, 
                                          col_labels=sigma, ind_labels=r)   

            # case 2.2: sigma is a 1-dim vector and r is scalar
            #
            # make sigma and tau coordinated np.ndarray or pd.DataFrames
            # and S, K and r coordinated accordingly
            elif iterable_sigma:
                coord_params = coordinate(x=sigma, y=tau, 
                                          x_name="sigma", y_name=time_name,
                                          others_scalar={"S": S, "K": K, "r": r}, 
                                          np_output=np_output, 
                                          col_labels=sigma, ind_labels=time_param)   

            # case 2.3: r is a 1-dim vector and sigma is scalar
            #
            # make r and tau coordinated np.ndarray or pd.DataFrames
            # and S, K and sigma coordinated accordingly
            elif iterable_r:
                coord_params = coordinate(x=r, y=tau, 
                                          x_name="r", y_name=time_name,
                                          others_scalar={"S": S, "K": K, "sigma": sigma}, 
                                          np_output=np_output, 
                                          col_labels=r, ind_labels=time_param)

        # return coordinated parameters
        return {"S": coord_params["S"], 
                "K": coord_params["K"],
                "tau": coord_params[time_name], 
                "sigma": coord_params["sigma"], 
                "r": coord_params["r"], 
                "np_output": np_output}

    def d1_and_d2(self, *args, **kwargs):
        """
        Utility method to compute d1 and d2 terms of Black-Scholes pricing formula
        """
        
        # parsing optional parameters
        S     = args[0] if len(args) > 0 else kwargs['S'] if 'S' in kwargs else self.get_S()
        tau   = args[1] if len(args) > 1 else kwargs['tau'] if 'tau' in kwargs else self.get_tau()
        K     = args[2] if len(args) > 2 else kwargs['K'] if 'K' in kwargs else self.get_K()
        r     = args[3] if len(args) > 3 else kwargs['r'] if 'r' in kwargs else self.get_r()
        sigma = args[4] if len(args) > 4 else kwargs['sigma'] if 'sigma' in kwargs else self.get_sigma()
            
        # compute d1 and d2
        d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        return d1, d2
    
    #
    # Template methods
    # 
    
    # upper price limit template
    def price_upper_limit(self):
        raise NotImplementedError()     

    # lower price limit template
    def price_lower_limit(self):
        raise NotImplementedError()     

    #
    # Public methods
    # 

    def payoff(self, *args, **kwargs):
        """
        Calculates and returns the payoff of the option.
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py
            
        Can be called using (underlying, 
                             strike-price), 
        
        signature, where:

        - underlying can be specified either as the 1st positional argument or as keyboard argument 'S'. 
          It's value can be:
        
            - Empty: .get_S() is used,
            - A number (e.g. S=100),
            - A List of numbers (allowed only if parameter 'K' is a scalar)       

        - strike-price can be specified as keyboard argument 'K'. 
          It's value can be:
        
            - Empty: .get_K() is used,
            - A number (e.g. K=100),
            - A List of numbers (allowed only if parameter 'S' is a scalar)       
        """
        
        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)
        
        # underlying value and strike price
        S = param_dict["S"]
        K = param_dict["K"]
                
        # call case
        if self.get_type() == 'call':
            return self.call_payoff(S=S, K=K)
        # put case
        else:
            return self.put_payoff(S=S, K=K)
                
    def price(self, *args, **kwargs):
        """
        Calculates and returns the price of the option. 
        
        Usage examples: 
            - example_options.py
            - example_options_other_params.py
            
        Can be called using (underlying, 
                             strike-price, 
                             time-parameter, 
                             sigma, 
                             short-rate)
        
        signature, where:

        - underlying can be specified either as the 1st positional argument or as keyboard argument 'S'. 
          It's value can be:
        
            - Empty: .get_S() is used,
            - A number (e.g. S=100),
            - A List of numbers (allowed only if parameter 'K' is a scalar)       

        - strike-price can be specified as keyboard argument 'K'. 
          It's value can be:
        
            - Empty: .get_K() is used,
            - A number (e.g. K=100),
            - A List of numbers (allowed only if parameter 'S' is a scalar)       

        - time-parameter can be specified either as the 2nd positional argument or as keyboard argument 't' or 'tau'. 
          If tau==0, the payoff of the option is returned, its price otherwise. 
          It's value can be:
        
            - Empty: .get_tau() is used,
            - A single (e.g. t='15-05-2020') / Iterable (e.g. pd.date_range) valuation date(s): 
              accepted types are either a 'dd-mm-YYYY' String or a dt.datetime object
            - A single (e.g. tau=0.5) / Iterable time-to-maturity value(s) 

        - sigma can be specified as keyboard argument 'sigma'. 
          It's value can be:
        
            - Empty: .get_sigma() is used,
            - A volatility value (e.g. 0.2 for 20% per year)
            - An iterable:
                
                - if sigma_axis == False (default): an iterable of the same 
                  shape of x-axis variable (S or K) x tau (if both the x-axis
                  and tau are iterable variables) or of the same shape of x-axis 
                  (tau) if the x-axis (tau) is iterable but tau (x-axis) is scalar.
                  In this case, volatility parameter is distributed along the 
                  other(s) vectorial dimension(s).
                  
                - if sigma_axis == True: an iterable of arbitrary lenght.
                  In this case, the x-axis dimension is spanned by sigma parameter.
                  This setup is mutually exclusive w.r.t. to the r_axis == True
                  setup.

        - short-rate can be specified as keyboard argument 'r'. 
          It's value can be:
        
            - Empty: .get_r() is used,
            - A short-rate value (e.g. 0.05 for 5% per year)
            - An iterable:
                
                - if r_axis == False (default): an iterable of the same 
                  shape of x-axis variable (S or K) x tau (if both the x-axis
                  and tau are iterable variables) or of the same shape of x-axis 
                  (tau) if the x-axis (tau) is iterable but tau (x-axis) is scalar.
                  In this case, short-rate parameter is distributed along the 
                  other(s) vectorial dimension(s).
                  
                - if r_axis == True: an iterable of arbitrary lenght. 
                  In this case, the x-axis dimension is spanned by sigma parameter.                
                  This setup is mutually exclusive w.r.t. to the sigma_axis == True
                  setup.
        """
                       
        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)

        # underlying value, strike-price, time-to-maturity, volatility and short-rate
        S = param_dict["S"]
        K = param_dict["K"]
        tau = param_dict["tau"]
        sigma = param_dict["sigma"]
        r = param_dict["r"]
        np_output = param_dict["np_output"]
        
        #
        # for tau==0 output the payoff, otherwise price
        #
        
        if np_output:
            # initialize an empty structure to hold prices
            price = np.empty_like(S, dtype=float)
            # filter positive times-to-maturity
            tau_pos = tau > 0
        else:
            # initialize an empty structure to hold prices
            price = pd.DataFrame(index=S.index, columns=S.columns)
            # filter positive times-to-maturity
            tau_pos = tau.iloc[:,0] > 0
        
        # call case
        if self.get_type() == 'call':
            # tau > 0 case
            price[tau_pos] = self.call_price(S=S[tau_pos], K=K[tau_pos], tau=tau[tau_pos], sigma=sigma[tau_pos], r=r[tau_pos])
            # tau == 0 case
            price[~tau_pos] = self.call_payoff(S=S[~tau_pos], K=K[~tau_pos])  
        # put case
        else:
            # tau > 0 case
            price[tau_pos] = self.put_price(S=S[tau_pos], K=K[tau_pos], tau=tau[tau_pos], sigma=sigma[tau_pos], r=r[tau_pos])
            # tau == 0 case
            price[~tau_pos] = self.put_payoff(S=S[~tau_pos], K=K[~tau_pos])  
            
        return price

    def PnL(self, *args, **kwargs):
        """
        Calculates and returns the P&L generated owning an option.
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py        
        
        Can be called with the same signature of the .price() public method.

        We distinguish two cases:
            
            1) if tau==0, this is the P&L at option's expiration. 
               That is, the PnL if the option is kept until maturity. 
               It is computed as:
        
                   P&L = payoff - initial price
                   
            2) if tau > 0, this is the P&L as if the option position is closed before maturity, 
               when the time-to-maturity is tau. It is computed as:
                  
                  P&L = current price - initial price
        
        The choice between returning the payoff and current price is delegated 
        to .price() method.
        """
                
        return self.price(*args, **kwargs) - scalarize(self.get_initial_price())
  
    def implied_volatility(self, *args, iv_estimated=0.25, epsilon=1e-8, 
                           minimization_method="Newton", max_iter = 100, **kwargs):
        """
        Calculates and returns the Black-Scholes Implied Volatility of the option.
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py
            - example_options_IV.py
            
        Implements two minimization routines:
            
            - Newton (unconstrained) method;
            - Least-Squares constrained method. 
            
        Can be called with the same signature of the .price() public method 
        with additional optional parameters:
            
            - iv_estimated: an initial guess for implied volatility;
            - target_price: target price to use for implied volatility calculation;
            - epsilon: minimization stopping threshold;
            - minimization_method: minimization methot to use;
            - max_iter: maximum number of iterations.
        """
        
        # preliminary consistency check
        if ('sigma_axis' in kwargs) and (kwargs['sigma_axis'] == True):
            raise NotImplementedError(".implied_volatility() method not implemented for x-axis spanned by 'sigma' parameter.")
                            
        # target price
        target_price = kwargs["target_price"] if "target_price" in kwargs else self.price(*args, **kwargs)
        
        # shape of output IV
        output_shape = target_price.shape
            
        # delete "np_output" from kwargs if it exists, to do calculations 
        # with np.ndarrays (returns True if not in kwargs)
        np_output = kwargs.pop("np_output", True)

        # casting output as pd.DataFrame, if necessary
        if not np_output:
            ind_output=target_price.index
            col_output=target_price.columns
            target_price = target_price.values.squeeze()

        # delete "sigma" from kwargs if it exists
        kwargs.pop("sigma", None)
        
        if minimization_method == "Newton":
            
            # initial guess for implied volatility: iv_{n} and iv_{n+1}
            # iv_{n+1} will be iteratively updated
            iv_n = coordinate_y_with_x(x=target_price, y=iv_estimated, np_output=True)
            iv_np1 = coordinate_y_with_x(x=target_price, y=iv_estimated, np_output=True)
            
            # stopping criterion: 
            #
            # - SRSR > epsilon threshold or 
            # - maximum iterations exceeded
            # 
            # where: SRSR is the Sum of Relative Squared Residuals between
            # n-th and (n+1)-th iteration solutions, defined as: 
            #
            # SRSR = \Sum_{i} ((x_{n+1} - x_{n})/x_{n})**2 (NaN excluded)
            
            # SRSR is initialized at value greater than epsilon by construction
            SRSR = epsilon + 1
            
            # iterations counter initialized at 1
            iter_num = 1
            
            while (SRSR > epsilon) and (iter_num <= max_iter):
                
                # update last solution found
                iv_n = iv_np1
                
                # function to minimize at last solution found
                f_iv_n = self.price(*args, sigma=iv_n, **kwargs) - target_price

                # derivative w.r.t. to sigma (that is, Vega) at last solution found
                df_div_n = self.vega(*args, sigma=iv_n, factor=1.0, **kwargs)
                
                # new solution found
                iv_np1 = iv_n - f_iv_n/df_div_n

                # calculation of new value for stopping metrics
                SRSR = np.nansum(((iv_np1 - iv_n)/iv_n)**2)

                # iteration counter update
                iter_num += 1
                
            print("\nTermination value for Sum of Relative Squared Residuals \
                  \nbetween n-th and (n+1)-th iteration solutions metric \
                  \n(NaN excluded): {:.1E} (eps = {:.1E}). Iterations: {} \n"\
                  .format(SRSR, epsilon, iter_num))
        
        elif minimization_method == "Least-Squares":
            #
            # See documentation for scipt.optmize.least_squares function
            #
            # at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
            #
            
            # minimization function (function of implied volatility only)
            f = lambda iv: (self.price(*args, sigma=iv, **kwargs) - target_price).flatten() 
            
            # initial implied volatility guess
            iv0 = np.repeat(iv_estimated, repeats=target_price.size)
            
            # positivity bounds: iv > 0
            iv_bounds = (0.0, np.inf)
            
            # minimization method: Trust-Region-Reflective algorithm
            min_method='trf'
            
            # tolerance for termination by the change of the cost function
            cost_tolerance = kwargs["cost_tolerance"] if "cost_tolerance" in kwargs else 1e-12
            
            # tolerance for termination by the change of the solution found
            sol_tolerance = kwargs["sol_tolerance"] if "sol_tolerance" in kwargs else 1e-12
            
            # optimization
            res = sc_opt.least_squares(fun=f, x0=iv0, bounds=iv_bounds, method=min_method,
                                ftol=cost_tolerance, xtol=sol_tolerance)
            
            # output message
            print("\nTermination message: " + res.message + " Success? {}".format(res.success))
            
            # optimal iv found
            iv_np1 = res.x
              
        # output reshape and cast as pd.DataFrame, if needed
        iv_np1 = iv_np1.reshape(output_shape)
        if not np_output:
            iv_np1 = pd.DataFrame(data=iv_np1, index=ind_output, columns=col_output)

        return iv_np1

    def delta(self, *args, **kwargs):
        """
        Calculates and returns the Gamma of the option.
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py
            - example_options_numeric_analytic_greeks_comparison.py
            
        Can be called with the same signature of the .price() public method.
        """
                       
        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)

        # underlying value, strike-price, time-to-maturity, volatility and short-rate
        S = param_dict["S"]
        K = param_dict["K"]
        tau = param_dict["tau"]
        sigma = param_dict["sigma"]
        r = param_dict["r"]
                
        # call case
        if self.get_type() == 'call':
            return self.call_delta(S=S, K=K, tau=tau, sigma=sigma, r=r)
        # put case
        else:
            return self.put_delta(S=S, K=K, tau=tau, sigma=sigma, r=r)

    def theta(self, *args, **kwargs):
        """
        Calculates and returns the Theta of the option. 
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py
            - example_options_numeric_analytic_greeks_comparison.py
            
        Can be called with the same signature of the .price() public method.

        Optionally, the theta can be rescaled using the "factor" keyboard parameter. 
        By default it is scaled to consider variation of +1 calendar day of t (not +1 year).
        """
                       
        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)

        # underlying value, strike-price, time-to-maturity volatility and short-rate
        S = param_dict["S"]
        K = param_dict["K"]
        tau = param_dict["tau"]
        sigma = param_dict["sigma"]
        r = param_dict["r"]
        
        # rescaling factor
        rescaling_factor = kwargs["factor"] if "factor" in kwargs else 1.0/365.0
                
        # call case
        if self.get_type() == 'call':
            return self.call_theta(S=S, K=K, tau=tau, sigma=sigma, r=r) * rescaling_factor
        # put case
        else:
            return self.put_theta(S=S, K=K, tau=tau, sigma=sigma, r=r) * rescaling_factor

    def gamma(self, *args, **kwargs):
        """
        Calculates and returns the Gamma of the option. 
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py
            - example_options_numeric_analytic_greeks_comparison.py
            
        Can be called with the same signature of the .price() public method.
        """
                       
        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)

        # underlying value, strike-price, time-to-maturity volatility and short-rate
        S = param_dict["S"]
        K = param_dict["K"]
        tau = param_dict["tau"]
        sigma = param_dict["sigma"]
        r = param_dict["r"]
                
        # call case
        if self.get_type() == 'call':
            return self.call_gamma(S=S, K=K, tau=tau, sigma=sigma, r=r)
        # put case
        else:
            return self.put_gamma(S=S, K=K, tau=tau, sigma=sigma, r=r)
          
    def vega(self, *args, **kwargs):
        """
        Calculates and returns the Vega of the option. 
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py
            - example_options_numeric_analytic_greeks_comparison.py
            
        Can be called with the same signature of the .price() public method.

        Optionally, the vega can be rescaled using the "factor" keyboard parameter. 
        By default it is scaled to consider variation of +1% of sigma (not +100%).
        """
                       
        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)

        # underlying value, strike-price, time-to-maturity volatility and short-rate
        S = param_dict["S"]
        K = param_dict["K"]
        tau = param_dict["tau"]
        sigma = param_dict["sigma"]
        r = param_dict["r"]
                
        # rescaling factor
        rescaling_factor = kwargs["factor"] if "factor" in kwargs else 0.01

        # call case
        if self.get_type() == 'call':
            return self.call_vega(S=S, K=K, tau=tau, sigma=sigma, r=r) * rescaling_factor
        # put case
        else:
            return self.put_vega(S=S, K=K, tau=tau, sigma=sigma, r=r) * rescaling_factor

    def rho(self, *args, **kwargs):
        """
        Calculates and returns the Rho of the option. 
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py
            - example_options_numeric_analytic_greeks_comparison.py
            
        Can be called with the same signature of the .price() public method.

        Optionally, the rho can be rescaled using the "factor" keyboard parameter. 
        By default it is scaled to consider variation of +1% of r (not +100%).
        """
                       
        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)

        # underlying value, strike-price, time-to-maturity volatility and short-rate
        S = param_dict["S"]
        K = param_dict["K"]
        tau = param_dict["tau"]
        sigma = param_dict["sigma"]
        r = param_dict["r"]
                
        # rescaling factor
        rescaling_factor = kwargs["factor"] if "factor" in kwargs else 0.01

        # call case
        if self.get_type() == 'call':
            return self.call_rho(S=S, K=K, tau=tau, sigma=sigma, r=r) * rescaling_factor
        # put case
        else:
            return self.put_rho(S=S, K=K, tau=tau, sigma=sigma, r=r) * rescaling_factor

#-----------------------------------------------------------------------------#
        
class PlainVanillaOption(EuropeanOption):
    """
    PlainVanillaOption class implementing payoff and pricing of plain-vanilla call and put options.
    Inherits from EuropeanOption base-class. Put price is calculated using put-call parity
    
    Attributes:
    -----------
        mkt_env (MarketEnvironment): Instance of MarketEnvironment class
        type (str):                  From 'type' attribute of EuropeanOption base class.
        S_t (float):                 'S' attribute of mkt_env.
        K (float):                   From 'K' attribute of EuropeanOption base class.
        t (str; dt.datetime):        't' attribute of mkt_env.
        T (str; dt.datetime):        From 'T' attribute of EuropeanOption base class.
        tau (float):                 time to maturity in years, computed as tau=T-t by time_to_maturity() method
        r (float):                   'r' attribute of mkt_env.
        sigma (float):               'sigma' attribute of mkt_env.
    
    Public Methods:
    --------   

        public methods inherited from EuropeanOption class

        price_upper_limit: float 
            Overridden method. Returns the upper limit for a vanilla option price.

        price_lower_limit: float 
            Overridden method. Returns the lower limit for a vanilla option price.
                        
    Usage examples: 
    --------   
        
        - example_options.py
        - example_options_other_params.py
        - example_options_IV.py
        - example_options_numeric_analytic_greeks_comparison.py

    Instantiation
    --------   
        - default: PlainVanillaOption(mkt_env) is equivalent to 
                   PlainVanillaOption(mkt_env, option_type='call', K=100.0, T="31-12-2020")

        - general: PlainVanillaOption(mkt_env, option_type='call' or 'put' String, K=Float, T="DD-MM-YYYY" String)

    where: mkt_env is a MarketEnvironment object.
    """
    
    # initializer with optional *args and **kwargs parameters
    def __init__(self, *args, **kwargs):  
        
        # calling the EuropeanOption initializer
        super(PlainVanillaOption, self).__init__(*args, **kwargs)
        
        # info strings
        self.__info = r"Plain Vanilla {} [K={:.1f}, T={} (tau={:.2f}y)]".format(self.get_type(), self.get_K(), datetime_obj_to_date_string(self.get_T()), self.get_tau())
        self.__mkt_info = r"[S_t={:.1f}, r={:.1f}%, sigma={:.1f}%, t={}]".format(self.get_S(), self.get_r()*100, self.get_sigma()*100, datetime_obj_to_date_string(self.get_t()))
        
        # initial price of the option (as scalar value)
        self.__initial_price = self.price()
        
        # informations dictionary
        self.__docstring_dict = {
            'call':{
                'price_upper_limit': r"Upper limit: $S_t$",
                'payoff':            r"Payoff: $max(S-K, 0)$",
                'price_lower_limit': r"Lower limit: $max(S_t - K e^{-r \tau}, 0)$"
            },
            'put': {
                'price_upper_limit': r"Upper limit: $K e^{-r \tau}$",
                'payoff':            r"Payoff: $max(K-S, 0)$",
                'price_lower_limit': r"Lower limit: $max(K e^{-r \tau} - S_t, 0)$"}
        }
                
    def __repr__(self):
        return r"PlainVanillaOption('{}', S_t={:.1f}, K={:.1f}, t={}, T={}, tau={:.2f}y, r={:.1f}%, sigma={:.1f}%)".\
                format(self.get_type(), self.get_S(), self.get_K(), self.get_t().strftime("%d-%m-%Y"), 
                       self.get_T().strftime("%d-%m-%Y"), self.get_tau(), self.get_r()*100, self.get_sigma()*100)
    
    #
    # getters
    #
    
    def get_info(self):
        return self.__info
    
    def get_mkt_info(self):
        return self.__mkt_info
    
    def get_initial_price(self):
        return self.__initial_price

    def get_docstring(self, label):
        return self.__docstring_dict[self.get_type()][label] 

    #
    # Public methods
    # 
    
    def call_payoff(self, S, K):
        """Plain-Vanilla call option payoff
        """
        # Function np.maximum(arr, x) returns the array of the maximum 
        # between each element of arr and x
        return np.maximum(S-K, 0.0)

    def put_payoff(self, S, K):
        """Plain-Vanilla put option payoff"""
        return np.maximum(K-S, 0.0)
        
    def price_upper_limit(self, *args, **kwargs):
        """
        Calculates and returns the upper limit of the Plain-Vanilla option price. 
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py

        Can be called with the same signature of the .price() public method.
        """

        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)
        
        # underlying value, strike-price, time-to-maturity volatility and short-rate
        S = param_dict["S"]
        K = param_dict["K"]
        tau = param_dict["tau"]
        r = param_dict["r"]

        if self.get_type() == 'call':
            # call case
            return self.call_price_upper_limit(S=S)
        else:
            # put case
            return self.put_price_upper_limit(S=S, K=K, tau=tau, r=r)
            
    def call_price_upper_limit(self, S):
        """Plain-Vanilla call option price upper limit"""
        return S
    
    def put_price_upper_limit(self, S, K, tau, r):
        """Plain-Vanilla call option price upper limit"""
        return K*np.exp(-r * tau)

    def price_lower_limit(self, *args, **kwargs):
        """
        Calculates and returns the lower limit of the Plain-Vanilla option price. 
       
        Usage example: 
            - example_options.py
            - example_options_other_params.py

        Can be called with the same signature of the .price() public method.
        """

        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)

        # underlying value, strike-price, time-to-maturity and short-rate
        S = param_dict["S"]
        K = param_dict["K"]
        tau = param_dict["tau"]
        r = param_dict["r"]
                                       
        # call case
        if self.get_type() == 'call':
            return self.call_price_lower_limit(S=S, K=K, tau=tau, r=r)
        # put case
        else:
            return self.put_price_lower_limit(S=S, K=K, tau=tau, r=r)
            
    def call_price_lower_limit(self, S, K, tau, r):
        """Plain-Vanilla call option price lower limit"""
        # Function np.maximum(arr, x) returns the array of the maximum 
        # between each element of arr and x
        return np.maximum(S - K*np.exp(-r * tau), 0.0)
        
    def put_price_lower_limit(self, S, K, tau, r):
        """Plain-Vanilla put option price lower limit"""
        return np.maximum(K*np.exp(-r * tau) - S, 0.0)
                                                 
    def call_price(self, S, K, tau, sigma, r):
        """"Plain-Vanilla call option price """
        
        # get d1 and d2 terms
        d1, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute price
        price = S * stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * tau) * stats.norm.cdf(d2, 0.0, 1.0)
                           
        return price
    
    def put_price(self, S, K, tau, sigma, r):
        """ Plain-Vanilla put option price from Put-Call parity relation: Call + Ke^{-r*tau} = Put + S"""
        return self.call_price(S=S, K=K, tau=tau, sigma=sigma, r=r) + K * np.exp(-r * tau) - S     
    
    def call_delta(self, S, K, tau, sigma, r):
        """"Plain-Vanilla call option Delta """
        
        # get d1 term
        d1, _ = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute delta
        delta = stats.norm.cdf(d1, 0.0, 1.0)
        
        # stats.norm.cdf down-cast pd.DataFrames to np.ndarray    
        if isinstance(S, pd.DataFrame):
            delta = pd.DataFrame(data=delta, index=S.index, columns=S.columns)
                           
        return delta

    def put_delta(self, S, K, tau, sigma, r):
        """"Plain-Vanilla put option Delta """
        
        return self.call_delta(S=S, K=K, tau=tau, sigma=sigma, r=r) - 1.0

    def call_theta(self, S, K, tau, sigma, r):
        """"Plain-Vanilla call option Theta """
        
        # get d1 and d2 terms
        d1, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute theta
        theta = - (S * sigma * stats.norm.pdf(d1, 0.0, 1.0) / (2.0 * np.sqrt(tau))) - r * K * np.exp(-r * tau) * stats.norm.cdf(d2, 0.0, 1.0)
                           
        return theta

    def put_theta(self, S, K, tau, sigma, r):
        """"Plain-Vanilla put option Theta """
        
        # get d1 and d2 terms
        d1, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute theta
        theta = - (S * sigma * stats.norm.pdf(d1, 0.0, 1.0) / (2.0 * np.sqrt(tau))) + r * K * np.exp(-r * tau) * stats.norm.cdf(-d2, 0.0, 1.0)
        
        return theta

    def call_gamma(self, S, K, tau, sigma, r):
        """"Plain-Vanilla call option Gamma """
        
        # get d1 term
        d1, _ = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute gamma
        gamma = stats.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(tau))
        
        return gamma
        
    def put_gamma(self, S, K, tau, sigma, r):
        """"Plain-Vanilla put option Gamma """
        
        return self.call_gamma(S=S, K=K, tau=tau, sigma=sigma, r=r)

    def call_vega(self, S, K, tau, sigma, r):
        """"Plain-Vanilla call option vega """
        
        # get d1 term
        d1, _ = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)
        
        # compute vega
        vega = S * np.sqrt(tau) * stats.norm.pdf(d1, 0.0, 1.0)
                           
        return vega
    
    def put_vega(self, S, K, tau, sigma, r):
        """Plain-Vanilla put option vega """
        
        return self.call_vega(S=S, K=K, tau=tau, sigma=sigma, r=r)
        
    def call_rho(self, S, K, tau, sigma, r):
        """"Plain-Vanilla call option Rho """
        
        # get d2 term
        _, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute rho
        rho = tau * K * np.exp(-r * tau) * stats.norm.cdf(d2, 0.0, 1.0)
        
        return rho

    def put_rho(self, S, K, tau, sigma, r):
        """Plain-Vanilla put option Rho """

        # get d2 term
        _, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute rho
        rho = - tau * K * np.exp(-r * tau) * stats.norm.cdf(-d2, 0.0, 1.0)
        
        return rho
#-----------------------------------------------------------------------------#

class DigitalOption(EuropeanOption):
    """
    DigitalOption class implementing payoff and pricing of digital call and put options.
    Inherits from EuropeanOption base-class. Put price is calculated using put-call parity
    
    Attributes:
    -----------
        mkt_env (MarketEnvironment): Instance of MarketEnvironment class
        Q (float):                   cash amount
        type (str):                  From 'type' attribute of EuropeanOption base class.
        S_t (float):                 'S' attribute of mkt_env.
        K (float):                   From 'K' attribute of EuropeanOption base class.
        t (str; dt.datetime):        't' attribute of mkt_env.
        T (str; dt.datetime):        From 'T' attribute of EuropeanOption base class.
        tau (float):                 time to maturity in years, computed as tau=T-t by time_to_maturity() method
        r (float):                   'r' attribute of mkt_env.
        sigma (float):               'sigma' attribute of mkt_env.
    
    Public Methods:
    --------   
 
        public methods inherited from EuropeanOption class

        price_upper_limit: float 
            Overridden method. Returns the upper limit for a vanilla option price.

        price_lower_limit: float 
            Overridden method. Returns the lower limit for a vanilla option price.
            
    Usage examples: 
    --------   
        
        - example_options.py
        - example_options_other_params.py
        - example_options_IV.py
        - example_options_numeric_analytic_greeks_comparison.py

    Instantiation
    --------   
        - default: DigitalOption(mkt_env) is equivalent to 
                   DigitalOption(mkt_env, cash_amount=1.0, option_type='call', K=100.0, T="31-12-2020")

        - general: DigitalOption(mkt_env, cash_amount=Float, option_type='call' or 'put' String, K=Float, T="DD-MM-YYYY" String)

    where: mkt_env is a MarketEnvironment object.
    """

    # initializer with optional *args and **kwargs parameters and default cash_amount
    # default keyword arguments (like cash_amount here) must go after args list argument in function def
    def __init__(self, *args, cash_amount=1.0, **kwargs):  
        
        # calling the EuropeanOption initializer
        super(DigitalOption, self).__init__(*args, **kwargs)
        
        # amount of cash in case of payment
        self.__Q = cash_amount    
        
        # info strings
        self.__info = r"CON {} [K={:.1f}, T={} (tau={:.2f}y), Q={:.1f}]".format(self.get_type(), self.get_K(), datetime_obj_to_date_string(self.get_T()), self.get_tau(), self.get_Q())
        self.__mkt_info = r"[S_t={:.1f}, r={:.1f}%, sigma={:.1f}%, t={}]".format(self.get_S(), self.get_r()*100, self.get_sigma()*100, datetime_obj_to_date_string(self.get_t()))
        
        # initial price of the option
        self.__initial_price = self.price()

        # informations dictionary
        self.__docstring_dict = {
            'call':{
                'price_upper_limit': r"Upper limit: $Q e^{-r \tau}$",
                'payoff':            r"Payoff: $Q$ $I(S > K)$",
                'price_lower_limit': r"Lower limit: $0$"
            },
            'put': {
                'price_upper_limit': r"Upper limit: $Q e^{-r \tau}$",
                'payoff':            r"Payoff: $Q$ $I(S \leq K)$",
                'price_lower_limit': r"Lower limit: $0$"}
        }        
                
    def __repr__(self):
        return r"DigitalOption('{}', cash={:.1f}, S_t={:.1f}, K={:.1f}, t={}, T={}, tau={:.2f}y, r={:.1f}%, sigma={:.1f}%)".\
                format(self.get_type(), self.get_Q(), self.get_S(), self.get_K(), self.get_t().strftime("%d-%m-%Y"), 
                       self.get_T().strftime("%d-%m-%Y"), self.get_tau(), self.get_r()*100, self.get_sigma()*100)
    
    #
    # getters
    #
    
    def get_info(self):
        return self.__info
    
    def get_mkt_info(self):
        return self.__mkt_info
    
    def get_Q(self):
        return self.__Q
    
    def get_initial_price(self):
        return self.__initial_price
    
    def get_docstring(self, label):
        return self.__docstring_dict[self.get_type()][label] 
    
    #
    # setters
    #

    def set_Q(self, cash_amount):
        self.__Q = cash_amount

    #
    # Public methods
    # 
    
    def call_payoff(self, S, K):
        """ CON call option payoff"""
        # Function np.heaviside(arr, x) returns:
        #        
        #    0 if arr < 0
        #    x if arr == 0
        #    1 if arr > 0
        return np.heaviside(S-K, 0.0)
        
    def put_payoff(self, S, K):
        """ CON put option payoff"""
        return np.heaviside(K-S, 1.0)
        
    def price_upper_limit(self, *args, **kwargs):
        """
        Calculates and returns the upper limit of the CON option price. 
       
        Usage example: 
            - example_options.py
            - example_options_other_params.py

        Can be called with the same signature of the .price() public method.
        """

        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)
        
        # time-to-maturity volatility and short-rate
        tau = param_dict["tau"]
        r = param_dict["r"]
        
        # the same for call and put
        return self.get_Q()*np.exp(-r * tau)
        
    def price_lower_limit(self, *args, **kwargs):
        """
        Calculates and returns the lower limit of the CON option price. 
        
        Usage example: 
            - example_options.py
            - example_options_other_params.py

        Can be called with the same signature of the .price() public method.
       """

        # process input parameters
        param_dict = self.process_pricing_parameters(*args, **kwargs)
        
        # underlying value
        S = param_dict["S"]
        
        # the same for call and put
        return 0.0*S
       
    def call_price(self, S, K, tau, sigma, r):
        """ CON call option Black-Scholes price"""
                
        Q = self.get_Q()
        
        # get d2 term
        _, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute price
        price = Q * np.exp(-r * tau) * stats.norm.cdf(d2, 0.0, 1.0)

        return price
    
    def put_price(self, S, K, tau, sigma, r):
        """ CON put option price from Put-Call parity relation: CON_Call + CON_Put = Qe^{-r*tau}"""
        return self.get_Q() * np.exp(- r * tau) - self.call_price(S=S, K=K, tau=tau, sigma=sigma, r=r)        

    def call_delta(self, S, K, tau, sigma, r):
        """ CON call option Black-Scholes Delta"""
                
        Q = self.get_Q()
        
        # get d2 term
        _, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute delta
        delta = Q * np.exp(-r * tau) * stats.norm.pdf(d2, 0.0, 1.0) / (S * sigma * np.sqrt(tau))

        return delta

    def put_delta(self, S, K, tau, sigma, r):
        """ CON put option Black-Scholes Delta"""
        
        return - self.call_delta(S=S, K=K, tau=tau, sigma=sigma, r=r)

    def call_theta(self, S, K, tau, sigma, r):
        """ CON call option Black-Scholes Theta"""
                
        Q = self.get_Q()
        
        # get d1 and d2 terms
        d1, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute theta
        theta = Q * np.exp(- r * tau) * (((d1 * sigma * np.sqrt(tau) - 2.0 * r *tau)/(2.0 * sigma * tau * np.sqrt(tau))) * stats.norm.pdf(d2, 0.0, 1.0) + r * stats.norm.cdf(d2, 0.0, 1.0))

        return theta

    def put_theta(self, S, K, tau, sigma, r):
        """ CON put option Black-Scholes Theta"""
        
        Q = self.get_Q()

        return - self.call_theta(S=S, K=K, tau=tau, sigma=sigma, r=r) + r * Q * np.exp(- r * tau)

    def call_gamma(self, S, K, tau, sigma, r):
        """ CON call option Black-Scholes Gamma"""
                
        Q = self.get_Q()
        
        # get d1 and d2 terms
        d1, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute gamma
        gamma = - (d1 * Q * np.exp(- r * tau) * stats.norm.pdf(d2, 0.0, 1.0)) / (S*S * sigma*sigma * tau)

        return gamma

    def put_gamma(self, S, K, tau, sigma, r):
        """ CON put option Black-Scholes Gamma"""
        
        return - self.call_gamma(S=S, K=K, tau=tau, sigma=sigma, r=r)
    
    def call_vega(self, S, K, tau, sigma, r):
        """ CON call option Black-Scholes Vega"""
                
        Q = self.get_Q()
        
        # get d1 and d2 terms
        d1, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute vega
        vega = - (d1 * Q * np.exp(- r * tau) * stats.norm.pdf(d2, 0.0, 1.0)) / (sigma)

        return vega
    
    def put_vega(self, S, K, tau, sigma, r):
        """ CON put option Black-Scholes Vega"""
        
        return - self.call_vega(S=S, K=K, tau=tau, sigma=sigma, r=r)
    
    def call_rho(self, S, K, tau, sigma, r):
        """CON call option Rho """
        
        Q = self.get_Q()
        
        # get d2 term
        _, d2 = self.d1_and_d2(S=S, K=K, tau=tau, sigma=sigma, r=r)

        # compute rho
        rho = Q * np.exp(- r * tau) * (((np.sqrt(tau) * stats.norm.pdf(d2, 0.0, 1.0))/(sigma)) - tau * stats.norm.cdf(d2, 0.0, 1.0))

        return rho

    def put_rho(self, S, K, tau, sigma, r):
        """Plain-Vanilla put option Rho """
        
        Q = self.get_Q()

        return - self.call_rho(S=S, K=K, tau=tau, sigma=sigma, r=r) - tau * Q * np.exp(- r * tau)

