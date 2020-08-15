<p align="center">
  <img src="images/logo_pyBlackScholesAnalytics.png" width="500" title="hover text" stlye="max-width:10%;">
</p>

# pyBlackScholesAnalytics

[**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) is a Python package for implementing analytics for options and option strategies under the Black-Scholes Model for educational purposes.

# Summary

[**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) package is a Python package designed to use the well known Black-Scholes model 
to evaluate price, P&L and greeks of European options (both plain-vanilla and simple equity exotics 
such as cash-or-nothing Digital options), as well as simple option strategies built on them.

The package has been developed as a spin-off development of the ["IT for Business and Finance"](https://github.com/gabrielepompa88/IT-For-Business-And-Finance-2019-20) class held at the University of Siena for the Master degree in Finance in 2020.

[**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) places itself in the middle between the coding style and level of a master student and that required for a junior quant at an investment bank. The aim is to address the gap between the two providing a playground 
for students to master financial concepts related to options and option strategies and implementing a dedicated comprehensive
object-oriented architecture.

# Contents:

The current version of the package features the following modules:

- `options`: definitions for `EuropeanOption` abstract base-class as well 
as `PlainVanillaOption` and `DigitalOption` derived classes

- `portfolio`: definition of `Portfolio` class implementing analytics for portfolios of options

- `plotter`: definitions for `Plotter` abstract base-class as well 
as `OptionPlotter` and `PortfolioPlotter` derived classes

- `utils`: definition of general utility functions

- `numeric_routines`: definition of `NumericGreeks` class implementing option greeks through finite-difference methods

# Resources 

[**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) comes with several [examples](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/tree/master/pyblackscholesanalytics) (`example_*.py` files) to explore the features of the package. A series of 4 Jupyter notebooks introduces step-by step the ideas of the Object-Oriented paradygm as improvements over sequential implementations:
- [
Derivatives Analytics - Introduction to Object Oriented Programming](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Introduction_to_OOP.ipynb)
- [
Derivatives Analytics - Inheritance and Polymorphism](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Inheritance_and_Polymorphism.ipynb)
- [Derivatives Analytics - Objects Composition](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Objects_Composition.ipynb)
- [Derivatives Analytics - Options Greeks](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Options_Greeks.ipynb)

Finally, YouTube video-lectures are provided for each notebook: 

- []()

# Contacts

This is still an alpha version, so if you have comments or suggestions you can reach Gabriele Pompa (_gabriele.pompa@gmail.com_). If you wish to contribute, please contact me through [GitHub/gabrielepompa88](https://github.com/gabrielepompa88). If you are interested but feel a bit new to Python, I can recommend the open ["IT for Business and Finance"](https://github.com/gabrielepompa88/IT-For-Business-And-Finance-2019-20) as a reasonable starting point. 

Thank you in advance for your attention.
