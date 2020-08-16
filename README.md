<p align="center">
  <img src="images/logo_pyBlackScholesAnalytics.png" width="500" title="hover text" stlye="max-width:10%;">
</p>

# pyBlackScholesAnalytics

[**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) is a Python package for implementing analytics for options and option strategies under the Black-Scholes Model for educational purposes.

# Summary

[**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) package is a Python package designed to use the well known Black-Scholes model 
to evaluate price, P&L and greeks of European options (both plain-vanilla and simple equity exotics 
such as cash-or-nothing Digital options), as well as simple option strategies built on them.

The package has been developed as a spin-off project of the ["IT for Business and Finance"](https://github.com/gabrielepompa88/IT-For-Business-And-Finance-2019-20) class held at the University of Siena for the Master degree in Finance in 2020.

[**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) places itself in the middle between the coding style and level of a master student and that required for a junior quant at an investment bank. The aim is to address the gap between the two providing a playground 
for students to master financial concepts related to options and option strategies and implementing a dedicated comprehensive
object-oriented architecture.

# Contents:

The current version (0.0.1) of the package features the following modules:

- `options`: definitions for `EuropeanOption` abstract base-class as well 
as `PlainVanillaOption` and `DigitalOption` derived classes

- `portfolio`: definition of `Portfolio` class implementing analytics for portfolios of options

- `plotter`: definitions for `Plotter` abstract base-class as well 
as `OptionPlotter` and `PortfolioPlotter` derived classes

- `utils`: definition of general utility functions

- `numeric_routines`: definition of `NumericGreeks` class implementing option greeks through finite-difference methods

# Resources 

**Examples**: [**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) comes with several [examples](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/tree/master/examples) that can be used as entry-point to explore the package. 

**Tutorials**: as far as the educational purpose is concerned, I find the [**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) package itself helpful as much as the way in which its final version can be progressively built. In my experience, the constructive approach is ubiquitous in the real life of a Quant/Strat: a business need is first formulated by a trader or another stakeholder, then tackled by the Quant/Strat team with an _ad hoc_ analysis, then a tactic short-term implementation of the response is produced and, finally, a strategic and robust long-term solution is designed and implemented. For this reason, the package is complemented by a series of 4 [Jupyter Notebook tutorials](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/tree/master/Notebook_Tutorials) and [Youtube Video Playlists](https://www.youtube.com/channel/UC77o5i2ePrXjwlQQLQeYsBw/playlists) presenting it step-by step in a constructive way building on the ideas of the Object-Oriented paradygm as improvements over sequential implementations of the same financial concepts:

| Tutorial description |      Jupyter Notebook      |  Youtube Video Playlist |
|:-------------------|:--------------------------:|:------------------:|
| ciao | [Derivatives Analytics - Introduction to Object Oriented Programming](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Introduction_to_OOP.ipynb) | [OOP Introduction]( https://www.youtube.com/playlist?list=PLRIS0g8TmV1NI7lr2I7BF9wdJ_PK9mVAz) |
| ciao | [Derivatives Analytics - Inheritance and Polymorphism](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Inheritance_and_Polymorphism.ipynb) | [Inheritance and Polymorphism]() |
| ciao | [Derivatives Analytics - Objects Composition](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Objects_Composition.ipynb) | [Objects Composition]() |
| ciao | [Derivatives Analytics - Options Greeks](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Options_Greeks.ipynb) | [Options Greeks]() |

# Contacts

This is still the first version of this package, so if you find errors, have comments or suggestions you can reach Gabriele Pompa (_gabriele.pompa@gmail.com_). If you wish to contribute, please contact me through [GitHub/gabrielepompa88](https://github.com/gabrielepompa88). If you are interested but feel a bit new to Python, I can recommend the open ["IT for Business and Finance"](https://github.com/gabrielepompa88/IT-For-Business-And-Finance-2019-20) as a reasonable starting point. 

Thank you in advance for your attention.
