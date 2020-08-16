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

The current version (0.0.1) of the package features the following sub-packages:

- `options`: definitions for `EuropeanOption` abstract base-class as well 
as `PlainVanillaOption` and `DigitalOption` derived classes

- `portfolio`: definition of `Portfolio` class implementing analytics for portfolios of options

- `plotter`: definitions for `Plotter` abstract base-class as well 
as `OptionPlotter` and `PortfolioPlotter` derived classes

- `utils`: definition of general utility functions

- `numeric_routines`: definition of `NumericGreeks` class implementing option greeks through finite-difference methods

# Resources 

As far as the educational purpose is concerned, I find the [**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) package itself helpful as much as the way in which its final version can be progressively built. In my experience, the constructive approach is ubiquitous in the real life of a Quant/Strat: a business need is first formulated by a trader or another stakeholder, then tackled by the Quant/Strat team with an _ad hoc_ analysis, then a tactic short-term implementation of the response is produced and, finally, a strategic and robust long-term solution is designed and implemented. For this reason, the package is complemented by a series of 4 Tutorials in the form of [Jupyter Notebooks](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/tree/master/Notebook_Tutorials) and [Youtube Videos](https://www.youtube.com/channel/UC77o5i2ePrXjwlQQLQeYsBw/playlists). These tutorial aim to present the package step-by step in a constructive way building on the ideas of the Object-Oriented paradygm as improvements over sequential implementations of the same financial concepts. 

Moreover several [examples](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/tree/master/examples) showcase the features of [**pyBlackScholesAnalytics**](https://github.com/gabrielepompa88/pyBlackScholesAnalytics) package and can be used as entry-point to begin the exploration of the package.

## Tutorials

|  | Jupyter Notebook | Youtube Video Playlist |
|:-------------------|:--------------------------:|:------------------:|
| **Derivatives Analytics - Introduction to Object Oriented Programming**: in this tutorial we introduce _Object-Oriented_ Programming in Python. We first make a non-financial example, developing the intuition behind the need of a change of programming paradigm to be able to cohordinate together different pieces of code. Once we have established the intuition, we then leverage on some basic financial knowledge to develop a Black-Scholes pricer for European call Options, first, and then a more general pricer for Plain-Vanilla put Options as well. | [:notebook:](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Introduction_to_OOP.ipynb) | [:tv:]( https://www.youtube.com/playlist?list=PLRIS0g8TmV1NI7lr2I7BF9wdJ_PK9mVAz) |
| **Derivatives Analytics - Inheritance and Polymorphism**: in this tutorial we introduce _Inheritance_ and _Polymorphism_ in Python which are two milestones in Object-Oriented programming. We present these concepts introducing Digital cash-or-nothing Options and observing their similarities with Plain-Vanilla Options. Inheritance and Polymorphism allow us to leverage on the financial analogies between these two contracts and eventually represent them more efficiently as derived classes of a unique `EuropeanOption` abstract base class. | [:notebook:](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Inheritance_and_Polymorphism.ipynb) | [:tv:]() |
| **Derivatives Analytics - Objects Composition**: in this tutorial we introduce _Composition_ which is an additional way to model relationships among objects, alternatively to Inheritance. We present this relationship introducing a common `MarketEnvironment` under which Option contracts are emitted, a `Portfolio` class is designed to aggregate Options and another `Plotter` class is designed to provide visualization routines. Finally, we examine several Option Strategies which uses the architecture implemented so far. | [:notebook:](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Objects_Composition.ipynb) | [:tv:]() |
| **Derivatives Analytics - Options Greeks**: in this tutorial we introduce Option Greeks. That is, the derivatives of an option price with respect to its pricing parameters. We provide both a numeric computation using _finite-difference_ methods implemented in `NumericGreeks` class and their analytic expression using the Black-Scholes model. Finally, all features introduced in this and previous tutorial are collected in the `pyBlackScholesAnalytics` Python package which is hereby introduced. | [:notebook:](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/Notebook_Tutorials/Derivatives_Analytics___Options_Greeks.ipynb) | [:tv:]() |

## Examples

| <!-- -->  | <!-- --> |
|:-------------------:|:--------------------------|
| [options.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/options.py) | This example shows basic usage of `PlainVanillaOption` and `DigitalOption` classes. Basic instantiation examples are provided with combinations of the underlying level (S) and time parameter (t/tau). Price, P&L, first-order greeks as well as Black-Scholes implied-volatility are computed for Plain-Vanilla and Digital Option contracts. |
| [options_other_params.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/options_other_params.py) | This example shows usage of `PlainVanillaOption` and `DigitalOption` classes. Instantiation examples are provided involving combinations of the underlying level (S), strike-price (K), time parameter (t/tau), as well as underlying volatility (sigma) and short-rate (r) parameters. Price, P&L, first-order greeks as well as Black-Scholes implied-volatility are computed for Plain-Vanilla and Digital Option contracts. |
| [options_IV.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/options_IV.py) | This example shows usage of `PlainVanillaOption` and `DigitalOption` classes to compute of Black-Scholes implied volatility surfaces for Plain-Vanilla and Digital Option contracts. |
| [options_plot.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/options_plot.py) | This example shows basic integration of `PlainVanillaOption` and `DigitalOption` classes with `Plotter` class. Price, P&L and first-order greeks plots are shown for Plain-Vanilla and Digital Option contracts. |
| [options_plot_other_params.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/options_plot_other_params.py) | This example shows integration of `PlainVanillaOption` and `DigitalOption` classes with `Plotter` class. Price, P&L and first-order greeks plots are shown for Plain-Vanilla and Digital Option contracts against underlying level (S), strike-price (K), volatility (sigma) and short-rate (r). |
| [options_plot_IV.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/options_plot_IV.py) | This example shows integration of `PlainVanillaOption` and `DigitalOption` classes with `Plotter` class (in particular `.plot_IV()` method). Focus is on the visualization of Black-Scholes implied-volatility for Plain-Vanilla and Digital Option contracts both as a line plot and surface-plot. |
| [options_plot_surface.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/options_plot_surface.py) | This example shows integration of `PlainVanillaOption` and `DigitalOption` classes with `Plotter` class. Price, P&L and first-order greeks plots are shown for Plain-Vanilla and Digital Option contracts as underlying level (S), strike-price (K), volatility (sigma) and short-rate (r) surface-plots Vs time parameter. |
| [options_numeric_greeks.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/options_numeric_greeks.py) | This example provides an example of first-order numeric greeks implemented in the `NumericGreeks` class using finite-difference methods for Plain-Vanilla and Digital Option contracts. |
| [options_numeric_analytic_greeks_comparison.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/options_numeric_analytic_greeks_comparison.py) | This example provides a comparison of first-order greeks for Plain-Vanilla and Digital Option contracts implemented either through finite-difference methods in `NumericGreeks` class or using their analytic expression implemented in `PlainVanillaOption` and `DigitalOption` classes. |
| [portfolio.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/portfolio.py) | This example shows basic usage of `Portfolio` class to construct a derivative portfolio of Plain-Vanilla and Digital Option contracts. Basic instantiation examples are provided with combinations of the underlying level (S) and time parameter (t/tau). Price, P&L, first-order greeks are computed for constructed portfolio and benchmarked with the corresponding metrics calculated combining constituent options metrics. |
| [portfolio_single_strike.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/portfolio_single_strike.py) | This example shows basic usage of `Portfolio` class to construct a derivative portfolio of Plain-Vanilla and Digital Option contracts. Basic instantiation examples are provided with combinations of the underlying level (S), strike-price (K), time parameter (t/tau) as well as underlying volatility (sigma) and short-rate (r) parameters. Price, P&L, first-order greeks are computed for single-strike portfolio. |
| [portfolio_multi_strikes.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/portfolio_multi_strikes.py) | This example shows basic usage of `Portfolio` class to construct a derivative portfolio of Plain-Vanilla and Digital Option contracts. Basic instantiation examples are provided with combinations of the underlying level (S), time parameter (t/tau) as well as underlying volatility (sigma) and short-rate (r) parameters. Price, P&L, first-order greeks are computed for multi-strike portfolio. |
| [bull_spread.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/bull_spread.py) | This example shows usage of `Portfolio` class to create a Bull-Spread option strategy. Basic instantiation examples are provided with combinations of the underlying level (S) and time parameter (t/tau). Price, P&L, first-order greeks are computed and plotted using the `Plotter` class as line plots and surface-plots Vs time parameter. |
| [bull_spread_other_params.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/bull_spread_other_params.py) | This example shows usage of `Portfolio` class to create a Bull-Spread option strategy. Instantiation examples are provided with combinations of the underlying level (S) and time parameter (t/tau) as well as underlying volatility (sigma) and short-rate (r) parameters. Price, P&L, first-order greeks are computed and plotted using the `Plotter` class as line plots and surfaces-plots Vs time parameter. |
| [calendar_spread.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/calendar_spread.py) | This example shows usage of `Portfolio` class to create a Calendar-Spread option strategy. Basic instantiation examples are provided with combinations of the underlying level (S) and time parameter (t). Price, P&L, first-order greeks are computed and plotted using the `Plotter` class as line plots and surface-plots Vs time parameter. |
| [calendar_spread_other_params.py](https://github.com/gabrielepompa88/pyBlackScholesAnalytics/blob/master/examples/calendar_spread_other_params.py) | This example shows usage of `Portfolio` class to create a Calendar-Spread option strategy. Instantiation examples are provided with combinations of the underlying level (S) and time parameter (t) as well as underlying volatility (sigma) and short-rate (r) parameters. Price, P&L, first-order greeks are computed and plotted using the `Plotter` class as line plots and surfaces-plots Vs time parameter. |

# Contacts

This is still the first version of this package, so if you find errors, have comments or suggestions you can reach Gabriele Pompa (_gabriele.pompa@gmail.com_). If you wish to contribute, please contact me through [GitHub/gabrielepompa88](https://github.com/gabrielepompa88). If you are interested but feel a bit new to Python, I can recommend the open ["IT for Business and Finance"](https://github.com/gabrielepompa88/IT-For-Business-And-Finance-2019-20) as a reasonable starting point. 

Thank you in advance for your attention.
