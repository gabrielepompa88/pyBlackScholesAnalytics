import holoviews as hv
import numpy as np
import re
from functools import reduce
from bokeh.plotting import show
from bokeh.palettes import Blues
from .plotter import Plotter
from ..utils.utils import *

hv.extension('bokeh')

class HoloviewsOptionPlotter(Plotter):
    
    def plot_single_time(self, x_axis_dict, time, time_label, plot_metrics, plot_price_limits):
        x_id = x_axis_dict.pop('x_axis', 'x-id not found')

        # x-axis values
        x = x_axis_dict[x_id]

        # define the figure
        curves = []

        # plot the price for different x-axis values        
        price = hv.Curve((x, getattr(self.fin_inst, plot_metrics)(**{x_id: x, 't': time})), label=time_label).opts(
            tools=['hover']
        )
        curves.append(price)
        # precompute emission level metrics (exploiting vectorization)
        if x_id == 'S':
            x_emission = self.fin_inst.get_S()
        elif x_id == 'K':
            x_emission = self.fin_inst.get_K()
        elif x_id == 'sigma':
            x_emission = self.fin_inst.get_sigma()
        elif x_id == 'r':
            x_emission = self.fin_inst.get_r()

        emission_metric = getattr(self.fin_inst, plot_metrics)(**{x_id: x_emission, 't': time})

        # blue dot at original emission level of the x-axis for reference
        dot = hv.Points((x_emission, emission_metric),
                label=r"Emission level " + x_id + r"={:.2f}".format(x_emission)).opts(line_width=5)
        curves.append(dot)

        if x_id in ['S', 'K']:

            if plot_price_limits:
                # plot the upper limit, the price and the lower limit for different x-axis values
                upper_limit_a = hv.Curve((x, self.fin_inst.price_upper_limit(**{x_id: x, 't': time})), 'k-.',
                        label="Upper Limit").opts(
                            line_dash='dashdot',
                            color="black"
                        )
                curves.append(upper_limit_a)
                upper_limit_b = hv.Curve((x, self.fin_inst.price_lower_limit(**{x_id: x, 't': time})), 'k--',
                        label="Lower limit").opts(
                            line_dash='dashed',
                            color="black"
                        )
                curves.append(upper_limit_b)



            # plot the red payoff line for different x-axis values
            if plot_metrics in ['price', 'PnL']:
                payoff_line = hv.Curve((x, getattr(self.fin_inst, plot_metrics)(**{x_id: x, 'tau': 0.0})), 'r-',
                        label=plot_metrics + r" at maturity").opts(
                            color="red"
                        )
                curves.append(payoff_line)
            
            # plot a dot to highlight the strike position and a reference zero line
            strike = hv.Scatter((self.fin_inst.get_K(), 0), 'k.', label="Strike K={}".format(self.fin_inst.get_K()))
            strike.opts(line_width=5, color="black", fill_color="black")
            curves.append(strike)

            zero_line = hv.Curve((x, np.zeros_like(x)), 'k--').opts(
                            line_dash='dashed',
                            color="black"
                        )
            curves.append(zero_line)
        
        curves = reduce((lambda x,y: x*y), curves)

        # set axis labels 
        curves.opts(
            legend_position='top_left',
            width=1000,
            height=600 ,            
            xlabel=x_id + r" at " + time_label,
            ylabel='Black-Scholes {}'.format(plot_metrics),
            gridstyle={
                'grid_line_color': 'lightgray',
                'xgrid_line_dash': [4, 4]
            },            
            show_grid=True,
            )  

        curves.opts(title=plot_metrics + " of a " + self.get_title())

        show(hv.render(curves))


    def plot_multi_time(self, x_axis_dict, times, time_labels, plot_metrics):
        """
        Plot Portfolio values against underlying value(s), possibly at multiple dates.

        See Instantiation and Usage examples in class docstring.
        """

        # identifier of the x-axis
        x_id = x_axis_dict.pop('x_axis', 'x-id not found')

        # other x-axis parameters
        sigma_axis = x_axis_dict.pop('sigma_axis', False)
        r_axis = x_axis_dict.pop('r_axis', False)

        # x-axis values
        x = x_axis_dict[x_id]

        # number of times-to-maturity considered
        n_times = len(times)

        curves = []
        # precompute surface (exploiting vectorization)
        surface_metrics = getattr(self.fin_inst, plot_metrics)(
            **{x_id: x, 't': times, 'sigma_axis': sigma_axis, 'r_axis': r_axis})

        # plot the price for different underlying values, one line for each different date
        for i in range(n_times):
        # plot the price for different x-axis values        
            price = hv.Curve((x, surface_metrics[i, :]), label=re.sub('\\$|\\\\','', time_labels[i])).opts(
                tools=['hover'], color=hv.Cycle('Category20b')
            )
            curves.append(price)

        # precompute emission level metrics (exploiting vectorization)
        if x_id == 'S':
            x_emission = self.fin_inst.get_S()
        elif x_id == 'K':
            x_emission = self.fin_inst.get_K()
        elif x_id == 'sigma':
            x_emission = self.fin_inst.get_sigma()
        elif x_id == 'r':
            x_emission = self.fin_inst.get_r()

        emission_metrics = getattr(self.fin_inst, plot_metrics)(**{x_id: x_emission, 't': times})

        # blue dot at original emission level of the x-axis for reference
        dot = hv.Points((x_emission + np.zeros_like(emission_metrics), emission_metrics),
                label=r"Emission level " + x_id + r"={:.2f}".format(x_emission)).opts(line_width=5)
        curves.append(dot)
                
        # part meaningful only if the x-axis is 'S' or 'K'
        if x_id in ['S', 'K']:
            
            # if defined, plot the red payoff line for different underlying values
            if plot_metrics in ['price', 'PnL']:
                payoff_line = hv.Curve((x, getattr(self.fin_inst, plot_metrics)(
                    **{x_id: x, 'tau': 0.0, 'sigma_axis': sigma_axis, 'r_axis': r_axis})), 'r-',
                        label=plot_metrics + r" at maturity").opts(color="red")                        
                curves.append(payoff_line)

            # plot a dot to highlight the strike position and a reference zero line            
            strike = hv.Scatter((self.fin_inst.get_K(), 0), 'k.', label="Strike K={}".format(self.fin_inst.get_K()))
            strike.opts(line_width=5, color="black", fill_color="black")
            curves.append(strike)

            zero_line = hv.Curve((x, np.zeros_like(x)), 'k--').opts(
                            line_dash='dashed',
                            color="black"
                        )
            curves.append(zero_line)

        curves = reduce((lambda x,y: x*y), curves)

        # set axis labels 
        curves.opts(
            legend_position='top_left',
            width=1000,
            height=600 ,            
            xlabel=x_id + r" at different dates",
            ylabel='Black-Scholes {}'.format(plot_metrics),
            gridstyle={
                'grid_line_color': 'lightgray',
                'xgrid_line_dash': [4, 4]
            },            
            show_grid=True,
            )  

        curves.opts(title=plot_metrics + " of a " + self.get_title())

        show(hv.render(curves))


class HoloviewsPortfolioPlotter(Plotter):
    def plot_single_time(self, x_axis_dict, time, time_label, plot_metrics, plot_price_limits):    
        x_id = x_axis_dict.pop('x_axis', 'x-id not found')

        # x-axis values
        x = x_axis_dict[x_id]

        # define the figure
        curves = []

        # plot the price for different x-axis values        
        price = hv.Curve((x, getattr(self.fin_inst, plot_metrics)(**{x_id: x, 't': time})), label=time_label).opts(
            tools=['hover']
        )
        curves.append(price)
        # precompute emission level metrics (exploiting vectorization)
        if x_id == 'S':
            x_emission = self.fin_inst.get_S()
        elif x_id == 'K':
            x_emission = self.fin_inst.get_K()
        elif x_id == 'sigma':
            x_emission = self.fin_inst.get_sigma()
        elif x_id == 'r':
            x_emission = self.fin_inst.get_r()

        emission_metric = getattr(self.fin_inst, plot_metrics)(**{x_id: x_emission, 't': time})

        # blue dot at original emission level of the x-axis for reference
        dot = hv.Points((x_emission, emission_metric),
                label=r"Emission level " + x_id + r"={:.2f}".format(x_emission)).opts(line_width=5)
        curves.append(dot)

        if x_id in ['S', 'K']:

            # plot the red payoff line for different x-axis values
            if (not self.fin_inst.is_multi_horizon) and (plot_metrics in ['price', 'PnL']):
                payoff_line = hv.Curve((x, getattr(self.fin_inst, plot_metrics)(**{x_id: x, 'tau': 0.0})), 'r-',
                        label=plot_metrics + r" at maturity").opts(
                            color="red"
                        )
                curves.append(payoff_line)
            
            # plot a dot to highlight the strike position and a reference zero line
            strikes = self.fin_inst.get_K()
            strike = hv.Scatter((strikes, np.zeros_like(strikes)), 'k.', label="Strike K={}".format(strikes))
            strike.opts(line_width=5, color="black", fill_color="black")
            curves.append(strike)

            zero_line = hv.Curve((x, np.zeros_like(x)), 'k--').opts(
                            line_dash='dashed',
                            color="black"
                        )
            curves.append(zero_line)
        
        curves = reduce((lambda x,y: x*y), curves)

        # set axis labels 
        curves.opts(
            legend_position='top_left',
            width=1000,
            height=600 ,            
            xlabel=x_id + r" at " + time_label,
            ylabel='Black-Scholes {}'.format(plot_metrics),
            gridstyle={
                'grid_line_color': 'lightgray',
                'xgrid_line_dash': [4, 4]
            },            
            show_grid=True,
            )  

        curves.opts(title=plot_metrics + " of a " + self.get_title())

        show(hv.render(curves))

    def plot_multi_time(self, x_axis_dict, times, time_labels, plot_metrics):
        """
        Plot Portfolio values against underlying value(s), possibly at multiple dates.

        See Instantiation and Usage examples in class docstring.
        """

        # identifier of the x-axis
        x_id = x_axis_dict.pop('x_axis', 'x-id not found')

        # other x-axis parameters
        sigma_axis = x_axis_dict.pop('sigma_axis', False)
        r_axis = x_axis_dict.pop('r_axis', False)

        # x-axis values
        x = x_axis_dict[x_id]

        # number of times-to-maturity considered
        n_times = len(times)

        curves = []
        # precompute surface (exploiting vectorization)
        surface_metrics = getattr(self.fin_inst, plot_metrics)(
            **{x_id: x, 't': times, 'sigma_axis': sigma_axis, 'r_axis': r_axis})

        # plot the price for different underlying values, one line for each different date
        for i in range(n_times):
        # plot the price for different x-axis values        
            price = hv.Curve((x, surface_metrics[i, :]), label=re.sub('\\$|\\\\','', time_labels[i])).opts(
                tools=['hover'], color=hv.Cycle('Category20b')
            )
            curves.append(price)

        # precompute emission level metrics (exploiting vectorization)
        if x_id == 'S':
            x_emission = self.fin_inst.get_S()
        elif x_id == 'K':
            x_emission = self.fin_inst.get_K()
        elif x_id == 'sigma':
            x_emission = self.fin_inst.get_sigma()
        elif x_id == 'r':
            x_emission = self.fin_inst.get_r()

        emission_metrics = getattr(self.fin_inst, plot_metrics)(**{x_id: x_emission, 't': times})

        # blue dot at original emission level of the x-axis for reference
        dot = hv.Points((x_emission + np.zeros_like(emission_metrics), emission_metrics),
                label=r"Emission level " + x_id + r"={:.2f}".format(x_emission)).opts(line_width=5)
        curves.append(dot)
                
         # if defined, plot the red payoff line for different underlying values
        if x_id in ['S', 'K']:
            
            if not self.fin_inst.is_multi_horizon:
                # if defined, plot the red payoff line for different underlying values
                if plot_metrics in ['price', 'PnL']:
                    payoff_line = hv.Curve((x, getattr(self.fin_inst, plot_metrics)(
                        **{x_id: x, 'tau': 0.0, 'sigma_axis': sigma_axis, 'r_axis': r_axis})), 'r-',
                            label=plot_metrics + r" at maturity").opts(color="red")                        
                    curves.append(payoff_line)

            # plot a dot to highlight the strike position and a reference zero line
            strikes = self.fin_inst.get_K()

            strike = hv.Scatter((strikes, np.zeros_like(strikes)), 'k.', label="Strike K={}".format(strikes))
            strike.opts(line_width=5, color="black", fill_color="black")
            curves.append(strike)

            zero_line = hv.Curve((x, np.zeros_like(x)), 'k--').opts(
                            line_dash='dashed',
                            color="black"
                        )
            curves.append(zero_line)

        curves = reduce((lambda x,y: x*y), curves)

        # set axis labels 
        curves.opts(
            legend_position='top_left',
            width=1000,
            height=600 ,            
            xlabel=x_id + r" at different dates",
            ylabel='Black-Scholes {}'.format(plot_metrics),
            gridstyle={
                'grid_line_color': 'lightgray',
                'xgrid_line_dash': [4, 4]
            },            
            show_grid=True,
            )  

        curves.opts(title=plot_metrics + " of a " + self.get_title())

        show(hv.render(curves))


    def plot_surf(self, x_axis_dict, times, time_labels, plot_metrics, view):
        """
        Plot Portfolio values as a surface of underlying value(s) and multiple dates.

        See Instantiation and Usage examples in class docstring.
        """
        curves = []

        # identifier of the x-axis
        x_id = x_axis_dict.pop('x_axis', 'x-id not found')

        # other x-axis parameters
        sigma_axis = x_axis_dict.pop('sigma_axis', False)
        r_axis = x_axis_dict.pop('r_axis', False)

        # x-axis values
        x = x_axis_dict[x_id]

        # number of times-to-maturity considered
        n_times = len(times)

        # define the figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.gca(projection='3d')

        # define a dense grid of times
        # in case of dates: from the most recent valuation date to expiration date
        # in case of times-to-maturity: from the biggest tau to 0 (that is, expiration)
        times_dense = self.make_dense(times, n=500)
        n_times_dense = len(times_dense)

        # if times are dates, we convert into their numeric representation. This is needed for plotting
        times_numeric = date_to_number(times)
        times_dense_numeric = date_to_number(times_dense)

        # precompute surface (exploiting vectorization)
        surface_metrics = getattr(self.fin_inst, plot_metrics)(
            **{x_id: x, 't': times_dense, 'sigma_axis': sigma_axis, 'r_axis': r_axis}, np_output=False)

        # grid points, if needed convert dates to numeric representation for plotting
        underlying_grid, time_grid = np.meshgrid(surface_metrics.columns, times_dense_numeric)

        # surface plot
        surf = hv.Surface((underlying_grid, time_grid, surface_metrics.values.astype('float64')))
        curves.append(surf)
        """
        # plot the price for different underlying values, one line for each different date
        plt.gca().set_prop_cycle(None)
        for i in range(n_times):
            ax.plot(x, np.repeat(times_numeric[i], repeats=len(x)), surface_metrics.loc[times[i], :], '-', lw=1.5,
                    label=time_labels[i], zorder=1 + i + 1)

        # precompute emission level metrics (exploiting vectorization)
        if x_id == 'S':
            x_emission = self.fin_inst.get_S()
        elif x_id == 'K':
            x_emission = self.fin_inst.get_K()
        elif x_id == 'sigma':
            x_emission = self.fin_inst.get_sigma()
        elif x_id == 'r':
            x_emission = self.fin_inst.get_r()
        """

        show(hv.render(curves))    