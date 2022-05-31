import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts

from holoviews.operation.timeseries import rolling
from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT

#from bokeh.server.server import Server

#hv.Store.set_current_backend('bokeh')
hv.extension('bokeh')

def get_curve(data, label=''):
    df = pd.DataFrame(data)
    df['date'] = df.date.astype('datetime64[ns]')
    return hv.Curve(df, ('date', 'Date'), ('adj_close', 'Price'), label=label)

hv.Dimension.type_formatters[np.datetime64] = '%Y'

aapl = get_curve(AAPL, label='AAPL')
goog = get_curve(GOOG, label='GOOG')
ibm  = get_curve(IBM, label='IBM')
msft = get_curve(MSFT, label='MSFT')

avg_curve = rolling(aapl, rolling_window=30).relabel('Average')
avg_scatter = hv.Scatter((np.array(AAPL['date'], dtype=np.datetime64), np.array(AAPL['adj_close'])), 
                         ('date', 'Date'), ('adj_close', 'Price'), label='close')

color_cycle = hv.Cycle(values=['#A6CEE3', '#B2DF8A','#33A02C', '#FB9A99'])
plot_opts = opts.Overlay(aspect=1, legend_position='top_left')
curve_opts = opts.Curve(color=color_cycle)

stocks = (aapl * goog * ibm * msft).opts(plot_opts, curve_opts)

appl_stats = (avg_scatter * avg_curve).opts(
    opts.Scatter(alpha=0.2, color='darkgrey'),
    opts.Curve(color='navy'), plot_opts)

from bokeh.plotting import show
show(hv.render(stocks + appl_stats))
#stocks + appl_stats                         