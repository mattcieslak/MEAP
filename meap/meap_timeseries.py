"""
Blatantly "borrowed" from the chaco annotated examples.

Lasso selection of data points

Draws a simple scatterplot of random data. Drag the mouse to use the lasso
selector, which allows you to circle all the points in a region.

Upon completion of the lasso operation, the indices of the selected points are
printed to the console.

Uncomment 'lasso_selection.incremental_select' line (line 74) to see the
indices of the selected points computed in real time.
"""

# Chaco spits out lots of deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from meap import MEAPView
from meap.io import PhysioData

# Enthought library imports
from enable.api import ComponentEditor
from traits.api import HasTraits, Instance,Str, Array, Event, Tuple
from traitsui.api import Item, Group
from chaco.data_range_1d import DataRange1D

# Chaco imports
from chaco.api import ArrayPlotData, Plot, jet
from chaco.tools.api import RangeSelection,RangeSelectionOverlay 

import logging
logger = logging.getLogger(__name__)

# Attributes to use for the plot view.
size=(650,650)
title="Scatter plot with selection"
bg_color="lightgray"

class BeatSelection(RangeSelection):
    left_button_selects = True

class MEAPTimeseries(HasTraits):
    physiodata = Instance(PhysioData)
    plot = Instance(Plot,transient=True)
    plotdata = Instance(ArrayPlotData,transient=True)
    selection = Instance(BeatSelection,transient=True)
    signal = Str
    selected_beats = Array
    index_datasourse = Instance(DataRange1D)
    marker_size = Array
    metadata_name = Str
    selected = Event
    selected_range = Tuple
    
    traits_view = MEAPView(
        Group(
            Item('plot', editor=ComponentEditor(size=size),
                 show_label=False),
            orientation = "vertical"),
        resizable=True, win_title=title
    )

    def __init__(self,**traits):
        super(MEAPTimeseries,self).__init__(**traits)
        self.metadata_name = self.signal + "_selection"

    def _selection_changed(self):
        selection = self.selection.selection
        logger.info("%s selection changed to %s", self.signal,
                str(selection))
        if selection is None or len(selection) == 0:
            return
        self.selected_range = selection
        self.selected = True

    def _plot_default(self):
        
        plot = Plot(self.plotdata)
        
        if self.signal in ("tpr","co","sv"):
            rc_signal = "resp_corrected_" + self.signal
            if rc_signal in self.plotdata.arrays and self.plotdata.arrays[rc_signal].size > 0:
                # Plot the resp-uncorrected version underneath
                plot.plot(("peak_times",self.signal),type="line",color="purple")
                signal = rc_signal
            signal = self.signal
        elif self.signal == "hr":
            plot.plot(("peak_times",self.signal),type="line",color="purple")
            signal = "mea_hr"
        else:
            signal = self.signal
        
        # Create the plot
        plot.plot(("peak_times", signal),type="line",color="blue")
        plot.plot(("peak_times", signal,"beat_type"),
                  type="cmap_scatter",
                  color_mapper=jet,
                  name="my_plot",
                  marker="circle",
                  border_visible=False,
                  outline_color="transparent",
                  bg_color="white",
                  index_sort="ascending",
                  marker_size=3,
                  fill_alpha=0.8
                  )
        
        # Tweak some of the plot properties
        plot.title = self.signal #"Scatter Plot With Lasso Selection"
        plot.title_position = "right"        
        plot.title_angle=270
        plot.line_width = 1
        plot.padding =20 
    
        # Right now, some of the tools are a little invasive, and we need the
        # actual ScatterPlot object to give to them
        my_plot = plot.plots["my_plot"][0]
    
        # Attach some tools to the plot
        self.selection = BeatSelection(component=my_plot)
        my_plot.active_tool = self.selection
        selection_overlay = RangeSelectionOverlay(component=my_plot)
        my_plot.overlays.append(selection_overlay)
    
        # Set up the trait handler for the selection
        self.selection.on_trait_change(self._selection_changed,
                                          'selection_completed')

        return plot

