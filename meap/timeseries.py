from traits.api import (HasTraits, Str, Array, Float,
          Bool, Enum, Instance, on_trait_change,Property,
          Button, List, CInt, CFloat)
from meap.gui_tools import (
    VGroup, HGroup, Item, HSplit, TableEditor, ObjectColumn, 
    OKButton, CancelButton, Plot, ArrayPlotData, RangeSelection, 
    RangeSelectionOverlay, marker_trait, LinePlot)
import numpy as np

from meap.gui_tools import ComponentEditor, ColorTrait, KeySpec
from meap.filters import smooth
from meap.gui_tools import MEAPView, messagebox
from meap import SUPPORTED_SIGNALS, colors


from scipy.stats.mstats import winsorize

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from meap import SEARCH_WINDOW
n_regions=0


class CensorSelection(RangeSelection):

    def moving_mouse_leave(self,event):
        return
    def moving_mouse_enter(self,event):
        return
    def moving_left_up(self,event):
        return
    def moving_mouse_move(self,event):
        return

class CensorRegion(HasTraits):
    start_time = Float(-1.)
    end_time = Float(-1.)
    viz = Instance(RangeSelectionOverlay,transient=True)
    metadata_name = Str
    plot = Instance(LinePlot,transient=True)

    traits_view = MEAPView(
        Item("start_time"),
        Item("end_time")
        )

    def set_limits(self,start,end):
        """
        """
        self.plot.index.metadata[self.metadata_name] = start, end

    @on_trait_change("plot.index.metadata")
    def metadata_chaned(self):
        try:
            if self.plot.index.metadata[self.metadata_name] is None:
                return
            st,end = self.plot.index.metadata[self.metadata_name]
        except KeyError:
            return
        self.start_time = st
        self.end_time = end

    def _viz_default(self):
        self.plot.active_tool = CensorSelection(self.plot,
                                               selection_mode="append",
                                               append_key=KeySpec("control"),
                                               left_button_selects = True,
                                               metadata_name=self.metadata_name)
        rso = RangeSelectionOverlay(component=self.plot,
                        metadata_name=self.metadata_name)
        self.plot.overlays.append(rso)
        return rso

def overlaps(a,b):
    # requires a[0] <= b[0]
    if a[0] == b[0]:
        return True
    if b[0] < a[1] :
        return True
    return False

def censor_region_overlappers(c_regs):
    if len(c_regs) == 0: return []
    # Make sure they don't overlap
    if not type(c_regs[0]) == type(()):
        censor_regions = [ (ci.start_time,ci.end_time) for ci in c_regs ]
    else:
        censor_regions = c_regs

    # Sort them according to the start_time
    censor_regions = sorted(censor_regions,key=lambda x: x[0] )

    # Loop over the intervals and merge those that overlap
    i = 0
    while i < len(censor_regions):
        j = i + 1
        while j < len(censor_regions):
            logger.info("%d,%d", i, j)
            if overlaps(censor_regions[i],censor_regions[j]):
                reg1 = censor_regions.pop(i)
                reg2 = censor_regions.pop(j-1)
                start_time = min(min(reg1),min(reg2))
                end_time   = max(max(reg1),max(reg2))
                censor_regions.insert(i, (start_time, end_time))
            else:
                j +=1
            logger.info(censor_regions)
        i += 1
    return censor_regions

censor_table = TableEditor(
    columns=[ObjectColumn(name="start_time",editable=True),
             ObjectColumn(name="end_time",editable=True)
            ],
    auto_size=True,
    show_toolbar=True,
    deletable=True,
    #edit_view="traits_view",
    #row_factory=CensorRegion
    )

class TimePoint(HasTraits):
    name = Str
    time = CFloat
    value = CFloat
    index = CInt
    #beat = Instance(meap.beat.HeartBeat)
    applies_to = Enum("ecg","dzdt","bp","systolic","diastolic","doppler")
    point_type = Enum("max", "min", "inflection",
                    "average","geom_trick","classifier")
    needs_attention = Bool(False)
    physiodata = Instance("meap.io.PhysioData")


    def __init__(self,**traits):
        super(TimePoint,self).__init__(**traits)
        self.physiodata = self.beat.physiodata
        index_array = getattr(self.physiodata, self.name+"_indices", None)

        # For fast time-to-index conversion
        if self.applies_to in ("systolic", "diastolic", "bp"):
            self.offset = self.physiodata.bp_pre_peak
        elif self.applies_to == "dzdt_karcher":
            self.offset = int(self.physiodata.dzdt_karcher_mean_time[0])
        else:
            self.offset = getattr(self.physiodata,self.applies_to + "_pre_peak")

        # Initialize to whatever's in the physiodata array
        if self.beat.id is not None and self.beat.id > -1:
            try:
                self.set_index(index_array[self.beat.id])
            except Exception, e:
                logger.warn("Error setting %s:\n%s",self.name, e)


    def mark_similar(self,unmarked_beat, smoothing_window_len=21,search_window=SEARCH_WINDOW):
        """
        Extracts a time point specific to ``unmarked_beat`` that
        has similar signal property (min/max/inflection) within
        a specific time window
        """
        # Get the necessary info from the unmarked beat
        unmarked_point = getattr(unmarked_beat, self.name)
        ts = getattr(unmarked_beat, self.applies_to+"_signal")
        # Define the search area
        window_min_idx = self.index - search_window
        window_max_idx = self.index + search_window + 1
        if self.index < search_window and self.point_type != "average":
            logger.warn( "timepoint too close to 0 for a symmetric search window")
            return
        if self.name == "q":
            search_window = 9
            smoothing_window_len = 0
        search_region = ts[
            (self.index-search_window):(self.index+search_window+1)
            ]

        # Smooth the timeseries if requested
        if smoothing_window_len > 0:
            search_region = smooth(search_region,
                        window_len=smoothing_window_len)
        # find the actual time of the point
        if self.point_type == "max":
            t_ind = window_min_idx + search_region.argmax()
        elif self.point_type == "min":
            t_ind = window_min_idx + search_region.argmin()
        elif self.point_type == "inflection":
            diff = np.ediff1d(search_region,to_begin=0)
            diff2 = np.ediff1d(diff,to_begin=0)
            t_ind = window_min_idx + np.abs(diff2).argmax()
        elif self.point_type == "geom_trick":
            r_idx = unmarked_beat.r.index
            c_idx = unmarked_beat.c.index
            if not r_idx <= c_idx:
                unmarked_beat.b.needs_attention = True
                return self
            roi = ts[r_idx:c_idx]
            line = np.interp(np.arange(len(roi)),
                             [0,len(roi)],
                             [roi[0],roi[-1]]
                            )
            t_ind = r_idx + np.argmax(line-roi)

        # Average is a special case.
        elif self.point_type == "average":
            unmarked_point.value = ts.mean()
            unmarked_point.set_index(0)
            return

        # Check that we aren't hitting an edge of the search reg
        if t_ind in (window_min_idx,window_max_idx):
            bnum = unmarked_beat.id if unmarked_beat.id is not None else -1
            logger.warn("[%d] %s point detected at edge of search region",
                        bnum, self.name)
            unmarked_point.needs_attention = True
        unmarked_point.set_index(t_ind)
        index_array = getattr(self.physiodata, self.name+"_indices", None)
        assert index_array[unmarked_beat.id] == t_ind
        return True

    def set_time(self,time):
        """
        If all we have is a time, adjust the index and value to match
        """
        self.time = time
        self.set_index(int(time) + self.offset)

    def set_index(self,index):
        """
        If all we have is a time, adjust the index and value to match
        """
        ts = getattr(self.beat, self.applies_to+"_signal")
        index_array = getattr(self.physiodata, self.name+"_indices", None)
        self.time = float(index) - self.offset
        self.index = index
        self.value = ts[index]
        if index_array is not None and self.beat.id > -1:
            index_array[self.beat.id] = self.index



class KarcherTimePoint(TimePoint):
    def __init__(self,**traits):
        """
        
        """
        super(KarcherTimePoint,self).__init__(**traits)
        self.physiodata = self.beat.physiodata
        index_array = self.physiodata.karcher_b_indices
        karcher_time = self.physiodata.dzdt_karcher_mean_time
        self.offset = int(karcher_time[0])
        # Initialize to whatever's in the physiodata array
        if self.beat.id is not None and self.beat.id > -1:
            try:
                self.set_index(index_array[self.beat.id])
            except Exception, e:
                logger.warn("Error setting %s:\n%s",self.name, e)
                
    def mark_similar(self,*args,**kwargs):
        pass

    def set_time(self,time):
        """
        If all we have is a time, adjust the index and value to match
        """
        self.time = time
        self.set_index(int(time) + self.offset)

    def set_index(self,index):
        """
        If all we have is a time, adjust the index and value to match
        """
        ts = self.beat.dzdt_signal
        index_array = getattr(self.physiodata,"karcher_b_indices")
        self.time = float(index) - self.offset
        self.index = index
        self.value = ts[index]
        index_array[self.beat.id] = self.index

            
class TimeSeries(HasTraits):
    # Holds data
    name=Str
    contains = Enum(SUPPORTED_SIGNALS)
    plot = Instance(Plot,transient=True)
    data=Array
    time = Property(Array,
            depends_on=["start_time","sampling_rate","data"])

    # Plotting options
    visible = Bool(True)
    ymax = Float()
    ymin = Float()
    line_type = marker_trait
    line_color = ColorTrait("blue")
    plot_type = Enum("line","scatter")
    censored_regions = List(Instance(CensorRegion))
    b_add_censor = Button(label="Add CensorRegion", transient=True)
    b_zoom_y = Button(label="Zoom y", transient=True)
    b_info = Button(label="Info", transient=True)
    b_clear_censoring = Button(label="Clear Censoring", transient=True)
    renderer = Instance(LinePlot,transient=True)

    # For the winsorizing steps
    winsor_swap = Array
    winsor_min = Float(0.005)
    winsor_max = Float(0.005)
    winsorize = Bool(False)
    winsor_enable = Bool(True)


    def __init__(self,**traits):
        """
        Class to represent data collected over time
        """
        super(TimeSeries,self).__init__(**traits)
        if not self.contains in self.physiodata.contents:
            raise ValueError("Signal not found in data")
        self.name = self.contains

        # First, check whether it's already winsorized
        winsorize_trait = self.name + "_winsorize"
        if getattr(self.physiodata, winsorize_trait):
            self.winsor_enable = False
        self.winsor_min = getattr(self.physiodata,self.name + "_winsor_min")
        self.winsor_max = getattr(self.physiodata,self.name + "_winsor_max")

        # Load the actual data
        self.data = getattr(self.physiodata, self.contains + "_data")
        self.winsor_swap = self.data.copy()

        self.sampling_rate = getattr(self.physiodata,
                                    self.contains + "_sampling_rate")
        self.sampling_rate_unit = getattr(self.physiodata,
                                    self.contains + "_sampling_rate_unit")
        self.start_time = getattr(self.physiodata,
                                    self.contains + "_start_time")
        """
        The censored regions are loaded from physiodata INITIALLY.
        from that point on the censored regions are accessed from
        physiodata's
        """

        self.line_color = colors[self.contains]
        self.n_censor_intervals = 0
        for (start,end), source in zip(self.physiodata.censored_intervals,
                                       self.physiodata.censoring_sources):
            if str(source) == self.contains:
                self.censored_regions.append(
                    CensorRegion(
                        start_time=start,
                        end_time=end,
                        metadata_name=self.__get_metadata_name())
                    )

    def __get_metadata_name(self):
        name = self.contains + "%03d"%self.n_censor_intervals
        self.n_censor_intervals += 1
        return name

    def _winsorize_changed(self):
        if self.winsorize:
            logger.info("Winsorizing %s with limits=(%.5f%.5f)",self.name,self.winsor_min, self.winsor_max)
            # Take the original data and replace it with the winsorized version
            self.data = np.array(winsorize(self.winsor_swap,
                                  limits=(self.winsor_min,self.winsor_max)))
            setattr(self.physiodata,self.name+"_winsor_min",self.winsor_min)
            setattr(self.physiodata,self.name+"_winsor_max",self.winsor_max)

        else:
            logger.info("Restoring %s to its original data", self.name)
            self.data = self.winsor_swap.copy()
        setattr(self.physiodata, self.contains + "_data", self.data)
        setattr(self.physiodata,self.name+"_winsorize", self.winsorize)
        self.plot.range2d.y_range.low = self.data.min()
        self.plot.range2d.y_range.high = self.data.max()
        self.plot.request_redraw()


    def __str__(self):
        descr = "Timeseries: %s\n" % self.name
        descr += "-" * len(descr) + "\n\t" + \
               "\n\t".join([
                   "Sampling rate: %.4f%s" % (self.sampling_rate,self.sampling_rate_unit),
                   "N samples: %d" % self.data.shape[0],
                   "N censored intervals: %d" % len(self.censored_regions),
                   "Start time: %.3f" % self.start_time
                   ])
        return descr

    def _plot_default(self):
        # Create plotting components
        plotdata = ArrayPlotData(time=self.time,data=self.data)
        plot = Plot(plotdata)
        self.renderer = plot.plot(("time","data"),color=self.line_color)[0]
        plot.title = self.name
        plot.title_position = "right"
        plot.title_angle = 270
        plot.line_width=1
        plot.padding=25
        plot.width = 400

        # Load the censor regions and add them to the plot
        for censor_region in self.censored_regions:
            # Make a censor region on the Timeseries object
            censor_region.plot = self.renderer
            censor_region.viz
            censor_region.set_limits(censor_region.start_time,
                                     censor_region.end_time)
        return plot

    def _get_time(self):
        # Ensure the properties that depend on time are updated
        return np.arange(len(self.data)) / self.sampling_rate + self.start_time

    def _b_info_fired(self):
        messagebox(str(self))

    def _b_zoom_y_fired(self):
        ui = self.edit_traits(view="zoom_view", kind="modal")
        if ui.result:
            self.plot.range2d.y_range.low = self.ymin
            self.plot.range2d.y_range.high = self.ymax

    def _b_clear_censoring_fired(self):
        for reg in self.censored_regions:
            self.renderer.overlays.remove(reg.viz)
        self.censored_regions = []
        self.plot.request_redraw()
        self.n_censor_intervals = 0

    def _b_add_censor_fired(self):
        self.censored_regions.append(
                CensorRegion(plot=self.renderer,
                    metadata_name=self.__get_metadata_name())
        )
        self.censored_regions[-1].viz

    buttons = VGroup(
            Item("b_add_censor",show_label=False),
            #Item("b_info"),
            #Item("b_zoom_y"),
            Item("winsorize",enabled_when="winsor_enable"),
            Item("winsor_max",enabled_when="winsor_enable",format_str="%.4f",width=25),
            Item("winsor_min",enabled_when="winsor_enable",format_str="%.4f",width=25),
            Item("b_clear_censoring",show_label=False),
            show_labels=True,show_border=True)
    widgets = HSplit(
        Item('plot',editor=ComponentEditor(), show_label=False),
        buttons)

    zoom_view = MEAPView(
            HGroup("ymin", "ymax"),buttons = [OKButton,CancelButton])

    traits_view = MEAPView(widgets, width=500, height=300, resizable=True )
