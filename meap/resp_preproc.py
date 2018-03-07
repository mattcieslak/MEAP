#!/usr/bin/env python
from traits.api import (HasTraits, Str, Array, Float,cached_property,
          Bool, Enum, Instance, on_trait_change,Property,
          Range, DelegatesTo, Int, Button, List)

from meap.io import PhysioData
from meap.filters import (bandpass, smooth, regress_out,
                          censor_peak_times, legendre_detrend, lowpass, find_peaks)
from meap import fail, MEAPView, messagebox
from meap.timeseries import TimeSeries
import numpy as np

# Needed for Tabular adapter
from traitsui.api import (Group, View, Item, TableEditor,
        ObjectColumn,VSplit, RangeEditor)
from traitsui.menu import OKButton, CancelButton
from chaco.api import Plot, ArrayPlotData, VPlotContainer
from chaco.scatterplot import ScatterPlot
from chaco.lineplot import LinePlot

from enable.component_editor import ComponentEditor
from scipy.interpolate import interp1d
from scipy.stats.mstats import winsorize


class RespirationProcessor(HasTraits):
    # Parameters
    resp_polort = DelegatesTo("physiodata")
    resp_high_freq_cutoff = DelegatesTo("physiodata")

    time = DelegatesTo("physiodata", "processed_respiration_time")

    # Data object and arrays to save
    physiodata = Instance(PhysioData) 
    respiration_cycle = DelegatesTo("physiodata")
    respiration_amount = DelegatesTo("physiodata")
    
    # For visualization
    plots = Instance(VPlotContainer,transient=True)

    # Respiration plot
    resp_plot = Instance(Plot,transient=True)
    resp_plot_data = Instance(ArrayPlotData,transient=True)
    resp_signal = Array() # Could be filtered z0 or resp belt
    raw_resp_signal = Array() # Could be filtered z0 or resp belt
    resp_signal = Property(Array,
            depends_on="physiodata.resp_polort,physiodata.resp_high_freq_cutoff")
    polyfilt_resp = Array()
    lpfilt_resp = DelegatesTo("physiodata", "processed_respiration_data")
    resp_inhale_begin_times = DelegatesTo("physiodata")
    resp_inhale_begin_values = Array
    resp_exhale_begin_times = DelegatesTo("physiodata")
    resp_exhale_begin_values = Array
    
    z0_plot = Instance(Plot,transient=True)
    z0_plot_data = Instance(ArrayPlotData,transient=True)
    raw_z0_signal = Array()
    resp_corrected_z0 = DelegatesTo("physiodata")

    dzdt_plot = Instance(Plot,transient=True)
    dzdt_plot_data = Instance(ArrayPlotData,transient=True)
    raw_dzdt_signal = Array()
    resp_corrected_dzdt = DelegatesTo("physiodata")
    
    start_time = Range(low=0,high=10000.0, initial=0.0)
    window_size = Range(low=1.0,high=300.0,initial=30.0)

    state = Enum("unusable", "z0", "resp", "z0_resp")
    dirty = Bool(True)

    b_process = Button(label="Process Respiration")
    
    def __init__(self,**traits):
        super(RespirationProcessor,self).__init__(**traits)
        resp_inc = "respiration" in self.physiodata.contents
        z0_inc = "z0" in self.physiodata.contents
        if z0_inc and resp_inc:
            self.state = "z0_resp"
            #messagebox("Using both respiration belt and ICG")
            z0_signal = self.physiodata.z0_data
            dzdt_signal = self.physiodata.dzdt_data
            resp_signal = self.physiodata.respiration_data
        elif resp_inc and not z0_inc:
            self.state = "resp"
            #messagebox("Only respiration belt data will be used.")
            resp_signal = self.physiodata.resp_data
            z0_signal = None
            dzdt_signal = None
        elif z0_inc and not resp_inc:
            self.state = "z0"
            #messagebox("Using only z0 channel to estimate respiration")
            resp_signal = self.physiodata.z0_data
            z0_signal = self.physiodata.z0_data
            dzdt_signal = self.physiodata.dzdt_data
            self.resp_polort = 1
        else:
            self.state = "unusable"
            #messagebox("No respiration belt or z0 channels found")

        # Establish  the maximum shared length of resp_inhale_begin_values
        signals = [sig for sig in (resp_signal, z0_signal,dzdt_signal) if \
                sig is not None]
        if len(signals) > 1:
            minlen = min([len(sig) for sig in signals])
        else:
            minlen = len(signals[0])

        # Establish a time array
        if resp_inc:
            self.time = TimeSeries(physiodata = self.physiodata, 
                            contains="respiration").time[:minlen]
        else:
            self.time = TimeSeries(physiodata = self.physiodata, 
                            contains="z0").time[:minlen]

        # Get out the final signals
        if z0_inc:
            self.raw_resp_signal = z0_signal[:minlen].copy()
            self.raw_resp_signal[:50] = self.raw_resp_signal.mean()
            self.raw_z0_signal = z0_signal[:minlen].copy()
            self.raw_z0_signal[:50] = self.raw_z0_signal.mean()
            self.raw_dzdt_signal = dzdt_signal[:minlen].copy()
            self.raw_dzdt_signal[:50] = self.raw_dzdt_signal.mean()
        if resp_inc:
            self.raw_resp_signal = resp_signal[:minlen].copy()
        # if data already exists, it can't be dirty
        if self.physiodata.resp_exhale_begin_times.size > 0: self.dirty = False
        self.on_trait_change(self._parameter_changed,"resp_polort,resp_high_freq_cutoff")
        
    @cached_property
    def _get_resp_signal(self):
        resp_signal = winsorize(self.raw_resp_signal)
        return (resp_signal - resp_signal.mean()) / resp_signal.std()

    def _parameter_changed(self):
        self.dirty = True

    def _b_process_fired(self):
        self.process()
        
    def process(self):
        """
        processes the respiration timeseries
        """
        sampling_rate = 1./(self.time[1] - self.time[0])
        # Respiration belts can lose tension over time. 
        # This removes linear trends
        if self.resp_polort > 0:
            pfit = legendre_detrend(self.resp_signal, self.resp_polort)
            if not pfit.shape == self.time.shape:
                messagebox("Legendre detrend failed")
                return
            self.polyfilt_resp = pfit
        else:
            self.polyfilt_resp = self.resp_signal    
        
        lpd = lowpass( self.polyfilt_resp, self.resp_high_freq_cutoff,
                      sampling_rate )
        if not lpd.shape == self.time.shape: 
            messagebox("lowpass filter failed")
            return
        
        self.lpfilt_resp = (lpd - lpd.mean()) / lpd.std()
        
        resp_inhale_begin_indices, resp_exhale_begin_indices = find_peaks(
                self.lpfilt_resp, maxima=True, minima=True)
        self.resp_inhale_begin_times = self.time[resp_inhale_begin_indices]
        self.resp_exhale_begin_times = self.time[resp_exhale_begin_indices]
        self.resp_corrected_z0 = regress_out(self.raw_z0_signal, self.lpfilt_resp)
        self.resp_corrected_dzdt = regress_out(self.raw_dzdt_signal, self.lpfilt_resp)
        
        # update the scatterplot if we're interactive
        if self.resp_plot_data is not None:
            self.resp_plot_data.set_data("inhale_times",self.resp_inhale_begin_times)
            self.resp_plot_data.set_data("inhale_values",self.lpfilt_resp[
                                            resp_inhale_begin_indices])
            self.resp_plot_data.set_data("exhale_times",self.resp_exhale_begin_times)
            self.resp_plot_data.set_data("exhale_values",self.lpfilt_resp[
                                            resp_exhale_begin_indices])
            self.resp_plot_data.set_data("lpfilt_resp",self.lpfilt_resp )
            self.dzdt_plot_data.set_data("cleaned_data",self.resp_corrected_dzdt)
            self.z0_plot_data.set_data("cleaned_data",self.resp_corrected_z0)
        else:
            print "plot data is none"
            
        # Update the respiration cycles cached_pro
        times = np.concatenate([np.zeros(1), self.resp_exhale_begin_times,
                                self.resp_inhale_begin_times, 
                                self.time[-1,np.newaxis]])
        vals = np.concatenate([np.zeros(1), np.zeros_like(resp_exhale_begin_indices),
                    0.5*np.ones_like(resp_inhale_begin_indices), np.zeros(1)])
        srt = np.argsort(times)
        terp = interp1d(times[srt],vals[srt])
        self.respiration_cycle = terp(self.time)
        self.respiration_amount = np.abs(np.ediff1d(self.respiration_cycle,to_begin=0))

    def _resp_plot_default(self):
        # Create plotting components
        self.resp_plot_data = ArrayPlotData(
            time=self.time,
            inhale_times=self.resp_inhale_begin_times,
            inhale_values=self.resp_inhale_begin_values,
            exhale_times=self.resp_exhale_begin_times,
            exhale_values=self.resp_exhale_begin_values,
            filtered_resp=self.resp_signal,
            lpfilt_resp=self.lpfilt_resp
        )
        plot = Plot(self.resp_plot_data)
        plot.plot(("time","filtered_resp"),color="blue",line_width=1)
        plot.plot(("time","lpfilt_resp"),color="green",line_width=1)
        # Plot the inhalation peaks
        plot.plot(("inhale_times","inhale_values"),type="scatter",marker="square")
        plot.plot(("exhale_times","exhale_values"),type="scatter",marker="circle")
        plot.title = "Respiration"
        plot.title_position = "right"
        plot.title_angle = 270
        plot.padding=20
        return plot
        
    def _z0_plot_default(self):
        self.z0_plot_data = ArrayPlotData(
            time=self.time,
            raw_data=self.raw_z0_signal,
            cleaned_data=self.resp_corrected_z0
        )
        plot = Plot(self.z0_plot_data)
        plot.plot(("time","raw_data"),color="blue",line_width=1)
        plot.plot(("time","cleaned_data"),color="green",line_width=1)
        plot.title = "z0"
        plot.title_position = "right"
        plot.title_angle = 270
        plot.padding=20
        return plot
    
    def _dzdt_plot_default(self):
        """
        Creates a plot of the ecg_ts data and the signals derived during
        the Pan Tomkins algorithm
        """
        # Create plotting components
        self.dzdt_plot_data = ArrayPlotData(
            time=self.time,
            raw_data=self.raw_dzdt_signal,
            cleaned_data=self.resp_corrected_dzdt
        )
        plot = Plot(self.dzdt_plot_data)
        plot.plot(("time","raw_data"), color="blue", line_width=1)
        plot.plot(("time","cleaned_data"),color="green",line_width=1)
        plot.title = "dZ/dt"
        plot.title_position = "right"
        plot.title_angle = 270
        plot.padding=20
        return plot
    
    def _plots_default(self):
        plots_to_include = ()
        if self.state in ("z0_resp", "z0"):
            self.index_range = self.resp_plot.index_range
            self.dzdt_plot.index_range = self.index_range
            self.z0_plot.index_range = self.index_range
            plots_to_include = [self.resp_plot,self.z0_plot,self.dzdt_plot]
        elif self.state == "resp":
            self.index_range = self.resp_plot.index_range
            plots_to_include = [self.resp_plot]
        return VPlotContainer(*plots_to_include)
    
    @on_trait_change("window_size,start_time")
    def update_plot_range(self):
        self.resp_plot.index_range.high = self.start_time + self.window_size
        self.resp_plot.index_range.low = self.start_time

    proc_params_group = Group(
        Group(
              Item("resp_polort"),
              Item("resp_high_freq_cutoff"),
              Item("b_process",show_label=False,
                  enabled_when="state != unusable and dirty"),
              label="Resp processing options",
              show_border=True,
              orientation="vertical",
              springy=True
              ),
        orientation="horizontal")

    plot_group = Group(
            Group(
                Item("plots",editor=ComponentEditor(),
                     width=800,height=700),
                show_labels=False),
            Item("start_time"),
            Item("window_size")
        )
    traits_view = MEAPView(
        VSplit(
            plot_group,
            proc_params_group,
            ),
        resizable=True,
        win_title="Process Respiration Data",
        
        )
