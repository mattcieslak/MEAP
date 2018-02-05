#!/usr/bin/env python
from traits.api import (HasTraits, Array, cached_property,
          Bool, Instance, on_trait_change, Property, Float,
          Range, Int, DelegatesTo,  Button)

from meap.filters import (bandpass, smooth, times_contained_in,
        censor_peak_times, normalize,find_peaks)
from meap import MEAPView, messagebox
from meap.io import PhysioData, peak_stack
from meap.timeseries import TimeSeries
from skimage.filters import threshold_otsu

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time
import numpy as np

# Needed for Tabular adapter
from traitsui.api import Group, Item, VSplit, VGroup, HGroup
from traitsui.menu import OKButton, CancelButton
from chaco.api import ArrayPlotData,Plot, jet
#from chaco.tools.api import PanTool, ZoomTool
from chaco.scatterplot import ScatterPlot
from chaco.tools.api import RangeSelectionOverlay, ZoomTool
from enable.component_editor import ComponentEditor

# For the 3d peak window
from meap.peak_picker import PeakPickingTool, PeakPickingOverlay

# Dependencies to determine when new signals need to get generated
algorithm_parameters = (
     "physiodata.bandpass_min,physiodata.bandpass_max,"
     "physiodata.smoothing_window_len,physiodata.smoothing_window,"
     "physiodata.pt_adjust,physiodata.peak_threshold,"
     "physiodata.apply_filter,physiodata.apply_diff_sq,"
     "physiodata.apply_smooth_ma,physiodata.peak_window,"
     )
# An aux signal is a simultaneously recorded signal that has VERY EASY to
# find peaks in it.  It is used to constrain the search for QRS complexes.
aux_signal_dependencies = ("physiodata.secondary_heartbeat_window_len,"
     "physiodata.secondary_heartbeat_pre_msec,physiodata.secondary_heartbeat_window,"
     "physiodata.secondary_heartbeat_window_len,physiodata.secondary_heartbeat")

# QRS Power Signal is the result of applying a series of filters to the ECG
# It is highest when the local properties of the signal look like a QRS complex
qrs_power_signal_dependencies = ('physiodata.bandpass_min,physiodata.qrs_source_signal,'
       'physiodata.bandpass_max,physiodata.smoothing_window,physiodata.ecg2_weight,'
       'physiodata.smoothing_window_len,physiodata.apply_filter,physiodata.use_ECG2,'
       'physiodata.apply_diff_sq,physiodata.apply_smooth_ma')

peak_detection_parameters = ("physiodata.use_secondary_heartbeat,physiodata.pt_adjust,"
                             "physiodata.peak_threshold")

algorithm_parameters = ",".join([aux_signal_dependencies,qrs_power_signal_dependencies,
    peak_detection_parameters])

class StaticCensoredInterval(RangeSelectionOverlay):
    #Used to represent a region censored by the user
    axis = "index"
    border_width = 0.
    fill_color = "gray"
    alpha = 0.3

class AuxSignalWindow(RangeSelectionOverlay):
    # Represents a viable time interval based on aux signal peaks
    axis = "index"
    #border_width = 1.
    fill_color = "purple"
    border_color = "purple"
    alpha = 0.2


class ImageRangeSelector(ZoomTool):
    xmin = Float(0)
    xmax = Float(0)
    ymin = Float(0)
    ymax = Float(0)
    image_plot_selected = Int(0)
    enable_wheel=False
        
    def _end_select(self, event):
        self._screen_end = (event.x, event.y)
        start = np.array(self._screen_start)
        end = np.array(self._screen_end)
        if sum(abs(end - start)) < self.minimum_screen_delta:
            self._end_selecting(event)
            event.handled = True
            return
        low, high = self._map_coordinate_box(self._screen_start, self._screen_end)
        self.xmin, self.ymin = low
        self.xmax, self.ymax = high
        self.image_plot_selected += 1
        return self._end_selecting(event)

class PanTomkinsDetector(HasTraits):
    # Holds the data Source
    physiodata = Instance(PhysioData) 
    ecg_ts = Instance(TimeSeries)
    dirty = Bool(False)
    
    # For visualization
    plot = Instance(Plot,transient=True)
    plot_data = Instance(ArrayPlotData,transient=True)
    image_plot = Instance(Plot,transient=True)
    image_plot_data = Instance(ArrayPlotData,transient=True)
    image_selection_tool = Instance(ImageRangeSelector)
    image_plot_selected = DelegatesTo("image_selection_tool")
    peak_vis = Instance(ScatterPlot,transient=True)
    scatter = Instance(ScatterPlot,transient=True)
    start_time = Range(low=0,high=100000.0, initial=0.0)
    window_size = Range(low=1.0,high=100000.0,initial=30.0)
    peak_editor = Instance(PeakPickingTool,transient=True)
    show_censor_regions = Bool(True)

    # parameters for processing the raw data before PT detecting
    bandpass_min = DelegatesTo("physiodata")
    bandpass_max = DelegatesTo("physiodata") 
    smoothing_window_len = DelegatesTo("physiodata") 
    smoothing_window = DelegatesTo("physiodata") 
    pt_adjust = DelegatesTo("physiodata")
    peak_threshold = DelegatesTo("physiodata")
    apply_filter = DelegatesTo("physiodata")
    apply_diff_sq = DelegatesTo("physiodata")
    apply_smooth_ma = DelegatesTo("physiodata")
    peak_window = DelegatesTo("physiodata")
    qrs_source_signal = DelegatesTo("physiodata")
    qrs_power_signal = Property(Array, depends_on=qrs_power_signal_dependencies)
    
    # Use a secondary signal to limit the search range
    use_secondary_heartbeat = DelegatesTo("physiodata")
    secondary_heartbeat = DelegatesTo("physiodata")
    secondary_heartbeat_abs = DelegatesTo("physiodata")
    secondary_heartbeat_pre_msec = DelegatesTo("physiodata")
    secondary_heartbeat_window = DelegatesTo("physiodata")
    secondary_heartbeat_window_len = DelegatesTo("physiodata")
    secondary_heartbeat_n_likelihood_bins = DelegatesTo("physiodata")
    aux_signal = Property(Array)#, depends_on=aux_signal_dependencies)

    # Secondary ECG signal 
    can_use_ecg2 = Bool(False)
    use_ECG2 = DelegatesTo("physiodata")
    ecg2_weight = DelegatesTo("physiodata")
    
    # The results of the peak search
    peak_indices = DelegatesTo("physiodata")
    peak_times = DelegatesTo("physiodata")
    peak_values = Array
    dne_peak_indices = DelegatesTo("physiodata")
    dne_peak_times = DelegatesTo("physiodata")
    dne_peak_values = Array
    # Holds the timeseries for the signal vs noise thresholds
    thr_times = Array
    thr_vals = Array

    # for pulling out data for aggregating ecg_ts, dzdt_ts and bp beats
    ecg_matrix = DelegatesTo("physiodata")

    # UI elements
    b_detect = Button(label="Detect QRS", transient=True)
    b_shape = Button(label="Shape-Based Tuning", transient=True)

    def __init__(self,**traits):
        super(PanTomkinsDetector,self).__init__(**traits)
        self.ecg_ts = TimeSeries(physiodata=self.physiodata,contains="ecg")
        # relevant data to be plotted
        self.ecg_time = self.ecg_ts.time
        
        self.can_use_ecg2 = "ecg2" in self.physiodata.contents
        
        # Which ECG signal to use?
        if self.qrs_source_signal == "ecg" or not self.can_use_ecg2:
            self.ecg_signal = normalize(self.ecg_ts.data)
            if self.can_use_ecg2:
                self.ecg2_signal = normalize(self.physiodata.ecg2_data)
            else:
                self.ecg2_signal = None
        else:
            self.ecg_signal = normalize(self.physiodata.ecg2_data)
            self.ecg2_signal = normalize(self.physiodata.ecg_data)
        
        self.thr_times = self.ecg_time[np.array([0,-1])]
        self.censored_intervals = self.physiodata.censored_intervals
        # graphics containers
        self.aux_window_graphics = []
        self.censored_region_graphics = []
        # If peaks are already available in the mea.mat, 
        # they have already been marked
        if len(self.peak_times): 
            self.dirty = False
            self.peak_values = self.ecg_signal[self.peak_indices]
        if self.dne_peak_indices is not None and len(self.dne_peak_indices):
            self.dne_peak_values = self.ecg_signal[self.dne_peak_indices]
    
    def _b_shape_fired(self):
        self.shape_based_tuning()

    def shape_based_tuning(self):
        logger.info("Running shape-based tuning") 
        qrs_stack = peak_stack(self.peak_indices, self.qrs_power_signal,
                               pre_msec=300,post_msec=700,
                               sampling_rate=self.physiodata.ecg_sampling_rate)
        
        
        
    def _b_detect_fired(self):
        self.detect()
        

    def _show_censor_regions_changed(self):
        for crg in self.censored_region_graphics:
            crg.visible = self.show_censor_regions
        self.plot.request_redraw()
        
    @cached_property
    def _get_qrs_power_signal(self):
        if self.apply_filter:
            filtered_ecg = normalize(bandpass(self.ecg_signal, 
                    self.bandpass_min, self.bandpass_max,
                     self.ecg_ts.sampling_rate))
        else:
            filtered_ecg = self.ecg_signal

        # Differentiate and square the signal?
        if self.apply_diff_sq:
            # Differentiate the signal and square it
            diff_sq  = np.ediff1d(filtered_ecg, to_begin = 0)**2
        else:
            diff_sq = filtered_ecg

        # If we're to apply a Moving Average smoothing
        if self.apply_smooth_ma:
            # MA smoothing
            smooth_ma = smooth(diff_sq,window_len=self.smoothing_window_len,
                 window=self.smoothing_window)
        else:
            smooth_ma = diff_sq
        if not self.use_ECG2:
            # for visualization purposes    
            return normalize(smooth_ma)
        logger.info("Using 2nd ECG signal")
        
        # Use secondary ECG signal and combine QRS power
        if self.apply_filter:
            filtered_ecg2 = normalize(bandpass(self.ecg2_signal, 
                    self.bandpass_min, self.bandpass_max,
                     self.ecg_ts.sampling_rate))
        else:
            filtered_ecg2 = self.ecg2_signal
        if self.apply_diff_sq:
            diff_sq2  = np.ediff1d(filtered_ecg2, to_begin = 0)**2
        else:
            diff_sq2 = filtered_ecg2

        if self.apply_smooth_ma:
            smooth_ma2 = smooth(diff_sq2,window_len=self.smoothing_window_len,
                 window=self.smoothing_window)
        else:
            smooth_ma2 = diff_sq2
            
        return normalize(
            ((1-self.ecg2_weight)*smooth_ma + self.ecg2_weight*smooth_ma2)**2)

    def _get_aux_signal(self):
        if not self.use_secondary_heartbeat: return np.array([])
        sig = getattr(self.physiodata, self.secondary_heartbeat + "_data")
        if self.secondary_heartbeat_abs:
            sig = np.abs(sig)
        if self.secondary_heartbeat_window_len > 0:
            sig = smooth(sig,window_len = self.secondary_heartbeat_window_len,
                    window=self.secondary_heartbeat_window)
        return normalize(sig)

    @on_trait_change(algorithm_parameters)
    def params_edited(self):
        self.dirty = True

    def update_aux_window_graphics(self):
        if not hasattr(self,"ecg_trace") or not self.use_secondary_heartbeat: return
        #remove the old graphics
        for aux_window in self.aux_window_graphics:
            del self.ecg_trace.index.metadata[aux_window.metadata_name]
            self.ecg_trace.overlays.remove(aux_window)
        # add the new graphics
        for n, (start,end) in enumerate(self.aux_windows/1000.):
            window_key = "aux%03d" % n
            self.aux_window_graphics.append(
                AuxSignalWindow(
                    component=self.aux_trace,
                    metadata_name = window_key
                    )
            )
            self.ecg_trace.overlays.append(self.aux_window_graphics[-1])
            self.ecg_trace.index.metadata[window_key] = start, end
    
    def get_aux_windows(self):
        aux_peaks = find_peaks(self.aux_signal)
        aux_pre_peak = aux_peaks - self.secondary_heartbeat_pre_msec
        aux_windows = np.column_stack([aux_pre_peak, aux_peaks])
        return aux_windows

    def detect(self):
        """
        Implementation of the Pan Tomkins QRS detector
        """
        logger.info("Beginning QRS Detection")
        t0 = time.time()

        # The original paper used a different method for finding peaks
        smoothdiff = normalize(np.ediff1d(self.qrs_power_signal, to_begin=0))
        peaks = find_peaks(smoothdiff)
        peak_amps = self.qrs_power_signal[peaks]

        # Part 2: getting rid of useless peaks
        # ====================================
        # There are lots of small, irrelevant peaks that need to
        # be discarded. 
        # 2a) remove peaks occurring in censored intervals
        censor_peak_mask = censor_peak_times(self.ecg_ts.censored_regions, #TODO: switch to physiodata.censored_intervals   
                                             self.ecg_time[peaks])
        n_removed_peaks = peaks.shape[0] - censor_peak_mask.sum()
        logger.info("%d/%d potential peaks outside %d censored intervals", 
            n_removed_peaks,peaks.shape[0],len(self.ecg_ts.censored_regions))
        peaks = peaks[censor_peak_mask]
        peak_amps = peak_amps[censor_peak_mask]

        # 2b) if a second signal is used, make sure the ecg peaks are
        #     near aux signal peaks
        if self.use_secondary_heartbeat:
            self.aux_windows = self.get_aux_windows()
            aux_mask = times_contained_in(peaks, self.aux_windows)
            logger.info("Using secondary signal")
            logger.info("%d aux peaks detected",self.aux_windows.shape[0])
            logger.info("%d/%d peaks contained in second signal window",
                    aux_mask.sum(), peaks.shape[0])
            peaks = peaks[aux_mask]
            peak_amps = peak_amps[aux_mask]

        # 2c) use otsu's method to find a cutoff value
        peak_amp_thr = threshold_otsu(peak_amps)+self.pt_adjust
        otsu_mask = peak_amps > peak_amp_thr
        logger.info("otsu threshold: %.7f", peak_amp_thr)
        logger.info("%d/%d peaks survive", otsu_mask.sum(), peaks.shape[0])
        peaks = peaks[otsu_mask]
        peak_amps = peak_amps[otsu_mask]

        # 3) Make sure there is only one peak in each secondary window
        #    this is accomplished by estimating the distribution of peak
        #    times relative to aux window starts
        if self.use_secondary_heartbeat:
            npeaks = peaks.shape[0]
            # obtain a distribution of times and amplitudes
            rel_peak_times = np.zeros_like(peaks)
            window_subsets = []
            for start,end in self.aux_windows:
                mask = np.flatnonzero((peaks >= start) & (peaks <= end))
                if len(mask) == 0: continue
                window_subsets.append(mask)
                rel_peak_times[mask] = peaks[mask] - start
            
            # how common are relative peak times relative to window starts
            densities, bins = np.histogram(rel_peak_times,
                    range=(0,self.secondary_heartbeat_pre_msec),
                    bins=self.secondary_heartbeat_n_likelihood_bins, density=True)
            
            likelihoods = densities[
                np.clip(np.digitize(rel_peak_times,bins)-1, 0, 
                    self.secondary_heartbeat_n_likelihood_bins - 1)]
            _peaks = []
            # Pull out the maximal peak 
            for subset in window_subsets:
                # If there's only a single peak contained, no need for math
                if len(subset) == 1:
                    _peaks.append(peaks[subset[0]])
                    continue
                
                _peaks.append( peaks[subset[np.argmax(likelihoods[subset])]] )
            peaks = np.array(_peaks)
            logger.info("Only 1 peak per aux window allowed:" + \
                   " %d/%d peaks remaining",peaks.shape[0],npeaks)
            
        # Check that no two peaks are too close:
        peak_diffs = np.ediff1d(peaks,to_begin=500)
        peaks = peaks[peak_diffs > 200] # Cutoff is 300BPM

        # Stack the peaks and see if the original data has a higher value
        raw_stack = peak_stack(peaks, self.ecg_signal, 
                      pre_msec=self.peak_window, post_msec=self.peak_window,
                      sampling_rate=self.ecg_ts.sampling_rate)
        adj_factors = np.argmax(raw_stack,axis=1) - self.peak_window
        peaks = peaks + adj_factors
        
        self.peak_indices = peaks
        self.peak_values = self.ecg_signal[peaks]
        self.peak_times = self.ecg_time[peaks]
        self.thr_vals = np.array([peak_amp_thr]*2)
        t1 = time.time()
        # update the scatterplot if we're interactive
        if self.plot_data is not None:
            self.plot_data.set_data("peak_times",self.peak_times )
            self.plot_data.set_data("peak_values",self.peak_values )
            self.plot_data.set_data("qrs_power",self.qrs_power_signal)
            self.plot_data.set_data("aux_signal",self.aux_signal)
            self.plot_data.set_data("thr_vals",self.thr_vals)
            self.plot_data.set_data("thr_times",self.thr_vals)
            self.plot_data.set_data("thr_times",
                    np.array([self.ecg_time[0],self.ecg_time[-1]])),
            self.update_aux_window_graphics()
            self.plot.request_redraw()
            self.image_plot_data.set_data("imagedata", self.ecg_matrix)
            self.image_plot.request_redraw()
        else:
            print "plot data is none"
        logger.info("found %d QRS complexes in %.3f seconds", 
                len(self.peak_indices), t1 - t0)
        self.dirty = False
    
    def _plot_default(self):
        """
        Creates a plot of the ecg_ts data and the signals derived during
        the Pan Tomkins algorithm
        """
        # Create plotting components
        self.plot_data = ArrayPlotData(
            
            time=self.ecg_time,
            raw_ecg= self.ecg_signal,
            qrs_power = self.qrs_power_signal,
            aux_signal = self.aux_signal,
            # Ensembleable peaks
            peak_times=self.peak_times,
            peak_values=self.peak_values,
            # Non-ensembleable, useful for HR, HRV
            dne_peak_times=self.dne_peak_times,
            dne_peak_values=self.dne_peak_values,
            # Threshold slider bar
            thr_times = np.array([self.ecg_time[0],self.ecg_time[-1]]),
            thr_vals= np.array([0,0])
        )

        plot = Plot(self.plot_data, use_backbuffer=True)
        self.aux_trace = plot.plot(("time","aux_signal"),color="purple",
               alpha=0.6, line_width=2)[0]
        self.qrs_power_trace = plot.plot(
                ("time","qrs_power"), color="blue",alpha=0.8)[0]
        self.ecg_trace = plot.plot(("time","raw_ecg"), color="red",
                line_width=2)[0]
        
        # Load the censor regions and add them to the plot
        for n, (start,end) in enumerate(self.censored_intervals):
            censor_key = "censor%03d"%n 
            self.censored_region_graphics.append(
                StaticCensoredInterval(
                    component = self.qrs_power_trace,
                    metadata_name=censor_key
            ))
            self.ecg_trace.overlays.append(self.censored_region_graphics[-1])
            self.ecg_trace.index.metadata[censor_key] = start, end


        # Line showing 
        self.threshold_trace = plot.plot(("thr_times","thr_vals"),
                            color="black", line_width=3)[0]

        # Plot for plausible peaks
        self.ppeak_vis = plot.plot(("dne_peak_times","dne_peak_values"),
                type="scatter",marker="diamond")[0]
        
        # Make a scatter plot where you can edit the peaks
        self.peak_vis = plot.plot(("peak_times","peak_values"),type="scatter")[0]
        self.scatter = plot.components[-1]
        self.peak_editor = PeakPickingTool(self.scatter)
        self.scatter.tools.append(self.peak_editor)
        # when the user adds or removes a point, automatically extract 
        self.on_trait_event(self._change_peaks,"peak_editor.done_selecting")
        
        self.scatter.overlays.append(PeakPickingOverlay(component=self.scatter))
        
        return plot
    
    # TODO: have this update other variables in physiodata: hand_labeled, mea, etc 
    def _delete_peaks(self, peaks_to_delete):
        pass
    
    def _add_peaks(self, peaks_to_add):
        pass
    # End TODO
    
    def _change_peaks(self):
        if self.dirty:
            messagebox("You shouldn't edit peaks until you've run Detect QRS")
        interval = self.peak_editor.selection
        if interval is None: return
        mode = self.peak_editor.selection_purpose
        logger.info("PeakPickingTool entered %s mode over %s", mode, str(interval))
        if mode == "delete":
            # Do it for real peaks
            ok_peaks = np.logical_not(
                np.logical_and(
                self.peak_times > interval[0],
                self.peak_times < interval[1] 
                )
            )
            self.peak_indices = self.peak_indices[ok_peaks]
            self.peak_values = self.peak_values[ok_peaks]
            self.peak_times = self.peak_times[ok_peaks]
            # And do it for 
            ok_dne_peaks = np.logical_not(
                np.logical_and(
                self.dne_peak_times > interval[0],
                self.dne_peak_times < interval[1] 
                )
            )
            self.dne_peak_indices = self.dne_peak_indices[ok_dne_peaks]
            self.dne_peak_values = self.dne_peak_values[ok_dne_peaks]
            self.dne_peak_times = self.dne_peak_times[ok_dne_peaks]
        else:
            # Find the signal contained in the selection
            sig = np.logical_and(self.ecg_time > interval[0],
                                 self.ecg_time < interval[1])
            sig_inds = np.flatnonzero(sig)
            selected_sig = self.ecg_signal[sig_inds]
            # find the peak in the selected region
            peak_ind = sig_inds[0] + np.argmax(selected_sig)

            # If we're in add peak mode, always make sure
            # that only unique peaks get added:
            if mode == "add":
                real_peaks = np.sort(np.unique(
                        self.peak_indices.tolist() + [peak_ind]))
                self.peak_indices = real_peaks
                self.peak_values = self.ecg_signal[real_peaks]
                self.peak_times = self.ecg_time[real_peaks]
            else:
                real_dne_peaks = np.sort(np.unique(
                    self.dne_peak_indices.tolist() + [peak_ind]))
                self.dne_peak_indices = real_dne_peaks
                self.dne_peak_values = self.ecg_signal[real_dne_peaks]
                self.dne_peak_times = self.ecg_time[real_dne_peaks]

        # update the scatterplot if we're interactive
        if not self.plot_data is None:
            self.plot_data.set_data("peak_times",self.peak_times )
            self.plot_data.set_data("peak_values",self.peak_values )
            self.plot_data.set_data("dne_peak_times",self.dne_peak_times )
            self.plot_data.set_data("dne_peak_values",self.dne_peak_values )
            self.image_plot_data.set_data("imagedata",self.ecg_matrix)
            self.image_plot.request_redraw()
        
    @on_trait_change("window_size,start_time")
    def update_plot_range(self):
        self.plot.index_range.high = self.start_time + self.window_size
        self.plot.index_range.low = self.start_time

    @on_trait_change("image_plot_selected")
    def snap_to_image_selection(self):
        if not self.peak_times.size > 0: return
        tmin = self.peak_times[int(self.image_selection_tool.ymin)] -2.
        tmax =  self.peak_times[int(self.image_selection_tool.ymax)] +2.
        logger.info("selection tool sends data to (%.2f, %.2f)",tmin,tmax)
        self.plot.index_range.low = tmin - 2.
        self.plot.index_range.high = tmax + 2.



    def _image_plot_default(self):
        # for image plot
        img = self.ecg_matrix
        if self.ecg_matrix.size == 0:
            img = np.zeros((100,100))
        self.image_plot_data = ArrayPlotData(imagedata=img)
        plot = Plot(self.image_plot_data)
        self.image_selection_tool= ImageRangeSelector(component=plot,
                            tool_mode="range",axis="value", always_on=True)
        plot.img_plot(
            "imagedata", colormap=jet, name="plot1",origin="bottom left")[0]
        plot.overlays.append(self.image_selection_tool)
        return plot

    detection_params_group = VGroup(
        HGroup(Group(
              VGroup(
              Item("use_secondary_heartbeat"),
              Item("peak_window"),
              Item("apply_filter"),
              Item("bandpass_min"),
              Item("bandpass_max"),
              Item("smoothing_window_len"),
              Item("smoothing_window"),
              Item("pt_adjust"),
              Item("apply_diff_sq"),
              label = "Pan Tomkins"),

              VGroup(# Items for MultiSignal detection
              Item("secondary_heartbeat"),
              Item("secondary_heartbeat_pre_msec"),
              Item("secondary_heartbeat_abs"),
              Item("secondary_heartbeat_window"),
              Item("secondary_heartbeat_window_len"),
              label = "MultiSignal Detection",
              enabled_when = "use_secondary_heartbeat"
              ),
              
              VGroup(# Items for two ECG Signals
              Item("use_ECG2"),
              Item("ecg2_weight"),
              label = "Multiple ECG",
              enabled_when = "can_use_ecg2"
              ),

              label="Beat Detection Options",
              layout="tabbed",
              show_border=True,
              springy=True
              ),
              Item("image_plot",editor=ComponentEditor(),width=300),
              show_labels=False),
        Item("b_detect", enabled_when="dirty"),
        Item("b_shape"),
        show_labels = False
        )

    plot_group = Group(
            Group(
                Item("plot",editor=ComponentEditor(),width=800),
                show_labels=False),
            Item("start_time"),
            Item("window_size")
        )
    traits_view = MEAPView(
        VSplit(
            plot_group,
            detection_params_group,
            #orientation="vertical"
            ),
        resizable=True,
        buttons=[OKButton,CancelButton],
        title="Detect Heart Beats"
        )
