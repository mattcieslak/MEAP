#!/usr/bin/env python
from traits.api import (HasTraits, Str, Array, Float,
          Bool, Enum, Instance, on_trait_change,File,Property,
          Range, DelegatesTo, Int, Button, List, Any)

from meap.filters import bandpass, smooth
from meap import fail, __version__, ParentButton
from meap.io import PhysioData, peak_stack
from meap.timeseries import TimeSeries
from skimage.filters import threshold_otsu

import time
import numpy as np

# Needed for Tabular adapter
from traitsui.api import (Group, View, Item, TableEditor,
        ObjectColumn,VSplit, RangeEditor)
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from traitsui.menu import OKButton, CancelButton
#from mayavi.core.ui.api import SceneEditor
#from mayavi.tools.mlab_scene_model import MlabSceneModel
from chaco.api import Plot, ArrayPlotData, VPlotContainer,Plot
from chaco.tools.api import PanTool, ZoomTool
from chaco.tools.line_inspector import LineInspector
from chaco.scatterplot import ScatterPlot
from chaco.lineplot import LinePlot
import numpy as np

from enable.component_editor import ComponentEditor
from enable.api import ColorTrait
from chaco.api import marker_trait
from scipy.stats import nanmean, nanstd

# For the 3d peak window
from mayavi import mlab
from mayavi.sources.api import ArraySource
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.filters.api import WarpScalar, PolyDataNormals
from mayavi.modules.surface import Surface
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk
from meap.peak_picker import PeakPickingTool, PeakPickingOverlay
from meap.filters import censor_peak_times, min_separated_times
from meap import MEAPView


class MRI_ECG_Detector(HasTraits):
    # Holds the data Source
    physiodata = Instance(PhysioData) 
    ecg_ts = DelegatesTo("physiodata")
    ecg_pre_peak = DelegatesTo("physiodata")
    ecg_post_peak = DelegatesTo("physiodata")
    censored_regions = DelegatesTo("physiodata")
    
    # For visualization
    plot = Instance(Plot,transient=True)
    plot_data = Instance(ArrayPlotData,transient=True)
    peak_vis = Instance(ScatterPlot,transient=True)
    scatter = Instance(ScatterPlot,transient=True)
    ecg_trace = Instance(LinePlot,transient=True)
    filt_trace =Instance(LinePlot,transient=True) 
    diffsq_trace =Instance(LinePlot,transient=True) 
    smooth_ma_trace =Instance(LinePlot,transient=True) 
    threshold_trace =Instance(LinePlot,transient=True) 
    start_time = Range(low=0,high=10000.0, initial=0.0)
    window_size = Range(low=1.0,high=300.0,initial=30.0)
    # tool for editing the r peaks
    peak_editor = Instance(PeakPickingTool,transient=True)
    
    
    
    surf_source = Instance(Source,transient=True)
    beat_surf = Instance(PipelineBase,transient=True)
    scene3d = Instance(MlabSceneModel,transient=True)


    # parameters for processing the raw data before PT detecting
    mhd_bandpass_min = DelegatesTo("physiodata")
    mhd_bandpass_max = DelegatesTo("physiodata")
    mhd_smoothing_window_len = DelegatesTo("physiodata")
    mhd_smoothing_window =DelegatesTo("physiodata") 
    combined_smoothing_window_len =DelegatesTo("physiodata") 
    combined_smoothing_window =DelegatesTo("physiodata") 
    weight_magnetohydrodynamics =DelegatesTo("physiodata") 
    qrs_to_mhd_ratio =DelegatesTo("physiodata") 
    
    bandpass_min =DelegatesTo("physiodata")
    bandpass_max =DelegatesTo("physiodata") 
    smoothing_window_len =DelegatesTo("physiodata") 
    smoothing_window =DelegatesTo("physiodata") 
    pt_adjust = DelegatesTo("physiodata")
    peak_threshold = DelegatesTo("physiodata")
    apply_filter = DelegatesTo("physiodata")
    apply_diff_sq = DelegatesTo("physiodata")
    apply_smooth_ma = DelegatesTo("physiodata")
    peak_window = DelegatesTo("physiodata")

    # Filtering operation produces a new TimeSeries
    filtered_ecg = Instance(TimeSeries)
    diff_sq = Instance(TimeSeries)

    # The results of the peak search
    peak_indices = DelegatesTo("physiodata")
    peak_times = DelegatesTo("physiodata")
    
    # Holds the timeseries for the signal vs noise thresholds
    thr_times = Array
    thr_vals = Array

    # for pulling out data for aggregating ecg_ts, dzdt_ts and bp beats
    ecg_matrix = DelegatesTo("physiodata")


    # UI elements
    b_detect = Button(label="Detect QRS",transient=True)

    def __init__(self,**traits):
        super(MRI_ECG_Detector,self).__init__(**traits)
        self.detect()
        # Set one point at the beginning and one at the end
        self.thr_times = self.ecg_ts.time[np.array([0,-1])]
        self.on_trait_change(self.detect,
            "bandpass_min,bandpass_max,smoothing_window_len,peak_window," + \
            "pt_adjust,apply_filt,apply_diff_sq,smoothing_window," + \
            "mhd_bandpass_min,mhd_bandpass_max,mhd_smoothing_window_len," + \
            "qrs_to_mhd_ratio,combined_smoothing_window_len" )
        
    def _scene3d_default(self):
        return MlabSceneModel()
    
    def _beat_surf_default(self):
        return mlab.surf(self.ecg_matrix,warp_scale="auto")
    
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        self.scene3d.mlab.view(40, 50)
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
        
    def detect(self):
        """
        Implementation of the Pan Tomkins QRS detector
        """
        t0 = time.time()
        included_indices = censor_peak_times(self.censored_regions,self.ecg_ts.time)
        # The bandpass filter is intended to reduce noise
        # from electrical outlets, muscle noise, T wave
        # interference and
        if self.apply_filter:
            # There need to be different filters depending on the sampling rate
            if self.ecg_ts.sampling_rate < 2000.:
                print "Using classic filter mode for sampling rate",self.ecg_ts.sampling_rate
                self.mhd_filtered_ecg = self.ecg_ts.new_from_me(
                        newdata=bandpass(self.ecg_ts.data, self.mhd_bandpass_min, 
                                         self.mhd_bandpass_max,
                        self.ecg_ts.sampling_rate)
                )
                self.filtered_ecg = self.ecg_ts.new_from_me(
                        newdata=bandpass(self.ecg_ts.data, self.bandpass_min, self.bandpass_max,
                        self.ecg_ts.sampling_rate)
                )
                
            self.filtered_ecg.data = self.filtered_ecg.data/self.filtered_ecg.data.max()
            self.mhd_filtered_ecg.data = self.mhd_filtered_ecg.data/self.mhd_filtered_ecg.data.max()
        else:
            self.filtered_ecg = self.ecg_ts
            self.mhd_filtered_ecg = self.ecg_ts

        # Differentiate and square the signal?
        if self.apply_diff_sq:
            # Differentiate the signal and square it
            self.diff_sq  = self.filtered_ecg.new_from_me(
                 np.ediff1d(self.filtered_ecg.data,
                    to_begin = 0)**2
            )
            self.diff_sq.data = self.diff_sq.data/self.diff_sq.data.max()
            # Differentiate the signal and square it
            self.mhd_diff_sq  = self.mhd_filtered_ecg.new_from_me(
                 np.ediff1d(self.mhd_filtered_ecg.data,
                    to_begin = 0)**2
            )
            self.mhd_diff_sq.data = self.mhd_diff_sq.data/self.mhd_diff_sq.data.max()
            self.diff_sq.data = self.diff_sq.data/self.diff_sq.data.max()
        else:
            self.diff_sq = self.filtered_ecg
            self.mhd_diff_sq = self.mhd_filtered_ecg

        # If we're to apply a Moving Average smoothing
        if self.apply_smooth_ma:
            # MA smoothing
            self.smooth_ma = self.diff_sq.new_from_me(
               smooth(
                 self.diff_sq.data,
                 window_len=self.smoothing_window_len,
                 window=self.smoothing_window
                 )
            )
            self.mhd_smooth_ma = self.mhd_diff_sq.new_from_me(
               smooth(
                 self.mhd_diff_sq.data,
                 window_len=self.mhd_smoothing_window_len,
                 window=self.mhd_smoothing_window
                 )
            )
            self.smooth_ma.data = self.smooth_ma.data/self.smooth_ma.data.max()
            self.mhd_smooth_ma.data = self.mhd_smooth_ma.data/self.mhd_smooth_ma.data.max()

        else:
            self.smooth_ma = self.diff_sq
            self.mhd_smooth_ma = self.mhd_diff_sq
            
        # Use the MHD low-freq deflection to find peaks?
        if self.weight_magnetohydrodynamics:
            combined = self.smooth_ma.data*self.qrs_to_mhd_ratio + \
                 self.mhd_smooth_ma.data *(1-self.qrs_to_mhd_ratio)

            if self.combined_smoothing_window > 0:
                combined = smooth( combined, 
                                   window_len=self.combined_smoothing_window_len,
                                   window=self.combined_smoothing_window)
                
            self.mhd_qrs = self.smooth_ma.new_from_me(
                            combined
                            )
        else:
            self.mhd_qrs = self.smooth_ma
                
        
        smoothdiff = np.diff(self.mhd_qrs.data)
        peaks = np.flatnonzero(( smoothdiff[:-1] > 0 ) & (smoothdiff[1:] < 0 ))
        
        print "ignoring", len(self.censored_regions), "censored regions"
        peaks = peaks[censor_peak_times(self.censored_regions, self.ecg_ts.time[peaks])]        
        peaks = peaks[min_separated_times(self.ecg_ts.time[peaks], 
                        self.ecg_post_peak / self.ecg_ts.sampling_rate )]
        
        peak_amps = self.mhd_qrs.data[peaks]
        peak_amp_thr = threshold_otsu(peak_amps)+self.pt_adjust
        real_peaks = peaks[ peak_amps > peak_amp_thr ]
        
        # Stack the peaks and see if the original data has a higher value
        raw_stack = peak_stack(real_peaks, self.ecg_ts.data, 
                      pre_msec=self.peak_window, post_msec=1,
                      sampling_rate=self.ecg_ts.sampling_rate)
        adj_factors = np.argmax(raw_stack,axis=1) - self.peak_window
        real_peaks = real_peaks + adj_factors
        
        self.peak_indices = real_peaks
        self.peak_values = self.ecg_ts.data[real_peaks]
        self.peak_times = self.ecg_ts.time[real_peaks]
        self.thr_vals = np.array([peak_amp_thr]*2)
        t1 = time.time()
        # update the scatterplot if we're interactive
        if not self.plot_data is None:
            self.plot_data.set_data("peak_times",self.peak_times )
            self.plot_data.set_data("peak_values",self.peak_values )
            self.plot_data.set_data("smooth_ma",self.smooth_ma.data)
            self.plot_data.set_data("mhd_smooth_ma",self.mhd_smooth_ma.data)
            self.plot_data.set_data("mhd_qrs",self.mhd_qrs.data)
            self.plot_data.set_data("thr_vals",self.thr_vals)
        else:
            print "plot data is none"
        #self.extract_beats()
        
        # if we're interactive, 3d plot the beats
        if not self.surf_source is None:
            #self.surf_source = Source
            self.beat_surf.mlab_source.reset(self.ecg_matrix)
            
        print "found", len(self.peak_indices), "QRS complexes in %.3f seconds" % (t1 - t0)
    
        
    
    def _plot_default(self):
        """
        Creates a plot of the ecg_ts data and the signals derived during
        the Pan Tomkins algorithm
        """
        # Create plotting components
        self.plot_data = ArrayPlotData(
            time=self.ecg_ts.time,
            raw_data=self.ecg_ts.data,
            smooth_ma = self.smooth_ma.data,
            mhd_smooth_ma = self.mhd_smooth_ma.data,
            mhd_qrs = self.mhd_qrs.data,
            peak_times=self.peak_times,
            peak_values=self.peak_values,
            thr_times = self.thr_times,
            thr_vals= self.thr_vals
            
        )
        plot = Plot(self.plot_data, use_backbuffer=True)
        self.ecg_trace = plot.plot(("time","raw_data"),color="green")[0]
        self.smooth_ma_trace = plot.plot(("time","smooth_ma"),color="red")[0]
        self.mhd_smooth_ma_trace = plot.plot(("time","mhd_smooth_ma"),color="magenta")[0]
        self.mhd_qrs_trace = plot.plot(("time","mhd_qrs"),color="blue")[0]
        
        self.threshold_trace = plot.plot(("thr_times","thr_vals"),
                            color="black", line_width=3)[0]

        # Make a scatter plot where you can edit the peaks
        self.peak_vis = plot.plot(("peak_times","peak_values"),type="scatter")[0]
        self.scatter = plot.components[-1]
        self.peak_editor = PeakPickingTool(self.scatter)
        self.scatter.tools.append(self.peak_editor)
        # when the user adds or removes a point, automatically extract 
        self.on_trait_event(self._change_peaks,"peak_editor.done_selecting")
        
        self.scatter.overlays.append(PeakPickingOverlay(component=self.scatter))
        # We're interactive so activate the 3d window
        self.scene3d
        self.beat_surf
        return plot
    
    def _change_peaks(self):
        interval = self.peak_editor.selection
        mode = self.peak_editor.selection_purpose
        if mode == "delete":
            print "delete mode over", interval
            ok_peaks = np.logical_not(
                np.logical_and(
                self.peak_times > interval[0],
                self.peak_times < interval[1] 
                )
            )
            self.peak_indices = self.peak_indices[ok_peaks]
            self.peak_values = self.peak_values[ok_peaks]
            self.peak_times = self.peak_times[ok_peaks]
            
        else:
            print "add mode"
            # Find the signal contained in the selection
            sig = np.logical_and(self.ecg_ts.time > interval[0],
                                 self.ecg_ts.time < interval[1])
            sig_inds = np.flatnonzero(sig)
            selected_sig = self.ecg_ts.data[sig_inds]
            # find the peak in the selected region
            peak_ind = sig_inds[0] + np.argmax(selected_sig)
            real_peaks = np.array(
                sorted(self.peak_indices.tolist() + [peak_ind]))
            self.peak_indices = real_peaks
            self.peak_values = self.ecg_ts.data[real_peaks]
            self.peak_times = self.ecg_ts.time[real_peaks]
        # update the scatterplot if we're interactive
        if not self.plot_data is None:
            self.plot_data.set_data("peak_times",self.peak_times )
            self.plot_data.set_data("peak_values",self.peak_values )
        
    
    @on_trait_change("window_size,start_time")
    def update_plot_range(self):
        self.plot.index_range.high = self.start_time + self.window_size
        self.plot.index_range.low = self.start_time

    detection_params_group = Group(
        Group(
              Item("peak_window"),
              Item("apply_filter"),
              Item("bandpass_min"),#editor=RangeEditor(enter_set=True)),
              Item("bandpass_max"),#editor=RangeEditor(enter_set=True)),
              Item("mhd_bandpass_min"),#editor=RangeEditor(enter_set=True)),
              Item("mhd_bandpass_max"),#editor=RangeEditor(enter_set=True)),
              Item("smoothing_window_len"),#editor=RangeEditor(enter_set=True)),
              Item("smoothing_window"),
              Item("mhd_smoothing_window_len"),#editor=RangeEditor(enter_set=True)),
              Item("mhd_smoothing_window"),
              Item("qrs_to_mhd_ratio"),
              Item("combined_smoothing_window_len"),
              Item("pt_adjust"),#editor=RangeEditor(enter_set=True)),
              Item("apply_diff_sq"),
              Item("weight_magnetohydrodynamics"),
              label="P/T processing options",
              show_border=True,
              orientation="vertical",
              springy=True
              ),
        Group(
              Item("scene3d",
                   editor=SceneEditor(scene_class=Scene),
                   height=200,width=200
                   ),
              show_labels=False,
              springy=True
            ),
        orientation="horizontal")

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
        buttons=[ParentButton,OKButton,CancelButton],
        win_title="Detect Heart Beats"
        )
