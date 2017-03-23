#!/usr/bin/env python
from traits.api import (HasTraits, Array,
          Bool, Enum, Instance, on_trait_change,Property,
          DelegatesTo, Button, List, cached_property )
from beat import HeartBeat, GlobalEnsembleAveragedHeartBeat
#from meap import TimeSeries
from meap.io import PhysioData

import time
import numpy as np
from meap import MEAPView
# Needed for Tabular adapter
from traitsui.api import ( Item, TableEditor,
        ObjectColumn,HGroup,VGroup)
from traitsui.extras.checkbox_column import CheckboxColumn
from chaco.api import Plot, ArrayPlotData, ScatterInspectorOverlay
from chaco.tools.api import ScatterInspector 

from enable.component_editor import ComponentEditor
from pyface.api import ProgressDialog
from sklearn.decomposition import FastICA


import logging
logger = logging.getLogger(__name__)

beat_table = TableEditor(
    columns =
    [   ObjectColumn(name="id",editable=False),
        CheckboxColumn(name="hand_labeled",editable=False)
    ],
    auto_size  = True,
    show_toolbar = True,
    edit_view="traits_view",
    row_factory=HeartBeat,
    orientation="horizontal",
    selected="selected_beat"
    )

from meap import outlier_feature_function


class BeatTrain(HasTraits):
    # Holds the peak labels for each detected beat
    physiodata = Instance(PhysioData)
    # stacks of waveforms
    z0_matrix = DelegatesTo("physiodata")
    ecg_matrix = DelegatesTo("physiodata")
    dzdt_matrix = DelegatesTo("physiodata")
    bp_matrix = DelegatesTo("physiodata") 
    systolic_matrix = DelegatesTo("physiodata")
    diastolic_matrix = DelegatesTo("physiodata")
    censored_intervals = DelegatesTo("physiodata")
    use_trimmed_co = DelegatesTo("physiodata")
    censored_secs_before = DelegatesTo("physiodata")

    # Physio measures derived by self.beats
    lvet = DelegatesTo("physiodata")
    co = DelegatesTo("physiodata") 
    resp_corrected_co = DelegatesTo("physiodata") 
    pep = DelegatesTo("physiodata") 
    sv = DelegatesTo("physiodata")
    resp_corrected_sv = DelegatesTo("physiodata")
    map = DelegatesTo("physiodata")
    systolic = DelegatesTo("physiodata")
    diastolic = DelegatesTo("physiodata")
    hr = DelegatesTo("physiodata")
    mea_hr = DelegatesTo("physiodata")
    tpr = DelegatesTo("physiodata")
    resp_corrected_tpr = DelegatesTo("physiodata")
    peak_times = DelegatesTo("physiodata")
    peak_indices = DelegatesTo("physiodata")
    global_ensemble_average = Property(Instance(GlobalEnsembleAveragedHeartBeat))
    rr_intervals = Property(Array)
    moving_ensembled = Bool(False)
    
    # The important trait:
    beats = List(Instance(HeartBeat))
    selected_beat = Instance(HeartBeat)
    parameter_plot_data = Instance(ArrayPlotData,transient=True)
    fitted = Bool(False)
    # traits for the summary plots
    plot_contents = Enum("LVET","PEP", "Stroke Volume")
    summary_inspector = Instance(ScatterInspector)
    outlier_plot = Instance(Plot,transient=True)
    outlier_plot_data = Instance(ArrayPlotData,transient=True)
    outlier_inspector = Instance(ScatterInspector)
    b_calculate = Button("Calculate")
    auto_calc_outliers = Bool(True)

    subset = Array
    beat_features = Array
    
    def __init__(self,**traits):
        super(BeatTrain,self).__init__(**traits)
        if self.auto_calc_outliers:
            self.calculate_outliers()  
        # Initialize the heartrate
        self.get_heartrate()
    
    def _get_global_ensemble_average(self):
        return GlobalEnsembleAveragedHeartBeat(physiodata=self.physiodata)

    def _outlier_plot_data_default(self):
        return ArrayPlotData(
            x1=np.array([]),        
            x2=np.array([])
        )
        

    def __len__(self):
        if self.subset.size == 0:
            return self.physiodata.peak_times.shape[0]
        return self.subset.shape[0]
    
    def calculate_outliers(self):
        logger.info("Extracting features for outlier detection")        
        feature_grabber = outlier_feature_function(self.physiodata.contents)
        
        self.beat_features = np.array(
            [feature_grabber(beat) for beat in self.beats if beat.usable]
        )
        usable = np.array([beat.usable for beat in self.beats])
        self.usable_beats = [beat.id for beat in self.beats if beat.usable]
        if self.beat_features.size == 0:
            return
        
        if not self.fitted:
            self.fits = FastICA(n_components=2).fit(self.beat_features)
            self.fitted = True
        
        transform_2d = self.fits.transform(self.beat_features)
        
        if self.outlier_plot is not None:
            x,y = transform_2d.T
            beat_ids = np.array([int(b.id) for b in self.beats])
            usable = self.physiodata.usable[beat_ids] > 0
            self.outlier_plot_data.set_data("x1",x)
            self.outlier_plot_data.set_data("x2",y)
        
    def _outlier_plot_default(self):
        plot = Plot(self.outlier_plot_data, 
                    use_backbuffer=True)
        self.outlier_markers = plot.plot(
                ("x1","x2"),type="scatter",
                name="outlier_plot",
                marker="square",
                color="orange")
        plot.padding=0
        # Taken from online examples
        my_plot = plot.plots["outlier_plot"][0]
        self.outlier_inspector = ScatterInspector(my_plot, selection_mode="toggle",
                                          persistent_hover=False)
        self.outlier_index_data = my_plot.index
        self.outlier_index_data.on_trait_change(self.outlier_plot_item_selected,"metadata_changed")
        my_plot.tools.append(self.outlier_inspector)
        my_plot.overlays.append(
            ScatterInspectorOverlay(my_plot,
                                    hover_color = "transparent",
                                    hover_marker_size = 10,
                                    hover_outline_color = "purple",
                                    hover_line_width = 2,
                                    selection_marker_size = 8,
                                    selection_color = "lightblue")
        )
        return plot
    
    @on_trait_change("beats.point_updated")
    def point_hand_labeled(self):
        self.update_param_plot()
        self.calculate_outliers()
        
    def set_beats(self,beats):
        self.beats = beats
        # The ICA model is no longer fitted to these beats
        self.fitted = False
        if self.outlier_plot is not None:
            self.update_param_plot()
            self.calculate_outliers()
            
    
    def outlier_plot_item_selected(self):
        """a point got clicked in the outlier detector"""
        sel_indices = self.outlier_index_data.metadata.get('selections', [])
        if not len(sel_indices):  return
        index = sel_indices[-1]
        #actual_index = [b.id for b in self.beats].index(index)
        self.selected_beat = self.beats[index]
        
    parameter_plot = Instance(Plot,transient=True)
    parameter_plot_data = Instance(ArrayPlotData)
    def _parameter_plot_data_default(self):
        return ArrayPlotData(
                    beat_id=np.array([b.id for b in self.beats]),        
                    param_value=np.array([]))
    
    def _parameter_plot_default(self):
        self.update_param_plot()
        plot = Plot(self.parameter_plot_data, 
                    use_backbuffer=True,
                    )
        param_markers = plot.plot(
                ("beat_id","param_value"),type="scatter",
                marker="square", name="param_plot",color="blue")
        
        my_plot = plot.plots["param_plot"][0]
        self.parameter_inspector = ScatterInspector(my_plot, selection_mode="toggle",
                                          persistent_hover=False)
        self.param_index_data = my_plot.index
        self.param_index_data.on_trait_change(self.parameter_plot_item_selected,"metadata_changed")
        my_plot.tools.append(self.parameter_inspector)
        my_plot.overlays.append(
            ScatterInspectorOverlay(my_plot,
                                    hover_color = "transparent",
                                    hover_marker_size = 10,
                                    hover_outline_color = "purple",
                                    hover_line_width = 2,
                                    selection_marker_size = 8,
                                    selection_color = "lightblue")
        )
        plot.padding = 30
        return plot
    
    @on_trait_change("plot_contents")
    def update_param_plot(self):
        if self.plot_contents == "LVET":
            grabber_func = lambda x : x.get_lvet()
        elif self.plot_contents == "PEP":
            grabber_func = lambda x : x.get_pep()
        elif self.plot_contents == "Stroke Volume":
            grabber_func = lambda x : x.get_sv()
        if hasattr(self,"parameter_plot_data"):
            self.parameter_plot_data.set_data("beat_id",
                np.array([b.id for b in self.beats if b.usable]))
            self.parameter_plot_data.set_data("param_value",
                np.array([grabber_func(b) for b in self.beats if b.usable]))
        
    def parameter_plot_item_selected(self):
        """a point got clicked in the parameter plotter"""
        sel_indices = self.param_index_data.metadata.get('selections', [])
        if not len(sel_indices):  return
        index = sel_indices[-1]
        self.selected_beat = self.beats[index]
            
    
    @on_trait_change("selected_beat")
    def set_table_index(self):
        selected_id = self.selected_beat.id
        logger.info("Selection changed to event %d", selected_id)
        self.outlier_index_data.metadata['selections'] = []
        self.param_index_data.metadata['selections'] = []
        actual_index = [b.id for b in self.beats].index(self.selected_beat.id)
        self.outlier_inspector._select(actual_index)
        self.parameter_inspector._select(actual_index)
    
    def get_HRV(self):
        return np.nanstd(self.rr_intervals)
        
    def mark_points(self,show_progressbar=False):
        """
        Loops over all the peaks detected by the beat detector. At
        each point, attempt to find a min/max that correspoinds to 
        """
        logger.info("Clearing previous point times")
        point_names = [pt.name for pt in self.beats[0].points]
        for point in point_names:
            setattr(self.physiodata, point+"_indices", 
                    -np.ones(self.physiodata.peak_times.shape,dtype=np.int))

        t0=time.time()
        if show_progressbar:
            progress = ProgressDialog(title="Marking Heartbeats", min=0,
                    max = len(self.peak_times), show_time=True,
                    message="Processing...")
            progress.open()
        # mark each beat
        for i,beat in enumerate(self.beats):
            beat.mark_points(waveform_prior=self.global_ensemble_average)
            if show_progressbar:
                (cont,skip) = progress.update(i)
        # finalize progressbar
        if show_progressbar:
            (cont,skip) = progress.update(i+1)

        t1= time.time()
        logger.info("point classification took %.3f", t1-t0)
        
    def __iter__(self):
        """
        generates a HeartBeat for each of the signals.
        """
        # Create HeartBeat objects for each Beat
        for beat in self.beats:
            yield beat

    def _fix_duplicated_beats_bug(self):
        # Check that no two peaks are too close:
        
        peak_diffs = np.ediff1d(self.physiodata.peak_indices, to_begin=500)
        peak_mask = peak_diffs > 200 # Cutoff is 300BPM
        if not np.any(~peak_mask): return
        logger.info("Dataset affected by 1.0.0b peak duplicate bug")
        self.physiodata.peak_indices = self.physiodata.peak_indices[peak_mask] 
        self.physiodata.peak_times = self.physiodata.peak_times[peak_mask] 
        # The index arrays will be messed up too
        for p in ["p","q","r","s","t","c","x","o","systole","diastole","b"]:
            setattr(self.physiodata, p + "_indices", 
                    getattr(self.physiodata,p+"_indices")[peak_mask])
        
    def _beats_default(self):
        self._fix_duplicated_beats_bug()
        b = []
        if self.physiodata is None: return b
        beat_indices = self.subset if self.subset.size > 0 \
                else np.arange(len(self.physiodata.peak_times))
        return [HeartBeat(id=n, physiodata=self.physiodata) for n in beat_indices]
    
    def get_heartrate(self):
        self.hr = 60/self.rr_intervals
        return self.hr
    
    def _get_rr_intervals(self):
        """
        Calculating RR intervals is complicated in the presence
        of censored intervals.
        """
        peak_times = self.physiodata.peak_times 
        dne_peak_times = self.physiodata.dne_peak_times
        if dne_peak_times.shape == (): 
            dne_peak_times = dne_peak_times.reshape((1,))

        all_peaks = np.unique(np.sort(np.concatenate((peak_times,dne_peak_times))))
        hr = np.ediff1d(all_peaks, to_begin=0)
        hr[0] = hr[1]
        censored_intervals = self.physiodata.censored_intervals
        if len(censored_intervals):
            censor_start, censor_end = censored_intervals.T # unpack columns
            # How long did each censor interval cover?
            censor_durations = censor_end - censor_start

            # How many seconds are censored between beats?
            censored_secs_before = np.zeros(len(all_peaks))

            # If two beats are separated by a censoring interval, 
            # one will precede a censor interval and the next will follow
            for beatnum, (beat1, beat2) in enumerate(zip(all_peaks[:-1], all_peaks[1:])):
                between_beats = \
                    np.logical_and( beat1 < censor_start, beat2 > censor_end )
                # sum of an all-false masked array is zero
                censored_secs_before[beatnum+1] = censor_durations[between_beats].sum()


            # Some rr intervals are separated by a censoring region and are
            # therefore a simple subtraction won't work.  
            if np.any(hr == 0): raise ValueError("Encountered hr == 0")
            invalids = ( censored_secs_before > 0 )
            if np.any(invalids):
                # Interpolate to fill the missing hr values
                valids = np.logical_not(invalids)
                interpolated_hr_vals = np.interp( peak_times[invalids],
                                peak_times[valids],hr[valids])
                hr[invalids] = interpolated_hr_vals
        if len(dne_peak_times):
            hr = np.interp(peak_times,all_peaks,hr)
            
        return hr
                
    def calculate_physio(self):
        """
        Creates arrays of beat properties from all detected beats.
        """
        
        hr = self.mea_hr if self.moving_ensembled else self.get_heartrate()
        if "lvet" in self.physiodata.calculable_indexes:
            self.lvet = np.array([pt.get_lvet() for pt in self.beats])
        if "sv" in self.physiodata.calculable_indexes:
            self.sv = np.array([pt.get_sv(rc=False) for pt in self.beats])
        if "resp_corrected_sv" in self.physiodata.calculable_indexes:
            self.resp_corrected_sv = np.array([pt.get_sv(rc=True) for pt in self.beats])
        if "map" in self.physiodata.calculable_indexes:
            self.map = np.array([pt.get_map() for pt in self.beats])
        if "pep" in self.physiodata.calculable_indexes:
            self.pep = np.array([pt.get_pep() for pt in self.beats])
        if "co" in self.physiodata.calculable_indexes:
            self.co = hr * self.sv / 1000
            normal_co = (self.co - self.co.mean())/ self.co.std()
            if self.use_trimmed_co:
                trimmed_co = self.co.copy()
                trimmed_co[np.abs(normal_co) > 3] = self.co.mean()
                trimmed_co[trimmed_co < 0] = trimmed_co.mean()
                self.co = trimmed_co
        if "resp_corrected_co" in self.physiodata.calculable_indexes:
            self.resp_corrected_co = hr * self.resp_corrected_sv / 1000
            normal_resp_corrected_co = (self.resp_corrected_co - self.resp_corrected_co.mean())/ self.resp_corrected_co.std()
            if self.use_trimmed_co:
                trimmed_resp_corrected_co = self.resp_corrected_co.copy()
                trimmed_resp_corrected_co[np.abs(normal_resp_corrected_co) > 3] = self.resp_corrected_co.mean()
                trimmed_resp_corrected_co[trimmed_resp_corrected_co < 0] = trimmed_resp_corrected_co.mean()
                self.resp_corrected_co = trimmed_resp_corrected_co
        if "tpr" in self.physiodata.calculable_indexes:
            self.tpr = self.map/self.co * 80
        if "resp_corrected_tpr" in self.physiodata.calculable_indexes:
            self.resp_corrected_tpr = self.map/self.resp_corrected_co * 80
            
        
    traits_view = MEAPView(
        HGroup(
            HGroup(
                Item("beats", editor=beat_table, show_label=False)
                ),
            VGroup(
                Item("outlier_plot",editor=ComponentEditor(), width=400, height=400),
                Item("plot_contents"),
                Item("parameter_plot",editor=ComponentEditor(), width=400, height=400),
                show_labels=False
                ),
            show_labels=False
            ),
        resizable=True,
        win_title = "Beat Train"
        )
    
class MEABeatTrain(BeatTrain):
    # stacks of waveforms
    z0_matrix = DelegatesTo("physiodata", "mea_z0_matrix")
    ecg_matrix = DelegatesTo("physiodata", "mea_ecg_matrix")
    dzdt_matrix = DelegatesTo("physiodata", "mea_dzdt_matrix")
    bp_matrix = DelegatesTo("physiodata", "mea_bp_matrix")
    systolic_matrix = DelegatesTo("physiodata","mea_systolic_matrix")
    diastolic_matrix = DelegatesTo("physiodata","mea_diastolic_matrix")
    moving_ensembled = Bool(True)

    def _beats_default(self):
        self._fix_duplicated_beats_bug()
        b = []
        if self.physiodata is None: return b
        beat_indices = self.subset if self.subset.size > 0 \
                else np.arange(len(self.physiodata.peak_times))
        return [HeartBeat(id=n, physiodata=self.physiodata, 
                           moving_ensembled=True) for n in beat_indices]

    def update_signals(self):
        """
        If the mea_``signal``_matrix is changed then each beat needs to 
        get its signals reset to be up to date with the physiodata object
        """
        logger.info("Resetting signal data in MEA beats")
        for beat in self.beats:
            beat._set_default_signals()
            beat.update_plot()
