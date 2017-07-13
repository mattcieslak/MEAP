#!/usr/bin/env python
from traits.api import (HasTraits,  Array,  File, cached_property,
          Bool, Enum, Instance, on_trait_change, Property,
          DelegatesTo, Int, Button, List, Set )
import os
import joblib

# Needed for Tabular adapter
from traitsui.api import Item,HGroup,VGroup, HSplit
from traitsui.menu import OKButton, CancelButton
from enable.component_editor import ComponentEditor
from chaco.api import Plot, ArrayPlotData, VPlotContainer,jet
import numpy as np
from pyface.api import ProgressDialog

from meap.beat_train import MEABeatTrain
from meap.meap_timeseries import MEAPTimeseries
from meap.beat import GlobalEnsembleAveragedHeartBeat
import time
from meap import MEAPView, ParentButton, messagebox
from meap.io import PhysioData
from meap.classifiers import BPointClassifier
from meap.filters import zero_crossing_index

import logging
logger = logging.getLogger(__name__)


class MovingEnsembler(HasTraits):
    physiodata = Instance(PhysioData)
    bpoint_classifier=Instance(BPointClassifier)
    
    # weighting traits
    mea_window_type = DelegatesTo("physiodata")
    mea_func_name = DelegatesTo("physiodata")
    mea_window_secs = DelegatesTo("physiodata")
    mea_exp_power = DelegatesTo("physiodata")
    mea_weight_direction= DelegatesTo("physiodata")
    mea_func_name = DelegatesTo("physiodata")
    mea_n_neighbors = DelegatesTo("physiodata")
    mea_weights = DelegatesTo("physiodata")
    mea_smooth_hr = DelegatesTo("physiodata")
    dirty = Bool(False,desc=("When a weighting parameter is changed"
        " but the new weighting hasn't been applied to the mea beats")    
    )
    b_apply_weighting = Button(label="Apply Moving Weighting")
    
    never = Bool(False)

    # Traits for the imageplot widget
    content_list = Property(List)
    def _get_content_list(self):
        return sorted(self.physiodata.contents)
    image_plot_signal = Enum(values = "content_list")
    image_plot_content = Enum("Original", "Moving Ensembled", "Residuals")
    image_plot = Instance(Plot,transient=True)
    image_plotdata = Instance(ArrayPlotData,transient=True)

    def _image_plot_content_changed(self):
        self.update_image_plot()

    def _image_plot_signal_changed(self):
        self.update_image_plot()

    # Bpoint classifier
    b_train = Button(label="Train Classifier")
    b_test_params = Button(label="Classifier Grid Search")
    b_test_performance = Button(label="Calculate Performance")
    b_save = Button(label="Save classifier")
    n_samples = Int(100)
    selected_beat_indices = Array
    selected_beats = Instance(MEABeatTrain)
    interactive = Bool(False)
    b_apply_clf = Button(label="Apply b-point classifier",transient=True)
    b_mark_it_zero = Button(label="Mark b at dZ/dt=0",transient=True)
    b_clear_custom_markings = Button(label="Clear custom annotations",transient=True)

    bpoint_classifier_file = DelegatesTo("physiodata")

    mea_beat_train = Instance(MEABeatTrain)
    edit_listening = Bool(False,desc="If true, update_plots is called"
            " when a beat gets hand labeled")

    # graphics items
    plot = Instance(VPlotContainer,transient=True)
    physio_signals_to_plot = Set(["hr","map","pep","sv","tpr","co"])
    physio_signals = List(Instance(MEAPTimeseries))    
    
    # tools for editing point markings
    b_train_clf = Button(label="Create training set",transient=True)
    saved_since_train = Bool(False)
    
    automark = Bool(True) # For debugging
    
    def __init__(self,**traits):
        super(MovingEnsembler,self).__init__(**traits)

        # Set the initial path to whatever's in the physiodata file
        logger.info("Initializing moving ensembler")
        # If we have to create new matrices due to empty or
        # invalid matrices in physiodata
        mat_rows = []
        stored_mea_mats_ok = True
        for signal in self.physiodata.contents:
            if signal == "respiration": continue
            matname = "%s_matrix" % signal
            if not hasattr(self.physiodata, matname):
                needs_empty = True
            else:
                mat_matrix = getattr(self.physiodata, matname)
                mea_matname = "mea_%s_matrix" % signal
                mea_matrix = getattr(self.physiodata, mea_matname)
                needs_empty = False # Does this signal need a new wmpty matrix?
    
                if not mea_matrix.ndim == 2:
                    needs_empty = True
                elif not mat_matrix.shape[0] == mea_matrix.shape[0] or \
                     not mat_matrix.shape[1] == mea_matrix.shape[1]:
                    needs_empty = True

            if needs_empty:
                stored_mea_mats_ok = False
                logger.info("creating empty %s matrix", signal)
                setattr(self.physiodata, mea_matname,
                        np.zeros_like(getattr(self.physiodata, matname)))
            mat_rows.append(getattr(self.physiodata, mea_matname).shape[0])
            mat_rows.append(getattr(self.physiodata, matname).shape[0])

        if not np.all(self.physiodata.hr.shape[0] == np.array(mat_rows)):
            logger.info("HR array mismatch with data matrices, fixing...")
            #self.mea_beat_train.hr = np.zeros_like(self.physiodata.peak_times)
            stored_mea_mats_ok = False
            
        if not stored_mea_mats_ok:
            if self.automark:
                self.apply_weighting("auto init")
            else:
                logger.info("Not automatically marking points")
        else:
            self.dirty = False
            
        self._init_bpoint_clf_name()
        
    @on_trait_change("bpoint_classifier_file")
    def _file_updated(self):
        logger.info("Checking for new bpoint classifier %s", self.bpoint_classifier_file)
        if os.path.exists(self.bpoint_classifier_file):
            self.bpoint_classifier = self._bpoint_classifier_default()
                
        
    def _init_bpoint_clf_name(self):
        """Loops over various possible directories to find where to write
        the bpoint_classifier file"""
        def ok_write(fname):
            dirname, basename = os.path.split(fname)
            if not os.path.exists(dirname):
                return False
            
            if basename.endswith("mea.mat"):
                _fname = basename[:-7] + "bpoint_classifier"
            elif basename.endswith("acq") or basename.endswith("mat"):
                _fname = basename[:-3] + "bpoint_classifier"
            else:
                _fname = basename + ".bpoint_classifier"
                
            return os.path.join(dirname,_fname)
        
        if hasattr(self.physiodata,"file_location") and ok_write(self.physiodata.file_location):
            self.bpoint_classifier_file = ok_write(self.physiodata.file_location)
        elif hasattr(self.physiodata,"original_file") and ok_write(self.physiodata.original_file):
            self.bpoint_classifier_file = ok_write(self.physiodata.original_file)
        else:
            self.bpoint_classifier_file = os.path.join(os.getcwd(),"meap.bpoint_classifier")
        
    def __get_matplot_data(self):
        signal = self.image_plot_signal
        mat = getattr(self.physiodata, signal + "_matrix")
        if self.image_plot_content == "Original":
            return mat
        else:
            mea_mat = getattr(self.physiodata, "mea_" +signal + "_matrix")
            if self.image_plot_content == "Moving Ensembled":
                return mea_mat
            else:
                return mat - mea_mat

    def _image_plot_default(self):
        # for image plot
        self.image_plotdata = ArrayPlotData(imagedata=self.__get_matplot_data())
        plot = Plot(self.image_plotdata)
        plot.img_plot(
            "imagedata", colormap=jet, name="plot1",origin="bottom left")[0]
        return plot

    def update_image_plot(self):
        logger.info("updating image plot") 
        if self.image_plotdata is None: return
        self.image_plotdata.set_data("imagedata", self.__get_matplot_data())
        self.image_plot.request_redraw()
    
    def _time_window_moving_ensemble(self):
        # Parameters for this run
        raw_hr = self.mea_beat_train.get_heartrate()
        peak_times = self.physiodata.peak_times
        window_secs = self.physiodata.mea_window_secs
        weight_function = self.physiodata.mea_func_name
        power = self.physiodata.mea_exp_power
        weighting_direction = self.physiodata.mea_weight_direction

        n_beats = len(peak_times)
        n_beats_averaged = np.zeros(n_beats)
        self.physiodata.mea_hr = np.zeros(n_beats)
        
        # If the window secs is set to 0 use the 
        if window_secs == 0.:
            # Apply the moving ensemble weighting
            for signal in self.physiodata.contents:
                setattr(self.physiodata, "mea_%s_matrix" % signal,
                    getattr(self.physiodata,"%s_matrix" % signal).copy())
            self.physiodata.mea_hr = raw_hr
            logger.info("Using raw beats instead of moving ensembling")
            return

        for beatnum in range(n_beats):
            beat_time = peak_times[beatnum]

            # establish the time window
            if weighting_direction == "before":
                start_time = beat_time -window_secs
                end_time = beat_time + 0.0005
            elif weighting_direction == "after":
                start_time = beat_time -0.0005 # be sure to include this beat
                end_time = beat_time + window_secs
            elif weighting_direction == "symmetric":
                start_time = beat_time -window_secs/2.
                end_time = beat_time + window_secs/2.
            else:
                raise ValueError("Impossible")

            contained_beats = np.flatnonzero( 
                    (peak_times > start_time) & (peak_times < end_time))
            n_beats_averaged[beatnum] = len(contained_beats)
            matrix_slice = slice(contained_beats[0],contained_beats[-1],None)

            # Get the neighbor weighting function
            if weight_function == "flat":
                weights = np.ones(n_beats,dtype=np.float) / n_beats_averaged
            elif weight_function in ("linear", "exponential"):
                seconds_away = np.abs(peak_times[matrix_slice] - beat_time )
                neighbor_nearness = 1 - seconds_away / window_secs
                if weight_function == "exponential":
                    neighbor_nearness = neighbor_nearness ** power
                weights = neighbor_nearness / neighbor_nearness.sum()

            # Apply the moving ensemble weighting
            for signal in self.physiodata.contents:
                getattr(self.physiodata, "mea_%s_matrix" % signal)[beatnum] = \
                    (getattr(self.physiodata,"%s_matrix" % signal)[matrix_slice] * \
                        weights.reshape(-1,1)).sum(0)
            if self.mea_smooth_hr:
                self.physiodata.mea_hr[beatnum] = (raw_hr[matrix_slice] * \
                                                            weights).sum()

    def _n_neighbor_moving_ensemble(self):
        return
        if self.mea_func_name == "linear":
            self.mea_weights = np.arange(self.mea_n_neighbors + 1, dtype=np.float) + 1
        elif self.mea_func_name == "exponential":
            self.mea_weights = np.exp(np.arange(self.mea_n_neighbors+1)+1)
        elif self.mea_func_name == "flat":
            self.mea_weights = np.ones(self.mea_n_neighbors + 1, dtype=np.float) / self.mea_n_neighbors
        # Apply the weighting function to the raw matrices
        if self.mea_n_neighbors == 0:
            logger.info("n_neighbors=0, using raw data")
            self.mea_beat_train.z0_matrix = self.physiodata.z0_matrix
            self.mea_beat_train.ecg_matrix = self.physiodata.ecg_matrix
            self.mea_beat_train.dzdt_matrix = self.physiodata.dzdt_matrix
            if self.physiodata.using_continuous_bp:
                self.mea_beat_train.bp_matrix = self.physiodata.bp_matrix
            else:
                self.mea_beat_train.systolic_matrix = self.physiodata.systolic_matrix
                self.mea_beat_train.diastolic_matrix = self.physiodata.diastolic_matrix
        else:
            logger.info("Moving ensemble averaging beats")
            self.mea_beat_train.z0_matrix = self.weight(
                                                    self.physiodata.z0_matrix)
            self.mea_beat_train.ecg_matrix = self.weight(
                                                    self.physiodata.ecg_matrix)
            self.mea_beat_train.dzdt_matrix = self.weight(
                                                    self.physiodata.dzdt_matrix)
            if self.physiodata.using_continuous_bp:
                self.mea_beat_train.bp_matrix = self.weight(
                                                    self.physiodata.bp_matrix)
            else:
                self.mea_beat_train.systolic_matrix = self.weight(
                                                    self.physiodata.systolic_matrix)
                self.mea_beat_train.diastolic_matrix = self.weight(
                                                    self.physiodata.diastolic_matrix)
    
    def apply_weighting(self,name):
        logger.info( 'meap param "%s" changed' % name)
        t0 = time.time()
        if self.mea_window_type == "Beats":
            self._n_neighbor_moving_ensemble()
        elif self.mea_window_type == "Seconds":
            self._time_window_moving_ensemble()
        logger.info("Updating MEA Beat train beats and plots")
        self.mea_beat_train.update_signals()
        
        # Use the MEA heart beats to calculate physio state
        self.physiodata.hand_labeled = np.zeros_like(self.physiodata.peak_indices)
        self.mea_beat_train.mark_points(show_progressbar=True)
        t1 = time.time()
        logger.info("Marked points for mea_beat_train in %.2f seconds", t1-t0)
        self.calculate_physio()
        self.dirty=False
        
    def _fix_garbled_mea_mats(self):
        if self.mea_window_type == "Beats":
            self._n_neighbor_moving_ensemble()
        elif self.mea_window_type == "Seconds":
            self._time_window_moving_ensemble()

    def _b_apply_weighting_fired(self):
        self.apply_weighting("button_request")
        
    # Functions involving b-point classification
    def _bpoint_classifier_default(self):
        if os.path.exists(self.bpoint_classifier_file):
            logger.info("attempting to load %s", self.bpoint_classifier_file)
            try:
                clf = joblib.load(self.bpoint_classifier_file)
                return BPointClassifier(
                    physiodata=self.physiodata, classifier=clf)
            except Exception, e:
                logger.info("unable to load classifier file")
                logger.info(e)
        logger.info("Loading new bpoint classifier (init)")
        return BPointClassifier(physiodata=self.physiodata)

    def _b_apply_clf_fired(self):
        if not self.bpoint_classifier.trained:
            messagebox("Classifier is not trained yet!")
            return

        progress = ProgressDialog(title="B-Point Classification", min=0,
                max = len(self.physiodata.peak_times), show_time=True,
                message="Classifying...")
        progress.open()
        for i, beat in enumerate(self.mea_beat_train.beats):
            if not beat.hand_labeled: 
                beat.b.set_index(int(self.bpoint_classifier.estimate_bpoint(beat.id)))
            (cont,skip) = progress.update(i)
        (cont,skip) = progress.update(i+1)
        self.calculate_physio()
        
    def _b_mark_it_zero_fired(self):
        progress = ProgressDialog(title="B-Point Classification", min=0,
                max = len(self.physiodata.peak_times), show_time=True,
                message="Classifying...")
        progress.open()
        for i, beat in enumerate(self.mea_beat_train.beats):
            if beat.hand_labeled: continue
            r_ind = self.physiodata.r_indices[i]
            c_ind = self.physiodata.c_indices[i]
            
            beat.b.set_index(r_ind + zero_crossing_index(beat.dzdt_signal[r_ind:c_ind]))
            (cont,skip) = progress.update(i)
        (cont,skip) = progress.update(i+1)
        self.calculate_physio()

    def _b_clear_custom_markings_fired(self):
        for beat in self.mea_beat_train.beats:
            beat.hand_marked = False
        self.update_plots()

    def _global_ensemble_default(self):
        return GlobalEnsembleAveragedHeartBeat(physiodata=self.physiodata)

    def _mea_beat_train_default(self):
        logger.info("creating default mea_beat_train")
        assert self.physiodata is not None
        mbt = MEABeatTrain(physiodata=self.physiodata)
        return mbt

    def _weighting_func_default(self):
        return WeightingFunction(physiodata=self.physiodata)

    def _plot_default(self):
        """
        Instead of defining these in __init__, only
        construct the plots when a ui is requested
        """
        self.interactive = True
        # Create plotting components
        pdata = dict([(sig,getattr(self.mea_beat_train,sig)) for sig in \
                self.physio_signals_to_plot | set(("resp_corrected_sv", 
                    "resp_corrected_co", "resp_corrected_tpr"))])

        self.plotdata = ArrayPlotData(
            peak_times=self.mea_beat_train.peak_times.flatten(),
            beat_type = self.physiodata.hand_labeled,
            mea_hr = self.physiodata.mea_hr, # Plot the mea_hr instead of hr
            **pdata
        )
        
        physio_signals = []
        container = VPlotContainer(
            resizable="hv", bgcolor="lightgray", fill_padding=True, padding=10
        )

        for sig in self.physio_signals_to_plot & self.physiodata.calculable_indexes:
            physio_signals.append(
                MEAPTimeseries( signal=sig, plotdata=self.plotdata
                                )
                )
            container.add(physio_signals[-1].plot)
        container.padding_top =10 
        self.physio_signals = physio_signals

        return container

    def calculate_physio(self):
        logger.info("Calculating physio on mea_beat_train")
        if not self.edit_listening:
            self.edit_listening = True
            self.on_trait_change(self.update_labeled,
                    "mea_beat_train.beats.point_updated")
        self.mea_beat_train.calculate_physio()
        self.update_plots()

    def update_labeled(self):
        self.calculate_physio()

    def update_plots(self):
        if self.interactive:
            # update the meap physio traces            
            self.plotdata.set_data("map",self.physiodata.map)
            self.plotdata.set_data("co",self.physiodata.co)
            self.plotdata.set_data("resp_corrected_co",self.physiodata.resp_corrected_co)
            self.plotdata.set_data("tpr",self.physiodata.tpr)
            self.plotdata.set_data("resp_corrected_tpr",self.physiodata.resp_corrected_tpr)
            self.plotdata.set_data("sv",self.physiodata.sv)
            self.plotdata.set_data("resp_corrected_sv",self.physiodata.resp_corrected_sv)
            self.plotdata.set_data("pep",self.physiodata.pep)
            self.plotdata.set_data("lvet",self.physiodata.lvet)
            self.plotdata.set_data("hr",self.physiodata.hr)
            self.plotdata.set_data("beat_type", self.physiodata.hand_labeled)
            for subplot in self.physio_signals:
                subplot.plot.request_redraw()
            self.update_image_plot()

    @on_trait_change("physio_signals.selected_range")
    def beats_selected(self,obj,name,new):
        if name == "selected_range":
            selection = np.flatnonzero(( self.physiodata.peak_times >= new[0] ) \
                                     & ( self.physiodata.peak_times <= new[1] ))
            logger.info("Beats selected on %s:\n%s", obj.signal, str(selection))
            beat_viewer = MEABeatTrain(physiodata=self.physiodata)
            beats = [self.mea_beat_train.beats[n] for n in selection]
            beat_viewer.set_beats(beats)
            beat_viewer.edit_traits(kind="livemodal")

    def _selected_beats_default(self):
        return MEABeatTrain(physiodata=self.physiodata)

    def _b_train_fired(self):
        self.bpoint_classifier.train()
        self.saved_since_train = False

    @on_trait_change("n_samples")
    def select_new_samples(self):
        nbeats = len(self.mea_beat_train.beats)
        nsamps = min(self.n_samples,nbeats)
        selected_beats = np.random.choice(nbeats,size=nsamps,replace=False)
        self.selected_beats.set_beats([self.mea_beat_train.beats[n] for n in selected_beats])

    
    @on_trait_change( 
       ("physiodata.mea_window_type, physiodata.mea_func_name, "
        "physiodata.mea_window_secs, "
        "physiodata.mea_exp_power, physiodata.mea_weight_direction, "
        "physiodata.mea_func_name, "
        "physiodata.mea_n_neighbors, physiodata.mea_weights, physiodata.mea_smooth_hr")
    )
    def params_edited(self):
        self.dirty = True

    def _b_train_clf_fired(self):
        self.select_new_samples()
        self.edit_traits(view="training_view")

    def _b_test_performance_fired(self):
        self.bpoint_classifier.check_performance()

    def _b_save_fired(self):
        if self.bpoint_classifier.save():
            self.saved_since_train = True
            
        
    diagnostic_plots = VGroup(
        HGroup(
            Item("image_plot_signal", label = "Signal"),
            Item("image_plot_content", label = "Visualize")
        ),
        Item("image_plot",editor=ComponentEditor(size=(100,100)),
                 show_label=False))
    weighting_widgets = VGroup(
        HGroup(Item("mea_window_type"),Item("mea_smooth_hr")),
        HGroup(Item("mea_func_name"),
        Item("mea_n_neighbors",visible_when="mea_window_type=='Beats'"),
        Item("mea_window_secs",visible_when="mea_window_type=='Seconds'")),
        Item("mea_weight_direction"),
        Item("b_apply_weighting",show_label=False, enabled_when="dirty")
        )
    window_widgets = VGroup(diagnostic_plots,
                            weighting_widgets,show_labels=False)
    traits_view = MEAPView(
        HSplit(
            Item("plot",editor=ComponentEditor(),show_label=False),
            VGroup(
                window_widgets,
                Item('bpoint_classifier_file'),
                VGroup(
                     HGroup(Item("b_train_clf",show_label=False),
                            Item("b_apply_clf",show_label=False)),
                     HGroup(Item("b_mark_it_zero",show_label=False),
                           Item("b_clear_custom_markings",show_label=False)),
                      )
                )
        ),
        resizable=True, 
        win_title="Physio Timeseries",
        width=800, height=700,
        buttons = [ParentButton,OKButton,CancelButton]
    )
    

    training_view = MEAPView(
        VGroup(
        HGroup("n_samples", 
            Item("saved_since_train",
                 label="Saved since training?",enabled_when='never'),
            Item("b_save",show_label=False),
            Item("b_train",show_label=False),
            Item("b_test_performance", show_label=False)),
        HGroup(Item("selected_beats",style="custom",show_label=False))),
        resizable=True
    )
