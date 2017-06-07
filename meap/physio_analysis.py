#!/usr/bin/env python
from traits.api import (HasTraits, Str, Array, Float,CFloat, Dict,
          Bool, Enum, Instance, on_trait_change,File,Property,
          Range, DelegatesTo, Int, Button, List, Color,Set,Event,CBool )
import numpy as np

from traitsui.api import (Group, View, Item, TableEditor, VSplit,
        ObjectColumn, ExpressionColumn,HSplit, EnumEditor, SetEditor,
        Handler)
from traitsui.table_column import ObjectColumn
from meap.beat_train import BeatTrain
from meap.beat import EnsembleAveragedHeartBeat, GlobalEnsembleAveragedHeartBeat
from meap.io import PhysioData, load_from_disk
from meap import messagebox
from numpy import nanmean, nanstd
from collections import defaultdict
from meap import MEAPView
from pyface.api import ProgressDialog
import logging
logger = logging.getLogger(__name__)
import joblib

from meap.classifiers import BPointClassifier
from meap import outlier_feature_function_for_physio_experiment

# Stuff for excel
import os
import os.path as op

from chaco.api import ArrayPlotData, ColorBar, \
                                 ColormappedSelectionOverlay, HPlotContainer, \
                                 jet,gist_rainbow, LinearMapper, Plot, ScatterInspectorOverlay
from chaco.tools.api import PanTool, ZoomTool
from chaco.tools.line_inspector import LineInspector
from chaco.tools.scatter_inspector import ScatterInspector
from chaco.scatterplot import ScatterPlot
from chaco.lineplot import LinePlot

from enable.component_editor import ComponentEditor
from enable.api import ColorTrait
from chaco.api import marker_trait, DataRange1D, gist_rainbow
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA

from traitsui.extras.checkbox_column \
    import CheckboxColumn

from traitsui.color_column \
     import ColorColumn
from enable.api \
     import ColorTrait

import pandas as pd
from sklearn.linear_model import LinearRegression


condition_table = TableEditor(
    columns = [
        #CheckboxColumn(name="plot"),
        ObjectColumn(name="name"),
        ColorColumn(name="color")
    ]
)

index_mapping = {
    "mea_hr":"mea_hr",
    "resp_corrected_co":"mea_co_rc",
    "resp_corrected_sv":"mea_sv_rc",
    "resp_corrected_tpr":"mea_tpr_rc",
    "hr":"raw_hr",
    "tpr":"mea_tpr",
    "sv":"mea_sv",
    "co":"mea_co",
    "pep":"mea_pep",
    "lvet":"mea_lvet"
    
}



class EventType(HasTraits):
    name = Str
    color = ColorTrait("red")
    plot = Bool(False)
    
class Subject(HasTraits):
    name = Str("")
    color = ColorTrait("red")
    plot = Bool(True)
    
    
def discrete_colors(n):
    # generates n rgba tuples from a colormap
    dr1d = DataRange1D(low=0,high=1)
    cmap = gist_rainbow(dr1d)
    colors = cmap.map_uint8(np.linspace(0,1,n))/255.
    return map(tuple,colors)
    
def mea_content_grabber(plotter):
    if plotter.plot_contents == "Pre-Ejection Period" :
        return lambda x: x.pep
    if plotter.plot_contents == "Stroke Volume":
        return lambda x: x.sv
    if plotter.plot_contents == "Cardiac Output":
        return lambda x: x.co
    if plotter.plot_contents == "Total Peripheral Resistance":
        return lambda x: x.tpr
    if plotter.plot_contents == "Heart Rate":
        return lambda x: x.hr
    if plotter.plot_contents == "Mean Arterial Pressure":
        return lambda x: x.map
    if plotter.plot_contents == "Systolic BP":
        return lambda x: x.sbp
    if plotter.plot_contents == "Diastolic BP":
        return lambda x: x.dbp
    
class MEAPlot(HasTraits):
    #event = Instance(ExpEvent)
    container = Instance(HPlotContainer)
    plot_contents = Enum("Pre-Ejection Period", "Stroke Volume",
                         "Cardiac Output", "Total Peripheral Resistance", "Heart Rate",
                         "Mean Arterial Pressure", "Systolic BP", "Diastolic BP" )
    plot = Instance(Plot)
    def _plot_default(self):
        x,y = self.get_xy()
        self.plot_data = ArrayPlotData(
            time=x,        
            value=y,
        )
        plot = Plot(self.plot_data)
        plot.plot(("time","value"))
        plot.padding = 30
        return plot

    def get_xy(self):    
        if self.plot_contents == "Stroke Volume":
            grabber_func = lambda x : x.sv
        elif self.plot_contents == "Pre-Ejection Period":
            grabber_func = lambda x : x.pep
        elif self.plot_contents == "Cardiac Output":
            grabber_func = lambda x : x.co
            
        return self.event.moving_ensemble_times, grabber_func(self.event)
    
    @on_trait_change("plot,plot_contents,plot_grouping")
    def update_plot_contents(self):
        if self.plot is None:
            return
        
        x,y = self.get_xy()
        
        self.plot_data.set_data("time",x)
        self.plot_data.set_data("value",y)

    plot_group =  Group(
                    Item("plot_contents"),
                    Item("plot", editor=ComponentEditor(),
                         width=600,height=400),
                    show_labels=False
               )
    traits_view = View(plot_group)
    
class GroupMEAPlot(HasTraits):
    #analysis = Instance(PhysioExperiment)
    plot_contents = Enum("Pre-Ejection Period", "Stroke Volume",
                         "Cardiac Output", "Total Peripheral Resistance", "Heart Rate",
                         "Mean Arterial Pressure", "Systolic BP", "Diastolic BP" )
    conditions = DelegatesTo("analysis")
    
    plot = Instance(Plot)
    
    plotdata = Instance(ArrayPlotData)
    plot_grouping = Enum("Group Result", "Line per-subject" )
    subjects = Property(List(Instance(Subject)))
    b_summary = Button(label="Summarize")
    merge_all_conditions = Bool(False)
    def _get_subjects(self):
        subjs = [Subject(name=s) for s in self.analysis.subjects]
        n_subjs = len(subjs)
        subj_colors = discrete_colors(n_subjs)
        for subj,color  in zip(subjs,subj_colors): subj.color = color
        return subjs
    
    def _b_summary_fired(self):
        self.update_plot_content()
        
    def _plot_default(self):
        self.plotdata = ArrayPlotData()
        plot = Plot(self.plotdata)
        return plot
    
    def clear_plot(self):
        for k in self.plotdata.arrays.keys():
            self.plotdata.del_data(k)
            
        for p in self.plot.plots.keys():
            self.plot.delplot(p)
    
    def update_plot_content(self):
        self.clear_plot()
        grabber = mea_content_grabber(self)
        
        # Which subjects will get plotted?
        selected_subjects = set([subj.name for subj in self.subjects if subj.plot])
        if not len(selected_subjects): return
        
        # Which conditions should get plotted? 
        if self.merge_all_conditions:
            selected_conditions = set([cond.name for cond in self.conditions])
        else:
            selected_conditions = set([cond.name for cond in self.conditions if cond.plot])
        if not len(selected_conditions): return
            
        # Where should the color come from?
        if self.plot_grouping == "Group Result":
            color_source = self.conditions
            evt_attr = "condition"
        elif self.plot_grouping == "Line per-subject":
            color_source = self.subjects
            evt_attr = "subject"
            
        def get_color(event):
            this_event = [x for x in color_source if x.name == getattr(event,evt_attr) ]
            assert len(this_event) == 1
            this_event = this_event[0]
            return this_event.color
            
        #Loop over every event. Determine whether it should get plotted  
        # and what its color should be
        for event in self.analysis.events:
            if not event.subject in selected_subjects: continue
            if not event.condition in selected_conditions: continue
            x = event.moving_ensemble_times
            y = grabber(event)
            x_key = 't.%d' % event.event_id
            y_key = 'y.%d' % event.event_id
            self.plotdata.set_data(x_key, x)
            self.plotdata.set_data(y_key, y)
            self.plot.plot((x_key, y_key), color=get_color(event))            
            
        
            
    def plot_group_results(self):
        grabber = mea_content_grabber(self)
        selected_subjects = set([subj for subj in self.subjects if subj.plot])
        if not len(selected_subjects): return
        collected = defaultdict(list)
        for event in self.analysis.events:
            collected[event.condition] .append(
                (event.moving_ensemble_times, grabber(event)))
        
            
    analysis_option_group = Group("plot_contents",
                          "merge_all_conditions", "plot_grouping",
                          Item("b_summary",show_label=False))
    
    cond_and_subj_picker = Group(
                    Item("conditions", editor=condition_table, show_label=False),
                    Item("subjects", editor=condition_table, show_label=False),
                    orientation = "horizontal")
    
    plot_options_group = Group(
        analysis_option_group, cond_and_subj_picker,orientation='horizontal',show_labels=False)
        

    plot_group =  VSplit(
                    plot_options_group,
                    Item("plot", editor=ComponentEditor(),
                         width=600,height=400),
                    orientation="vertical",
                    show_labels=False)
    traits_view = View(plot_group)

class ExpEvent(HasTraits):
    onset = Float    # Seconds
    duration = Float # Seconds
    condition = EventType
    beat_indices = Property(Array)
    event_id = Int
    ea_hr = Float
    ea_tpr = Float
    ea_svk = Float
    ea_sv = Float
    ea_svsb = Float
    ea_co = Float
    ea_pep = Float
    ea_map = Float
    ea_hrv = Float
    ea_lvet = Float
    ea_sbp = Float
    ea_dbp = Float
    ea_nbreaths = Float
    ea_z0 = Float
    ea_dZdt_max = Float
    failed_marking = Bool(False)
    hand_labeled = CBool(False)
    physio_indexes = set(["hand_labeled", "usable"])
    ensemble_average_points = ["p_time","q_time","r_time","s_time",
                "t_time","c_time","x_time","o_time","systole_time",
                "diastole_time","b_time"]
    header = List
    ensemble_averaged_heartbeat = Instance(EnsembleAveragedHeartBeat)
    usable = CBool(True)
    physiodata = Instance(PhysioData)
    viable_time_interval = Property(Float)
    
    moving_ensemble_times = Array
    mea_hr = Array 
    mea_tpr = Array
    mea_sv = Array
    mea_co = Array
    mea_pep = Array
    mea_map = Array
    mea_lvet = Array
    mea_dbp = Array
    mea_sbp = Array
    
    def __init__(self,**traits):
        super(ExpEvent,self).__init__(**traits)
        self.custom_times = {}        
        for k,v in traits.iteritems():
            if k.endswith("_time"):
                self.custom_times[k] = v
        
    @on_trait_change("ensemble_averaged_heartbeat.point_updated")
    def _hb_changed(self,obj,name,new):
        self.hand_labeled = True
        self.calculate_physio()
        logger.info("Updating Beat features")
        #self.parent.parent.calculate_outliers()
        self.parent.parent.update_param_plot()
    
    def _get_viable_time_interval(self):
        # figure out the viable time before calculating HR
        viable_time_interval = censored_leftovers(
                            self.physiodata.censored_intervals,
                            self.onset, self.onset + self.duration
                            )
        if viable_time_interval < 1: return -1
        return viable_time_interval
    
    def get_hrv(self):
        """
        Calculates the SDSD of heartbeats. Censored intervals 
        are accounted for.
        """
        if self.viable_time_interval < 5:
            return -1
        peak_times = self.physiodata.peak_times[self.beat_indices]
        ci = self.physiodata.censored_intervals
        if ci.size == 0 or np.all(peak_times < ci.min()) or np.all(peak_times > ci.max()):
            # If there's no overlap don't even bother checking            
            ok_diffs = np.diff(peak_times)
        else:
            # Check to see which overlaps
            pt = np.vstack([peak_times[:-1],peak_times[1:]]).T
            bad_diffs = np.zeros(len(pt))
            if ci.ndim == 1: ci = ci.reshape(-1,2)
            for _ci in ci:
                bad_diffs = bad_diffs + \
                  np.logical_and(pt[:,0] > _ci[0], pt[:,1] < _ci[1])
            ok_diffs = np.diff(peak_times)[np.logical_not(bad_diffs)]
        hrv = np.std(ok_diffs)
        logger.info("using %d beats of %d, HRV=%.2f"%(
            len(ok_diffs), len(peak_times)-1, hrv))
        return hrv
        
    def _get_beat_indices(self):
        return np.flatnonzero(
            (self.parent.btrain.peak_times > self.onset) & \
            (self.parent.btrain.peak_times < (self.onset + self.duration))
            )
    
    def get_datadict(self,other_attrs=[]):
        out = {}
        
        # Extract the physio measurements
        for measure in self.physio_indexes:
            out[measure] =  getattr(self, measure,np.nan)
            
        # Extract point times
        for pt in ExpEvent.ensemble_average_points:
            pointname = pt.split("_")[0]
            try: 
                point = getattr(self.ensemble_averaged_heartbeat,pointname)
                out[pt] = point.time
            except Exception,e:
                out[pt] = -99
                
        # Other attributes from usr interaction
        out['usable'] = int(self.usable)
        out['hand_labeled'] = int(self.hand_labeled)
        
        # extract custom attruibutes
        for attr in other_attrs:
            if attr in out:
                logger.warn("%s conflicts with a default MEAP output column: ignoring",attr)
                continue
            out[attr] = getattr(self,attr)
                
        return out
    
        
    def _ensemble_averaged_heartbeat_default(self):
        if self.parent is None or self.parent.physiodata is None:
            return
        # Extract the beats that occur during the event
        ea = EnsembleAveragedHeartBeat(
            physiodata = self.parent.physiodata,
            subset = self.beat_indices,
            global_ensemble=False,
            marking_strategy="custom peaks",
        )
        return ea
    
    def ensemble_average(self,bpoint_classifier=None):
        logger.info("event %d: %d beats over %.2f seconds. %.2f seconds usable",
                    self.event_id,len(self.beat_indices),self.duration,self.viable_time_interval)
        # Account for any of the event time that may have been censored
        # TODO: add do-not-ensemble peaks here
        self.ea_hr = 60.*len(self.beat_indices)/self.viable_time_interval
        # attach the results
        self.ea_hrv = self.get_hrv()
        
        # TODO: insert bpoint classifier here
        # It should also go in the datadict
        try:
            self.ensemble_averaged_heartbeat.mark_points(
                waveform_prior=self.parent.global_ensemble)
        except Exception, e:
            logger.warn(e)
        if self.hand_labeled:
            for point, point_time in self.custom_times.iteritems():
                try:
                    getattr(self.ensemble_averaged_heartbeat,
                        point.split("_")[0]).set_time(point_time)
                except AttributeError:
                    continue
        if bpoint_classifier is not None:
            logger.info("Using bpoint classifier")
            self.ensemble_averaged_heartbeat.b.set_index(
                        bpoint_classifier.estimate_bpoint(
                        beat_obj=self.ensemble_averaged_heartbeat))
            
        self.calculate_physio()
        self.sync_trait("usable", self.ensemble_averaged_heartbeat)
        self.failed_marking = self.ensemble_averaged_heartbeat.failed_marking
        self.sync_trait("failed_marking",self.ensemble_averaged_heartbeat)
            
    def get_nbreaths(self):
        #if self.physiodata.respiration_cycle.size == 0 or \
        #   self.physiodata.respiration_amount.size ==0:
        return -1
        #epoch_indices = np.logical_and(self.physiodata.respiration_ts.time > self.onset,
        #    self.physiodata.respiration_ts.time < (self.onset + self.duration ))
        #return self.physiodata.respiration_amount[epoch_indices].sum()
    
    def calculate_physio(self):
        
        if "map" in self.physiodata.calculable_indexes:
            self.ea_dbp = self.ensemble_averaged_heartbeat.get_diastolic_bp()
            self.ea_sbp = self.ensemble_averaged_heartbeat.get_systolic_bp()
            self.ea_map = self.ensemble_averaged_heartbeat.get_map() 
            self.physio_indexes.update(["ea_dbp","ea_sbp","ea_map"])
        if "sv" in self.physiodata.calculable_indexes:
            self.ea_sv = self.ensemble_averaged_heartbeat.get_sv()
            self.ea_svk = self.ensemble_averaged_heartbeat._get_kubicek_sv()
            self.ea_svsb = self.ensemble_averaged_heartbeat._get_sramek_bernstein_sv()
            self.ea_z0 = self.ensemble_averaged_heartbeat.base_impedance
            self.physio_indexes.update(["ea_sv","ea_svk","ea_svsb","ea_z0"])
        if "pep" in self.physiodata.calculable_indexes:
            self.ea_pep = self.ensemble_averaged_heartbeat.get_pep()
            self.ea_lvet = self.ensemble_averaged_heartbeat.get_lvet()
            self.ea_eef = self.ensemble_averaged_heartbeat.get_eef()
            self.ea_dZdt_max = self.ensemble_averaged_heartbeat.dZdt_max
            self.physio_indexes.update(["ea_pep","ea_lvet","ea_eef","ea_dZdt_max"])
        if "co" in self.physiodata.calculable_indexes:
            self.ea_co = self.ea_hr * self.ea_sv / 1000
            self.physio_indexes.update(["ea_co"])
        if "tpr" in self.physiodata.calculable_indexes:
            if self.ea_co == 0:
                self.ea_tpr = 0
            else:
                self.ea_tpr = self.ea_map/self.ea_co * 80
            self.physio_indexes.update(["ea_tpr"])
        
        if "resp_corrected_sv" in self.physiodata.calculable_indexes:
            self.ea_sv_rc = self.ensemble_averaged_heartbeat.get_sv(rc=True)
            self.ea_svk_rc = self.ensemble_averaged_heartbeat._get_kubicek_sv(rc=True)
            self.ea_svsb_rc = self.ensemble_averaged_heartbeat._get_sramek_bernstein_sv(rc=True)
            self.ea_z0_rc = self.ensemble_averaged_heartbeat.resp_corrected_base_impedance
            self.physio_indexes.update(["ea_sv_rc","ea_svk_rc","ea_svsb_rc","ea_z0_rc"])
        if "resp_corrected_co" in self.physiodata.calculable_indexes:
            self.ea_co_rc = self.ea_hr * self.ea_sv_rc / 1000
            self.physio_indexes.update(["ea_co_rc"])
        if "resp_corrected_tpr" in self.physiodata.calculable_indexes:
            if self.ea_co_rc == 0:
                self.ea_tpr_rc = 0
            else:
                self.ea_tpr_rc = self.ea_map/self.ea_co_rc * 80
            self.physio_indexes.update(["ea_tpr_rc"])
        
        # Respiration? Not for now.
        self.ea_nbreaths = self.get_nbreaths()
        
        # Calculate the moving ensemble stuff if it's there
        if not self.parent.mea_computed: return
        data_begin = self.onset 
        data_end = self.onset + self.duration
        me_beats = np.flatnonzero(
                           ( self.parent.physiodata.peak_times > data_begin ) & \
                           ( self.parent.physiodata.peak_times < data_end ))
        # Get slopes and intercepts from each epoch
        times = self.physiodata.peak_times[me_beats]
        times = (times - times.mean()).squeeze()
        if len(times) < 8: return
        for physio_name, stats_name in index_mapping.iteritems():
            try:
                data = getattr(self.physiodata,physio_name)[me_beats]
            except Exception:
                continue
            # Fit a simple linear model and add the output
            linreg = LinearRegression(fit_intercept=True)
            linreg.fit(times.reshape(-1,1),data)
            error = linreg.score(times.reshape(-1,1),data)
            setattr(self,stats_name+"_slope", linreg.coef_[0])
            setattr(self,stats_name+"_error", error)
            setattr(self,stats_name+"_intercept",linreg.intercept_)
            self.physio_indexes.update([stats_name+"_error",
                    stats_name+"_slope",stats_name+"_intercept"])


    traits_view = MEAPView(
        Group(
            Item("ensemble_averaged_heartbeat",
                 style="custom",
                 show_label=False)
        )
    )
MEAPlot.add_class_trait("event", Instance(ExpEvent))
    
    
class SpreadsheetColumn(ObjectColumn):
    def get_cell_color(self, object):
        if not object.usable:
            return "red"
        if object.hand_labeled:
            return "gray"
        return "white"
    
class MarkingWarningColumn(ObjectColumn):
    def get_cell_color(self, object):
        if object.failed_marking:
            return "magenta"
        return "white"
    
class OutlierColumn(ObjectColumn):
    def get_cell_color(self, object):
        if not object.usable:
            return "red"
        if object.hand_labeled:
            return "gray"
        return "white"
    
    
events_table = TableEditor(
    columns= \
    [
        SpreadsheetColumn(name="subject",label="Subject", editable=False),
        MarkingWarningColumn(name="event_id", format="%d",editable=False,width=1),
        SpreadsheetColumn(name="condition",label="Condition", editable=False),
        SpreadsheetColumn(name="onset", editable=False),
        SpreadsheetColumn(name="duration", editable=False),
    ],
    other_columns = \
    [
        SpreadsheetColumn(name="ea_hr",label="HR",width=2,format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_pep",label="PEP",width=2,format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_lvet",label="LVET",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_tpr",label="TPR",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_map",label="MAP",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_hrv",label="HRV",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_co",label="CO",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_sbp",label="SBP",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_dbp",label="DBP",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_svk",label="SVk",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_svsb",label="SVsb",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_z0",label="z0",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_dZdt_max",label="max(dZdt)",format="%.3f", editable=False),
        SpreadsheetColumn(name="ea_nbreaths",label="Breaths",format="%.3f", editable=False),
    ],
    deletable=False,
    sort_model=False,
    auto_size=True,
    orientation="vertical",
    edit_view="ExpEvent",
    selected="selected_event"

)
    

def censored_leftovers(censor_intervals, start_time, end_time):
    # GUARANTEE THAT `censor_intervals` is non overlapping
    whole_time = end_time - start_time
    for censor_start, censor_end in censor_intervals.reshape(-1,2):
        if censor_end < start_time: continue
        if censor_start > end_time: continue
        
        start_overlap = max(start_time,censor_start)
        end_overlap = min(end_time, censor_end)
        
        censored_duration = end_overlap - start_overlap
        whole_time -= censored_duration
    
    return whole_time

class PhysioFileAnalysis(HasTraits):
    # Holds the data from
    physiodata = Instance(PhysioData)
    # Holds the unique eveny types for this experiment
    conditions = List(Instance(EventType))
    # Holds the individual events
    events = List(Instance(ExpEvent))
    interactive = Bool(True)
    btrain = Instance(BeatTrain)
    infile = File
    global_ensemble = Instance(GlobalEnsembleAveragedHeartBeat)
    mea_computed = Property(Bool)
    
    def __init__(self,**traits):
        super(PhysioFileAnalysis,self).__init__(**traits)
        
    def get_bpoint_classifier(self):
        def can_read(fname):
            dirname, basename = os.path.split(fname)
            if not os.path.exists(dirname):
                return 
            
            if basename.endswith("mea.mat"):
                _fname = basename[:-7] + "bpoint_classifier"
            elif basename.endswith("acq") or basename.endswith("mat"):
                _fname = basename[:-3] + "bpoint_classifier"
            else:
                _fname = basename + ".bpoint_classifier"
            
            clf_fname = os.path.join(dirname,_fname)
            if not op.exists(clf_fname): return
            
            try:
                clf = joblib.load(clf_fname)
                return clf
            except Exception:
                return None
            
        clf = None
        # Try the explicitly saved file
        if self.physiodata.bpoint_classifier_file:
            clf = can_read(self.physiodata.bpoint_classifier_file)
        
        # Try one in the same directory
        if clf is None and hasattr(self.physiodata,"file_location"):
            clf = can_read(self.physiodata.file_location)
        # Try original file
        if clf is None and hasattr(self.physiodata,"original_file"):
            clf = can_read(self.physiodata.original_file)
        # Return None 
        if clf is None: return
        return BPointClassifier(physiodata=self.physiodata, classifier=clf)
            
    
    
    def _global_ensemble_default(self):
        return GlobalEnsembleAveragedHeartBeat(physiodata=self.physiodata)
    
    def _btrain_default(self):
        return BeatTrain(physiodata=self.physiodata,
                         auto_calc_outliers=False)
    
    def _get_mea_computed(self):
        if self.physiodata is None:
            return False
        return self.physiodata.mea_hr.size > 0
        
    def _physiodata_default(self):
        pd = load_from_disk(self.infile)
        for ev in self.events:
            ev.physiodata = pd
        return pd
    
    def ensemble_average(self):
        """
        Computes cv properties from epochs of data 
        """
        beats = self.btrain
        bpoint_classifier = self.get_bpoint_classifier()
        # For each event
        for ev_num, event in enumerate(self.events):
            event.ensemble_average(bpoint_classifier=bpoint_classifier)
            
ExpEvent.add_class_trait("parent",Instance(PhysioFileAnalysis))
        


class PhysioExperiment(HasTraits):
    # Holds the set of multiple events spanning multiple files
    input_path = File
    # output excel file
    output_xls = File
    # Which columns contain numeric data?
    floats = Set(["onset","duration"])
    # Holds the unique eveny types for this experiment
    conditions = List(Instance(EventType))
    # Holds the individual events
    events = List(Instance(ExpEvent))
    events_processed=Bool(False)
    selected_event=Instance(ExpEvent)
    usable_events = List([])
    # Holds the individual acq files
    physio_files = List(Instance(PhysioFileAnalysis))
    # Interactively process the data?
    interactive = Bool(False)
    header = List
    b_run = Button(label="Run!")
    b_save = Button(label="Save Results")
    b_run_mea = Button(label="Run!")
    export_wide = Bool(False)
    
    # traits for the summary plots
    plot_grouping = Enum("Subject", "Data File", "All")
    plt_subject = Str("")
    subjects = List([""])
    plt_data_file = Str("")
    data_files = List([""])
    plot_contents = Enum("LVET","PEP", "B-point")
    summary_inspector = Instance(ScatterInspector)
    
    # Features to use for the outlier analysis
    beat_features = Array
    outlier_plot = Instance(Plot,transient=True)
    outpier_plot_data = Instance(ArrayPlotData,transient=True)
    parameter_plot = Instance(Plot,transient=True)
    parameter_plot_data = Instance(ArrayPlotData,transient=True)
    fitted = Bool(False)
    subject_colors = Array
    outlier_inspector = Instance(ScatterInspector)
    
    def _b_save_fired(self):
        self.save_ensemble_averages()
    
    def _b_run_fired(self):
        self.run()
        #self.on_trait_event(self._hb_changed,
        #        "events.ensemble_averaged_heartbeat.point_edited")
    
    @on_trait_change("input_path")
    def open_xls(self,fpath=""):
        if not fpath:
            fpath = self.input_path
        # Open the file
        if not op.exists(fpath): return
        try:
            xls = pd.read_excel(fpath)
        except Exception, e:
            messagebox("Could not read excel file %s" % fpath) 
            
        logger.info("Loading " + fpath)
        available_columns = xls.columns
        lower_columns = [colname.lower() for colname in available_columns]
        required_columns = ['subject', 'session', 'condition', 'duration', 'onset', 'file']
        to_rename = {}
        for column in required_columns:
            if not column in lower_columns:
                messagebox("Experiment spreadsheet must have a %s column" % column)
                #return
            orig_index = lower_columns.index(column)
            if not available_columns[orig_index] == column:
                to_rename[available_columns[index]] = column
        # Rename any columns if they are capitalized
        if len(to_rename):
            xls.rename(columns=to_rename, inplace=True)
        xls['subject'] = xls['subject'].astype(str)
        xls['onset'] = xls['onset'].astype(np.float)
        xls['duration'] = xls['duration'].astype(np.float)
        
        events = []
        for rnum,row in xls.iterrows():
            if not op.exists(row.file):
                logger.info("ignoring event %d, file missing %s" % (rnum,row.file))
                continue
            events.append( ExpEvent( **row.to_dict() ) )
            
        if len(events) == 0:
            messagebox("No events with existing mea.mat files found.\nCheck your spreadsheet")
            return
                
        self.events = events
        self.header = map(str,xls.columns)
        self._process_events()
        
        
    def _process_events(self):
        """
        Loads each unique mea.mat file and attaches its events to it.
        """
        # Find all the unique event types, color code them
        conds = sorted(set([e.condition for e in self.events]))
        cond_colors = discrete_colors(len(conds))
        for c,col in zip(conds,cond_colors):
            self.conditions.append(
                EventType(name=c, color=col)
            )
        cond_colors = discrete_colors(len(self.conditions))
        
        # Find all of the unique acq files that exist
        acq_files = sorted(list(set([e.file for e in self.events])))
        self.data_files = [os.path.basename(a) for a in acq_files]
        self.physio_files = []
        ev_num = 0
        subjects = set()
        
        for i, acq_file in enumerate(acq_files):
            logger.info("Loading %s", acq_file)
            events=[e for e in self.events if e.file == acq_file]
            self.physio_files.append(
                PhysioFileAnalysis(
                    events=events,
                    infile=acq_file,
                    interactive=self.interactive,
                    parent=self
                    )
                )
            for event in self.physio_files[-1].events:
                event.event_id = ev_num
                event.parent = self.physio_files[-1]
                ev_num += 1
            subjects.update([event.subject])
        
        self.usable_events = [evt.event_id for evt in self.events if evt.usable]
        self.subjects = sorted(list(subjects))
        self.events_processed = True
        color_lut = dict([(c,i) for i,c in enumerate(self.subjects)])
        self.subject_colors = np.array([color_lut[e.subject] for e in self.events])
            
            
    def run(self):
        # read the excel file
        #self.open_xls()
        # organize all the data from it
        if not self.events_processed:
            self._process_events()
        # Run the pipeline on each acq_file
        self.ensemble_average()

            
    def ensemble_average(self):
        """
        Iterate over physio files, calculate ensemble averages per 
        event.
        """
        progress = ProgressDialog(title="Processing Events", min=0,
                max = len(self.physio_files), show_time=True,
                message="Calculating...")
        progress.open()
        # Loop over physio files
        for i, pf in enumerate(self.physio_files):
            progress.message = op.split(pf.infile)[-1]
            (cont,skip) = progress.update(i)
            # Process them            
            pf.ensemble_average()
        (cont,skip) = progress.update(i+1)
        
        self.calculate_outliers()
            
    def calculate_outliers(self):
        feature_grabber = outlier_feature_function_for_physio_experiment(self)
        # Update the ICA plot and outlier est.
        logger.info("Extracting features for outlier detection")        
        self.beat_features = np.array(
            [feature_grabber(event) for event in self.events if event.usable]
        )
        usable = np.array([evt.ensemble_averaged_heartbeat.usable for evt in self.events])
        self.usable_events = [evt.event_id for evt in self.events if evt.usable]
        
        if self.beat_features.size == 0:
            return
        if not self.fitted:
            logger.info("Initializing FastICA 2D mapping")
            self.fits = FastICA(n_components=2).fit(self.beat_features)
            self.fitted = True
        
        transform_2d = self.fits.transform(self.beat_features)
        dists = np.sqrt((transform_2d**2).sum(1))
        for evt,dist in zip(self.events,dists):
            evt.oddity = dist
            
        self.oddities=dists
        if self.outlier_plot is not None:
            x,y = transform_2d.T
            self.outlier_plot_data.set_data("x1",x)
            self.outlier_plot_data.set_data("x2",y)
            self.outlier_plot_data.set_data("oddity",self.subject_colors)
            
    
    @on_trait_change("selected_event")
    def set_table_index(self):
        selected_id = self.selected_event.event_id
        logger.info("Selection changed to event %d", selected_id)
        self.outlier_index_data.metadata['selections'] = []
        self.param_index_data.metadata['selections'] = []
        if not selected_id in self.usable_events:
            logger.info("Event %d is not usable, therefore not plotted",
                        selected_id)
            return
        actual_index = self.usable_events.index(selected_id)
        self.outlier_inspector._select(actual_index)
        self.parameter_inspector._select(actual_index)
            
    def outlier_plot_item_selected(self):
        """a point got clicked in the outlier detector"""
        sel_indices = self.outlier_index_data.metadata.get('selections', [])
        if not len(sel_indices):  return
        index = sel_indices[-1]
        actual_id = self.usable_events.index(index)
        logger.info("Outlier plot point %d selected, mapping to event %d",
                    index, actual_id)
        # Is is the event we're already editing?        
        if self.selected_event.event_id == actual_id:
            logger.info("Selected already-editing event")
            return
        self.selected_event = self.events[actual_id]

        
    def _outlier_plot_default(self):
        self.outlier_plot_data = ArrayPlotData(
            x1=np.array([]),        
            x2=np.array([]),
            oddity=np.array([])
        )
        plot = Plot(self.outlier_plot_data, 
                    use_backbuffer=True)
        self.outlier_markers = plot.plot(
                ("x1","x2","oddity"),type="cmap_scatter",
                name="outlier_plot",
                marker="square",
                color_mapper=gist_rainbow)
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
    
    def parameter_plot_item_selected(self):
        """a point got clicked in the parameter plotter"""
        sel_indices = self.param_index_data.metadata.get('selections', [])
        if not len(sel_indices):  return
        index = sel_indices[-1]
        self.param_index_data.metadata['selections'] = [index]
        self.selected_event = self.events[index]
    
    def _parameter_plot_default(self):
        self.parameter_plot_data = ArrayPlotData(
            event_id=np.array([]),        
            param_value=np.array([]),
            point_color=np.array([])
        )
        plot = Plot(self.parameter_plot_data, 
                    use_backbuffer=True,
                    )
        param_markers = plot.plot(
                ("event_id","param_value","point_color"),type="cmap_scatter",
                marker="square", name="param_plot",
                color_mapper=gist_rainbow)
        
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
    
    @on_trait_change("plt_subject,plt_data_file,plot_contents,plot_grouping")
    def update_param_plot(self):
        if self.outlier_plot is None:
            return
        if self.plot_contents == "LVET":
            grabber_func = lambda x : x.ea_lvet
        elif self.plot_contents == "PEP":
            grabber_func = lambda x : x.ea_pep
        elif self.plot_contents == "B-point":
            grabber_func = lambda x : x.ensemble_averaged_heartbeat.b.time

        if self.plot_grouping == "Data File":
            lookup_func = lambda x : os.path.basename(x.file) == self.plt_data_file
        elif self.plot_grouping == "Subject":
            lookup_func = lambda x : x.subject == self.plt_subject
        elif self.plot_grouping == "All":
            lookup_func = lambda x: True
        
        events = []
        for event in self.events:
            if lookup_func(event):
                events.append(event)
        self.parameter_plot_data.set_data("event_id",np.array(
            [e.event_id for e in events]))
        self.parameter_plot_data.set_data("param_value",
            np.array([grabber_func(e) for e in events]))
        
        # If there's more than 1 subject use that as the color
        if self.plot_grouping == "All":
            color_lut = dict([(c,i) for i,c in enumerate(self.subjects)])
            self.parameter_plot_data.set_data("point_color",
                np.array([color_lut[e.subject] for e in events]))
        elif self.plot_grouping == "Subject":
            color_lut = dict([(c,i) for i,c in enumerate( \
                set([e.file for e in events]) ) ])
            self.parameter_plot_data.set_data("point_color",
                np.array([color_lut[e.file] for e in events]))
        else:
            self.parameter_plot_data.set_data("point_color",
                np.array([e.oddity for e in events]))
        
    def export_wide_format(self,df):
        data = df        
        output_path = ".".join(self.output_xls.split(".")[:-1])+".wide_format.xlsx"
        id_col = df.condition        
        if "trialnum" in df.columns:
            id_col = id_col + "." + df.trialnum
        if "timepoint" in df.columns:
            id_col = id_col + "." + df.timepoint
        data['condition'] = id_col
        m_data =pd.melt(data,id_vars=['subject','condition'])
        m_data['all'] = m_data.condition + "." + m_data.variable
        m_data.drop("condition",axis=1,inplace=True)
        m_data.drop("variable",axis=1,inplace=True)
        dupes = m_data.duplicated()
        if np.any(dupes):
            logger.warning("Dupliucates found in data, Check your results!!!")
            m_data = m_data.drop_duplicates(["subject","all"])
        final = m_data.pivot("subject","all","value")
        logger.info("saving to %s", output_path)
        final.to_excel(output_path,index=False)
        logger.info("Done.")
    
    def save_ensemble_averages(self,output_path=""):
        """
        uses pandas to write out an excel file
        """
        #self.ensemble_average()
        if not output_path:
            output_path = self.output_xls
        df = pd.DataFrame(
            [evt.get_datadict(other_attrs=self.header) for evt in self.events])
        df.to_excel(output_path,index=False)
        if self.export_wide:
            self.export_wide_format(df)
    
    # Ensemble average widgets
    ensemble_average_group =    HSplit(
            Group(
                Group(
                    Item("input_path",label="Input XLS"), 
                    Item("output_xls",label="Output XLS"),
                    Item("b_run", show_label=False),
                    Item("b_save", show_label=False),
                    Item("export_wide"),
                    orientation="horizontal"),
            Item(
                 "events",
                 show_label=False,
                 editor=events_table
                 ),
            ),
          Group(
              Item("outlier_plot", editor=ComponentEditor(),
                         width=400,height=400),
              Group(
                  Group(
                      Item("plot_contents"),Item("plot_grouping"), 
                      Item("plt_subject",visible_when="plot_grouping=='Subject'",
                           editor=EnumEditor(name="subjects")),
                      Item("plt_data_file",visible_when="plot_grouping=='Data File'",
                           editor=EnumEditor(name="data_files")),
                      show_labels=False,orientation="horizontal"
                    ),
                    Item("parameter_plot", editor=ComponentEditor(),
                         width=400,height=400),
                    show_labels=False
               ),
                    show_labels=False
                   ),
          show_labels=False
          )
    
    traits_view = MEAPView(
        Group(ensemble_average_group,show_labels=False),
          resizable=True,
          win_title="Physio Analysis"
    )

    

PhysioFileAnalysis.add_class_trait("parent",Instance(PhysioExperiment))
GroupMEAPlot.add_class_trait("analysis",Instance(PhysioExperiment))