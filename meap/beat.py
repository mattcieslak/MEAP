#!/usr/bin/env python
from traits.api import (HasTraits, Str, Array, Float,
          Bool, CBool, Enum, Instance, Property,
          DelegatesTo, Int, Button, List, Event,
          Either,CInt)
from meap.filters import smooth, normalize, template_to_target_dtw, dtw_mapping
from meap import BLOOD_RESISTIVITY, MEAPView, ENSEMBLE_SIGNALS,colors
from meap.timeseries import TimePoint
from meap.io import PhysioData
from meap.point_marker2 import PointDraggingTool, BTool, BMarker
from pyface.api import ProgressDialog
import numpy as np
from traitsui.api import ( Item, VGroup, HGroup, Group,
     TableEditor, HSplit, ObjectColumn )
from traitsui.menu import OKButton, CancelButton

from chaco.api import Plot, ArrayPlotData, HPlotContainer, VPlotContainer
from enable.component_editor import ComponentEditor
from enable.api import Component
from chaco.api import  DataLabel, ScatterPlot, ArrayDataSource

import logging
logger=logging.getLogger(__name__)

class TimepointColumn(ObjectColumn):
    def get_cell_color(self, object):
        if object.needs_attention:
            return "red"
        return "white"

plt_offsets = {
        "ecg": 0.0,
        "dzdt":1.1,
        "sbp":2.2,
        "dbp":2.2,
        "bp":2.2,
        "z0":3.3
        }

plt_scalars = {
        "ecg": 1.,
        "dzdt":1,
        "sbp":0.5,
        "dbp":0.5,
        "bp":0.5,
        "z0":0.5
        }
            

heuristic_table = TableEditor(
    columns =
    [ TimepointColumn(name="name",editable=False),
      TimepointColumn(name="time",editable=True),
      TimepointColumn(name="point_type",editable=True)
    ],
    auto_size=True
)


class HeartBeat(HasTraits):
    physiodata=Instance(PhysioData)
    id=Either(None,CInt)
    # Signals from which to derive points
    dzdt_signal = Array
    resp_corrected_dzdt_signal = Array
    dzdt_time = Array
    z0_signal = Array
    resp_corrected_z0_signal = Array
    z0_time = Array
    ecg_signal = Array
    ecg_time = Array
    bp_signal  = Array
    bp_time  = Array
    systolic_signal  = Array
    systolic_time  = Array
    diastolic_signal  = Array
    diastolic_time  = Array
    base_impedance = Property(Float)
    dZdt_max = Property(Float)
    resp_corrected_base_impedance = Property(Float)
    resp_corrected_dZdt_max = Property(Float)
    marking_strategy = Enum("custom peaks", "heuristic","dtw")
    points = List(Instance(TimePoint))
    # Important points on the ecg
    p = Instance(TimePoint)
    p_label = Instance(DataLabel,transient=True)
    q = Instance(TimePoint)
    q_label = Instance(DataLabel,transient=True)
    r = Instance(TimePoint)
    r_label = Instance(DataLabel,transient=True)
    s = Instance(TimePoint)
    s_label = Instance(DataLabel,transient=True)
    t = Instance(TimePoint)
    t_label = Instance(DataLabel,transient=True)
    # Important points on the icg
    b = Instance(TimePoint)
    b_label = Instance(DataLabel,transient=True)
    c = Instance(TimePoint)
    c_label = Instance(DataLabel,transient=True)
    x = Instance(TimePoint)
    x_label = Instance(DataLabel,transient=True)
    o = Instance(TimePoint)
    o_label = Instance(DataLabel,transient=True)
    # Blood pressure points
    systole = Instance(TimePoint)
    systole_label = Instance(DataLabel,transient=True)
    diastole = Instance(TimePoint)
    diastole_label = Instance(DataLabel,transient=True)
    usable = CBool(True)
    moving_ensembled = CBool(False)
    
    # When in the experiment did the beat happen?  
    start_time = Float

    # arrays holding traces for the plot and their points
    plt_ecg = Array
    plt_dzdt =Array
    plt_bp =Array
    plt_systolic = Array
    plt_diastolic = Array
    #point_times = Array
    #point_values = Array
    #plt_point_values = Array

    # gets computed
    oddity_index = Int(0)

    # Overlap plotting
    plot_data = Instance( ArrayPlotData, transient=True)
    plot = Instance(Plot,transient=True)
    component = Instance(Component,transient=True)
    scatter = Instance(ScatterPlot,transient=True)
    point_picker = Instance(PointDraggingTool,transient=True)

    # dZ/dt plot items
    dzdt_plotdata = Instance( ArrayPlotData, transient=True)
    dzdt_plot = Instance(HPlotContainer,transient=True)
    dzdt_component = Instance(Component,transient=True)
    dzdt_point_picker = Instance(PointDraggingTool,transient=True)


    # Point classifier params
    apply_ecg_smoothing = Bool(True)
    ecg_smoothing_window_len = Int(40)
    apply_dzdt_smoothing = Bool(True)
    dzdt_smoothing_window_len = Int(40)
    apply_bp_smoothing = Bool(True)
    bp_smoothing_window_len = Int(80)
    failed_marking = Bool(False)

    point_vis = Instance(ScatterPlot,transient=True)
    title = Str("Heartbeat")
    point_updated = Event
    
    # Did a human being label this point?
    hand_labeled = CBool(False)

    # Traits from ICG editor
    btool_t = Float(0.)
    btool_t_selection = Float(0.)


    def __init__(self,**traits):
        super(HeartBeat,self).__init__(**traits)
        """
        If a non-negative id is passed, signals will automatically be collected
        from the physiodata object. Otherwise each signal must be passed 
        explicitly during construction (eg ``dzdt_signal = some_array``)

        Times are automatically generated based on the config options in 
        physiodata.  If an array is passed explicitly that doesn't match
        the physiodata settings an exception is raised.
        """
        # If an id is passed, get the signals available for that beat
        self.plotdata = None
        self._set_default_signals()
        self._set_default_times()
        self._set_points()

    def _set_default_signals(self):
        if self.id < 0: return
        mat_prefix= "mea_" if self.moving_ensembled else ""
        if self.id is not None:
            for signal in self.physiodata.contents & ENSEMBLE_SIGNALS:
                if signal in ("ecg", "ecg2"):
                    if not signal == self.physiodata.qrs_source_signal: continue
                if signal == self.physiodata.qrs_source_signal:
                    self.ecg_signal = \
                        getattr(self.physiodata,mat_prefix + signal+"_matrix")[self.id]
                else:
                    setattr(self, signal + "_signal", 
                        getattr(self.physiodata,mat_prefix + signal+"_matrix")[self.id])

    def _set_default_times(self):
        # Set the default times to be consistent with physiodata
        for signal in self.physiodata.contents:
            if signal == "respiration": continue
            if signal.startswith("ecg"):
                _signal = "ecg" 
            else:
                _signal = signal if not signal in \
                    ("z0","resp_corrected_z0","resp_corrected_dzdt") else "dzdt" 
            pre_msec = getattr(self.physiodata, _signal + "_pre_peak")
            sig = getattr(self, _signal + "_signal")
            setattr(self, signal + "_time", np.arange(sig.shape[0])-pre_msec)

    def _set_points(self):
        # Create points from each signal
        pts = self._default_physio_timepoints()
        for pname, pobj in pts.iteritems():
            setattr(self,pname,pobj)
        points = []
        for p in ["p","q","r","s","t","c","x","o","systole","diastole","b"]:
            pobj =  getattr(self,p)
            if pobj is not None:
                points.append(pobj) 
        self.points=points

    def _default_physio_timepoints(self):
        points = {}
        if "ecg" in self.physiodata.contents:
            # Important points on the ecg
            points.update({
             "p": TimePoint(name="p", applies_to="ecg",point_type="max",beat=self),
             "q": TimePoint(name="q", applies_to="ecg",point_type="min",beat=self),
             "r": TimePoint(name="r", applies_to="ecg",point_type="max",beat=self),
             "s": TimePoint(name="s", applies_to="ecg",point_type="min",beat=self),
             "t": TimePoint(name="t", applies_to="ecg",point_type="max",beat=self)})
        if "dzdt" in self.physiodata.contents:
            points.update({
             # Important points on the icg
             "b": TimePoint(name="b", applies_to="dzdt",point_type="geom_trick",beat=self),
             "c": TimePoint(name="c", applies_to="dzdt",point_type="max",beat=self),
             "x": TimePoint(name="x", applies_to="dzdt",point_type="min",beat=self),
             #"y": TimePoint(name="y", applies_to="dzdt",point_type="min",beat=self),
             "o": TimePoint(name="o", applies_to="dzdt",point_type="max",beat=self)
             })
        if "bp" in self.physiodata.contents:
            points["systole"]  = TimePoint(
                  name="systole", applies_to="bp",point_type="max",beat=self)
            points["diastole"] = TimePoint(
                  name="diastole", applies_to="bp",point_type="min",beat=self)
        if "sbp" in self.physiodata.contents and "dbp" in self.physiodata.contents:
            points["systole"] = TimePoint(name="systole", 
                  applies_to="systolic", point_type="average",beat=self)
            points["diastole"] = TimePoint(name="diastole", 
                  applies_to="diastolic", point_type="average",beat=self)
        return points

    def update_plot(self):
        if self.plotdata is None: return
        for signal in ("dzdt", "ecg", "z0"):
            if not signal in self.physiodata.contents: continue
            scalar,offset = plt_scalars[signal], plt_offsets[signal]
            setattr(self, "plt_" + signal, 
                  offset + scalar * normalize(getattr(self, signal+"_signal")))
            self.plotdata[signal+"_data"] = getattr(self,"plt_"+signal)
        self.plot.request_redraw()

    def _usable_changed(self):
        if self.id is not None and self.id > -1:
            self.physiodata.usable[self.id] = int(self.usable)
        
    def _get_base_impedance(self):
        if self.b.index >= self.x.index:
            self.usable = False
            return np.nan
        sig = self.physiodata.mea_z0_matrix[self.id] if self.moving_ensembled \
                else self.z0_signal
        return sig[self.b.index:self.x.index].mean()

    def _get_resp_corrected_base_impedance(self):
        if self.b.index >= self.x.index:
            self.usable = False
            return np.nan
        sig = self.physiodata.mea_resp_corrected_z0_matrix[self.id] if self.moving_ensembled \
                else self.resp_corrected_z0_signal
        return sig[self.b.index:self.x.index].mean()
        
    def mark_points(self,waveform_prior=None):
        if not self.usable: return
        if self.marking_strategy in ("custom peaks","dtw") and waveform_prior is None:
            raise ValueError("A HeartBeat object is required for custom point marking")

        if self.hand_labeled:
            logger.info("Loading stored point markings for %d",self.id)
            self.load_point_markings()

        # Make an initial guess for the points
        elif self.marking_strategy == "heuristic":
            logger.warn("Using heuristics to mark points")
            self.heuristic_mark_ecg_points()
            self.heuristic_mark_dzdt_points()
            self.heuristic_mark_bp_points()

        elif self.marking_strategy == "custom peaks":
            self.mark_custom_points(waveform_prior=waveform_prior)

        elif self.marking_strategy == "dtw":
            self.warp_timepoints(waveform_prior=waveform_prior)
            
    def load_point_markings(self):
        if self.id < 0: return
        # ECG
        self.p.set_index(self.physiodata.p_indices[self.id])
        self.q.set_index(self.physiodata.q_indices[self.id])
        self.r.set_index(self.physiodata.r_indices[self.id])
        self.s.set_index(self.physiodata.s_indices[self.id])
        self.t.set_index(self.physiodata.t_indices[self.id])
        # ICG
        self.b.set_index(self.physiodata.b_indices[self.id])
        self.c.set_index(self.physiodata.c_indices[self.id])
        self.x.set_index(self.physiodata.x_indices[self.id])
        self.o.set_index(self.physiodata.o_indices[self.id])
        if self.physiodata.using_continuous_bp:
            self.systole.set_index(self.physiodata.systole_indices[self.id])
            self.diastole.set_index(self.physiodata.diastole_indices[self.id])
            
    def _get_point_times(self):
        return np.array(
            [p.time for p in self.points])
    
    def _get_point_values(self):
        self.point_values = np.array(
            [p.value for p in self.points])

    def heuristic_mark_dzdt_points(self,caution=False):
        # find dzdt peaks
        if self.apply_dzdt_smoothing:
            data = smooth(self.dzdt_signal,window_len=self.dzdt_smoothing_window_len)
        else:
            data = self.dzdt_signal
        # Differentiate twice
        ediff = np.ediff1d(data,to_begin=0)
        e2diff = np.ediff1d(ediff,to_begin=0)
        # find zero crossings
        xings = np.flatnonzero(np.sign(ediff[:-1]) != np.sign(ediff[1:]))
        dzdt_peaks = self.dzdt_signal[xings]

        # the highest value should be the R peak
        c_index = xings[dzdt_peaks.argmax()]
        self.c.index = c_index
        self.c.time = self.dzdt_time[c_index]
        self.c.value = self.dzdt_signal[c_index]

        # the q point is the minimum immediately preceding the r peak
        try:
            b_index = (c_index-300) + e2diff[(c_index-300):c_index].argmax()
        except Exception, e:
            logger.warn(e)
            b_index = c_index-40
        self.b.index = b_index
        self.b.time = self.dzdt_time[b_index]
        self.b.value = self.dzdt_signal[b_index]

        # X is the trough following the c point
        x_index = c_index + data[c_index:(c_index+250)].argmin()
        self.x.index = x_index
        self.x.time = self.dzdt_time[x_index]
        self.x.value = self.dzdt_signal[x_index]

        # the s point is the minimum immediately following the r peak
        o_index = x_index + data[x_index:(x_index+100)].argmax()
        self.o.index = o_index
        self.o.time = self.dzdt_time[o_index]
        self.o.value = self.dzdt_signal[o_index]
        
    def heuristic_mark_ecg_points(self,caution=False):
        # find ecg peaks
        if self.apply_ecg_smoothing:
            data = smooth(self.ecg_signal,window_len=self.ecg_smoothing_window_len)
        else:
            data = self.ecg_signal
        # Differentiate twice
        ediff = np.ediff1d(data,to_begin=0)
        e2diff = np.ediff1d(ediff,to_begin=0)
        # find zero crossings
        xings = np.flatnonzero(np.sign(ediff[:-1]) != np.sign(ediff[1:]))
        ecg_peaks = self.ecg_signal[xings]


        # the highest value should be the R peak
        r_index = xings[ecg_peaks.argmax()]
        if caution:
            if np.sign(e2diff[r_index]) != -1.0:
                raise ValueError("R wave is concave up?!?!")
        self.r.index = r_index
        self.r.time = self.ecg_time[r_index]
        self.r.value = self.ecg_signal[r_index]

        # the q point is the minimum immediately preceding the r peak
        pre_r_xings = xings[xings<r_index]
        
        # The q point is the first dip before
        q_index = pre_r_xings[-1]
        self.q.index = q_index
        self.q.time = self.ecg_time[q_index]
        self.q.value = self.ecg_signal[q_index]

        # Take the concave down point nearest to the R peak
        p_index = pre_r_xings[(data[pre_r_xings]).argmax()]
        self.p.index = p_index
        self.p.time = self.ecg_time[p_index]
        self.p.value = self.ecg_signal[p_index]

        # the s point is the minimum immediately following the r peak
        post_r_xings = xings[xings > r_index]
        # find the concave up peaks
        post_conc_up = np.flatnonzero(e2diff[post_r_xings] > 0 )
        if not len(post_conc_up):
            s_index = r_index + 10
        else:
            # Take the concave up point nearest to the R peak
            s_index = post_r_xings[post_conc_up[0]]
        self.s.index = s_index
        self.s.time = self.ecg_time[s_index]
        self.s.value = self.ecg_signal[s_index]
        # the minimum value should be the

        try:
            t_index = s_index + 100 + data[(s_index+100):].argmax()
        except ValueError:
            t_index = len(data) -1
            self.t.needs_attention = True
        self.t.index = t_index
        self.t.time = self.ecg_time[t_index]
        self.t.value = self.ecg_signal[t_index]

    def heuristic_mark_bp_points(self,caution=False):
        if not "bp" in self.physiodata.contents:
            return
        self.systole.index = self.r.index
        self.diastole.index = self.r.index
        self.diastole.time = self.r.time
        self.systole.time = self.r.time
        self.systole.value = self.systolic_signal.mean()
        self.diastole.value = self.diastolic_signal.mean()
            
        # find ecg peaks
        if self.apply_bp_smoothing:
            data = smooth(self.bp_signal,window_len=self.bp_smoothing_window_len)
        else:
            data = self.bp_signal
        # Differentiate twice
        ediff = np.ediff1d(data,to_begin=0)
        # find zero crossings
        xings = np.flatnonzero(np.sign(ediff[:-1]) != np.sign(ediff[1:]))
        # limit it to only peaks after R
        xings = xings[xings > self.r.index]
            
        bp_peaks = self.bp_signal[xings]
        bp_peak_order = len(bp_peaks) - bp_peaks.argsort().argsort()

        # find the two biggest peaks
        try:
            biggest_peak_index = xings[bp_peak_order==1][0]
            next_biggest_peak_index = xings[bp_peak_order==2][0]
            systole_index = min(biggest_peak_index, next_biggest_peak_index)
        except Exception,e:
            logger.warn("Unable to locate systole:\n%s",e)
            systole_index = self.r.index
        self.systole.index = systole_index
        self.systole.value = self.bp_signal[systole_index] #*120
        self.systole.time = self.bp_time[systole_index]
        # Look for the trough between the r peak and the systole
        
        try:
            diastole_window = self.bp_signal[self.r.index:systole_index]
            diastole_index = diastole_window.argmin() + self.r.index
        except Exception, e:
            logger.warn("Unable to locate diastole")
            diastole_index = self.r.index
            
        self.diastole.index = diastole_index
        self.diastole.value = self.bp_signal[diastole_index] #*120
        self.diastole.time = self.bp_time[diastole_index]
        
    def mark_custom_points(self,waveform_prior):
        """
        A list of points is used to find corresponding points
        on this waveform
        """
        for point in waveform_prior.points:
            point.mark_similar(self)

    def warp_timepoints(self,waveform_prior):
        """
        Uses dynamic time warping to match a template waveform to 
        this waveform.
        """
        dists = []
        for signal in self.physiodata.contents:
            k = getattr(self, "dtw_%s_k" % signal)
            constraint = getattr(self, "dtw_%s_constraint" % signal)
            penalty = getattr(self, "dtw_%s_warping_penalty" % signal)
            template = getattr(waveform_prior, signal+"_signal")
            target = getattr(self, signal + "_signal")
            mapping, dist = dtw_mapping(template, target,k=k,
                    constraint=constraint,penalty=penalty)
            dists.append(dist)
            for p in self.points:
                if not p.applies_to == signal: continue

    def get_sv(self,rc=False):
        if self.b.index >= self.x.index:
            return -2 
        if self.physiodata.stroke_volume_equation == "Kubicek":
            return self._get_kubicek_sv(rc=rc)
        elif self.physiodata.stroke_volume_equation == "Sramek-Bernstein":
            return self._get_sramek_bernstein_sv(rc=rc)    
        else:
            raise ValueError(
                "stroke volume equation must be 'Kubicek' or 'Sramek-Bernstein'")

    def get_eef(self):
        # estimated ejection fraction
        return self.get_pep() / self.get_lvet()
        
    def _get_dZdt_max(self):
        return self.dzdt_signal[self.b.index:self.x.index].max()

    def _get_resp_corrected_dZdt_max(self):
        return self.resp_corrected_dzdt_signal[self.b.index:self.x.index].max()
        
    def _get_kubicek_sv(self, rc=False):
        # electrode distance from the subject:
        l = self.physiodata.subject_l
        # lvet IN SECONDS
        LVET = self.get_lvet() / 1000.
        base_impedance = self.resp_corrected_base_impedance if rc else self.base_impedance 
        dzdt_max = self.resp_corrected_dZdt_max if rc else self.dZdt_max 
        return BLOOD_RESISTIVITY * (l**2 / base_impedance**2) * dzdt_max * LVET
    
    def _get_sramek_bernstein_sv(self, rc=False):
        LVET = self.get_lvet() / 1000.
        l = self.physiodata.subject_l
        base_impedance = self.resp_corrected_base_impedance if rc else self.base_impedance 
        dzdt_max = self.resp_corrected_dZdt_max if rc else self.dZdt_max 
        return l**3 / 4.2 * LVET * dzdt_max / base_impedance
    
    def get_lvet(self):
        """ Returns LVET in milliseconds"""
        if self.x.time <= self.b.time:
            self.failed_marking = True
            return np.nan
        return self.x.time - self.b.time
    
    def get_map(self):
        return 2*self.diastole.value/3. + self.systole.value/3.
    
    def get_systolic_bp(self):
        return self.systole.value
    
    def get_diastolic_bp(self):
        return self.diastole.value
    
    def get_pep(self):
        return self.b.time - self.r.time

    def update_timepoint(self):
        """ The user has updated a point time through the GUI.
        Now, we update the values 
        """
        if self.point_vis is None:
            print "no plot to edit"
            return
        tp = self.point_picker.currently_dragging_point
        label = getattr(self,tp+"_label")
        point = getattr(self, tp)
        point.time = self.point_picker.current_time
        point.index = self.point_picker.current_index
        
        # If the point is for bp and we're using CNAP, the actual value matters
        if self.physiodata.using_continuous_bp and tp in ("systole","diastole"):
            point.value = self.bp_signal[self.point_picker.current_index]
            label.data_point = (point.time, self.point_picker.current_value)
        else:
            point.value = self.point_picker.current_value
            label.data_point = (point.time, point.value)
        rand_adj = np.random.rand()/1000
        label.label_position = (-20+rand_adj,20+rand_adj)
        label.request_redraw()
        self.hand_labeled = True
        #set hand labeled in physiodata
        self.physiodata.hand_labeled[self.id] = 1
        
    def _point_updated(self):
        # Triggers listener on beat_train 
        self.point_updated = True

    def _dzdt_signal_changed(self):
        # recompute the dzdt tangent for plotting
        self.ddzdt_signal = np.ediff1d(smooth(self.dzdt_signal,15), to_begin=0)
        self.dddzdt_signal = np.ediff1d(smooth(self.ddzdt_signal,15), to_begin=0)
    
    def _btool_t_selection_changed(self):
        logger.info("[in HeartBeat] b set to %.2f" % self.btool_t_selection)
        self.b.set_time(self.btool_t_selection)
        if self.plotdata is not None:
            self.plotdata.set_data('point_values', self._get_plt_point_values())
            self.plotdata.set_data('point_times', self._get_point_times())
            rand_adj = np.random.rand()/1000
            self.b_label.label_position = (-20+rand_adj,20+rand_adj)
            self.b_label.request_redraw()
            self.hand_labeled = True
            #set hand labeled in physiodata
            self.physiodata.hand_labeled[self.id] = 1
            self.plot.request_redraw()
            self.point_updated=True

    def _dzdt_plot_default(self):
        """
        Instead of defining these in __init__, only
        construct the plots when a ui is requested
        """
        #self._dzdt_signal_changed()
        _time = ArrayDataSource(self.dzdt_time,sort_order="ascending")
        self.dzdt_plotdata = ArrayPlotData(time = _time, 
                                           dzdt_signal = self.dzdt_signal,
                                           ddzdt_signal = self.ddzdt_signal,
                                           dddzdt_signal = self.dddzdt_signal
                                         )
        # Create the plots and tools/overlays
        main_plot = Plot(self.dzdt_plotdata,title="dZ/dt: R to C",padding=20)
        main_plot_line = main_plot.plot(("time","dzdt_signal"), line_width=3, 
                color=colors['dzdt'])[0]
        main_plot_btool = BTool(line_plot=main_plot_line, component=main_plot)
        main_plot.overlays.append(main_plot_btool)
        main_plot_btool.sync_trait("time",self, "btool_t")
        main_plot_btool.sync_trait("selected_time",self, "btool_t_selection")
        main_plot_bmarker = BMarker(line_plot=main_plot_line, component=main_plot, 
                                    selected_time=self.b.time,line_style='dash',color='maroon')
        main_plot.overlays.append(main_plot_bmarker)
        self.b.sync_trait("time", main_plot_bmarker)

        submain_plot = Plot(self.dzdt_plotdata,title="dZ/dt: Full Signal",padding=20)
        submain_plot_line = submain_plot.plot(("time","dzdt_signal"), line_width=3, 
                color=colors['dzdt'])[0]
        submain_plot_btool = BTool(line_plot=submain_plot_line, component=submain_plot)
        submain_plot_bmarker = BMarker(line_plot=submain_plot_line, component=submain_plot, 
                                    selected_time=self.b.time, line_style='dash',color='maroon')
        submain_plot.overlays.append(submain_plot_btool)
        submain_plot_btool.sync_trait("time",self, "btool_t")
        submain_plot_btool.sync_trait("selected_time",self, "btool_t_selection")
        submain_plot_bmarker = BMarker(line_plot=submain_plot_line, component=submain_plot, 
                                    selected_time=self.b.time,line_style='dash',color='maroon')
        submain_plot.overlays.append(submain_plot_bmarker)
        self.b.sync_trait("time", submain_plot_bmarker)
        
        d1_plot = Plot(self.dzdt_plotdata,title="First Derivative",padding=20)
        d1_plot_line = d1_plot.plot(("time","ddzdt_signal"),line_width=3,
                color=colors['dzdt'])[0]
        d1_plot_btool = BTool(line_plot=d1_plot_line, component=d1_plot)
        d1_plot.overlays.append(d1_plot_btool)
        d1_plot_btool.sync_trait("time",self, "btool_t")
        d1_plot_btool.sync_trait("selected_time",self, "btool_t_selection")
        d1_plot_bmarker = BMarker(line_plot=d1_plot_line, component=d1_plot, 
                                    selected_time=self.b.time,line_style='dash',color='maroon')
        d1_plot.overlays.append(d1_plot_bmarker)
        self.b.sync_trait("time", d1_plot_bmarker)

        d2_plot = Plot(self.dzdt_plotdata,title="Second Derivative",padding=20)
        d2_plot_line = d2_plot.plot(("time","dddzdt_signal"),line_width=3,
                color=colors['dzdt'])[0]
        d2_plot_btool = BTool(line_plot=d2_plot_line, component=d2_plot)
        d2_plot.overlays.append(d2_plot_btool)
        d2_plot_btool.sync_trait("time",self, "btool_t")
        d2_plot_btool.sync_trait("selected_time",self, "btool_t_selection")
        d2_plot_bmarker = BMarker(line_plot=d2_plot_line, component=d2_plot, 
                                    selected_time=self.b.time,line_style='dash',color='maroon')
        d2_plot.overlays.append(d2_plot_bmarker)
        self.b.sync_trait("time", d2_plot_bmarker)

        # Set a limited window for some of the plots
        xmin = self.r.time
        xmax = self.c.time
        for plt in [main_plot,d1_plot,d2_plot]:
            plt.index_mapper.range.low = xmin
            plt.index_mapper.range.high = xmax

        # Create containers
        vcon_left = VPlotContainer()
        vcon_left.add(main_plot)
        vcon_right = VPlotContainer(stack_order="top_to_bottom")
        vcon_right.add(submain_plot, d1_plot, d2_plot)
        container = HPlotContainer(vcon_left, vcon_right)
        return container

    def _get_plt_point_values(self):
        # Configure the timepoints we'll be plotting
        return np.array(
            [getattr(self,"plt_"+ p.applies_to)[p.index] for p in self.points ])

    def _plot_default(self):
        """
        Instead of defining these in __init__, only
        construct the plots when a ui is requested
        """
        
        plot_ymax=0
        plt_data = {}
        for signal in self.physiodata.contents & ENSEMBLE_SIGNALS:
            if signal.startswith("resp_corrected"):
                # TODO: perhaps make this smarter
                continue
            if signal in ("ecg", "ecg2"):
                if not signal == self.physiodata.qrs_source_signal: continue
                signal = "ecg"
            scalar,offset = plt_scalars[signal], plt_offsets[signal]

            setattr(self, "plt_" + signal, 
              offset + scalar * normalize(getattr(self, signal+"_signal")))
            plot_ymax=max(plot_ymax,offset+scalar)
            plt_data[signal+"_time"] = getattr(self, signal+"_time")
            plt_data[signal+"_data"] = getattr(self, "plt_"+signal)
        
        point_vals = self._get_plt_point_values()
        for p, pn in zip(self.points,point_vals):
            p.normed_value = pn
        plt_data['point_times'] = self._get_point_times()
        plt_data['point_values'] = point_vals

        # Create plotting components
        self.plotdata = ArrayPlotData(**plt_data)

        plot = Plot(self.plotdata)
        for signal in self.physiodata.contents & ENSEMBLE_SIGNALS:
            if signal.startswith("resp_corrected"):
                # TODO: perhaps make this smarter
                continue
            if signal in ("ecg", "ecg2"):
                if not signal == self.physiodata.qrs_source_signal: continue
                signal = "ecg"
            plot.plot((signal+"_time",signal+"_data"), line_width=3,
                      color=colors[signal])
            
        plot.range2d.y_range.low  = -0.1
        plot.range2d.y_range.high = plot_ymax
        if self.id > 0:
            plot.title = plot.title + " #%d" % self.id
            
        # add labels for the points
        for point in self.points:
            setattr(self,point.name + "_label",
                    DataLabel(
                       component=plot, data_point=(point.time, point.normed_value),
                       #border_padding=5,
                       marker_color="green",
                       marker_size=0,
                       show_label_coords=False,
                       label_style='bubble',
                       label_position=(-20, 20),
                       label_text=point.name.upper(),
                       border_visible=False,
                       font='modern 16',
                       bgcolor=colors[point.applies_to],                        
                       arrow_visible=True
                        )
                    )
            plot.overlays.append(getattr(self,point.name + "_label"))
        
        # Plot the points as a scatterplot
        self.point_vis = plot.plot(("point_times","point_values"),
                                   type="scatter",
                                   marker_size=10,
                                   marker="plus"
                                   )[0]
        self.scatter = plot.components[-1]
        self.point_picker = PointDraggingTool(self.scatter,beat=self )
        self.scatter.tools.append(self.point_picker)
        self.on_trait_event(self.update_timepoint,"point_picker.point_changed")
        self.on_trait_event(self._point_updated,"point_picker.point_edited")
        plot.padding =12
        return plot

    traits_view = MEAPView(
        HSplit(
            Group(
            Item("dzdt_plot",editor=ComponentEditor(),label="ICG Editor"),
            VGroup(
            Item("plot",editor=ComponentEditor(),show_label=False),
            Item("usable"), label="All Signals",show_left=False,springy=True),            
            show_labels=False,
            layout="tabbed")),
            width=800, height=500, resizable=True,
            win_title="Fix Point Marking"
            )

TimePoint.add_class_trait("beat",Instance(HeartBeat))
PointDraggingTool.add_class_trait("beat",Instance(HeartBeat))
    
class EnsembleAveragedHeartBeat(HeartBeat):
    marking_strategy = "custom points"
    subset = Array

    def __init__(self,**traits):
        """
        If a non-negative id is passed, signals will automatically be collected
        from the physiodata object. Otherwise each signal must be passed 
        explicitly during construction (eg ``dzdt_signal = some_array``)
        """
        # If an id is passed, get the signals available for that beat
        traits["id"] = None
        super(EnsembleAveragedHeartBeat,self).__init__(**traits)

    def _set_default_signals(self):
        for signal in self.physiodata.contents:
            if signal == "respiration": continue
            if signal in ("ecg", "ecg2"):
                if not signal == self.physiodata.qrs_source_signal: continue
                
            if signal == self.physiodata.qrs_source_signal:
                mat = getattr(self.physiodata, signal+"_matrix")
            else:
                mat = getattr(self.physiodata, signal+"_matrix")
            
            # Compute the average
            if self.subset.size == 0:
                sig = mat.mean(0)
            else:
                sig = mat[self.subset].mean(0)

            if signal == self.physiodata.qrs_source_signal:
                self.ecg_signal = sig
            else:
                setattr(self, signal + "_signal", sig)
        
class GlobalEnsembleAveragedHeartBeat(EnsembleAveragedHeartBeat):
    marking_strategy = "heuristic"
    use_all_beats = Bool(True)
    min_beats = Int(1)
    max_beats = Property(Int)
    n_random_beats = Int(100)
    b_template = Button(label="Optimize Templates")
    b_enhance = Button(label="ENHANCE")
    def _b_enhance_fired(self):
        self.edit_traits(view="dtw_view")
    # DTWEA parameters
    dtw_ecg_warping_penalty = DelegatesTo('physiodata')
    dtw_ecg_n_iterations = DelegatesTo('physiodata')
    dtw_ecg_constraint = DelegatesTo('physiodata')
    dtw_ecg_metric = DelegatesTo('physiodata')
    dtw_ecg_k = DelegatesTo('physiodata')
    dtw_ecg_used = DelegatesTo('physiodata')
    dtw_ecg_signal = Array

    dtw_z0_warping_penalty = DelegatesTo('physiodata')
    dtw_z0_n_iterations = DelegatesTo('physiodata')
    dtw_z0_constraint = DelegatesTo('physiodata')
    dtw_z0_metric = DelegatesTo('physiodata')
    dtw_z0_k = DelegatesTo('physiodata')
    dtw_z0_used = DelegatesTo('physiodata')
    dtw_z0_signal = Array
    
    dtw_dzdt_warping_penalty = DelegatesTo('physiodata')
    dtw_dzdt_n_iterations = DelegatesTo('physiodata')
    dtw_dzdt_constraint = DelegatesTo('physiodata')
    dtw_dzdt_metric = DelegatesTo('physiodata')
    dtw_dzdt_k = DelegatesTo('physiodata')
    dtw_dzdt_used = DelegatesTo('physiodata')
    dtw_dzdt_signal = Array

    # dtw plot items
    z0_dtw_plot_data = Instance( ArrayPlotData, transient=True)
    z0_dtw_plot = Instance(Plot,transient=True)
    ecg_dtw_plot_data = Instance( ArrayPlotData, transient=True)
    ecg_dtw_plot = Instance(Plot,transient=True)
    dzdt_dtw_plot_data = Instance( ArrayPlotData, transient=True)
    dzdt_dtw_plot = Instance(Plot,transient=True)

    b_dtw_ecg = Button(label="Apply")
    b_dtw_z0 = Button(label="Apply")
    b_dtw_dzdt = Button(label="Apply")
    b_apply_dtws = Button(label="Apply Enhancements")


    def _b_dtw_ecg_fired(self): self.dtw_ensemble_average("ecg")
    def _b_dtw_z0_fired(self): self.dtw_ensemble_average("z0")
    def _b_dtw_dzdt_fired(self): self.dtw_ensemble_average("dzdt")

    def __init__(self,**traits):
        if "subset" in traits and traits["subset"] is not None:
            del traits["subset"]
            logger.warn("subset arg to GlobalEnsembleAveragedHeartBeat ignored")
        super(GlobalEnsembleAveragedHeartBeat,self).__init__(**traits)
        self.p.set_time(self.physiodata.ens_avg_p_time)
        self.q.set_time(self.physiodata.ens_avg_q_time)
        self.r.set_time(self.physiodata.ens_avg_r_time)
        self.s.set_time(self.physiodata.ens_avg_s_time)
        self.t.set_time(self.physiodata.ens_avg_t_time)
        self.b.set_time(self.physiodata.ens_avg_b_time)
        self.x.set_time(self.physiodata.ens_avg_x_time)
        self.c.set_time(self.physiodata.ens_avg_c_time)
        self.o.set_time(self.physiodata.ens_avg_o_time)

        if self.systole is not None:
            self.systole.set_time(self.physiodata.ens_avg_systole_time)
            self.systole.sync_trait("time",self.physiodata,"ens_avg_systole_time")
        if self.diastole is not None:
            self.diastole.set_time(self.physiodata.ens_avg_diastole_time)
            self.diastole.sync_trait("time",self.physiodata,"ens_avg_diastole_time")
        # Sync the values so that edits in the GUI are matched in
        # self.physiodata
        self.p.sync_trait("time",self.physiodata,"ens_avg_p_time")
        self.q.sync_trait("time",self.physiodata,"ens_avg_q_time")
        self.r.sync_trait("time",self.physiodata,"ens_avg_r_time")
        self.s.sync_trait("time",self.physiodata,"ens_avg_s_time")
        self.t.sync_trait("time",self.physiodata,"ens_avg_t_time")
        self.b.sync_trait("time",self.physiodata,"ens_avg_b_time")
        self.x.sync_trait("time",self.physiodata,"ens_avg_x_time")
        self.c.sync_trait("time",self.physiodata,"ens_avg_c_time")
        self.o.sync_trait("time",self.physiodata,"ens_avg_o_time")
        self.physiodata.ens_avg_ecg_signal = self.ecg_signal
        self.physiodata.ens_avg_dzdt_signal = self.dzdt_signal
    
    def _get_max_beats(self):
        return len(self.physiodata.peak_times)

    def _n_random_beats_changed(self):
        self.subset = np.random.choice(self.max_beats,self.n_random_beats,
                                       replace=False)
        for signal in self.physiodata.contents & ENSEMBLE_SIGNALS:
            setattr(self, signal + "_signal", 
                getattr(self.physiodata, signal+"_matrix")[self.subset].mean(0))
            scalar,offset = plt_scalars[signal], plt_offsets[signal]
            setattr(self, "plt_" + signal, 
              offset + scalar * normalize(getattr(self, signal+"_signal")))
            if self.plotdata is not None:
                self.plotdata[signal+"_data"] = getattr(self, "plt_"+signal)
        self.plot.request_redraw()

    def dtw_ensemble_average(self,signal):
        plot = getattr(self, signal+"_dtw_plot")
        plotdata = getattr(self, signal+"_dtw_plot_data")
        niters = getattr(self, "dtw_%s_n_iterations" % signal)
        k = getattr(self, "dtw_%s_k" % signal)
        constraint = getattr(self, "dtw_%s_constraint" % signal)
        penalty = getattr(self, "dtw_%s_warping_penalty" % signal)
        matrix = getattr(self.physiodata, signal + "_matrix")

        if len(plot.components):
            plot.remove(*plot.components)
        
        # apply a warp to each heartbeat
        eas = [matrix.mean(0)]
        progress = ProgressDialog(title="DTW Ensemble Averaging", min=0,
                max = niters*matrix.shape[0],show_time=True,
                message="Processing %s..."%signal)
        progress.open()
        i = 0
        for niter, alpha in enumerate(np.linspace(0.3,1,niters)):
            new_beats = np.zeros_like(matrix)
            for j, beat in enumerate(matrix):
                new_beats[j],cost = template_to_target_dtw(beat,eas[-1],
                        penalty=penalty, constraint=constraint, k=k)
                i += 1
                (cont,skip) = progress.update(i)
            eas.append(new_beats.mean(0))
            plotkey = "iter%d" % niter
            plotdata.set_data(plotkey, eas[-1])
            plot.plot(("time", plotkey),color="blue",alpha=alpha,line_width=3)
            plot.request_redraw()

        # Replot the original signal
        plot.plot(("time", "signal"), color="green", line_width=5,
                alpha=0.6)
        setattr(self, "dtw_%s_signal"%signal, eas[-1])
        plot.request_redraw()

    def _ecg_dtw_plot_default(self):
        return self.get_dtw_plot("ecg")
    def _z0_dtw_plot_default(self):
        return self.get_dtw_plot("z0")
    def _dzdt_dtw_plot_default(self):
        return self.get_dtw_plot("dzdt")

    def _ecg_dtw_plot_data_default(self):
        return self.get_dtw_plot_data("ecg")
    def _z0_dtw_plot_data_default(self):
        return self.get_dtw_plot_data("z0")
    def _dzdt_dtw_plot_data_default(self):
        return self.get_dtw_plot_data("dzdt")

    def get_dtw_plot_data(self,signal):
        plt_data = {
            'time': getattr(self,signal+'_time'),
            'signal': getattr(self,signal+"_signal")}
        return ArrayPlotData(**plt_data)

    def get_dtw_plot(self, signal):
        plotdata = getattr(self, signal+"_dtw_plot_data")
        plot = Plot(plotdata)
        plot.plot(("time","signal"),color="green",line_width=3)
        plot.padding =20
        return plot

    def _b_apply_dtws_fired(self):
        for signal in ("dzdt", "ecg", "z0"):
            if not signal in self.physiodata.contents: continue
            # If selected to be used, apply it to self
            if getattr(self, "dtw_%s_used" % signal):
                logger.info("applying dtw enhancement to %s", signal)
                setattr(self, signal + "_signal", 
                        getattr(self,"dtw_%s_signal"%signal))
        self.update_plot()

    dtw_view = MEAPView(
        VGroup(
            VGroup(
        HGroup(VGroup(
        Item('dtw_ecg_warping_penalty'),
        Item('dtw_ecg_n_iterations'),
        Item('dtw_ecg_constraint'),
        Item('dtw_ecg_metric'),
        Item('dtw_ecg_k'),
        Item('dtw_ecg_used'),
        Item('b_dtw_ecg',show_label=False)
        ),
        Item("ecg_dtw_plot", editor=ComponentEditor(),show_label=False),
        label="ECG"
        ),

        HGroup(VGroup(
        Item('dtw_z0_warping_penalty'),
        Item('dtw_z0_n_iterations'),
        Item('dtw_z0_constraint'),
        Item('dtw_z0_metric'),
        Item('dtw_z0_k'),
        Item('dtw_z0_used'),
        Item('b_dtw_z0', show_label=False)
        ),
        Item("z0_dtw_plot", editor=ComponentEditor(),show_label=False),
        label="z0"
        ),

        HGroup( VGroup(
        Item('dtw_dzdt_warping_penalty'),
        Item('dtw_dzdt_n_iterations'),
        Item('dtw_dzdt_constraint'),
        Item('dtw_dzdt_metric'),
        Item('dtw_dzdt_k'),
        Item('dtw_dzdt_used'),
        Item('b_dtw_dzdt', show_label=False),
        ),
        Item("dzdt_dtw_plot", editor=ComponentEditor(),show_label=False),
        label="dZ/dt"
        ),
        layout="tabbed"
        ),
        Item("b_apply_dtws",show_label=False),
        )
        )