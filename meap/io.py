from meap import (fail, __version__, ENSEMBLE_SIGNALS, 
                  SMOOTHING_WINDOWS)
import os
import tempfile

from traits.api import (HasTraits, CStr, Array, CFloat, CInt,
                        Bool, Enum, Instance, File,Property,
                        Range,Int, List, PrototypedFrom,cached_property,
                        CBool, CArray, Set)
from traitsui.api import Item, Group

from scipy.io.matlab import savemat, loadmat
import numpy as np
import ConfigParser

import logging
logger = logging.getLogger(__name__)


class MEAPConfig(HasTraits):

    # Parameters for point marking
    apply_ecg_smoothing = CBool(True)
    ecg_smoothing_window_len = Int(5) # NOTE: Changed in 1.1 from 20
    apply_imp_smoothing = CBool(True)
    imp_smoothing_window_len = Int(40)
    apply_bp_smoothing = CBool(True)
    bp_smoothing_window_len = Int(80)

    # Parameters for waveform extraction
    peak_window = CInt(80) #Range(low=5,high=400, value= 200)
    ecg_pre_peak = CInt(300) #Range(low=50, high=500, value=300)
    ecg_post_peak = CInt(400) #Range(low=100, high=700, value=400)
    dzdt_pre_peak = CInt(300) #Range(50, 500, value=300)
    dzdt_post_peak = CInt(700) #Range(100, 1000, value=700)
    doppler_pre_peak = CInt(300) #Range(50, 500, value=300)
    doppler_post_peak = CInt(700) #Range(100, 1000, value=700)
    bp_pre_peak = CInt(300) #Range(50, 500, value=300)
    bp_post_peak = CInt(1000) #Range(100, 2500, value=1200)
    stroke_volume_equation = Enum("Kubicek","Sramek-Bernstein")
    extraction_group = Group(
        Group(
            Item("peak_window"),
            Item("ecg_pre_peak"),
            Item("ecg_post_peak"),
            Item("dzdt_pre_peak"),
            Item("dzdt_post_peak"),
            Item("bp_pre_peak"),
            Item("bp_post_peak"),
            Item("stroke_volume_equation"),
            orientation="vertical",
            show_border=True,
            label = "Waveform Extraction Parameters",
            springy=True
            ),
        Group(
            Item("mhd_bandpass_min"),
            Item("mhd_bandpass_max"),
            Item("mhd_smoothing_window_len"),
            Item("mhd_smoothing_window"),
            Item("qrs_to_mhd_ratio"),
            Item("combined_smoothing_window_len"),
            Item("combined_smoothing_window"),
            enabled_when="subject_in_mri"
        )
    )

    # Respiration analysis parameters
    process_respiration = CBool(True)
    resp_polort = CInt(7)
    resp_high_freq_cutoff = CFloat(0.35)
    regress_out_resp = Bool(False)

    # parameters for processing the raw data before PT detecting
    # MRI-specific    
    subject_in_mri = CBool(False)
    peak_detection_algorithm = Enum("Pan Tomkins 83", "Multisignal", "ECG2")

    # PanTomkins algorithm
    bandpass_min = Range(low=1,high=200,initial=5,value=5)
    bandpass_max = Range(low=1,high=200,initial=15,value=15)
    smoothing_window_len = Range(low=10,high=1000,initial=100,value=100)
    smoothing_window = Enum(SMOOTHING_WINDOWS)
    pt_adjust = Range(low=-2.,high=2.,value=0.00)
    peak_threshold=CFloat
    apply_filter = CBool(True)
    apply_diff_sq = CBool(True)
    apply_smooth_ma = CBool(True)
    pt_params_group = Group(
        Item("apply_filter"),
        Item("bandpass_min"),#editor=RangeEditor(enter_set=True)),
        Item("bandpass_max"),#editor=RangeEditor(enter_set=True)),
        Item("smoothing_window_len"),#editor=RangeEditor(enter_set=True)),
        Item("smoothing_window"),
        Item("pt_adjust"),#editor=RangeEditor(enter_set=True)),
        Item("apply_diff_sq"),
        Item("subject_in_mri"),
        label="R peak detection options",
        show_border=True,
        orientation="vertical",
        springy=True
    )

    # Second signal heartbeat detection
    use_secondary_heartbeat = CBool(False)
    secondary_heartbeat = Enum("dzdt", "pulse_ox", "bp")
    secondary_heartbeat_pre_msec = CInt(400)
    secondary_heartbeat_abs = CBool(True)
    secondary_heartbeat_window = Enum(SMOOTHING_WINDOWS)
    secondary_heartbeat_window_len = CInt(801)
    secondary_heartbeat_n_likelihood_bins = CInt(15)

    # ECG2 parameters
    use_ECG2 = CBool(False)
    qrs_signal_source = Enum("ecg", "ecg2")
    ecg2_weight = Range(low=0., high=1.,value=0.5)

    # DTWEA parameters
    dtw_ecg_warping_penalty = CFloat(0.1)
    dtw_ecg_n_iterations = CInt(5)
    dtw_ecg_constraint = Enum(("itakura",'slanted_band','sakoe_chiba','None')) 
    dtw_ecg_metric = Enum(('euclidean', 'sqeuclidean', 'cosine'))
    dtw_ecg_k = CInt(50)
    dtw_ecg_used = CBool(False)

    dtw_z0_warping_penalty = CFloat(0.1)
    dtw_z0_n_iterations = CInt(5)
    dtw_z0_constraint = Enum(("itakura",'slanted_band','sakoe_chiba','None')) 
    dtw_z0_metric = Enum(('euclidean', 'sqeuclidean', 'cosine'))
    dtw_z0_k = CInt(50)
    dtw_z0_used = CBool(False)

    dtw_dzdt_warping_penalty = CFloat(0.1)
    dtw_dzdt_n_iterations = CInt(5)
    dtw_dzdt_constraint = Enum(("itakura",'slanted_band','sakoe_chiba','None')) 
    dtw_dzdt_metric = Enum(('euclidean', 'sqeuclidean', 'cosine'))
    dtw_dzdt_k = CInt(50)
    dtw_dzdt_used = CBool(False)


    # Moving Ensembling Parameters
    mea_window_type = Enum("Seconds","Beats")
    mea_n_neighbors = Range(low=0, high=60, value=8)
    mea_window_secs = Range(low=1., high=60, value=15.)
    mea_exp_power = Enum(2,3,4,5,6)
    mea_func_name = Enum("linear","exponential","flat")
    mea_weight_direction = Enum("symmetric", "before", "after")
    mea_smooth_hr = CBool(True)
    use_trimmed_co = CBool(True)

    #bpoint classifier parameters
    bpoint_classifier_pre_point_msec = CInt(20)
    bpoint_classifier_post_point_msec = CInt(20)
    bpoint_classifier_sample_every_n_msec = CInt(1)
    bpoint_classifier_false_distance_min = CInt(5)
    bpoint_classifier_use_bpoint_prior = CBool(True)
    bpoint_classifier_include_derivative = CBool(True)


def load_config(config_path):
    cpr = ConfigParser.ConfigParser()
    cpr.read(config_path)
    config = MEAPConfig()
    for trait in config.editable_traits():
        try:
            cpr.get("MEAP",trait)
        except ConfigParser.NoOptionError:
            logger.warn("No option for %s privided in %s. Usinf default %s",
                        trait, config_path, str(getattr(config,trait)))
            continue
        typ = type(getattr(config,trait))
        if typ is bool:
            setattr(config,trait,cpr.getboolean("MEAP",trait))
        elif typ is int:
            setattr(config,trait,cpr.getint("MEAP",trait))
        elif typ is float:
            setattr(config,trait,cpr.getfloat("MEAP",trait))
        elif typ is str:
            setattr(config,trait,cpr.get("MEAP",trait))
        else:
            raise ValueError("Unable to read %s trait of type %s" %(trait,typ))
    return config



MSEC_PER_SEC=1000.
def peak_stack(peak_indices, values, pre_msec=300, 
               post_msec=700, sampling_rate=1000):
    if not len(values): return np.array([])
    arrays = []
    samples_per_msec = sampling_rate // MSEC_PER_SEC
    pre = int(pre_msec * samples_per_msec)
    post = int(post_msec * samples_per_msec)
    for peak in peak_indices:
        if peak - pre < 0:
            _data = values[:(peak+post)]
            arrays.append(np.hstack([np.zeros(pre-peak,),_data]))
        elif peak+post >= values.shape[0]:
            _data = values[(peak-pre):]
            arrays.append(np.hstack([_data,np.zeros(peak+post-values.shape[0],)]))
        else:
            arrays.append(values[(peak-pre):(peak+post)])
    return np.vstack(arrays)

class PhysioData(HasTraits):
    """
    Contains the parameters needed to run a MEAP session
    """

    contents = Property(Set)
    def _get_contents(self):
        """
        Assuming this object is already initialized, this trait
        will check for which data are available. For each signal
        type if the raw timeseries is available, 
        """
        contents = set()
        for signal in ENSEMBLE_SIGNALS | set(('respiration',)):
            attr = signal+"_data"
            if not hasattr(self, attr): continue
            if getattr(self,attr).size > 0:
                contents.update((signal,))

        # Check for respiration-corrected versions of z0 and dzdt
        for signal in ["resp_corrected_z0", "resp_corrected_dzdt"]:
            if not hasattr(self, signal): continue
            if getattr(self,signal).size > 0:
                contents.update((signal,))

        return contents

    calculable_indexes = Property(Set)
    @cached_property
    def _get_calculable_indexes(self):
        """
        Determines, based on content, which indexes are possible
        to calculate.
        """
        # Signals
        has_ecg = "ecg" in self.contents
        has_z0 = "z0" in self.contents
        has_dzdt = "dzdt" in self.contents
        has_resp = "respiration" in self.contents
        has_systolic = "systolic" in self.contents
        has_diastolic = "diastolic" in self.contents
        has_bp = "bp" in self.contents
        has_resp_corrected_z0 = self.resp_corrected_z0.size > 0
        has_l = self.subject_l > 1


        # Indexes
        has_hr = False
        has_lvet = False
        has_sv = False
        has_map = False
        has_co = False

        ix = set()
        if has_ecg:
            has_hr = True
            ix.update(("hr","hrv"))
        if has_ecg and has_dzdt:
            has_lvet = True
            ix.update(("pep", "lvet", "eef"))
        if has_lvet and has_l and has_z0:
            has_sv = True
            ix.update(("sv",))
            if has_resp_corrected_z0:
                ix.update(("resp_corrected_sv",))
        if has_bp or has_systolic and has_diastolic:
            has_map = True
            ix.update(("map",))
        if has_hr and has_sv:
            has_co = True
            ix.update(("co",))
            if has_resp_corrected_z0:
                ix.update(("resp_corrected_co",))
        if has_co and has_map:
            ix.update(("tpr",))
            if has_resp_corrected_z0:
                ix.update(("resp_corrected_tpr",))
        if has_resp:
            ix.update(("nbreaths"))
        return ix

    meap_version = CStr(__version__)
    original_file = File
    file_location = File

    # -- Censored Epochs --
    censored_intervals = Array
    censoring_sources = List

    @cached_property
    def _get_censored_regions(self):
        censor_regions = []
        for signal in self.contents:
            censor_regions += getattr(self, signal+"_ts").censored_regions

    # MEA Weighting function
    mea_window_type = PrototypedFrom("config")
    mea_n_neighbors = PrototypedFrom("config")
    mea_window_secs = PrototypedFrom("config")
    mea_exp_power = PrototypedFrom("config")
    mea_func_name = PrototypedFrom("config")
    mea_weight_direction = PrototypedFrom("config")
    use_trimmed_co = PrototypedFrom("config")
    mea_smooth_hr = PrototypedFrom("config")
    mea_weights = Array

    use_secondary_heartbeat = PrototypedFrom("config")
    secondary_heartbeat = PrototypedFrom("config")
    secondary_heartbeat_pre_msec = PrototypedFrom("config")
    secondary_heartbeat_abs = PrototypedFrom("config")
    secondary_heartbeat_window = PrototypedFrom("config")
    secondary_heartbeat_window_len = PrototypedFrom("config")
    secondary_heartbeat_n_likelihood_bins = PrototypedFrom("config")

    use_ECG2 = PrototypedFrom("config")
    ecg2_weight = PrototypedFrom("config")
    qrs_signal_source = PrototypedFrom("config")

    # Bpoint classifier options
    bpoint_classifier_pre_point_msec = PrototypedFrom("config")
    bpoint_classifier_post_point_msec = PrototypedFrom("config")
    bpoint_classifier_sample_every_n_msec =PrototypedFrom("config")
    bpoint_classifier_false_distance_min =PrototypedFrom("config")
    bpoint_classifier_use_bpoint_prior =PrototypedFrom("config")
    bpoint_classifier_include_derivative =PrototypedFrom("config")

    # DTWEA parameters
    dtw_ecg_warping_penalty = PrototypedFrom('config')
    dtw_ecg_n_iterations = PrototypedFrom('config')
    dtw_ecg_constraint = PrototypedFrom('config')
    dtw_ecg_metric = PrototypedFrom('config')
    dtw_ecg_k = PrototypedFrom('config')
    dtw_ecg_used = PrototypedFrom('config')

    dtw_z0_warping_penalty = PrototypedFrom('config')
    dtw_z0_n_iterations = PrototypedFrom('config')
    dtw_z0_constraint = PrototypedFrom('config')
    dtw_z0_metric = PrototypedFrom('config')
    dtw_z0_k = PrototypedFrom('config')
    dtw_z0_used = PrototypedFrom('config')

    dtw_dzdt_warping_penalty = PrototypedFrom('config')
    dtw_dzdt_n_iterations = PrototypedFrom('config')
    dtw_dzdt_constraint = PrototypedFrom('config')
    dtw_dzdt_metric = PrototypedFrom('config')
    dtw_dzdt_k = PrototypedFrom('config')
    dtw_dzdt_used = PrototypedFrom('config')
    # Impedance Data
    z0_winsor_min = CFloat(0.005)
    z0_winsor_max = CFloat(0.005)
    z0_winsorize = CBool(False)
    z0_included = CBool(False)
    z0_decimated = CBool(False)
    z0_channel_name = CStr("")
    z0_sampling_rate = CFloat(1000)
    z0_sampling_rate_unit = CStr("Hz")
    z0_unit = CStr("Ohms")
    z0_start_time = CFloat(0.)
    z0_data = Array
    mea_z0_matrix = Array
    z0_matrix = Property(Array,depends_on="peak_indices")
    def _get_z0_matrix(self):
        if self.peak_indices.size == 0: return np.array([])
        return peak_stack(self.peak_indices,self.z0_data,
                          pre_msec=self.dzdt_pre_peak,post_msec=self.dzdt_post_peak,
                          sampling_rate=self.z0_sampling_rate)

    mea_resp_corrected_z0_matrix = Array
    resp_corrected_z0_matrix = Property(Array,depends_on="peak_indices")
    def _get_resp_corrected_z0_matrix(self):
        if self.peak_indices.size == 0 or self.resp_corrected_z0.size == 0:
            return np.array([])
        return peak_stack(self.peak_indices,self.resp_corrected_z0,
                          pre_msec=self.dzdt_pre_peak,post_msec=self.dzdt_post_peak,
                          sampling_rate=self.z0_sampling_rate)

    dzdt_winsor_min = CFloat(0.005)
    dzdt_winsor_max = CFloat(0.005)
    dzdt_winsorize = CBool(False)
    dzdt_included = CBool(False)
    dzdt_decimated = CBool(False)
    dzdt_channel_name = CStr("")
    dzdt_sampling_rate = CFloat(1000)
    dzdt_sampling_rate_unit = CStr("Hz")
    dzdt_unit = CStr("Ohms/Sec")
    dzdt_start_time = CFloat(0.)
    dzdt_data = Array
    dzdt_matrix = Property(Array,depends_on="peak_indices")
    mea_dzdt_matrix = Array
    @cached_property
    def _get_dzdt_matrix(self):
        if self.peak_indices.size == 0: return np.array([])
        return peak_stack(self.peak_indices,self.dzdt_data,
                          pre_msec=self.dzdt_pre_peak,post_msec=self.dzdt_post_peak,
                          sampling_rate=self.dzdt_sampling_rate)
    
    # Doppler radar
    doppler_winsor_min = CFloat(0.005)
    doppler_winsor_max = CFloat(0.005)
    doppler_winsorize = CBool(False)
    doppler_included = CBool(False)
    doppler_decimated = CBool(False)
    doppler_channel_name = CStr("")
    doppler_sampling_rate = CFloat(1000)
    doppler_sampling_rate_unit = CStr("Hz")
    doppler_unit = CStr("Ohms/Sec")
    doppler_start_time = CFloat(0.)
    doppler_data = Array
    doppler_matrix = Property(Array,depends_on="peak_indices")
    mea_doppler_matrix = Array
    @cached_property
    def _get_doppler_matrix(self):
        if self.peak_indices.size == 0: return np.array([])
        return peak_stack(self.peak_indices,self.doppler_data,
                          pre_msec=self.doppler_pre_peak,post_msec=self.doppler_post_peak,
                          sampling_rate=self.doppler_sampling_rate)
    
    # Respiration
    resp_corrected_dzdt_matrix = Property(Array,depends_on="peak_indices")
    mea_resp_corrected_dzdt_matrix = Array
    @cached_property
    def _get_resp_corrected_dzdt_matrix(self):
        if self.peak_indices.size == 0 or self.resp_corrected_dzdt.size == 0:
            return np.array([])
        return peak_stack(self.peak_indices,self.resp_corrected_dzdt,
                          pre_msec=self.dzdt_pre_peak,post_msec=self.dzdt_post_peak,
                          sampling_rate=self.dzdt_sampling_rate)

    # ECG
    ecg_included = CBool(False)
    ecg_winsor_min = CFloat(0.005)
    ecg_winsor_max = CFloat(0.005)
    ecg_winsorize = CBool(False)
    ecg_decimated = CBool(False)
    ecg_channel_name = CStr("")
    ecg_sampling_rate = CFloat(1000)
    ecg_sampling_rate_unit = CStr("Hz")
    ecg_unit = CStr("V")
    ecg_start_time = CFloat(0.)
    ecg_data = Array
    ecg_matrix = Property(Array,depends_on="peak_indices")
    mea_ecg_matrix = Array
    @cached_property
    def _get_ecg_matrix(self):
        if self.peak_indices.size == 0: return np.array([])
        return peak_stack(self.peak_indices,self.ecg_data,
                          pre_msec=self.ecg_pre_peak,post_msec=self.ecg_post_peak,
                          sampling_rate=self.ecg_sampling_rate)

    # ECG Secondary (eg from EEG)
    ecg2_included = CBool(False)
    ecg2_winsor_min = CFloat(0.005)
    ecg2_winsor_max = CFloat(0.005)
    ecg2_winsorize = CBool(False)
    ecg2_decimated = CBool(False)
    ecg2_channel_name = CStr("")
    ecg2_sampling_rate = CFloat(1000)
    ecg2_sampling_rate_unit = CStr("Hz")
    ecg2_unit = CStr("V")
    ecg2_start_time = CFloat(0.)
    ecg2_data = Array
    ecg2_matrix = Property(Array,depends_on="peak_indices")
    mea_ecg2_matrix = Array
    @cached_property
    def _get_ecg2_matrix(self):
        if self.peak_indices.size == 0: return np.array([])
        return peak_stack(self.peak_indices,self.ecg2_data,
                          pre_msec=self.ecg_pre_peak,post_msec=self.ecg_post_peak,
                          sampling_rate=self.ecg_sampling_rate)

    # Blood pressure might come from a CNAP
    using_continuous_bp = CBool(False)
    bp_included = CBool(False)
    bp_winsor_min = CFloat(0.005)
    bp_winsor_max = CFloat(0.005)
    bp_winsorize = CBool(False)
    bp_decimated = CBool(False)
    bp_channel_name = CStr("")
    bp_sampling_rate = CFloat(1000)
    bp_sampling_rate_unit = CStr("Hz")
    bp_unit = CStr("mmHg")
    bp_start_time = CFloat(0.)
    bp_data = Array
    bp_matrix = Property(Array,depends_on="peak_indices")
    mea_bp_matrix = Array
    @cached_property
    def _get_bp_matrix(self):
        return peak_stack(self.peak_indices,self.bp_data,
                          pre_msec=self.bp_pre_peak,post_msec=self.bp_post_peak,
                          sampling_rate=self.bp_sampling_rate)

    # Or two separate channels
    systolic_included = CBool(False)
    systolic_winsor_min = CFloat(0.005)
    systolic_winsor_max = CFloat(0.005)
    systolic_winsorize = CBool(False)
    systolic_decimated = CBool(False)
    systolic_channel_name = CStr("")
    systolic_sampling_rate = CFloat(1000)
    systolic_sampling_rate_unit = CStr("Hz")
    systolic_unit = CStr("mmHg")
    systolic_start_time = CFloat(0.)
    systolic_data = Array
    systolic_matrix = Property(Array,
                               depends_on="peak_indices,bp_pre_peak,bp_post_peak")
    mea_systolic_matrix = Array
    @cached_property
    def _get_systolic_matrix(self):
        if self.peak_indices.size == 0 or self.using_continuous_bp: 
            return np.array([])
        return peak_stack(self.peak_indices,self.systolic_data,
                          pre_msec=self.bp_pre_peak,post_msec=self.bp_post_peak,
                          sampling_rate=self.bp_sampling_rate)

    diastolic_included = CBool(False)
    diastolic_winsor_min = CFloat(0.005)
    diastolic_winsor_max = CFloat(0.005)
    diastolic_winsorize = CBool(False)
    diastolic_decimated = CBool(False)
    diastolic_channel_name = CStr("")
    diastolic_sampling_rate = CFloat(1000)
    diastolic_sampling_rate_unit = CStr("Hz")
    diastolic_unit = CStr("Ohms")
    diastolic_start_time = CFloat(0.)
    diastolic_data = Array
    diastolic_matrix = Property(Array,
                                depends_on="peak_indices,bp_pre_peak,bp_post_peak")
    mea_diastolic_matrix = Array
    @cached_property
    def _get_diastolic_matrix(self):
        if self.peak_indices.size == 0 or not ("dbp" in self.contents): 
            return np.array([])
        return peak_stack(self.peak_indices,self.diastolic_data,
                          pre_msec=self.bp_pre_peak,post_msec=self.bp_post_peak,
                          sampling_rate=self.bp_sampling_rate)

    respiration_included = CBool(False)
    respiration_winsor_min = CFloat(0.005)
    respiration_winsor_max = CFloat(0.005)
    respiration_winsorize = CBool(False)
    respiration_decimated = CBool(False)
    respiration_channel_name = CStr("")
    respiration_sampling_rate = CFloat(1000)
    respiration_sampling_rate_unit = CStr("Hz")
    respiration_unit = CStr("Ohms")
    respiration_start_time = CFloat(0.)
    respiration_data = Array
    respiration_cycle = Array
    respiration_amount = Array
    resp_corrected_z0 = Array
    resp_corrected_dzdt = Array
    processed_respiration_data = Array
    processed_respiration_time = Array

    # -- Event marking signals (experiment and mri-related)
    mri_trigger_times = Array
    mri_trigger_included = CBool(False)
    mri_trigger_decimated = CBool(False)
    mri_trigger_channel_name = CStr("")
    mri_trigger_sampling_rate = CFloat(1000)
    mri_trigger_sampling_rate_unit = CStr("Hz")
    mri_trigger_unit = CStr("V")
    mri_trigger_start_time = CFloat(0.)
    event_names = List
    event_sampling_rate = CFloat(1000)
    event_included = CBool(True)
    event_decimated = CBool(False)
    event_start_time = CFloat(0.)
    event_sampling_rate_unit = "Hz"
    event_unit = CStr("Hz")

    # -- results of peak detection
    peak_times = Array
    peak_indices = CArray(dtype=np.int)
    # Non-markable heartbeats
    dne_peak_times = Array
    dne_peak_indices = CArray(dtype=np.int)
    # Any custom labels for heartbeats go here
    hand_labeled = Instance(np.ndarray) # An array of beat indices, each corresponding 
    def _hand_labeled_default(self):
        return np.zeros_like(self.peak_indices)
    # Is the beat usable for analysis?
    usable = Instance(np.ndarray)
    def _usable_default(self):
        return np.ones(len(self.peak_indices),dtype=np.int)

    p_indices = Instance(np.ndarray)
    def _p_indices_default(self):
        return np.zeros_like(self.peak_indices)
    q_indices = Instance(np.ndarray)
    def _q_indices_default(self):
        return np.zeros_like(self.peak_indices)
    r_indices = Instance(np.ndarray)
    def _r_indices_default(self):
        return np.zeros_like(self.peak_indices)
    s_indices = Instance(np.ndarray)
    def _s_indices_default(self):
        return np.zeros_like(self.peak_indices)
    t_indices = Instance(np.ndarray)
    def _t_indices_default(self):
        return np.zeros_like(self.peak_indices)
    b_indices = Instance(np.ndarray)
    def _b_indices_default(self):
        return np.zeros_like(self.peak_indices)
    c_indices = Instance(np.ndarray)
    def _c_indices_default(self):
        return np.zeros_like(self.peak_indices)
    x_indices = Instance(np.ndarray)
    def _x_indices_default(self):
        return np.zeros_like(self.peak_indices)
    o_indices = Instance(np.ndarray)
    def _o_indices_default(self):
        return np.zeros_like(self.peak_indices)
    systole_indices = Instance(np.ndarray)
    def _systole_indices_default(self):
        return np.zeros_like(self.peak_indices)
    diastole_indices = Instance(np.ndarray)
    def _diastole_indices_default(self):
        return np.zeros_like(self.peak_indices)

    # --- Subject information
    subject_age = CFloat(0.)
    subject_gender = Enum("M","F")
    subject_weight = CFloat(0.,label="Weight (lbs)")
    subject_height_ft = Int(0,label="Height (ft)",
                            desc="Subject's height in feet")
    subject_height_in = Int(0,label = "Height (in)",
                            desc="Subject's height in inches")
    subject_electrode_distance_front = CFloat(0.,
                                              label="Impedance electrode distance (front)")
    subject_electrode_distance_back = CFloat(0.,
                                             label="Impedance electrode distance (back)")
    subject_electrode_distance_right = CFloat(0.,
                                              label="Impedance electrode distance (back)")
    subject_electrode_distance_left = CFloat(0.,
                                             label="Impedance electrode distance (back)")
    subject_resp_max = CFloat(0.,label="Respiration circumference max (cm)")
    subject_resp_min = CFloat(0.,label="Respiration circumference min (cm)")
    subject_in_mri = CBool(False,label="Subject was in MRI scanner")
    subject_control_base_impedance = CFloat(0.,label="Control Imprdance",
                                            desc="If in MRI, store the z0 value from outside the MRI")

    subject_l = Property(CFloat,depends_on=
                         "subject_electrode_distance_front," + \
                         "subject_electrode_distance_back," + \
                         "subject_electrode_distance_right," + \
                         "subject_electrode_distance_left," + \
                         "subject_height_ft"
                         )
    @cached_property
    def _get_subject_l(self):
        """
        Uses information from the subject measurements to define the 
        l variable for calculating stroke volume. 

        if left and right electrode distances are provided, use the average
        if front and back electrode distances are provided, use the average
        if subject height in feet and inches is provided, use the estimate of
             l = 0.17 * height
        Otherwise return the first measurement found in front,back,left,right
        If nothing is found, returns 1

        """
        front = self.subject_electrode_distance_front
        back = self.subject_electrode_distance_back
        left = self.subject_electrode_distance_left
        right = self.subject_electrode_distance_right
        if left > 0 and right > 0:
            return (left + right) / 2.
        if front > 0 and back > 0:
            return (front + back) / 2.
        if self.subject_height_ft > 0:
            return (12*self.subject_height_ft + \
                    self.subject_height_in) * 2.54 * 0.17
        for measure in (front, back, left, right):
            if measure > 0.: return measure
        return 1

    # --- From the global configuration
    config = Instance(MEAPConfig)
    apply_ecg_smoothing = PrototypedFrom("config")
    ecg_smoothing_window_len = PrototypedFrom("config")
    apply_imp_smoothing = PrototypedFrom("config")
    imp_smoothing_window_len = PrototypedFrom("config")
    apply_bp_smoothing = PrototypedFrom("config")
    bp_smoothing_window_len = PrototypedFrom("config")
    regress_out_resp = PrototypedFrom("config")

    # parameters for processing the raw data before PT detecting
    subject_in_mri = PrototypedFrom("config")
    peak_detection_algorithm  = PrototypedFrom("config")
    # PanTomkins parameters
    qrs_source_signal = Enum("ecg", "ecg2")
    bandpass_min = PrototypedFrom("config")
    bandpass_max =PrototypedFrom("config") 
    smoothing_window_len = PrototypedFrom("config")
    smoothing_window = PrototypedFrom("config")
    pt_adjust = PrototypedFrom("config")
    peak_threshold = PrototypedFrom("config")
    apply_filter = PrototypedFrom("config")
    apply_diff_sq = PrototypedFrom("config")
    apply_smooth_ma = PrototypedFrom("config")
    peak_window = PrototypedFrom("config")

    # Parameters for waveform extraction
    ecg_pre_peak = PrototypedFrom("config")
    ecg_post_peak = PrototypedFrom("config")
    dzdt_pre_peak = PrototypedFrom("config")
    dzdt_post_peak = PrototypedFrom("config")
    bp_pre_peak = PrototypedFrom("config")
    bp_post_peak = PrototypedFrom("config")
    doppler_pre_peak = PrototypedFrom("config")
    doppler_post_peak = PrototypedFrom("config")
    stroke_volume_equation = PrototypedFrom("config")

    # parameters for respiration analysis
    process_respiration = PrototypedFrom("config")
    resp_polort = PrototypedFrom("config")
    resp_high_freq_cutoff = PrototypedFrom("config")
    resp_inhale_begin_times = Array
    resp_exhale_begin_times = Array

    # Time points of the global ensemble average
    ens_avg_ecg_signal = Array
    ens_avg_dzdt_signal = Array
    ens_avg_bp_signal = Array
    ens_avg_systolic_signal = Array
    ens_avg_diastolic_signal = Array
    ens_avg_p_time = CFloat
    ens_avg_q_time = CFloat
    ens_avg_r_time = CFloat
    ens_avg_s_time = CFloat
    ens_avg_t_time = CFloat
    ens_avg_b_time = CFloat
    ens_avg_c_time = CFloat
    ens_avg_x_time = CFloat
    ens_avg_y_time = CFloat
    ens_avg_o_time = CFloat
    ens_avg_systole_time = CFloat
    ens_avg_diastole_time = CFloat
    using_hand_marked_point_priors = CBool(False)

    # MEA Physio timeseries
    lvet = Array
    co = Array
    resp_corrected_co = Array
    pep = Array
    sv = Array
    resp_corrected_sv = Array
    map = Array    
    systolic = Array
    diastolic = Array
    hr = Array
    mea_hr = Array
    tpr = Array
    resp_corrected_tpr = Array

    def _config_default(self):
        return MEAPConfig()

    # Storing and accessing the bpoint classifier
    bpoint_classifier_file = File

    def save(self,outfile):
        # Populate matfile-friendly data structures for censoring regions
        tmp = tempfile.NamedTemporaryFile()
        save_attrs = []
        for k in self.editable_traits():
            if k.endswith("ts"):
                continue
            if k == "bpoint_classifier":
                continue
            if k == "bpoint_classifier_file":
                continue
            if k in ("censored_regions","event_names"):
                continue
            v = getattr(self,k)
            if type(v) == np.ndarray:
                if v.size == 0: continue
            if type(v) is set: continue
            save_attrs.append(k)
        savedict = dict([(k,getattr(self,k)) \
                         for k in save_attrs if not (getattr(self,k) is None)])
        savedict["censoring_sources"] = np.array(self.censoring_sources)
        for evt in self.event_names:
            savedict[evt] = getattr(self,evt)
        savedict["event_names"] = np.array(self.event_names)
        for k,v in savedict.iteritems():
            try:
                savemat( tmp, {k:v}, long_field_names=True)
            except Exception, e:
                logger.warn("unable to save %s because of %s", k,e)
        tmp.close()
        savemat(outfile, savedict,long_field_names=True)
        if not os.path.exists(outfile+".mat"):
            logger.critical("failed to save %s.mat", outfile)

def load_from_disk(matfile, config=None,verbose=False):
    """
    Loads a .mea file from disk
    """
    if not os.path.exists(matfile):
        fail("No file exists at %s" % matfile )
    #logger.info("Loading " + matfile)
    m = loadmat(matfile)
    pd = PhysioData(config=config, empty=True)
    cfg = MEAPConfig()
    config_loadable_traits = cfg.editable_traits()
    loadable_traits = pd.editable_traits()
    traits = {}
    cfg_traits = {}
    for k,v in m.iteritems():
        if k in loadable_traits or k.startswith("EVENT_"):
            if k == "config": continue
            if k in config_loadable_traits:
                tdict = cfg_traits
                obj = cfg
            else:
                tdict = traits
                obj = pd
            if verbose:
                logger.info("Extracting attribute: " + k)

            # Skip over the traits we generate on-the-fly
            if k.endswith("matrix") and not k.startswith("mea"): continue
            if k == "censored_regions": continue
            if k == "subject_l": continue
            if k == "base_impedance": continue
            if k == "bpoint_classifier_file": continue
            if k == "censored_intervals":
                tdict[k] = v # Dont's squeeze!!
                continue
            if k == "censoring_sources":
                tdict[k] = [str(d).strip() for d in v]
                continue

            # cast the squeezed matrices into their appropriate types
            if k.startswith("EVENT_"):
                logger.info("loading event data %s", k)
                tdict[k] = v.squeeze()
            else:
                cls = type(getattr(obj, k))
                if cls in (str,int,float,bool):
                    tdict[k] = cls(v.squeeze())
                elif cls is np.ndarray:
                    if k.endswith("indices") or k=="usable":
                        logger.info("casting %s to int",k)
                        tdict[k] = v.squeeze().astype(np.int)
                    else:
                        tdict[k] = v.squeeze()
        else:
            if verbose:
                logger.info("Ignoring attribute: " + k)
    cfg = MEAPConfig(**cfg_traits)
    if 'file_location' in traits:
        del traits['file_location']
    #if 'bpoint_classifier_file' in traits:
    #    del traits['bpoint_classifier_file']
    pd = PhysioData(config=cfg,file_location=matfile, **traits)
    return pd