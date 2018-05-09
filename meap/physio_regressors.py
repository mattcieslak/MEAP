#!/usr/bin/env python
from meap.gui_tools import MEAPView, messagebox
import nibabel as nib
from scipy.special import legendre
from sklearn.linear_model import LinearRegression
import numpy as np
from meap.io import PhysioData
from meap.filters import normalize
from meap.timeseries import TimeSeries
from traits.api import ( HasTraits, Enum, Array, Str, List,
        DelegatesTo, Instance, Int, File, Button, Bool, Float )
from meap.gui_tools import VGroup, Item
# Respiratory peaks
from scipy.stats.mstats import winsorize
import pandas as pd
import os.path as op

from scipy.interpolate import interp1d

import logging
logger=logging.getLogger(__name__)

def get_slicenum_volume(volume_shape, volume_affine, acquisition_type,
                        dicom_acquisition_direction):
    volume = np.zeros(volume_shape)
    if acquisition_type == "Coronal":
        direction = "A" if volume_affine[1,1] > 0 else "P"
        slice_ids = np.arange(volume_shape[1])
        if not direction == dicom_acquisition_direction[1].upper():
            logger.info("Slice acquisition order different from voxel order in y axis")
            slice_ids = slice_ids[::-1]
        else:
            logger.info("Slice acquisition consistent with voxel order in y axis")
            
    elif acquisition_type == "Axial":
        slice_ids = np.arange(volume_shape[2])
        direction = "S" if volume_affine[2,2] > 0 else "I"
        slice_ids = np.arange(volume_shape[2])
        if not direction == dicom_acquisition_direction[2].upper():
            logger.info("Slice acquisition order different from voxel order in z axis")
            slice_ids = slice_ids[::-1]
        else:
            logger.info("Slice acquisition consistent with voxel order in z axis")

    elif acquisition_type == "Saggittal":
        slice_ids = np.arange(volume_shape[0])
        direction = "R" if volume_affine[0,0] > 0 else "L"
        slice_ids = np.arange(volume_shape[0])
        if not direction == dicom_acquisition_direction[0].upper():
            logger.info("Slice acquisition order different from voxel order in z axis")
            slice_ids = slice_ids[::-1]
        else:
            logger.info("Slice acquisition consistent with voxel order in z axis")

    for slice_index, slicenum in enumerate(slice_ids):
        if acquisition_type == "Coronal":
            volume[:,slice_index,:] = slicenum
        elif acquisition_type == "Axial":
            volume[:,:,slice_index] = slicenum
        elif acquisition_type == "Saggittal":
            volume[slice_index,:,:] = slicenum

    return volume


def fourier_expand(phases,order=2):
    series = []
    for m in range(1,order+1):
        series += [
            np.cos(m*phases), np.sin(m*phases) ]
        
    for ser in series:
        ser[np.logical_not(np.isfinite(ser))] = 0
        
    return np.row_stack(series)

class FMRITool(HasTraits):
    physiodata = Instance(PhysioData)
    processed_respiration_data = DelegatesTo("physiodata")
    processed_respiration_time = DelegatesTo("physiodata")

    slice_times_txt = Str
    slice_times = Array
    slice_times_matrix = Array
    acquisition_type = Enum("Coronal", "Saggittal", "Axial")
    dicom_acquisition_direction = Enum("LPS+", "RAS+")
    unsure_about_direction = Bool(False)
    correction_type = Enum(values="possible_corrections")
    possible_corrections = List(["Slicewise Regression"])
    respiration_expansion_order = Int(4)
    drift_model_order = Int(6)
    add_back_intensity = Bool(True)
    cardiac_expansion_order = Int(3)
    interaction_expansion_order = Int(1)
    physio_design_file = File

    # Local arrays that have been restricted to during-scanning times
    r_times   = Array # Beat times occurring during scanning
    tr_onsets = Array # TR times
    resp_times = Array
    resp_signal = Array

    # Nifti Stuff
    fmri_file = File
    fmri_mask = File
    denoised_output = File
    regression_output = File
    nuisance_output_prefix = Str
    b_calculate = Button(label="RUN")
    write_4d_nuisance = Bool(False)
    write_4d_nuisance_for_each_modality = Bool(False)
    output_units = Enum("Percent Signal Change", "Raw BOLD")
    fmri_tr = Float

    interactive = Bool(False)
    missing_triggers = Int(0)
    missing_volumes = Int(0)
    
    traits_view = MEAPView(
        VGroup(
            VGroup(
            Item("fmri_file"), Item("fmri_mask"), 
            Item("slice_times_txt"), 
            Item("acquisition_type"),
            Item("dicom_acquisition_direction"),
            Item("unsure_about_direction"),
            Item("denoised_output"),
            Item("regression_output"),
            Item("nuisance_output_prefix"),
            Item("write_4d_nuisance"),
            Item("write_4d_nuisance_for_each_modality"),
            show_border=True,label="fMRI I/O"),
            VGroup(
                Item("cardiac_expansion_order"),
                Item("respiration_expansion_order"),
                Item("interaction_expansion_order"),
                Item("correction_type"),
                Item("drift_model_order"),
                Item("add_back_intensity"),
                Item("output_units"),
                Item("physio_design_file"),
                show_border=True,label="RETROICOR")
        ),
        Item("b_calculate",show_label=False),
        win_title="RETROICOR"
    )
    
    def _load_slice_timings(self):
        try:
            _slices = np.loadtxt(self.slice_times_txt)
        except Exception, e:
            messagebox("Unable to load slice timings:\n%s" %e)
            raise ValueError
        
        # Check if it's in the mid-TR format
        if np.any(_slices < 0):
            logger.info("Using the mid-TR slice convention")
            if self.fmri_tr <= 0:
                raise ValueError("fMRI file has to be set before using mid-TR")
            _slices = (_slices + 0.5) * self.fmri_tr
        
        # Check if the times are in msec or 
        if np.any(_slices) > 10:
            logger.info("Converting slice times from msec to sec")
            _slices = _slices / 1000.
        else:
            logger.info("Assuming slice timings are in seconds")
            
        # Slice times should be one row per TR, and one column
        # per slice.
        self.slice_times = _slices
        logger.info("Using slice timings of %s", str(self.slice_times))
        

    def _b_calculate_fired(self):
        # Load the slice timings data
        self._load_slice_timings()
        n_trs = len(self.tr_onsets)
        if not n_trs > 0:
            messagebox("A 4D NIfTI file must be specified")
            raise ValueError
        
        # Perform the analysis with the original slice times
        self.slice_times_matrix = np.row_stack([self.slice_times] * n_trs)
        logger.info("Computing slicewise regressors")
        slice_regressors = self.compute_slicewise_regressors()
        self.process_mri_slices(slice_regressors)
        
        # If unsure about the direction of the slices, re-run with the opposite 
        # slice direction
        if not self.unsure_about_direction: return
        
        # Make new output filenames
        if self.fmri_file.endswith(".nii"):
            basename = self.fmri_file[:-4]
        elif self.fmri_file.endswith(".nii.gz"):
            basename = self.fmri_file[:-7]
        self.denoised_output = basename + "_denoised_4D_reverse.nii.gz"
        self.regression_output = basename + "_reverse_r2.nii.gz"
        
        # 
        logger.info("Computing slicewise regressors in opposite direction")
        orig_timings = self.slice_times.copy()
        timing_range = orig_timings.max() - orig_timings.min()
        rev_times = (1 - (orig_timings - orig_timings.min()) / timing_range) \
                       * timing_range + orig_timings.min()        
        self.slice_times = rev_times
        self.slice_times_matrix = np.row_stack([self.slice_times] * n_trs)
        reverse_slice_regressors = self.compute_slicewise_regressors()
        self.process_mri_slices(reverse_slice_regressors)
        

    def _fmri_file_changed(self):
        # When is scanning occurring? In order to generate a 
        # valid design we'll have to match up to the number of TRs in the
        # nifti file.
        _img = nib.load(self.fmri_file)
        ntrs = _img.shape[-1]
        tr = _img.get_header().get_zooms()[-1]
        self.fmri_tr = tr
        # ACQ times of beats and  trigger times
        pd_triggers = self.physiodata.mri_trigger_times
        ntriggers = len(pd_triggers)
        if ntriggers > ntrs:
            logger.warn("More triggers in physiodata than volumes in fmri file")
            model_times = pd_triggers[:ntrs]
        elif ntriggers < ntrs:
            logger.warn("More volumes in fmri file than triggers in physiodata")
            missing_trs = ntrs - ntriggers
            model_times = np.concatenate([pd_triggers,
                pd_triggers[-1] + tr*(np.arange(missing_trs)+1)])
        else:
            model_times = pd_triggers
        self.tr_onsets = model_times

        peak_times = self.physiodata.peak_times 
        dne_peak_times = self.physiodata.dne_peak_times
        
        if dne_peak_times.size > 0 and dne_peak_times.ndim == 0:
            dne_peak_times = np.array([dne_peak_times])
            
        if len(dne_peak_times):
            _peaks = np.concatenate([peak_times, dne_peak_times])
        else:
            _peaks = peak_times
        all_peaks = np.unique(np.sort(_peaks))
        
        self.r_times = all_peaks[
                  (all_peaks >= self.tr_onsets[0]-5) * \
                  (all_peaks <= self.tr_onsets[-1]+5)]
        t = np.arange(len(self.physiodata.z0_data)) / \
                    self.physiodata.z0_sampling_rate + self.physiodata.z0_start_time
        self.time = t
        scanning_mask = (self.time >= self.tr_onsets[0]-3) * \
            (self.time <= self.tr_onsets[-1]+3)
        self.resp_times = self.processed_respiration_time[scanning_mask]
        self.resp_signal = self.processed_respiration_data[scanning_mask]
        if self.fmri_file.endswith(".nii"):
            basename = self.fmri_file[:-4]
        elif self.fmri_file.endswith(".nii.gz"):
            basename = self.fmri_file[:-7]
        self.denoised_output = basename + "_denoised_4D.nii.gz"
        self.regression_output = basename + "_r2.nii.gz"

    def heartbeat_phase(self,timepoint):
        """ Determines which pair of heartbeats contain this timepoint 
        and returns the phase at which ``timepoint`` occurs"""
        nearest_idx = np.argmin(np.abs(timepoint - self.r_times))
        nearest_beat_is_before = timepoint - self.r_times[nearest_idx] > 0
        if nearest_beat_is_before:
            if nearest_idx +1 == len(self.r_times): return 0
            previous_beat = self.r_times[nearest_idx]
            next_beat = self.r_times[nearest_idx + 1]
        else:
            previous_beat = self.r_times[nearest_idx-1]
            next_beat = self.r_times[nearest_idx]
        # Glover Eq. 2
        return 2*np.pi* (timepoint - previous_beat) / (next_beat - previous_beat)
    
    
    def compute_slicewise_regressors(self,N_BINS=100):
        bold = nib.load(self.fmri_file)
        if self.acquisition_type == "Coronal":
            expected_slices = bold.shape[1]
        elif self.acquisition_type == "Axial":
            expected_slices = bold.shape[0]
        elif self.acquisition_type == "Saggittal":
            expected_slices = bold.shape[2]

        # Check that the number of slice timings matches matrix
        if not (self.slice_times_matrix.ndim == 2 and \
                expected_slices == self.slice_times_matrix.shape[1]):
            messagebox("Acquisition of type %s requires %d slices. \
                    %d slice times were provided"%(self.acquisition_type,expected_slices,
                        self.slice_times_matrix.shape[1]))
            raise ValueError
        
        # Only needed when slice_times_matrix is manually specified
        num_trs = bold.shape[3]
        if self.slice_times_matrix.shape[0] < num_trs:
            messagebox("Not enough slice timings to cover the 4d time series")
            raise ValueError
        if self.slice_times_matrix.shape[0] > num_trs:
            self.slice_times_matrix = self.slice_times_matrix[:num_trs,:]            
            
        resp_hist, bins = np.histogram(self.resp_signal,N_BINS)
        resp_transfer_func = np.concatenate([
            [0],np.cumsum(resp_hist)/float(resp_hist.sum())])

        #kernel_size = self.physiodata.z0_sampling_rate - 1
        #resp_smooth = smooth(self.resp_signal,window_len=kernel_size, window="flat")
        resp_diff = np.ediff1d(self.resp_signal, to_begin=0)
        resp_phase = np.pi * resp_transfer_func[
            np.round(normalize(self.resp_signal) * N_BINS
                     ).astype(np.int)]*np.sign(resp_diff)
        
        # Make detrending regressors
        detrend_model = np.row_stack([ \
            legendre(x)(
                np.linspace(-1,1,self.slice_times_matrix.shape[0])) \
                        for x in range(self.drift_model_order)
            ]
        )
        detrend_columns = ["poly%02d"%polynum for polynum in range(
                                                self.drift_model_order)]
        
        # create the detrending 
        # At what phase did the TR occur?
        regressors = []
        for slicenum in range(self.slice_times_matrix.shape[1]):
            offsets = self.tr_onsets + self.slice_times_matrix[:,slicenum] # Both in seconds
            logger.info("Computing regressors for slice %d",slicenum)
            tr_cardiac_phase = np.array(
                    [self.heartbeat_phase(t) for t in offsets])

            tr_indices = np.array(
                [np.argmin(np.abs(t-self.resp_times)) for t in offsets]
            ).astype(np.int)
            tr_resp_phase = resp_phase[tr_indices]

            resp_regressors = fourier_expand(tr_resp_phase,
                                             self.respiration_expansion_order)
            columns = []
            columns += [("resp_ricor_cos%d"%(n+1), "resp_ricor_sin%d"%(n+1)) for n in \
                    range(self.respiration_expansion_order)]
            cardiac_regressors = fourier_expand(tr_cardiac_phase,
                                             self.cardiac_expansion_order)
            columns += [("cardiac_ricor_cos%d"%(n+1), "cardiac_ricor_sin%d"%(n+1)) for n in \
                    range(self.cardiac_expansion_order)]
            mult_plus = fourier_expand(tr_cardiac_phase + tr_resp_phase,
                                             self.interaction_expansion_order)
            columns += [("ix_plus_ricor_cos%d"%(n+1), "ix_plus_ricor_sin%d"%(n+1)) for n in \
                    range(self.interaction_expansion_order)]
            mult_minus = fourier_expand(tr_cardiac_phase - tr_resp_phase,
                                             self.interaction_expansion_order)
            columns += [("ix_minus_ricor_cos%d"%(n+1), "ix_minus_ricor_sin%d"%(n+1)) for n in \
                    range(self.interaction_expansion_order)]

            columns = detrend_columns + [item for sublist in columns for item in sublist]
            data=np.row_stack([detrend_model, resp_regressors, cardiac_regressors, mult_plus, mult_minus,
                               ]).T
            regressors.append(pd.DataFrame(data=data,columns=columns))
        return regressors

    def process_mri_slices(self, slice_regressors):
        # Load the bold data
        _img = nib.load(self.fmri_file)
        img = _img.get_data()

        if not op.exists(self.fmri_mask):
            raise ValueError("Mask file does not exist")
        _mask = nib.load(self.fmri_mask)
        mask = _mask.get_data()

        # Make a volume where each voxel contains its slice number
        slices = np.zeros(mask.shape)
        if self.acquisition_type == "Saggittal":
            nslices = slices.shape[0]
        elif self.acquisition_type == "Coronal":
            nslices = slices.shape[1]
        else:
            nslices = slices.shape[2]
        
        slices = get_slicenum_volume(_mask.shape, _img.affine,
                    self.acquisition_type, self.dicom_acquisition_direction)

        # Create a matrix of timeseries. also get a corresponding array of slice numbers
        ts = img[mask>0]
        slicenum = slices[mask>0].astype(np.int)

        # sometimes a trigger is sent before the scan is cancelled.
        design_len = np.array([x.shape[0] for x in slice_regressors])
        assert np.all(design_len == design_len[0])
        design_len = design_len[0]
        nvols = ts.shape[1]
        if nvols < design_len:
            for nslice in range(nslices):
                slice_regressors[nslice] = slice_regressors[nslice][:nvols]
        elif nvols > design_len:
            ts = ts[:,:design_len]
            
        # Holds the corrected output
        retroicorrected = np.zeros(ts.shape,dtype=np.float)
        
        # Holds a nuisance timeseries for each modality
        if self.write_4d_nuisance_for_each_modality:
            cardiac_fits = np.zeros_like(retroicorrected)
            resp_fits = np.zeros_like(retroicorrected)
            ix_fits = np.zeros_like(retroicorrected)
            drift_fits = np.zeros_like(retroicorrected)
        
        # Holds a nuisance timeseries for all modalities
        if self.write_4d_nuisance:
            fitts = np.zeros_like(retroicorrected)
        
        # Check again that it worked
        design_len = np.array([x.shape[0] for x in slice_regressors])
        assert np.all(design_len == design_len[0])
        design_len = design_len[0]
        if not ts.shape[1] == design_len: raise AttributeError
        
        if self.output_units == "Percent Signal Change":
            ts_mean = ts.mean(axis=1)
            ts = (ts - ts_mean[:,None]) / ts_mean[:,None] * 100
            
        # Compute the fits
        reg = LinearRegression(fit_intercept=False, normalize=False)
        voxel_fits = np.zeros(ts.shape[0])
        design_mats = [x.as_matrix() for x in slice_regressors]
        columns = slice_regressors[0].columns
        resp_mask = np.array([col.startswith("resp") for col in columns])
        cardiac_mask = np.array([col.startswith("cardiac") for col in columns])
        ix_mask = np.array([col.startswith("ix") for col in columns])
        detrend_mask = np.array([col.startswith("poly") for col in columns])
        ricor_mask = ix_mask | cardiac_mask | resp_mask        
        
        for nvox, voxel in enumerate(ts):
            if nvox % 10000 == 0 : print nvox
            # Get the design specific for this slice from
            whole_design = design_mats[slicenum[nvox]]
            reg.fit(whole_design, voxel)
            
            # Store the r^2
            voxel_fits[nvox] = reg.score(whole_design, voxel)
            # Store the corrected output
            if not self.add_back_intensity:
                retroicorrected[nvox] = voxel - reg.predict(whole_design)
            else:
                # Add back the intercept if requested
                retroicorrected[nvox] = voxel - reg.predict(whole_design) + reg.coef_[0]
            
            if self.write_4d_nuisance_for_each_modality:
                cardiac_fits[nvox] = np.dot(whole_design, reg.coef_ * cardiac_mask)
                resp_fits[nvox] = np.dot(whole_design, reg.coef_ * resp_mask)
                ix_fits[nvox] = np.dot(whole_design, reg.coef_ * ix_mask)
                drift_fits[nvox] = np.dot(whole_design, reg.coef_ * ix_mask)
                
            if self.write_4d_nuisance:
                fitts[nvox] = np.dot(whole_design, reg.coef_)

        # write the output
        def save_scalars(vec,filename):
            new_data = np.zeros_like(mask,dtype=np.float32)
            new_data[mask>0] = vec
            nib.Nifti1Image(new_data,_mask.get_affine()#,
                    #header=_mask.get_header()).to_filename(filename)
                    ).to_filename(filename)

        # write the output
        def save_vectors(vec,filename):
            out_shape = list(_img.shape)
            new_data = np.zeros(out_shape, dtype=np.float32)
            new_data[mask>0] = vec
            nib.Nifti1Image(new_data, _img.get_affine()
                    #header=_img.get_header()).to_filename(filename)
                    ).to_filename(filename)

        save_scalars(voxel_fits, self.regression_output)
        logger.info("Wrote r^2 results to %s", self.regression_output)
        save_vectors(retroicorrected, self.denoised_output)
        logger.info("Wrote physio-corrected 4D nifti to %s", self.denoised_output)
        
        if self.write_4d_nuisance_for_each_modality:
            cardiac_out = self.nuisance_output_prefix + "_Cardiac_fits.nii.gz"
            save_vectors(cardiac_fits, cardiac_out)
            logger.info("Wrote Cardiac fit to %s" % cardiac_out)
            
            resp_out = self.nuisance_output_prefix + "_Resp_fits.nii.gz"
            save_vectors(resp_fits, resp_out)
            logger.info("Wrote Respiration fit to %s" % resp_out)
            
            ix_out = self.nuisance_output_prefix + "_RCInteraction_fits.nii.gz"
            save_vectors(ix_fits, ix_out)
            logger.info("Wrote Resp/Cardiac interaction fit to %s" % ix_out)
            
            drift_out = self.nuisance_output_prefix + "_drift_fits.nii.gz"
            save_vectors(drift_fits, drift_out)
            logger.info("Wrote drift model fit to %s" % drift_out)
            
        if self.write_4d_nuisance:
            all_out = self.nuisance_output_prefix + "_Nuisance_fits.nii.gz"
            save_vectors(fitts, all_out)
            logger.info("Wrote omnibus nuisance fit to %s" % all_out)
            
        assert np.all(np.array(_img.shape) == np.array(nib.load(self.denoised_output).shape))
    

    def get_mea_regressors(self):
        # Get mri time info
        _img = nib.load(self.fmri_file)
        ntrs = _img.shape[-1]
        tr = _img.get_header().get_zooms()[-1]
        # ACQ times of beats and  trigger times
        ntriggers = len(self.tr_onsets)
        if ntriggers > ntrs:
            logger.warn("More triggers in physiodata than volumes in fmri file")
            wanted_t = self.tr_onsets[:ntrs]
        elif ntriggers < ntrs:
            logger.warn("More volumes in fmri file than triggers in physiodata")
            missing_trs = ntrs - ntriggers
            wanted_t = np.concatenate([self.tr_onsets,
                self.tr_onsets[-1] + tr*(np.arange(missing_trs)+1)])
        else:
            wanted_t = self.tr_onsets
        wanted_t = wanted_t + tr/2

        t = self.physiodata.peak_times
        models = {}
        for cvi in ["hr", "pep","lvet","sv_k_rc","co_k_rc"]:
            # Get the signal
            try:
                x = getattr(self.physiodata, cvi)
            except AttributeError:
                continue
            terp = interp1d(t,x,kind="linear", bounds_error=False)
            model = winsorize(terp(wanted_t),limits=(0.01,0.01))
            centered = np.array((model - np.nanmean(model))/np.nanstd(model))
            models[cvi] = centered
        df = pd.DataFrame(models)
        assert df.shape[0] == ntrs
        df["ntrs"] = ntrs
        return df

