#!/usr/bin/env python
from scipy.signal import convolve
import numpy as np
from skimage.filters import threshold_otsu
from scipy.signal import filtfilt
from scipy.signal import butter, decimate
from scipy.interpolate import interp1d
from scipy.special import legendre
from collections import defaultdict


    

def zero_crossing_index(data):
    crossings = np.flatnonzero(np.diff(np.signbit(data)))
    if len(crossings) == 0:
        return 0
    return crossings[-1]

def normalize(ts):
    return (ts-ts.min())/(ts.max() - ts.min())

def legendre_detrend(signal,polort):
    X = np.vstack([ \
        legendre(x)(
            np.linspace(-1,1,signal.shape[0])) for x in range(polort)
        ]
    ).T
    fit = np.linalg.lstsq(X,signal)
    
    return signal - np.dot(X,fit[0])

def regress_out(signal, noise):
    if signal.shape[0] < noise.shape[0]:
        noise = noise[:signal.shape[0]]
    elif signal.shape[0] > noise.shape[0]:
        noise = np.hstack([noise,
                 noise.mean()*np.ones(signal.shape[0] - noise.shape[0])])
        
    noise = noise-noise.mean()
    intercept = signal.mean()
    centered = signal - intercept
    
    fit = np.linalg.lstsq(noise[np.newaxis].T, centered)
    
    return centered - np.dot(noise[np.newaxis].T,fit[0]) + intercept

def censor_peak_times(censor_regions, peak_times):
    """ Takes a list of peak times and returns a
    binary mask where only times outside of censor regions
    are True
    """
    ok_indices =np.array([True] * len(peak_times)) 
    for censor_region in censor_regions:
        ok_indices[
            (peak_times >= censor_region.start_time) & 
            (peak_times <= censor_region.end_time)
            ] = False
        
    return ok_indices

def find_peaks(data, min_dist_samples=-1, maxima=True, minima=False):
    
    diff = np.ediff1d(data,to_begin=0)
    diff2 = np.ediff1d(diff,to_begin=0)
    diff2[:10] = 0
    peak_indices = np.flatnonzero(( diff[:-1] > 0 ) & (diff[1:] < 0 ) | ( diff[:-1] < 0 ) & (diff[1:] > 0 ))
    peak_amps = data[peak_indices]
    
    concavities = diff2[peak_indices]
    mins = peak_indices[concavities > 0]
    maxs = peak_indices[concavities < 0]
    if maxima and minima:
        return mins, maxs
    if maxima:
        return maxs
    if minima:
        return mins
    
    if apply_otsu:
        peak_amp_thr = threshold_otsu(peak_amps)
        peak_indices = peak_indices[ peak_amps > peak_amp_thr ]
    
    return peak_indices

def min_separated_times(times, min_sep):
    last_time = times[0]
    ok_indices = [0]
    
    for i,t in enumerate(times):
        if t - last_time > min_sep:
            last_time = t
            ok_indices.append(i)
    return np.array(ok_indices)

def times_contained_in(times, intervals):
    ok_mask = np.zeros(times.shape,dtype=np.bool)
    start,end = intervals.T
    for i,t in enumerate(times):
        ok_mask[i] = np.any((t >= start) & (t<=end))
    return ok_mask




def downsample(signal,orig_sampling_rate, 
               new_sampling_rate,ftype="iir",fwidth=1):
    """
    Downsamples the data to a new sampling rate. Useful for densely acquired
    signal
    """
    
    if new_sampling_rate > orig_sampling_rate:
        print "***WARNING:: UPSAMPLING***"
        return upsample(signal,orig_sampling_rate,new_sampling_rate)
    
    downsampling_factor = int(orig_sampling_rate / new_sampling_rate)
    return decimate(signal, downsampling_factor, n=fwidth, ftype=ftype)

def upsample(signal,orig_sampling_rate, new_sampling_rate):
    orig_times = np.arange(len(signal)) / float(orig_sampling_rate)
    new_times = np.linspace(0, orig_times[-1], len(signal)*10,endpoint=True)
    f = interp1d(orig_times,signal,assume_sorted=True,copy=False)
    return f(new_times)

def smooth(x,window_len=11,window='hanning'):
    """included in scipy documentation"""
    if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
            w=np.ones(window_len,'d')
    else:
            w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]
    
def bandpass(data, low_cutoff, high_cutoff, sampling_rate_hz):
    nyquist_freq =  sampling_rate_hz / 2.
    low = low_cutoff / nyquist_freq
    high = high_cutoff / nyquist_freq
    [b, a] = butter(4, [low, high], btype='band')
    
    return filtfilt(b, a, data)

def morlet(frequencies, scale, k0=6., 
           max_scale=300, n_scales=1000*0.17*2):
    # From wt.bases
    n=len(frequencies)
    expnt = -(scale * frequencies - k0) ** 2 / 2 * (frequencies > 0)
    norm = np.sqrt(scale * frequencies[1]) * (np.pi ** (-0.25)) * np.sqrt(n)
    daughter = norm * np.exp(expnt)
    daughter = daughter* (frequencies > 0)
    fourier_factor = 4 * np.pi / (k0 + np.sqrt(2+k0**2))
    coi=fourier_factor / np.sqrt(2)
    return {
        "fourier_factor":fourier_factor,
        "daughter":daughter,
        "coi":coi,
        "dof":2
    }
    
    
    
def lowpass(data, max_freq, sampling_rate):
    # Convert cutoff into  proportion of the Nyquist freq.
    nyquist_freq = sampling_rate / 2.
    cutoff_freq = max_freq / nyquist_freq
    
    b,a = butter(4, cutoff_freq)
    
    return filtfilt(b,a,data,padlen=200)
