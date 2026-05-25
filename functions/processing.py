import numpy as np
from obspy.signal import konnoohmachismoothing
from numpy.fft import rfft,rfftfreq


def nextpow2(n, minimum_power_of_two=2**15):  # 2**15 = 32768
    power_of_two = minimum_power_of_two
    while True:
        if power_of_two > n:
            return power_of_two
        power_of_two *= 2
def total_horizontal_energy(ns, ew, settings=None):
    """Computes the magnitude of sum of two orthoginal vectors."""
    return np.sqrt((ns*ns + ew*ew))

def hvcurves(data):
    '''
    Computing the smoothing matrix takes a while so I will look for a way of storing it as a variable later.
    '''
    bwidth = 40
    max_n_samples = data[0][-1].stats.npts
    good_n = nextpow2(max_n_samples)
    dt = data[0][-1].stats.npts
    fft_frq = rfftfreq(good_n, dt)
    smoothMatrix = konnoohmachismoothing.calculate_smoothing_matrix(fft_frq, bandwidth= bwidth, normalize = True)
    waveforms = []
    for window in data:
        fft_ns = np.abs(rfft(window[1].data, good_n))
        fft_ew = np.abs(rfft(window[0].data, good_n))
        h = total_horizontal_energy(fft_ns, fft_ew)
        v = np.abs(rfft(window[2].data, good_n))
        #SMOOTHING
        h_smooth = konnoohmachismoothing.apply_smoothing_matrix(h,smoothMatrix, count = 1)
        v_smooth = konnoohmachismoothing.apply_smoothing_matrix(v, smoothMatrix, count = 1)
        hvsr = h_smooth/v_smooth
        waveforms.append(hvsr)
    return waveforms