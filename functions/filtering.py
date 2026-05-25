import numpy as np
#import obspy
from scipy.signal import welch, butter, sosfiltfilt

# This is an implementation of the sta-lta window rejection 
components = ["E","N","Z"]

def sta_lta_window_rejection(records = [],
                             sta_seconds=1,
                             lta_seconds=30,
                             min_sta_lta_ratio=0.2,
                             max_sta_lta_ratio=2.5,
                             components=["E", "N", "Z"]):
    '''
    This code is just an implementation of the code made by Joseph Vantassel.
    All credits for him, https://github.com/jpvantassel/hvsrpy/ . 
    This function will return an array of the three components with the valid windows.
    The variables we will use are:
    -records: (Not sure) Here we will expect an array of cutted obspy objects .
    -sta_seconds: The window size of the short term average.
    -lta_seconds: The window size of the long term average.
    - min/max sta_lta_ratio: These are the thresholds criteria for rejecting a bad window.
    -components : An array of the components that are going to be analyzed.
    -dt_in_seconds: Our delta of T.
    -hvsr: This is a parameter that can be added if the stations are not aligned with the North direction.
    '''
    passing_records = []

    for record in records:
        record_is_valid = True  # assume good until a component fails

        for component in components:
            matching_trace = record.select(component=component)
            if not matching_trace:
                continue

            trace = matching_trace[0]
            n_samples = trace.stats.npts
            dt = trace.stats.delta
            total_duration = (n_samples - 1) * dt

            npts_in_sta = int(sta_seconds // dt)
            if npts_in_sta > n_samples:
                raise IndexError(
                    f"sta_seconds ({sta_seconds}) must be shorter than "
                    f"record length ({total_duration})."
                )

            n_sta_in_window = int(n_samples // npts_in_sta)
            short_timeseries = trace.data[:npts_in_sta * n_sta_in_window]
            sta_values = np.mean(
                np.abs(short_timeseries.reshape((n_sta_in_window, npts_in_sta))),
                axis=1
            )

            npts_in_lta = int(lta_seconds // dt)
            if npts_in_lta > n_samples:
                raise IndexError(
                    f"lta_seconds ({lta_seconds}) must be shorter than "
                    f"record length ({total_duration})."
                )
            lta = np.mean(np.abs(short_timeseries[:npts_in_lta]))

            ratios = sta_values / (lta + 1e-10)
            if np.max(ratios) > max_sta_lta_ratio or np.min(ratios) < min_sta_lta_ratio:
                record_is_valid = False
                break  # no need to check remaining components

        if record_is_valid:
            passing_records.append(record)  # appended exactly once per record

    return passing_records

def reject_low_frequency_windows(windows = [],
                                 lf_cutoff = 1.0,
                                 power_ratio_threshold = 0.30,
                                  amplitude_ratio_threshold = 1.2):
    horizontal_channels= ["HHN", "HHE"]
    vertical_channel = "HHZ"
    # Lowpass filter to isolate the slow oscillation for amplitude check
    def lowpass_envelope(data, fs, cutoff=0.5, order=4):
        sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
        return sosfiltfilt(sos, data)

    accepted = []

    for stream in windows:
        rejected = False

        z_tr   = stream.select(channel=vertical_channel)[0]
        z_data = z_tr.data
        fs     = z_tr.stats.sampling_rate

        # Lowpass Z for fair comparison
        z_smooth = lowpass_envelope(z_data, fs)
        z_ptp    = np.ptp(z_smooth)

        for channel in horizontal_channels:
            tr   = stream.select(channel=channel)[0]
            data = tr.data

            # --- Power ratio check ---
            nperseg = min(len(data) // 4, 1024)
            freqs, psd = welch(data, fs=fs, nperseg=nperseg, window="hann")
            lf_mask     = freqs < lf_cutoff
            total_power = np.trapezoid(psd, freqs)
            lf_power    = np.trapezoid(psd[lf_mask], freqs[lf_mask])
            ratio = lf_power / total_power if total_power > 0 else 1.0

            if ratio > power_ratio_threshold:
                rejected = True
                break

            # --- Smoothed amplitude ratio check ---
            h_smooth = lowpass_envelope(data, fs)
            h_ptp    = np.ptp(h_smooth)

            if z_ptp > 0 and (h_ptp / z_ptp) > amplitude_ratio_threshold:
                rejected = True
                break

        if not rejected:
            accepted.append(stream)

    return accepted