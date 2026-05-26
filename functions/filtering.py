import numpy as np
#import obspy
from scipy.signal import welch, butter, sosfiltfilt
from obspy.signal.trigger import classic_sta_lta

# This is an implementation of the sta-lta window rejection 
def sta_lta_window_rejection(records,
                             sta_seconds=1,
                             lta_seconds=20,
                             min_sta_lta_ratio=0.5,
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

    for i, record in enumerate(records):
        record_is_valid = True

        for component in components:
            matching_trace = record.select(component=component)
            if not matching_trace:
                # component missing entirely — skip or flag depending on your needs
                continue

            trace = matching_trace[0]
            dt    = trace.stats.delta
            npts  = trace.stats.npts

            npts_sta = int(sta_seconds / dt)
            npts_lta = int(lta_seconds / dt)

            # Guard: window must be longer than LTA
            if npts_lta >= npts:
                raise ValueError(
                    f"Window {i}: lta_seconds ({lta_seconds}s) must be shorter "
                    f"than the window length ({npts * dt:.1f}s)."
                )

            # --- sliding STA/LTA via ObsPy ---
            cft = classic_sta_lta(trace.data, npts_sta, npts_lta)

            # Skip warm-up zone where LTA isn't fully populated
            cft_valid = cft[npts_lta:]

            if cft_valid.size == 0:
                record_is_valid = False
                break

            cft_max = cft_valid.max()
            cft_min = cft_valid.min()

            if cft_max > max_sta_lta_ratio or cft_min < min_sta_lta_ratio:
                record_is_valid = False
                break

        if record_is_valid:
            passing_records.append(record)

    #print(f"Kept {len(passing_records)} / {len(records)} windows")
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

def iterative_hvsr_rejection(hvsr_curves, frequencies, n=2, max_iterations=50,
                             f_min=0.5, f_max=20.0):
    hvsr_curves = np.asarray(hvsr_curves)

    freq_mask = (frequencies >= f_min) & (frequencies <= f_max)
    if not np.any(freq_mask):
        raise ValueError(f"No frequencies found between {f_min} and {f_max} Hz.")

    masked_freqs = frequencies[freq_mask]

    masked_curves_all = hvsr_curves[:, freq_mask]
    peak_indices_all  = np.argmax(masked_curves_all, axis=1)
    f0_all            = masked_freqs[peak_indices_all]

    valid_start = (f0_all >= f_min) & (f0_all <= f_max)
    print(f"Pre-filter: {valid_start.sum()} / {len(hvsr_curves)} windows have f0 in [{f_min}, {f_max}] Hz")

    active_indices = valid_start.copy()

    for iteration in range(max_iterations):
        current_curves  = masked_curves_all[active_indices]  # use masked freqs only
        if len(current_curves) == 0:
            raise ValueError("All windows were rejected.")

        # --- BOX 1 ---
        peak_indices_b = np.argmax(current_curves, axis=1)
        f0_array_b     = masked_freqs[peak_indices_b]
        ln_f0_b        = np.log(f0_array_b)   

        mu_b    = np.mean(ln_f0_b)
        sigma_b = np.std(ln_f0_b)
        LM_b    = np.exp(mu_b)

        median_curve_b = np.median(current_curves, axis=0)
        f0_mc_b        = masked_freqs[np.argmax(median_curve_b)]
        d_b            = np.abs(LM_b - f0_mc_b)

        # --- BOX 2 ---
        lower_bound = np.exp(mu_b - n * sigma_b)
        upper_bound = np.exp(mu_b + n * sigma_b)

        current_active_idx  = np.where(active_indices)[0]
        new_active_indices  = np.zeros(hvsr_curves.shape[0], dtype=bool)

        for idx, f0_i in zip(current_active_idx, f0_array_b):
            if lower_bound <= f0_i <= upper_bound:
                new_active_indices[idx] = True

        # --- BOX 3 ---
        passed_curves = masked_curves_all[new_active_indices]
        if len(passed_curves) == 0:
            raise ValueError("All windows rejected during iteration.")

        peak_indices_e = np.argmax(passed_curves, axis=1)
        f0_array_e     = masked_freqs[peak_indices_e]
        ln_f0_e        = np.log(f0_array_e)

        mu_e    = np.mean(ln_f0_e)
        sigma_e = np.std(ln_f0_e)
        LM_e    = np.exp(mu_e)

        median_curve_e = np.median(passed_curves, axis=0)
        f0_mc_e        = masked_freqs[np.argmax(median_curve_e)]
        d_e            = np.abs(LM_e - f0_mc_e)

        # --- BOX 4 ---
        if d_b == 0:
            cond_d = (d_e == 0)
        else:
            cond_d = (np.abs(d_e - d_b) / d_b) < 0.01

        cond_sigma = np.abs(sigma_e - sigma_b) < 0.01

        active_indices = new_active_indices

        print(f"Iter {iteration+1}: kept {active_indices.sum()} windows | "
              f"f0 bounds [{lower_bound:.3f}, {upper_bound:.3f}] Hz | "
              f"sigma: {sigma_b:.4f}→{sigma_e:.4f} | "
              f"d: {d_b:.4f}→{d_e:.4f}")

        if cond_d and cond_sigma:
            print(f"Converged at iteration {iteration+1}.")
            break

    return hvsr_curves[active_indices]