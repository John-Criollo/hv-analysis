import numpy as np
import obspy

# This is an implementation of the sta-lta window rejection 
components = ["E","N","Z"]

def sta_lta_window_rejection(records,
                             sta_seconds = 1,
                             lta_seconds = 30,
                             min_sta_lta_ratio = 0.2,
                             max_sta_lta_ratio = 2.5,
                             components = ["E","N","Z"]):
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
    valid_window_boolean_mask = []
    for record in records:
        for component in components:
            matching_trace = record.select(component = component)
            if not matching_trace:
                continue

            trace = matching_trace[0]
            n_samples = trace.stats.npts
            dt = trace.stats.delta
            total_duration = (n_samples - 1) * dt

            # Now we compute sta values.
            npts_in_sta = int(sta_seconds // record.stats.delta)
            if npts_in_sta > n_samples:
                msg = "sta_seconds must be shorter than record length;"
                msg += f"sta_seconds is {sta_seconds} and "
                msg += f"record length is {total_duration}."
                raise IndexError(msg)
            
            n_sta_in_window = int(n_samples // npts_in_sta)
            short_timeseries = trace.data[:npts_in_sta*n_sta_in_window]
            sta_values = np.mean(np.abs(short_timeseries.reshape(
                (n_sta_in_window,npts_in_sta))), axis = 1)
            
            #Now we compute lta
            npts_in_lta =   int(lta_seconds // dt)
            if npts_in_lta > n_samples:
                msg = "lta_seconds must be shorter than record length; "
                msg += f"lta_seconds is {lta_seconds} and "
                msg += f"record length is {total_duration}."
                raise IndexError(msg)
            lta = np.mean(np.abs(short_timeseries[:npts_in_lta]))

            #Now we check the min and max sta-lta ratio
            ratios = sta_values / (lta + 1e-10) # We add a small epsilon so we dont divide by 0
            if (np.max(ratios) > max_sta_lta_ratio) or (np.min(ratios) < min_sta_lta_ratio):
                valid_window_boolean_mask.append(False)
                break
            else:
                # Executes only if the loop completed without hitting 'break'
                valid_window_boolean_mask.append(True)
                passing_records.append(record)
    return passing_records