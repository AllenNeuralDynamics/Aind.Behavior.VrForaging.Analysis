"""
Created on Fri Jun 23 10:39:15 2023

@author: tiffany.ona
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import butter, filtfilt, lfilter, iirnotch, find_peaks, savgol_filter


def moving_average(data, window_size):
    """
    Create a moving average of the data

    Parameters
    ----------
    data : pd.DataFrame
        dataframe that contains the signal data.
    window_size : int
        size of the window to apply the moving average.

    Returns
    -------
    smoothed_data : pd.DataFrame
        dataframe with the smoothed data.
    """

    window = np.ones(int(window_size)) / float(window_size)
    smoothed_data = np.convolve(data, window, "same")
    return smoothed_data


def apply_notch_filter(data, f_notch=60, Q=200, fs=250):
    """
    Apply a notch filter to remove 60 Hz noise

    Parameters
    ----------
    data : pd.DataFrame
        dataframe that contains the signal data.
    f_notch : int, optional
        frequency to apply the notch filter. The default is 60.
    Q : int, optional
        Q factor. The default is 200.

    Returns
    -------
    data : pd.DataFrame
        dataframe with the filtered data.
    """
    # Generate sample signal
    b, a = iirnotch(f_notch, Q, fs)

    # Apply the notch filter to the signal
    data = lfilter(b, a, data)

    return data


def findpeaks_and_plot(
    data,
    x,
    fig,
    range_plot=[120, 121],
    color="black",
    height=1.1,
    prominence=2.5,
    distance=30,
):
    """

    Parameters
    ----------
    data : pd.DataFrame
        dataframe that contains the signal data.
    x : pd.Series
        timing in seconds.
    fig : figure
        where things will be plotted.
    range_plot : list, optional
        The default is [120,121].indicates the range that wants to be displayed
    color : TYPE, optional
        The default is 'black'.
    label:  TYPE, optional
        The default is 'Pressure sensor'.

    The following are parameters related with finding the peaks
        height : TYPE, optional
            The default is 1.1.
        prominence : TYPE, optional
            The default is 2.5.
        distance : TYPE, optional
            The default is 30.


    Returns
    -------
    df_peaks : dataframe
        returns a dataframe with the peaks, amplitude, index_location and frequencies.

    """
    # ------   Find peaks for the pressure sensor
    peaks, properties = find_peaks(
        data, height=height, prominence=prominence, distance=distance
    )
    
    df_peaks = pd.DataFrame()
    df_peaks["heights_peaks"] = properties["peak_heights"]
    df_peaks["locations_peaks"] = x[peaks].values

    # Calculate the frequency per second of the peaks
    time_diffs = np.diff(df_peaks.locations_peaks)  # Time differences between consecutive troughs
    instantaneous_frequency = 1 / time_diffs  # Frequency in Hz
    df_peaks["instantaneous_frequency"] = np.insert(instantaneous_frequency, 0, 0)  # Insert 0 for the first value

    # ------   Find troughs for the pressure sensor
    inverted_data = -data
    troughs, trough_properties = find_peaks(
        inverted_data, height=height, prominence=prominence, distance=distance
    )

    # Create DataFrame for valleys
    df_troughs = pd.DataFrame()
    df_troughs["depths_troughs"] = -trough_properties["peak_heights"]
    df_troughs["locations_troughs"] = x[troughs].values
    
    # Calculate the frequency per second of the troughs
    time_diffs = np.diff(df_troughs.locations_troughs)  # Time differences between consecutive troughs
    instantaneous_frequency = 1 / time_diffs  # Frequency in Hz
    # # Assign the frequency values to the midpoint of the intervals
    # midpoints = df_troughs.locations_troughs[:-1] + time_diffs / 2
    df_troughs["instantaneous_frequency"] = np.insert(instantaneous_frequency, 0, 0)# Insert 0 for the first value

    # Plot the curve and peaks
    plt.plot(x, data, color=color, linewidth=0.7)
    plt.plot(df_peaks["locations_peaks"], properties["peak_heights"], "x", color=color)   
    plt.plot(df_troughs["locations_troughs"], df_troughs["depths_troughs"], "x", color='crimson')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Units")
    plt.xlim(range_plot)
    sns.despine()

    return df_peaks, df_troughs


def plot_FFT(data, fs=1000, color="black", label="test"):

    # Compute power spectrum for sensor
    signal_fft = np.fft.fft(data)
    power_spectrum = np.abs(signal_fft)

    # Frequency axis
    freq_axis = np.fft.fftfreq(len(data), 1 / fs)
    power_spectrum = moving_average(power_spectrum, window_size=100)

    n = len(data)
    # Plotting
    plt.semilogy(freq_axis, power_spectrum, color, label=label)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectrum")
    plt.title("Power Spectrum of the Signal")
    plt.xlim(-0.01, 125)
    plt.grid(True)
    sns.despine()


# Define a high-pass filter
def highpass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def lowpass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def butterworth_bandpass(data, lowcut=1.0, highcut=125.0, fs=1000.0, order=5):
    """
    Apply a Butterworth bandpass filter to the data.

    Parameters:
    - data: array-like, the input signal.
    - lowcut: float, the lower cutoff frequency of the bandpass filter.
    - highcut: float, the upper cutoff frequency of the bandpass filter.
    - fs: float, the sampling rate of the signal.
    - order: int, the order of the filter.

    Returns:
    - filtered_data: array-like, the filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def notch_filter(data, freq, fs=1000.0, quality_factor=30):
    nyquist = 0.5 * fs
    notch_freq = freq / nyquist
    b, a = iirnotch(notch_freq, quality_factor)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def filtering_standard(breathing, set_moving_average=False):
    # Sample breathing signal (replace with your actual data)
    time_diffs = breathing.index.to_series().diff().dropna().mean()
    sampling_interval = time_diffs.mean()
    fs = 1 / sampling_interval
    fs = np.round(fs, 2)

    # Apply Butterworth bandpass filter
    bandpassed_signal = butterworth_bandpass(breathing.data, lowcut=1.0, highcut=100.0, fs=fs, order=3)

    # Apply notch filter at 60 Hz
    final_filtered_signal = notch_filter(bandpassed_signal, freq=60, fs=fs, quality_factor=30)

    # Apply notch filter at 100 Hz
    # final_filtered_signal = bs.notch_filter(notched_signal_60, freq=120, fs=fs, quality_factor=30)

    # Apply Savitzky-Golay filter to smooth the signal
    smoothed_signal = savgol_filter(final_filtered_signal, window_length=35, polyorder=2)
    breathing['filtered_data'] = smoothed_signal

    if set_moving_average:
        slow_ther = moving_average(smoothed_signal, window_size=75)
        breathing['filtered_data'] = smoothed_signal-slow_ther
    return breathing

def plot_sniff_raster_simple(test_df, axes1, axes2, 
                           color, 
                           window: list = [-1, 5], 
                           range_step: float= 0.05, 
                           max_trial: int = 0, 
                           align: str = 'times'):
    test_df['new_trial'] = pd.factorize(test_df['total_sites'])[0]
    x= test_df[align]    
    y= test_df.new_trial + max_trial
    axes1.plot(x, y, 'o', color=color, markersize=1, marker='.')
    axes1.set_ylabel('Odor sites')
    
    # time_bins = np.arange(window[0], window[1], range_step)
    # heights, bin_edges, _ = axes2.hist(x, bins=time_bins, color=color, alpha=0.5, histtype='step', 
    #         edgecolor=color, weights=np.ones(len(x)) /  test_df.new_trial.nunique()/range_step)
    
    # Define overlapping bins
    overlap = range_step/2
    time_bins = np.arange(window[0], window[1] + range_step, overlap)

    # Calculate histogram values for each bin
    heights = []
    for i in range(len(time_bins) - 1):
        bin_start = time_bins[i]
        bin_end = time_bins[i] + range_step
        bin_count = ((x >= bin_start) & (x < bin_end)).sum()
        heights.append(bin_count / test_df.new_trial.nunique() / range_step)

    # # Plot the histogram with overlapping bins
    # axes2.plot(time_bins[:-1], heights, color=color, alpha=0.5, drawstyle='steps-post')

    # Apply rolling average
    heights_series = pd.Series(heights)
    heights_smoothed = heights_series.rolling(window=2, center=True).mean()
    axes2.plot(time_bins[:-1], heights_smoothed, color=color)

    axes2.set_ylabel('Frequency (Hz)')
    axes2.set_xlim(window[0], window[1])
    
def plot_sniff_raster_conditioned(raster, 
                                  velocity, 
                                  frequency_troughs: pd.DataFrame = None, 
                                  save = False):
    condition = 'is_choice'
    colors = ['crimson', 'steelblue']

    fig, ax = plt.subplots(3,4, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    axes1, axes2, axes4 = ax[0][0], ax[1][0], ax[2][0]
    for axes in ax.flatten():
        axes.vlines(0, 0, 1, transform=axes.get_xaxis_transform(), color='black', alpha=0.5, linewidth=0.5)
        
    raster_1 = raster.loc[(raster[condition] == 1)]
    raster_2 = raster.loc[(raster[condition] == 0)]
    color1 = colors[1]
    color2 = colors[0]

    plot_sniff_raster_simple(raster_1, axes1, axes2, color = color1)
    plot_sniff_raster_simple(raster_2, axes1, axes2, color = color2, max_trial = raster_1.total_sites.nunique()+5)

    # sns.lineplot(data=frequency_troughs, x='times', y='instantaneous_frequency', hue=condition, ax=axes3, palette= colors, legend=False)
    # axes3.set_ylabel('Frequency (Hz)')
    sns.lineplot(data=velocity, x='times', y='speed', hue=condition, ax=axes4, errorbar='sd', palette= colors)
    axes4.set_xlabel('Time from odor onset (s)')
    axes4.set_ylabel('Velocity (cm/s)')
    axes4.legend(loc='upper right')
    axes1.set_title('Stopped')

    condition = 'is_reward'
    colors = ['crimson', 'steelblue']

    axes1, axes2, axes4 = ax[0][1], ax[1][1], ax[2][1]
    for axes in ax.flatten():
        axes.vlines(0, 0, 1, transform=axes.get_xaxis_transform(), color='black', alpha=0.5, linewidth=0.5)
        
    raster_1 = raster.loc[(raster[condition] == 1)&(raster['is_choice']==1)]
    raster_2 = raster.loc[(raster[condition] == 0)&(raster['is_choice']==1)]
    color1 = colors[1]
    color2 = colors[0]

    plot_sniff_raster_simple(raster_1, axes1, axes2, color = color1)
    plot_sniff_raster_simple(raster_2, axes1, axes2, color = color2, max_trial = raster_1.total_sites.nunique()+5)

    # sns.lineplot(data=frequency_troughs, x='times', y='instantaneous_frequency', hue=condition, ax=axes3, palette= colors, legend=False)
    # axes3.set_ylabel('Frequency (Hz)')

    sns.lineplot(data=velocity.loc[(velocity['is_choice']==1)], x='times', y='speed', hue=condition, ax=axes4, errorbar='sd', palette= colors)
    axes4.set_xlabel('Time from odor onset (s)')
    axes4.set_ylabel('Velocity (cm/s)')
    axes4.legend(loc='upper right')
    axes1.set_title('Reward delivered')

    condition = 'site_number'
    colors = ['grey', 'black']

    axes1, axes2,  axes4 = ax[0][2], ax[1][2], ax[2][2]
    for axes in ax.flatten():
        axes.vlines(0, 0, 1, transform=axes.get_xaxis_transform(), color='black', alpha=0.5, linewidth=0.5)
        
    raster_1 = raster.loc[(raster[condition] == 0)&(raster['is_choice']==1)]
    raster_2 = raster.loc[(raster[condition] != 0)&(raster['is_choice']==1)]
    color1 = colors[1]
    color2 = colors[0]

    plot_sniff_raster_simple(raster_1, axes1, axes2, color = color1)
    plot_sniff_raster_simple(raster_2, axes1, axes2, color = color2, max_trial = raster_1.total_sites.nunique()+5)

    # sns.lineplot(data=frequency_troughs.loc[(frequency_troughs[condition] == 0)&(frequency_troughs['is_choice']==0)], x='times', y='instantaneous_frequency', ax=axes3, palette= colors, legend=False)
    # sns.lineplot(data=frequency_troughs.loc[(frequency_troughs[condition] != 0)&(frequency_troughs['is_choice']==0)], x='times', y='instantaneous_frequency', ax=axes3, palette= colors, legend=False)
    # axes3.set_ylabel('Frequency (Hz)')

    sns.lineplot(data=velocity.loc[(velocity[condition] == 0)&(velocity['is_choice']==1)], x='times', y='speed', ax=axes4, color=colors[0], errorbar='sd', legend=False, label='Visit 1')
    sns.lineplot(data=velocity.loc[(velocity[condition] != 0)&(velocity['is_choice']==1)], x='times', y='speed', ax=axes4, color=colors[1], errorbar='sd', legend=False, 
                 label='Other visits')
    axes4.set_xlabel('Time from odor onset (s)')
    axes4.set_ylabel('Velocity (cm/s)')
    axes4.legend(loc='upper right')
    axes1.set_title('Stopped - Visit number')
    
    condition = 'site_number'
    colors = ['grey', 'black']

    axes1, axes2, axes4 = ax[0][3], ax[1][3], ax[2][3]
    for axes in ax.flatten():
        axes.vlines(0, 0, 1, transform=axes.get_xaxis_transform(), color='black', alpha=0.5, linewidth=0.5)
        
    raster_1 = raster.loc[(raster[condition] == 0)&(raster['is_choice']==0)]
    raster_2 = raster.loc[(raster[condition] != 0)&(raster['is_choice']==0)]
    color1 = colors[1]
    color2 = colors[0]

    plot_sniff_raster_simple(raster_1, axes1, axes2, color = color1)
    plot_sniff_raster_simple(raster_2, axes1, axes2, color = color2, max_trial = raster_1.total_sites.nunique()+5)

    # sns.lineplot(data=frequency_troughs.loc[(frequency_troughs[condition] == 0)&(frequency_troughs['is_choice']==0)], x='times', y='instantaneous_frequency', ax=axes3, palette= colors, legend=False)
    # sns.lineplot(data=frequency_troughs.loc[(frequency_troughs[condition] != 0)&(frequency_troughs['is_choice']==0)], x='times', y='instantaneous_frequency', ax=axes3, palette= colors, legend=False)
    # axes3.set_ylabel('Frequency (Hz)')

    sns.lineplot(data=velocity.loc[(velocity[condition] == 0)&(velocity['is_choice']==0)], x='times', y='speed', ax=axes4, color=colors[0], errorbar='sd', legend=False, label='Visit 1')
    sns.lineplot(data=velocity.loc[(velocity[condition] != 0)&(velocity['is_choice']==0)], x='times', y='speed', ax=axes4, color=colors[1], errorbar='sd', legend=False, 
                 label='Last visit')

    axes4.set_xlabel('Time from odor onset (s)')
    axes4.set_ylabel('Velocity (cm/s)')
    axes4.legend(loc='upper right', title='Visit number')
    axes1.set_title('Not Stopped - Different visit number')
    
    mouse = raster.mouse.unique()[0]
    session = raster.session.unique()[0]
    plt.suptitle(f'{mouse} - {session}', y=1.05)
    
    sns.despine()
    plt.tight_layout()
    
    if save:
        save.savefig(fig)
        
    # plt.show()
    return fig

    


def plot_sniff_raster_odor_conditioned(raster, 
                                  velocity, 
                                  save = False):
    condition = 'is_choice'
    colors = ['steelblue', 'black', 'crimson',  'orangered']

    fig, ax = plt.subplots(3,3, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    for axes in ax.flatten():
        axes.vlines(0, 0, 1, transform=axes.get_xaxis_transform(), color='black', alpha=0.5, linewidth=0.5)
        
    for i, odor_label in enumerate(raster.odor_label.unique()):
        axes1, axes2, axes4 = ax[0][i], ax[1][i], ax[2][i]
            
        raster_1 = raster.loc[(raster['odor_label'] == odor_label)&(raster[condition] == 1)&(raster['site_number'] == 0)]
        raster_2 = raster.loc[(raster['odor_label'] == odor_label)&(raster[condition] == 1)&(raster['site_number'] != 0)]
        raster_3 = raster.loc[(raster['odor_label'] == odor_label)&(raster[condition] == 0)&(raster['site_number'] == 0)]
        raster_4 = raster.loc[(raster['odor_label'] == odor_label)&(raster[condition] == 0)&(raster['site_number'] != 0)]
        
        color1 = colors[0]
        color2 = colors[1]
        color3 = colors[2]
        color4 = colors[3]
        
        plot_sniff_raster_simple(raster_1, axes1, axes2, color = color1)
        running_max_trial = raster_1.total_sites.nunique()+5

        plot_sniff_raster_simple(raster_2, axes1, axes2, color = color2, max_trial = running_max_trial)
        running_max_trial += raster_2.total_sites.nunique()+5
        
        plot_sniff_raster_simple(raster_3, axes1, axes2, color = color3, max_trial = running_max_trial)
        running_max_trial += raster_3.total_sites.nunique()+5
        
        plot_sniff_raster_simple(raster_4, axes1, axes2, color = color4, max_trial = running_max_trial)

        axes2.set_ylim(0, 10)
        sns.lineplot(data=velocity.loc[(velocity.odor_label == odor_label)&(velocity['site_number'] == 0)], x='times', y='speed', hue=condition, ax=axes4, errorbar='sd', palette= {0: color3, 1: color1})
        sns.lineplot(data=velocity.loc[(velocity.odor_label == odor_label)&(velocity['site_number'] != 0)], x='times', y='speed', hue=condition, ax=axes4, errorbar='sd', palette= {0: color4, 1: color2})
        axes4.set_xlabel('Time from odor onset (s)')
        axes4.set_ylabel('Velocity (cm/s)')
        
        handles, labels = axes4.get_legend_handles_labels()
        new_labels = ['Non-stop first visit', 'Stop first visit', 'Non-stop subsequent visits', 'Stop subsequent visits']
        new_colors = [color3, color1, color4, color2]
        for handle, color in zip(handles, new_colors):
            handle.set_color(color)
        
        axes4.set_ylim(-10, 70)

        axes4.legend(handles=handles, labels=new_labels,  loc='upper center', bbox_to_anchor=(0.5, -0.2))
        axes1.set_title(f'Stopped - {odor_label}')

    mouse = raster.mouse.unique()[0]
    session = raster.session.unique()[0]
    plt.suptitle(f'{mouse} - {session}', y=1.05)

    sns.despine()
    plt.tight_layout()
    
    if save:
        save.savefig(fig)
        
    # plt.show()
    return fig


def plot_sniff_raster_conditioned_simple(raster, 
                                  velocity,
                                  condition = 'is_choice',
                                  condition_values = [1, 0],
                                  colors = ['crimson', 'steelblue'],
                                  all_axes = None,
                                  window = [-1,2]
                                  ):

    if all_axes is None:
        fig, ax = plt.subplots(3,1, figsize=(4, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1, 2]})
        axes1, axes2, axes4 = ax[0], ax[1], ax[2]
        
    for axes in ax.flatten():
        axes.vlines(0, 0, 1, transform=axes.get_xaxis_transform(), color='black', alpha=0.5, linewidth=0.5)
        
    raster_1 = raster.loc[(raster[condition] == condition_values[0])]
    raster_2 = raster.loc[(raster[condition] == condition_values[1])]
    
    color1 = colors[1]
    color2 = colors[0]

    plot_sniff_raster_simple(raster_1, axes1, axes2, color = color1)
    plot_sniff_raster_simple(raster_2, axes1, axes2, color = color2, max_trial = raster_1.total_sites.nunique()+5)

    # sns.lineplot(data=frequency_troughs, x='times', y='instantaneous_frequency', hue=condition, ax=axes3, palette= colors, legend=False)
    # axes3.set_ylabel('Frequency (Hz)')
    sns.lineplot(data=velocity, x='times', y='speed', hue=condition, ax=axes4, errorbar='sd', palette= colors)
    axes4.set_xlabel('Time from odor onset (s)')
    axes4.set_ylabel('Velocity (cm/s)')
    axes4.legend(loc='upper right')
    axes4.set_xlim(window)
    
    if condition == 'is_choice':
        axes4.fill_between((velocity.odor_to_stop.min(), velocity.odor_to_stop.quantile(0.5)), -10, 60, color='grey', alpha=0.1)
        axes1.fill_between((velocity.odor_to_stop.min(), velocity.odor_to_stop.quantile(0.5)), -1, raster.total_sites.max(), color='grey', alpha=0.1)
        axes2.fill_between((velocity.odor_to_stop.min(), velocity.odor_to_stop.quantile(0.5)), -2, 8, color='grey', alpha=0.1)
    else:
        axes4.fill_between((velocity.odor_to_water.min(), velocity.odor_to_water.quantile(0.8)), -10, 60, color='yellow', alpha=0.1)
        axes1.fill_between((velocity.odor_to_water.min(), velocity.odor_to_water.quantile(0.8)), -1, raster.total_sites.nunique(), color='yellow', alpha=0.1)
        axes2.fill_between((velocity.odor_to_water.min(), velocity.odor_to_water.quantile(0.8)), -2, 8, color='yellow', alpha=0.1)
         
    sns.despine()
    plt.tight_layout()
    return fig
