
"""
Created on Fri Jun 23 10:39:15 2023

@author: tiffany.ona
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
import seaborn as sns
from scipy.signal import find_peaks
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

def moving_average(data, window_size):
    '''
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
    '''
    
    window = np.ones(int(window_size)) / float(window_size)
    smoothed_data = np.convolve(data, window, 'same')
    return smoothed_data

def apply_filter(data, f_notch = 60, Q= 200):
    '''
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
    '''
    # Generate sample signal
    sampling_rate = 1000  # Sample rate (Hz)
    b, a = signal.iirnotch(f_notch, Q, sampling_rate)
    
    # Apply the notch filter to the signal
    data = signal.lfilter(b, a, data)
    
    return data

def findpeaks_and_plot(data, x, fig, range_plot = [120,121], color='black', heigth=1.1, prominence=2.5, distance=30, label = 'Pressure sensor'):
    '''

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
        heigth : TYPE, optional
            The default is 1.1.
        prominence : TYPE, optional
            The default is 2.5.
        distance : TYPE, optional
            The default is 30.
            

    Returns
    -------
    df_peaks : dataframe
        returns a dataframe with the peaks, amplitude, index_location and frequencies.

    '''
    # ------   Find peaks for the pressure sensor
    peaks, properties = find_peaks(data, height= heigth, prominence=prominence, distance=distance)
    ## ----------------------------
    freq = []
    df_peaks = pd.DataFrame()
    df_peaks['data'] = peaks
    
    for peak in peaks:
        number=df_peaks.loc[(df_peaks.data > peak-500)&(df_peaks.data < peak)].data.count()
        freq.append(number)
        
    df_peaks['freq'] = np.array(freq)/0.5
    df_peaks['heights'] = properties['peak_heights']
    df_peaks['locations'] = x[peaks].values
    
    print(peaks, properties)
    # Plot the curve and peaks
    plt.plot(x, data, label=label, color=color, linewidth=0.7)
    plt.plot(x[peaks], properties['peak_heights'], 'x', color=color)
    plt.vlines(x[peaks], ymin=min(data), ymax=max(data), color=color, linewidth=0.17)
    plt.xlabel('Time (s)')
    plt.ylabel('Units')
    plt.xlim(range_plot)
    plt.ylim(-20,20.5)
    
    return df_peaks


def plot_FFT(data, sampling_rate = 1000, color='black', label='test'):
    
    # Compute power spectrum for sensor
    signal_fft = np.fft.fft(data)
    power_spectrum = np.abs(signal_fft)
    
    # Frequency axis
    freq_axis = np.fft.fftfreq(len(data), 1 / sampling_rate)
    power_spectrum = moving_average(power_spectrum, window_size=100)

    # Plotting
    plt.semilogy(freq_axis, power_spectrum, color, label=label)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectrum')
    plt.title('Power Spectrum of the Signal')
    plt.xlim(-0.01,100)
