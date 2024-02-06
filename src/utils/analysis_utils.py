from scipy.signal import lfilter, firwin
import pandas as pd



def fir_filter(data, cutoff_hz, num_taps=61, nyq_rate=1000/2.):

    '''  
    Create a FIR filter and apply it to signal.
    
    nyq_rate (int) = The Nyquist rate of the signal.
    cutoff_hz (float) = The cutoff frequency of the filter: 5KHz
    numtaps (int) = Length of the filter (number of coefficients, i.e. the filter order + 1)
    '''
    
    # Use firwin to create a lowpass FIR filter
    fir_coeff = firwin(num_taps, cutoff_hz/nyq_rate)

    # Use lfilter to filter the signal with the FIR filter
    data["filtered_velocity"] = lfilter(fir_coeff, 1.0, data["velocity"].values)
    
    return data

def choose_cut(reward_sites: pd.DataFrame, number_skipped: int = 20):
    '''
    Choose the cut of the session based on the number of skipped sites
    
    Inputs:
    reward_sites: pd.DataFrame
        Dataframe with the reward sites
    number_skipped: int
        Number of skipped sites to choose the cut
        
    Returns:
    int
        The cut value of the session
        
    '''
    
    cumulative = 0
    for row,i in enumerate(reward_sites.has_choice):
        if int(i) == 0:
            cumulative += 1
        else:
            cumulative = 0
        
        if cumulative == number_skipped:
            return reward_sites.iloc[row].active_patch
        
    return max(reward_sites.active_patch)