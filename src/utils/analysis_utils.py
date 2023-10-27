from scipy.signal import lfilter, firwin

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