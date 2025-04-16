import os
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import firwin, filtfilt


def distinct_until_changed_state(
    onset_event: pd.DataFrame, offset_event: pd.DataFrame, flag: None
) -> pd.DataFrame:
    """
    Takes two DataFrame with events corresponding to onset and offset states of a digital IO and
    assembles it in a single DataFrame with the corresponding state transitions. Optionally, takes
    flag object to filter the events by a specific flag.
    """
    if flag is None:
        state = pd.concat(
            [
                onset_event[onset_event["Value"].assign(Value=True)],
                offset_event[offset_event["Value"].assign(Value=False)],
            ],
            axis=0,
            copy=True,
        ).sort_index()
    else:
        state = pd.concat(
            [
                onset_event[
                    onset_event["Value"].apply(lambda x: x.HasFlag(flag))
                ].assign(Value=True),
                offset_event[
                    offset_event["Value"].apply(lambda x: x.HasFlag(flag))
                ].assign(Value=False),
            ],
            axis=0,
            copy=True,
        ).sort_index()
    state = state.loc[state["Value"] - state["Value"].shift(1) != 0, :]
    return state


find_closest_modes = Literal["closest", "above_zero", "below_zero"]


def fir_filter(data, col, cutoff_hz, num_taps=61, nyq_rate=1000 / 2.0):
    """
    Create a FIR filter and apply it to signal.

    nyq_rate (int) = The Nyquist rate of the signal.
    cutoff_hz (float) = The cutoff frequency of the filter: 5KHz
    numtaps (int) = Length of the filter (number of coefficients, i.e. the filter order + 1)
    """

    # Use firwin to create a lowpass FIR filter
    fir_coeff = firwin(num_taps, cutoff_hz / nyq_rate)

    # Use lfilter to filter the signal with the FIR filter
    data['filtered_' + col] = filtfilt(fir_coeff, 1.0, data[col].values)

    return data


## ------------------------------------------------------------------------- ##


## ------------------------------------------------------------------------- ##
def compute_window(data, runningwindow, option, trial):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    end = False
    for i in range(len(data)):
        if data[trial].iloc[i] <= runningwindow:
            # Store the first index of that session
            if end == False:
                start = i
                end = True
            performance.append(round(np.mean(data[option].iloc[start : i + 1]), 2))
        else:
            end = False
            performance.append(
                round(np.mean(data[option].iloc[i - runningwindow : i]), 2)
            )
    return performance


## ------------------------------------------------------------------------- ##



def choose_cut(reward_sites: pd.DataFrame, number_skipped: int = 20):
    """
    Choose the cut of the session based on the number of skipped sites

    Inputs:
    reward_sites: pd.DataFrame
        Dataframe with the reward sites
    number_skipped: int
        Number of skipped sites to choose the cut

    Returns:
    int
        The cut value of the session

    """

    cumulative = 0
    for row, i in enumerate(reward_sites.is_choice):
        if int(i) == 0:
            cumulative += 1
        else:
            cumulative = 0

        if cumulative == number_skipped:
            return reward_sites.iloc[row].patch_number

    return max(reward_sites.patch_number)


def find_file(start_dir: str, filename_part: str):
    """
    Find a file in a directory

    Inputs:
    start_dir: str
        The directory to start the search. It doesn't go deeper from the first folder
    filename_part: str
        The part of the filename to search. This can be a substring of the filename

    Returns:
        str
            The full path of the file
    """
    for filename in os.listdir(start_dir):
        if filename_part in filename:
            return os.path.join(start_dir, filename)


# Define exponential function
def exponential_func(x, a, b):
    return a * np.exp(b * x)


def format_func(value, tick_number):
    return f"{value:.0f}"


def find_closest(
    query: float,
    array: ArrayLike,
    mode: find_closest_modes = "closest",
    tolerance: float = np.inf,
) -> Tuple[int, float]:
    """Returns the index and value of the closest element in array to query.

    Args:
        query (ArrayLike): Query value
        array (ArrayLike): Array where to find the closest value in
        mode (find_closest_modes, optional): Available methods to find the closest value. Defaults to "closest".

    Returns:
        Tuple[int, float]: Returns a tuple with the index and value of the closest element in array to query.
    """
    d = array - query
    if mode == "closest":
        pass
    elif mode == "above_zero":
        d[d < 0] = np.inf
    elif mode == "below_zero":
        d[d > 0] = np.inf
    arg_min = np.argmin(np.abs(d))
    if np.abs(d[arg_min]) >= tolerance:
        return (np.nan, np.nan)
    else:
        return (arg_min, array[arg_min])
