import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from typing import Literal, Tuple


def distinct_until_changed_state(onset_event: pd.DataFrame,
                                 offset_event: pd.DataFrame,
                                 flag: None) -> pd.DataFrame:
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
                ], axis=0, copy=True).sort_index()
    else:
        state = pd.concat(
            [
                onset_event[onset_event["Value"].apply(lambda x: x.HasFlag(flag))].assign(Value=True),
                offset_event[offset_event["Value"].apply(lambda x: x.HasFlag(flag))].assign(Value=False),
                ], axis=0, copy=True).sort_index()
    state = state.loc[state["Value"] - state["Value"].shift(1) != 0, :]
    return state


find_closest_modes = Literal["closest", "above_zero", "below_zero"]


def find_closest(query: ArrayLike,
                 array: ArrayLike,
                 mode: find_closest_modes = "closest",
                 tolerance: float = np.inf
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