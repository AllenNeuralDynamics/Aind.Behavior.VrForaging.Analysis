import pandas as pd


def distinct_until_changed_state(onset_event: pd.DataFrame,
                                 offset_event: pd.DataFrame,
                                 flag: None):
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
