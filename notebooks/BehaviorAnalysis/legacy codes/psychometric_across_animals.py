# Core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

## Plotting

## Harp/Bonsai
sys.path.append('../../src/')
from bonsai import load_bonsai_config
load_bonsai_config(r"C:\git\AllenNeuralDynamics\aind-vr-foraging\Bonsai")
import harp
import harp.processing
import data_io


def load_session_data(session_path: str | Path) -> Dict[str, data_io.DataStreamSource]:
    _out_dict = {}

    HarpBehavior = harp.HarpDevice("Behavior")
    _out_dict["harp_behavior"] = data_io.HarpSource(
        device=HarpBehavior,
        path=session_path / "Behavior",
        name="behavior",
        autoload=False)
    _out_dict["software_events"] = data_io.SoftwareEventSource(
        path=session_path / "SoftwareEvents",
        name="software_events",
        autoload=True)
    _out_dict["config"] = data_io.ConfigSource(
        path=session_path / "Config",
        name="config",
        autoload=True)
    _out_dict["operation_control"] = data_io.OperationControlSource(
        path=session_path / "OperationControl",
        name="config",
        autoload=False)
    return _out_dict

def parse_reward_sites(datasources: Dict[str, data_io.DataStreamSource]) -> pd.DataFrame:

    harp_behavior_data = datasources["harp_behavior"]
    software_events = datasources["software_events"]
    config = datasources["config"]
    operation_control = datasources["operation_control"]

    ## Find changes in reward available
    patches = software_events.streams.ActivePatch.data
    active_site = software_events.streams.ActiveSite.data

    reward_available_in_patch = software_events.streams.RewardAvailableInPatch.data
    give_reward = software_events.streams.GiveReward.data
    choice_feedback = software_events.streams.ChoiceFeedback.data

    reward_sites = active_site[active_site["data"].apply(lambda x: x['label']) == 'Reward']
    reward_sites["active_patch"] = -1
    reward_sites["visit_number"] = -1
    reward_sites["has_choice"] = False
    reward_sites["reward_delivered"] = 0
    reward_sites["past_no_reward_count"] = 0
    reward_sites["reward_available_in_patch"] = 0
    past_no_reward_counter = 0
    current_patch_idx = -1

    visit_number = 0
    for idx, event in enumerate(reward_sites.iterrows()):

        #active_patch
        arg_min, _ = harp.processing.find_closest(
            event[0],
            patches.index.values,
            mode="below_zero")
        if not (np.isnan(arg_min)):
            reward_sites.loc[event[0], "active_patch"] = arg_min
        if current_patch_idx != arg_min:
            current_patch_idx = arg_min
            visit_number = 0
        else:
            visit_number += 1
        reward_sites.loc[event[0], "visit_number"] = visit_number

        #available reward
        arg_min, _ = harp.processing.find_closest(
            event[0],
            reward_available_in_patch.index.values,
            mode="below_zero")
        if not (np.isnan(arg_min)):
            reward_sites.loc[event[0], "reward_available_in_patch"] = reward_available_in_patch.iloc[arg_min]["data"]

        # outcomes
        if idx < len(reward_sites) - 1:
            choice = choice_feedback.loc[(choice_feedback.index >= reward_sites.index[idx]) & (choice_feedback.index < reward_sites.index[idx+1])]
            reward_delivered = give_reward.loc[(give_reward.index >= reward_sites.index[idx]) & (give_reward.index < reward_sites.index[idx+1])]
        else: #account for the last trial
            choice = choice_feedback.loc[(choice_feedback.index >= reward_sites.index[idx])]
            reward_delivered = give_reward.loc[(give_reward.index >= reward_sites.index[idx])]

        reward_sites.loc[event[0], "has_choice"] = len(choice) > 0
        reward_sites.loc[event[0], "reward_delivered"] = reward_delivered.iloc[0]["data"] if len(reward_delivered) > 0 else np.nan
        reward_sites.loc[event[0], "past_no_reward_count"] = past_no_reward_counter
        if reward_sites.loc[event[0], "reward_delivered"] == 0:
            past_no_reward_counter += 1
        else:
            past_no_reward_counter = 0

    reward_sites["patch_label"] = reward_sites["active_patch"].apply(lambda x : patches.iloc[x]["data"]["label"])
    return reward_sites


def fit_psychometric(stim: np.array,
                     p_choice: np.array,
                     counts: Optional[np.array],
                     par0: np.array = np.array([1., 1.])):
    import numpy as np
    from scipy.optimize import curve_fit

    if np.shape(stim) != np.shape(p_choice):
        raise ValueError("stim and p_choice must have the same shape")
    if counts is None:
        counts = np.ones_like(stim)
    if np.shape(stim) != np.shape(counts):
        raise ValueError("stim and counts must have the same shape")

    def pf(x, alpha, beta):
        return 1. / (1 + np.exp(-(x-alpha)/beta))

    par, mcov = curve_fit(pf, stim, p_choice, par0)
    return par, mcov