import json
import os
from os import PathLike
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from harp.reader import create_reader

import data_io
from utils import processing

_SECONDS_PER_TICK = 32e-6
_payloadtypes = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.uint16),
    4: np.dtype(np.uint32),
    8: np.dtype(np.uint64),
    129: np.dtype(np.int8),
    130: np.dtype(np.int16),
    132: np.dtype(np.int32),
    136: np.dtype(np.int64),
    68: np.dtype(np.float32),
}


class TaskSchemaProperties:
    """This class is used to store the schema properties of the task configuration.

    tasklogic (str): The key used to access task logic data in the configuration.
    environment (str): The key used to access environment statistics in the configuration.
    reward_specification (str): The key used to access reward specifications in the configuration.
    odor_specifications (str): The key used to access odor specifications in the configuration.
    odor_index (str): The key used to access the odor index in the configuration.
    patches (list): A list of patches in the task configuration.
    """

    def __init__(self, data):
        self._data = data

        if "rig_input" in self._data["config"].streams.keys():
            self.rig = "rig_input"
        else:
            self.rig = "Rig"

        self._data["config"].streams[self.rig].load_from_file()

        if "TaskLogic" in self._data["config"].streams.keys():
            self.tasklogic = "TaskLogic"
        else:
            self.tasklogic = "tasklogic_input"

        self._data["config"].streams[self.tasklogic].load_from_file()

        if (
            "environment_statistics"
            in self._data["config"].streams[self.tasklogic].data
        ):
            self.environment = "environment_statistics"
            self.reward_specification = "reward_specification"
            self.odor_specifications = "odor_specification"
            self.odor_index = "index"
        else:
            self.environment = "environmentStatistics"
            self.reward_specification = "rewardSpecifications"
            self.odor_specifications = "odorSpecifications"
            self.odor_index = "odorIndex"

        self.patches = (
            self._data["config"]
            .streams[self.tasklogic]
            .data[self.environment]["patches"]
        )


class ContinuousData:
    def __init__(self, data, load_continuous: bool = True):

        self.data = data

        self.data["harp_behavior"].streams.OutputSet.load_from_file()
        self.data[
            "harp_behavior"
        ].streams.PulseSupplyPort0.load_from_file()  # Duration of each pulse
        self.data["harp_behavior"].streams.DigitalInputState.load_from_file()
        self.data["harp_behavior"].streams.AnalogData.load_from_file()
        if "rig_input" in self.data["config"].streams.keys():
            self.rig = "rig_input"
        else:
            self.rig = "Rig"
        self.data["config"].streams[self.rig].load_from_file()
        self.data["software_events"].streams.ChoiceFeedback.load_from_file()
        if load_continuous == True:
            self.encoder_data = self.encoder_loading()
            self.choice_feedback = self.choice_feedback_loading()
            self.lick_onset = self.lick_onset_loading()
            self.give_reward, self.pulse_duration, self.valve_output_pulse = (
                self.water_valve_loading()
            )
            # self.succesful_wait = self.succesfull_wait_loading()
            self.sniff_data_loading()

    def encoder_loading(self):
        ## Load data from encoder efficiently
        if "harp_board" in self.data["config"].streams[self.rig].data["treadmill"]:
            if (
                self.data["config"]
                .streams[self.rig]
                .data["treadmill"]["harp_board"]["device_type"]
                == "behavior"
            ):
                encoder_data = self.data["harp_behavior"].streams.AnalogData.data
        else:
            encoder_data = self.data["harp_behavior"].streams.AnalogData.data

        if "settings" in self.data["config"].streams[self.rig].data["treadmill"]:
            wheel_size = (
                self.data["config"]
                .streams[self.rig]
                .data["treadmill"]["settings"]["wheel_diameter"]
            )
            PPR = (
                self.data["config"]
                .streams[self.rig]
                .data["treadmill"]["settings"]["pulses_per_revolution"]
            )
            invert_direction = (
                self.data["config"]
                .streams[self.rig]
                .data["treadmill"]["settings"]["invert_direction"]
            )
        else:
            wheel_size = 15
            PPR = 8192.0
            invert_direction = True

        converter = wheel_size * np.pi / PPR * (-1 if invert_direction else 1)

        encoder_data["velocity"] = (encoder_data["Encoder"] * converter) * 1000
        self.encoder_data = processing.fir_filter(encoder_data, 5)

        # Load treadmill data
        # Maybe look at how the traces change with these two ways
        # encoder_data.index = pd.to_datetime(encoder_data.index, unit="s")
        # encoder_data['resample_velocity'] = encoder_data['velocity'].resample("33ms").sum().interpolate(method="linear") / 0.033
        # encoder_data.index = (encoder_data.index - pd.to_datetime(0))
        # encoder_data.index = encoder_data.index.total_seconds()
        self.encoder_data = encoder_data

        return self.encoder_data

    def choice_feedback_loading(self):
        # Find responses to Reward site
        self.choice_feedback = self.data["software_events"].streams.ChoiceFeedback.data
        return self.choice_feedback

    def lick_onset_loading(self):
        if "harp_lickometer" in self.data:
            self.data["harp_lickometer"].streams.LickState.load_from_file()
            lick_onset = (
                self.data["harp_lickometer"].streams.LickState.data["Channel0"] == True
            )
            self.lick_onset = lick_onset.loc[lick_onset == True]
        else:
            di_state = self.data["harp_behavior"].streams.DigitalInputState.data[
                "DIPort0"
            ]
            self.lick_onset = di_state.loc[di_state == True]
        return self.lick_onset

    def water_valve_loading(self):
        # Find give reward event
        give_reward = self.data["harp_behavior"].streams.OutputSet.data[["SupplyPort0"]]
        self.give_reward = give_reward.loc[give_reward.SupplyPort0 == True]

        # Find hardware reward events
        self.pulse_duration = np.median(
            self.data["harp_behavior"].streams.PulseSupplyPort0.data["PulseSupplyPort0"]
        )
        self.valve_output_pulse = self.data["harp_behavior"].streams.OutputSet.data[
            "SupplyPort0"
        ]

        return self.give_reward, self.pulse_duration, self.valve_output_pulse

    def sniff_data_loading(self):
        if "harp_sniffsensor" in self.data:
            self.data["harp_sniffsensor"].streams.RawVoltage.load_from_file()
            self.breathing = pd.DataFrame(
                index=self.data["harp_sniffsensor"]
                .streams.RawVoltage.data["RawVoltage"]
                .index,
                columns=["data"],
            )
            self.breathing["data"] = (
                self.data["harp_sniffsensor"]
                .streams.RawVoltage.data["RawVoltage"]
                .values
            )

        else:
            ## Breathing
            self.breathing = pd.DataFrame(
                index=self.data["harp_behavior"]
                .streams.AnalogData.data["AnalogInput0"]
                .index,
                columns=["data"],
            )
            self.breathing["data"] = (
                self.data["harp_behavior"]
                .streams.AnalogData.data["AnalogInput0"]
                .values
            )
        return self.breathing


class RewardFunctions:
    """
    This class is used to calculate and manage reward functions for amount, reward available or probability.

    Attributes:
        _data (dict): A dictionary containing the task configuration.
        reward_sites (DataFrame): A pandas DataFrame containing the reward sites data.
    """

    def __init__(self, data, reward_sites):
        """
        The constructor for reward_functions class.

        Parameters:
            data (dict): A dictionary containing the task configuration.
            reward_sites (DataFrame): A pandas DataFrame containing the reward sites data.
        """

        self._data = data
        self.reward_sites = reward_sites

        self.schema_properties = TaskSchemaProperties(self._data)

    def calculate_reward_functions(self):
        self.add_cumulative_rewards()
        self.reward_amount()
        self.reward_probability()
        self.reward_available()
        self.reward_sites.drop(columns=["cumulative_rewards"], inplace=True)
        return self.reward_sites

    def add_cumulative_rewards(self):
        """
        This method calculates the cumulative rewards for each patch in the reward sites.
        """

        previous_patch = -1
        cumulative_rewards = 0

        for index, row in self.reward_sites.iterrows():
            # Total number of rewards in the current patch ( accumulated)
            if row["active_patch"] != previous_patch:
                previous_patch = row["active_patch"]
                cumulative_rewards = 0

            self.reward_sites.loc[index, "cumulative_rewards"] = cumulative_rewards

            if row["reward_delivered"] != 0:
                cumulative_rewards += 1

    def reward_amount(self):
        """
        This method calculates the reward amount for each reward site based on the reward function specified in the task configuration.
        It creates a new column 'reward_amount' in the reward_sites DataFrame.

        Returns:
            DataFrame: The updated reward_sites DataFrame with the 'reward_amount' column.
        """

        # Create a curve for how the reward amount changes in time and create a column with the current value
        x = np.linspace(0, 500, 501)  # Generate 500 points between 0 and 500
        dict_odor = {}

        for patches in self.schema_properties.patches:
            if (
                "reward_function"
                not in patches[self.schema_properties.reward_specification]
            ):
                dict_odor[patches["label"]] = np.repeat(
                    patches[self.schema_properties.reward_specification]["amount"], 500
                )
                continue

            if (
                patches[self.schema_properties.reward_specification]["reward_function"][
                    "amount"
                ]["function_type"]
                == "ConstantFunction"
            ):
                odor_label = patches["label"]
                y = np.repeat(
                    patches[self.schema_properties.reward_specification][
                        "reward_function"
                    ]["amount"]["value"],
                    500,
                )
            else:

                odor_label = patches["label"]
                a = patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["amount"]["a"]
                b = patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["amount"]["b"]
                c = -patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["amount"]["c"]
                d = patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["amount"]["d"]

                # Generate x values
                y = a * pow(b, -c * x) + d

            dict_odor[odor_label] = y

        for index, row in self.reward_sites.iterrows():
            self.reward_sites.at[index, "reward_amount"] = np.around(
                dict_odor[row["odor_label"]][int(row["cumulative_rewards"])], 3
            )

        return self.reward_sites

    def reward_probability(self):
        """
        This method calculates the reward probability for each reward site based on the reward function specified in the task configuration.
        It creates a new column 'reward_probability' in the reward_sites DataFrame.
        """

        # Create a curve for how the reward probability changes in time and create a column with the current value
        x = np.linspace(0, 500, 501)  # Generate 100 points between 0 and 5
        dict_odor = {}

        for patches in self.schema_properties.patches:
            if (
                "reward_function"
                not in patches[self.schema_properties.reward_specification]
            ):
                dict_odor[patches["label"]] = np.repeat(
                    patches[self.schema_properties.reward_specification]["probability"],
                    500,
                )
                continue

            if (
                patches[self.schema_properties.reward_specification]["reward_function"][
                    "probability"
                ]["function_type"]
                == "ConstantFunction"
            ):
                odor_label = patches["label"]
                y = np.repeat(
                    patches[self.schema_properties.reward_specification][
                        "reward_function"
                    ]["probability"]["value"],
                    500,
                )
            else:

                odor_label = patches["label"]
                a = patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["probability"]["a"]
                b = patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["probability"]["b"]
                c = -patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["probability"]["c"]
                d = patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["probability"]["d"]

                # Generate x values
                y = a * pow(b, -c * x) + d

            dict_odor[odor_label] = y

        #### ----------- Need to add the modification for On Choice, right now specific for OnReward
        for index, row in self.reward_sites.iterrows():
            self.reward_sites.at[index, "reward_probability"] = np.around(
                dict_odor[row["odor_label"]][int(row["cumulative_rewards"])], 3
            )

    def reward_available(self):
        """
        This method calculates the reward availability for each reward site based on the reward function specified in the task configuration.
        It creates a new column 'reward_available' in the reward_sites DataFrame.

        Returns:
            DataFrame: The updated reward_sites DataFrame with the 'reward_available' column.
        """
        # Create a curve for how the reward available changes in time and create a column with the current value
        x = np.linspace(0, 500, 501)  # Generate 100 points between 0 and 5
        dict_odor = {}

        for patches in self.schema_properties.patches:

            # Segment for when the conventions were different. It was always a linear decrease.
            if (
                "reward_function"
                not in patches[self.schema_properties.reward_specification]
            ):
                if patches["patchRewardFunction"]["initialRewardAmount"] >= 100:
                    dict_odor[patches["label"]] = np.repeat(100, 500)
                else:
                    odor_label = patches["label"]
                    initial = patches["patchRewardFunction"]["initialRewardAmount"]
                    amount = patches[self.schema_properties.reward_specification][
                        "amount"
                    ]
                    y = initial - amount * x
                    dict_odor[odor_label] = y
                continue

            if (
                patches[self.schema_properties.reward_specification]["reward_function"][
                    "probability"
                ]["function_type"]
                == "ConstantFunction"
            ):
                odor_label = patches["label"]
                y = np.repeat(
                    patches[self.schema_properties.reward_specification][
                        "reward_function"
                    ]["probability"]["value"],
                    500,
                )
            else:
                odor_label = patches["label"]
                a = patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["probability"]["a"]
                b = patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["probability"]["b"]
                c = -patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["probability"]["c"]
                d = patches[self.schema_properties.reward_specification][
                    "reward_function"
                ]["probability"]["d"]

                # Generate x values
                y = a * pow(b, -c * x) + d

            dict_odor[odor_label] = y

        for index, row in self.reward_sites.iterrows():
            self.reward_sites.at[index, "reward_available"] = np.around(
                dict_odor[row["odor_label"]][int(row["cumulative_rewards"])], 3
            )

        return self.reward_sites


def read_harp_bin(file):

    data = np.fromfile(file, dtype=np.uint8)

    if len(data) == 0:
        return None

    stride = data[1] + 2
    length = len(data) // stride
    payloadsize = stride - 12
    payloadtype = _payloadtypes[data[4] & ~0x10]
    elementsize = payloadtype.itemsize
    payloadshape = (length, payloadsize // elementsize)
    seconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
    ticks = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
    seconds = ticks * _SECONDS_PER_TICK + seconds
    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data,
        offset=11,
        strides=(stride, elementsize),
    )

    if payload.shape[1] == 1:
        ret_pd = pd.DataFrame(payload, index=seconds, columns=["Value"])
        ret_pd.index.names = ["Seconds"]

    else:
        ret_pd = pd.DataFrame(payload, index=seconds)
        ret_pd.index.names = ["Seconds"]

    return ret_pd


## ------------------------------------------------------------------------- ##
def load_session_data(
    session_path: str | PathLike,
) -> Dict[str, data_io.DataStreamSource]:
    _out_dict = {}
    session_path = Path(session_path)
    HarpBehavior = create_reader(device=r"C:\git\harp-tech\device.behavior\device.yml")
    HarpOlfactometer = create_reader(
        device=r"C:\git\harp-tech\device.olfactometer\device.yml"
    )
    HarpLickometer = create_reader(
        device=r"C:\git\harp-tech\harp.device.lickety-split\software\bonsai\device.yml"
    )
    HarpSniffsensor = create_reader(
        device=r"C:\git\harp-tech\harp.device.sniff-detector\software\bonsai\device.yml"
    )

    if "Behavior.harp" in os.listdir(session_path):
        _out_dict["harp_behavior"] = data_io.HarpSource(
            device=HarpBehavior,
            path=session_path / "Behavior.harp",
            name="behavior",
            autoload=False,
        )
    else:
        behavior = 0
        print("Old behavior loading")
        _out_dict["harp_behavior"] = data_io.HarpSource(
            device=HarpBehavior,
            path=session_path / "Behavior",
            name="behavior",
            autoload=False,
        )

    if "Olfactometer.harp" in os.listdir(session_path):
        _out_dict["harp_olfactometer"] = data_io.HarpSource(
            device=HarpOlfactometer,
            path=session_path / "Olfactometer.harp",
            name="olfactometer",
            autoload=False,
        )

    if "Lickometer.harp" in os.listdir(session_path):
        _out_dict["harp_lickometer"] = data_io.HarpSource(
            device=HarpLickometer,
            path=session_path / "Lickometer.harp",
            name="lickometer",
            autoload=False,
        )

    if "SniffDetector.harp" in os.listdir(session_path):
        _out_dict["harp_sniffsensor"] = data_io.HarpSource(
            device=HarpSniffsensor,
            path=session_path / "SniffDetector.harp",
            name="sniffdetector",
            autoload=False,
        )

    _out_dict["software_events"] = data_io.SoftwareEventSource(
        path=session_path / "SoftwareEvents", name="software_events", autoload=True
    )

    # Load config old version
    if "config.json" in os.listdir(session_path):
        with open(str(session_path) + "\config.json", "r") as json_file:
            config = json.load(json_file)

    # Load new configuration
    else:
        _out_dict["config"] = data_io.ConfigSource(
            path=session_path / "Config", name="config", autoload=True
        )

    if "OperationControl.harp" in os.listdir(session_path):
        _out_dict["operation_control"] = data_io.OperationControlSource(
            path=session_path / "OperationControl.harp",
            name="operation_control",
            autoload=True,
        )
    else:
        pass

    if "UpdaterEvents" in os.listdir(session_path):
        _out_dict["updater_events"] = data_io.UpdaterEventSource(
            path=session_path / "UpdaterEvents", name="updater_events", autoload=True
        )
    else:
        pass
    return _out_dict


## ------------------------------------------------------------------------- ##
def odor_data_harp_olfactometer(data, reward_sites):
    """
    Process odor data from the Harp Olfactometer.

    Args:
        data (dict): A dictionary containing the data from the Harp Olfactometer.
        reward_sites (DataFrame): A DataFrame containing reward site information.

    Returns:
        DataFrame: A DataFrame containing the updated reward site information with odor onset and offset.

    Raises:
        AssertionError: If the odor labels do not match.

    """
    data["harp_olfactometer"].streams.OdorValveState.load_from_file()
    data["harp_olfactometer"].streams.EndValveState.load_from_file()

    schema_properties = TaskSchemaProperties(data)

    # Assign odor labels to odor indexes
    odor0 = False
    odor1 = False
    odor2 = False

    data["config"].streams[schema_properties.tasklogic].load_from_file()

    for patches in schema_properties.patches:
        if (
            patches[schema_properties.odor_specifications][schema_properties.odor_index]
            == 0
        ):
            odor0 = patches["label"]
        elif (
            patches[schema_properties.odor_specifications][schema_properties.odor_index]
            == 1
        ):
            odor1 = patches["label"]
        else:
            odor2 = patches["label"]

    # Selecting which odor valve is open before the end valves are opened
    OdorValveState = pd.DataFrame()
    OdorValveState["time"] = data[
        "harp_olfactometer"
    ].streams.OdorValveState.data.index.values
    data["harp_olfactometer"].streams.OdorValveState.data["Valve0"] = np.where(
        data["harp_olfactometer"].streams.OdorValveState.data["Valve0"] == True,
        odor0,
        data["harp_olfactometer"].streams.OdorValveState.data["Valve0"],
    )
    data["harp_olfactometer"].streams.OdorValveState.data["Valve1"] = np.where(
        data["harp_olfactometer"].streams.OdorValveState.data["Valve1"] == True,
        odor1,
        data["harp_olfactometer"].streams.OdorValveState.data["Valve1"],
    )
    data["harp_olfactometer"].streams.OdorValveState.data["Valve2"] = np.where(
        data["harp_olfactometer"].streams.OdorValveState.data["Valve2"] == True,
        odor2,
        data["harp_olfactometer"].streams.OdorValveState.data["Valve2"],
    )

    # Create a new dataframe to store the results
    OdorValveState = pd.DataFrame(columns=["time", "condition"])

    # Loop through each row and find the values that are not False
    for index, row in (
        data["harp_olfactometer"]
        .streams.OdorValveState.data[["Valve0", "Valve1", "Valve2"]]
        .iterrows()
    ):
        non_false_values = row[row != "False"].tolist()
        if non_false_values:  # Check if there are any non-False values
            OdorValveState = pd.concat(
                [
                    OdorValveState,
                    pd.DataFrame(
                        [[index, non_false_values[0]]], columns=["time", "condition"]
                    ),
                ]
            )

    EndValveState = pd.DataFrame()
    EndValveState["time"] = data[
        "harp_olfactometer"
    ].streams.EndValveState.data.index.values
    EndValveState["condition"] = np.where(
        data["harp_olfactometer"].streams.EndValveState.data["EndValve0"] == True,
        "EndValveOn",
        "EndValveOff",
    )

    odor_updates = pd.concat(
        [EndValveState[["time", "condition"]], OdorValveState[["time", "condition"]]]
    )
    odor_updates = odor_updates.sort_values(by="time")
    odor_updates = odor_updates[odor_updates["condition"] != False]

    odor_triggers = pd.DataFrame(columns=["odor_onset", "odor_offset", "condition"])
    onset = np.nan
    offset = np.nan
    first = True
    for i, row in odor_updates.iterrows():
        if row["condition"] == "EndValveOn":
            onset = row["time"]

        elif row["condition"] == "EndValveOff":
            offset = row["time"]

        elif (row["condition"] != "EndValveOn") or (row["condition"] != "EndValveOff"):
            if first:
                condition = row["condition"]
                first = False
                continue
            else:
                new_row = {
                    "odor_onset": onset,
                    "odor_offset": offset,
                    "condition": condition,
                }
                odor_triggers.loc[len(odor_triggers)] = new_row
                condition = row["condition"]

    if row["condition"] == "EndValveOn":
        new_row = {"odor_onset": onset, "odor_offset": np.nan, "condition": condition}
        odor_triggers.loc[len(odor_triggers)] = new_row

    try:
        assert np.any(
            odor_triggers["condition"].values == reward_sites["odor_label"].values
        )
        reward_sites["odor_onset"] = odor_triggers["odor_onset"].values
        reward_sites["odor_offset"] = odor_triggers["odor_offset"].values
    except:
        reward_sites = reward_sites.iloc[:-1]
        reward_sites["odor_onset"] = odor_triggers["odor_onset"].values
        reward_sites["odor_offset"] = odor_triggers["odor_offset"].values

    return reward_sites  ## ------------------------------------------------------------------------- ##


def parse_data_old(data, path):
    """
    Parses the data and extracts relevant information for analysis.

    Args:
        data (dict): The data dictionary containing the raw data.
        path (str): The path to the data files.

    Returns:
        pd.DataFrame: The parsed data in a pandas DataFrame format.
    """
    try:
        ## Load data from encoder efficiently
        data["harp_behavior"].streams.AnalogData.load_from_file()
        encoder_data = data["harp_behavior"].streams.AnalogData.data
    except:
        encoder_data = pd.DataFrame()
        encoder_data["Encoder"] = read_harp_bin(
            path + "\Behavior\Register__44" + ".bin"
        )[1]

    try:
        # Open and read the JSON file
        with open(str(path) + "\Config\TaskLogic.json", "r") as json_file:
            config = json.load(json_file)

    except:
        with open(str(path) + "\config.json", "r") as json_file:
            config = json.load(json_file)

    try:
        wheel_size = config.streams.Rig.data["treadmill"]["wheelDiameter"]
        PPR = -config.streams.Rig.data["treadmill"]["pulsesPerRevolution"]

    except:
        wheel_size = 15
        PPR = -8192.0

    perimeter = wheel_size * np.pi
    resolution = perimeter / PPR
    encoder_data["velocity"] = (encoder_data["Encoder"] * resolution) * 1000

    # Reindex the seconds so they are aligned to beginning of the session
    start_time = encoder_data.index[0]
    # encoder_data.index -= start_time

    # Get the first odor onset per reward site
    data["software_events"].streams.ActiveSite.load_from_file()
    active_site = data["software_events"].streams.ActiveSite.data

    # Use json_normalize to create a new DataFrame from the 'data' column
    df_normalized = pd.json_normalize(active_site["data"])
    df_normalized.index = active_site.index

    # Concatenate the normalized DataFrame with the original DataFrame
    active_site = pd.concat([active_site, df_normalized], axis=1)

    active_site["label"] = np.where(
        active_site["label"] == "Reward", "RewardSite", active_site["label"]
    )
    active_site.rename(columns={"startPosition": "start_position"}, inplace=True)
    # Rename columns

    active_site = active_site[["label", "start_position", "length"]]
    reward_sites = active_site[active_site["label"] == "RewardSite"]

    data["software_events"].streams.GiveReward.load_from_file()
    reward = data["software_events"].streams.GiveReward.data
    reward.fillna(0, inplace=True)

    try:
        data["software_events"].streams.ActivePatch.load_from_file()
        patches = data["software_events"].streams.ActivePatch.data

    except:
        patches = active_site.loc[active_site["label"] == "InterPatch"]
        patches.rename(columns={"label": "name"}, inplace=True)
        patches["name"] = np.where(
            patches["name"] == "InterPatch", "ActivePatch", patches["name"]
        )

    try:
        # Old way of obtaining the reward amount
        reward_available = event[1]["data"]["patchRewardFunction"][
            "initialRewardAmount"
        ]
    except:
        reward_available = config["environmentStatistics"]["patches"][0][
            "patchRewardFunction"
        ]["initialRewardAmount"]

    reward_updates = pd.concat([patches, reward])
    reward_updates.sort_index(inplace=True)
    reward_updates["current_reward"] = np.nan

    for event in reward_updates.iterrows():
        if event[1]["name"] == "GiveReward":  # update reward
            reward_available -= event[1]["data"]
        elif event[1]["name"] == "ActivePatch":  # reset reward
            try:
                # Old way of obtaining the reward amount
                reward_available = event[1]["data"]["patchRewardFunction"][
                    "initialRewardAmount"
                ]
            except:
                reward_available = config["environmentStatistics"]["patches"][0][
                    "patchRewardFunction"
                ]["initialRewardAmount"]
        else:
            raise ValueError("Unknown event type")
        reward_updates.at[event[0], "current_reward"] = reward_available

    for site in reward_sites.itertuples():
        arg_min, val_min = processing.find_closest(
            site.Index, reward_updates.index.values, mode="below_zero"
        )
        try:
            reward_sites.loc[site.Index, "reward_available"] = reward_updates[
                "current_reward"
            ].iloc[arg_min]
        except:
            reward_sites.loc[site.Index, "reward_available"] = reward_updates[
                "current_reward"
            ].iloc[arg_min]

    # Find responses to Reward site
    data["software_events"].streams.ChoiceFeedback.load_from_file()
    choiceFeedback = data["software_events"].streams.ChoiceFeedback.data

    reward_sites.loc[:, "active_patch"] = -1
    reward_sites.loc[:, "visit_number"] = -1
    reward_sites.loc[:, "has_choice"] = False
    reward_sites.loc[:, "reward_delivered"] = 0
    reward_sites.loc[:, "past_no_reward_count"] = 0
    past_no_reward_counter = 0
    current_patch_idx = -1

    visit_number = 0
    for idx, event in enumerate(reward_sites.iterrows()):
        arg_min, val_min = processing.find_closest(
            event[0], patches.index.values, mode="below_zero"
        )
        if not (np.isnan(arg_min)):
            reward_sites.loc[event[0], "active_patch"] = arg_min
        if current_patch_idx != arg_min:
            current_patch_idx = arg_min
            visit_number = 0
        else:
            visit_number += 1
        reward_sites.loc[event[0], "visit_number"] = visit_number

        if idx < len(reward_sites) - 1:
            choice = choiceFeedback.loc[
                (choiceFeedback.index >= reward_sites.index[idx])
                & (choiceFeedback.index < reward_sites.index[idx + 1])
            ]
            reward_in_site = reward.loc[
                (reward.index >= reward_sites.index[idx])
                & (reward.index < reward_sites.index[idx + 1])
            ]
        else:
            choice = choiceFeedback.loc[
                (choiceFeedback.index >= reward_sites.index[idx])
            ]
            reward_in_site = reward.loc[(reward.index >= reward_sites.index[idx])]
        reward_sites.loc[event[0], "has_choice"] = len(choice) > 0
        reward_sites.loc[event[0], "reward_delivered"] = (
            reward_in_site.iloc[0]["data"] if len(reward_in_site) > 0 else 0
        )
        reward_sites.loc[event[0], "past_no_reward_count"] = past_no_reward_counter
        if (
            reward_sites.loc[event[0], "reward_delivered"] == 0
            and reward_sites.loc[event[0], "has_choice"] == 1
        ):
            past_no_reward_counter += 1
        else:
            past_no_reward_counter = 0
    try:
        df_patch = pd.json_normalize(patches["data"])
        df_patch.reset_index(inplace=True)
        df_patch.rename(
            columns={
                "index": "active_patch",
                "label": "odor_label",
                "rewardSpecifications.amount": "amount",
            },
            inplace=True,
        )
        df_patch.rename(
            columns={"reward_specification.reward_function.amount.value": "amount"},
            inplace=True,
        )
    except:
        df_patch = pd.DataFrame(columns=["active_patch", "odor_label", "amount"])
        df_patch["active_patch"] = np.arange(len(patches))
        df_patch["odor_label"] = config["environmentStatistics"]["patches"][0]["label"]
        df_patch["amount"] = config["environmentStatistics"]["patches"][0][
            "rewardSpecifications"
        ]["amount"]

    reward_sites = pd.merge(
        reward_sites.reset_index(),
        df_patch[["odor_label", "active_patch", "amount"]],
        on="active_patch",
    )

    # Create new column for adjusted seconds to start of session
    reward_sites["adj_seconds"] = reward_sites["Seconds"] - start_time
    reward_sites.index = reward_sites["Seconds"]
    reward_sites.drop(columns=["Seconds"], inplace=True)

    # ---------------- Add water triggers times ---------------- #
    data["harp_behavior"].streams.OutputSet.load_from_file()
    water = data["harp_behavior"].streams.OutputSet.data[["SupplyPort0"]]
    reward_sites["next_index"] = reward_sites.index.to_series().shift(-1)
    reward_sites["water_onset"] = None

    # Iterate through the actual index of df1
    for value in water.index:
        # Check if the value is between 'Start' and 'End' in df2
        matching_row = reward_sites[
            (reward_sites.index <= value) & (reward_sites["next_index"].values >= value)
        ]

        # If a matching row is found, update the corresponding row in water with the index value
        if not matching_row.empty:
            matching_index = matching_row.index[
                0
            ]  # Assuming there's at most one matching row
            reward_sites.at[matching_index, "water_onset"] = value

    # ---------------------------------------------------- #

    # ---------------- Add odor triggers times ---------------- #

    odor_0 = data["harp_behavior"].streams.OutputSet.data["SupplyPort1"]
    odor_1 = data["harp_behavior"].streams.OutputSet.data["SupplyPort2"]

    odor_0 = odor_0.reset_index()
    odor_1 = odor_1.reset_index()

    odor_0["odor_onset"] = np.where(
        odor_0["SupplyPort1"] == 1,
        config["environmentStatistics"]["patches"][0]["label"],
        None,
    )
    odor_1["odor_onset"] = np.where(
        odor_1["SupplyPort2"] == 1,
        config["environmentStatistics"]["patches"][1]["label"],
        None,
    )

    odor_df = pd.concat(
        [odor_0[["Time", "odor_onset"]], odor_1[["Time", "odor_onset"]]]
    )
    odor_df.sort_index(inplace=True)
    odor_df.dropna(inplace=True)

    odor_df["time_diff"] = odor_df["Time"].diff()
    odor_df = odor_df.drop(
        index=odor_df.loc[(odor_df["time_diff"] < 1) & (odor_df.index > 0)].index
    )

    try:
        reward_sites["odor_onset"] = odor_df["Time"].values
    except:
        pass

    # ---------------- Add stop triggers times ---------------- #
    reward_sites["stop_time"] = None

    # Iterate through the actual index of df1
    for value in choiceFeedback.index:
        # Check if the value is between 'Start' and 'End' in df2
        matching_row = reward_sites[
            (reward_sites.index <= value) & (reward_sites["next_index"].values >= value)
        ]

        # If a matching row is found, update the corresponding row in water with the index value
        if not matching_row.empty:
            matching_index = matching_row.index[
                0
            ]  # Assuming there's at most one matching row
            reward_sites.at[matching_index, "stop_time"] = value

    reward_sites.drop(columns=["next_index"], inplace=True)
    # ---------------------------------------------------- #

    # Add colum for site number
    reward_sites.loc[:, "total_sites"] = np.arange(len(reward_sites))
    reward_sites.loc[:, "depleted"] = np.where(
        reward_sites["reward_available"] == 0, 1, 0
    )
    reward_sites.loc[:, "collected"] = np.where(
        (reward_sites["reward_delivered"] != 0), 1, 0
    )

    # reward_sites['next_visit_number'] = reward_sites['visit_number'].shift(-2)
    # reward_sites['last_visit'] = np.where(reward_sites['next_visit_number']==0, 1, 0)
    # reward_sites.drop(columns=['next_visit_number'], inplace=True)

    # reward_sites['last_site'] = reward_sites['visit_number'].shift(-1)
    # reward_sites['last_site'] = np.where(reward_sites['last_site'] == 0, 1,0)

    # reward_sites['next_patch'] = reward_sites['active_patch'].shift(1)
    # reward_sites['next_odor'] = reward_sites['odor_label'].shift(1)
    # reward_sites['same_patch'] = np.where((reward_sites['next_patch'] != reward_sites['active_patch'])&(reward_sites['odor_label'] == reward_sites['next_odor'] ), 1, 0)
    # reward_sites.drop(columns=['next_patch', 'next_odor'], inplace=True)

    encoder_data = processing.fir_filter(encoder_data, 5)

    if reward_sites.reward_available.max() >= 100:
        reward_sites["reward_available"] = 100

    return reward_sites, active_site, encoder_data, config


## ------------------------------------------------------------------------- ##
def parse_dataframe(data: pd.DataFrame):
    """
    Parse the data from the session and return the reward sites, active sites and encoder data

    Inputs:
    data: pd.DataFrame
        Data from the session

    Returns:
    reward_sites: pd.DataFrame
        Dataframe with the reward sites
    active_site: pd.DataFrame
        Dataframe with the active sites (Reward, interpatch and intersite instantiations)
    config: pd.DataFrame
        Dataframe with the configuration data

    """

    # Get the first odor onset per reward site
    data["software_events"].streams.ActiveSite.load_from_file()
    active_site_temp = data["software_events"].streams.ActiveSite.data

    # Use json_normalize to create a new DataFrame from the 'data' column
    active_site = pd.json_normalize(active_site_temp["data"])
    active_site.index = active_site_temp.index

    active_site["label"] = np.where(
        active_site["label"] == "Reward", "RewardSite", active_site["label"]
    )
    active_site.rename(columns={"startPosition": "start_position"}, inplace=True)

    # Crop and rename columns
    active_site = active_site[["label", "start_position", "length"]]
    reward_sites = active_site[active_site["label"] == "RewardSite"]

    # Patch initialization
    data["software_events"].streams.ActivePatch.load_from_file()
    patches = data["software_events"].streams.ActivePatch.data

    # Find responses to Reward site
    # Recover tones
    data["software_events"].streams.ChoiceFeedback.load_from_file()
    choiceFeedback = data["software_events"].streams.ChoiceFeedback.data

    # Recover water delivery
    data["harp_behavior"].streams.OutputSet.load_from_file()
    data["harp_behavior"].streams.OutputClear.load_from_file()
    water = data["harp_behavior"].streams.OutputSet.data[["SupplyPort0"]]

    # Successfull waits
    data["software_events"].streams.WaitRewardOutcome.load_from_file()
    succesfull_wait = pd.DataFrame(
        index=data["software_events"].streams.WaitRewardOutcome.data.index,
        columns=["data"],
    )
    new_data = pd.json_normalize(
        data["software_events"].streams.WaitRewardOutcome.data["data"]
    )["IsSuccessfulWait"]
    succesfull_wait["data"] = new_data.values
    succesfull_wait = succesfull_wait[succesfull_wait["data"] == True]

    reward_sites.loc[:, "active_patch"] = -1
    reward_sites.loc[:, "visit_number"] = -1
    reward_sites.loc[:, "has_choice"] = False
    reward_sites.loc[:, "reward_delivered"] = 0
    reward_sites.loc[:, "stop_cue"] = np.nan
    reward_sites.loc[:, "succesful_wait"] = np.nan
    reward_sites.loc[:, "water_onset"] = np.nan
    current_patch_idx = -1

    visit_number = 0
    for idx, event in enumerate(reward_sites.iterrows()):
        arg_min, val_min = processing.find_closest(
            event[0], patches.index.values, mode="below_zero"
        )
        if not (np.isnan(arg_min)):
            reward_sites.loc[event[0], "active_patch"] = arg_min
        if current_patch_idx != arg_min:
            current_patch_idx = arg_min
            visit_number = 0
        else:
            visit_number += 1
        reward_sites.loc[event[0], "visit_number"] = visit_number

        if idx < len(reward_sites) - 1:
            choice = choiceFeedback.loc[
                (choiceFeedback.index >= reward_sites.index[idx])
                & (choiceFeedback.index < reward_sites.index[idx + 1])
            ]
            reward_in_site = water.loc[
                (water.index >= reward_sites.index[idx])
                & (water.index < reward_sites.index[idx + 1])
            ]
            waits = succesfull_wait.loc[
                (succesfull_wait.index >= reward_sites.index[idx])
                & (succesfull_wait.index < reward_sites.index[idx + 1])
            ]

        else:  # Last odorsite of the session
            choice = choiceFeedback.loc[
                (choiceFeedback.index >= reward_sites.index[idx])
            ]
            reward_in_site = water.loc[(water.index >= reward_sites.index[idx])]
            waits = succesfull_wait.loc[
                (succesfull_wait.index >= reward_sites.index[idx])
            ]

        reward_sites.loc[event[0], "has_choice"] = len(choice) > 0
        reward_sites.loc[event[0], "stop_cue"] = (
            choice.index[0] if len(choice) > 0 else np.nan
        )
        reward_sites.loc[event[0], "reward_delivered"] = (
            1 if len(reward_in_site) > 0 else 0
        )
        reward_sites.loc[event[0], "water_onset"] = (
            reward_in_site.index[0] if len(reward_in_site) > 0 else np.nan
        )
        reward_sites.loc[event[0], "succesful_wait"] = (
            waits.index[0] if len(waits) > 0 else np.nan
        )

    # This is to recover the odor label,  there are easier ways but this one works with all the datasets so far
    df_patch = pd.json_normalize(patches["data"])
    df_patch.reset_index(inplace=True)
    df_patch.rename(
        columns={"index": "active_patch", "label": "odor_label"}, inplace=True
    )
    reward_sites = pd.merge(
        reward_sites.reset_index(),
        df_patch[["odor_label", "active_patch"]],
        on="active_patch",
    )
    reward_sites.index = reward_sites["Seconds"]
    reward_sites.drop(columns=["Seconds"], inplace=True)

    # ---------------- Add odor valve trigger times ---------------- #
    try:
        reward_sites = odor_data_harp_olfactometer(data, reward_sites)
    except:
        print("No olfactometer data - Old system?")
        pass
    # ---------------------------------------------------- #

    reward_sites = RewardFunctions(data, reward_sites).calculate_reward_functions()

    return reward_sites, active_site, data["config"]


## ------------------------------------------------------------------------- ##
