import json
import os
from os import PathLike
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import aind_vr_foraging_analysis.data_io as data_io
from aind_vr_foraging_analysis.utils import processing
from packaging.version import Version

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

from datetime import datetime
import pytz

def parse_user_date(user_date_str):
    """
    Parses a user-provided date string in the format 'YYYY-MM-DD' and returns a datetime.date object.

    Parameters:
    user_date_str (str): A string representing a date in the format 'YYYY-MM-DD'.

    Returns:
    datetime.date: The parsed date if the format is valid.
    None: If the input format is incorrect.
    """
    try:
        return datetime.strptime(user_date_str, "%Y-%m-%d").date()  # Convert user input to date
    except ValueError:
        return "Invalid date format"  # Return None if the format is incorrect

def extract_and_convert_time(filename):
    """
    Extracts a timestamp from a filename and converts it to a local date in the 'America/Los_Angeles' timezone.

    The filename must follow one of these formats:
    - 'prefix_YYYY-MM-DDTHHMMSSZ_suffix' (UTC timestamp, indicated by 'Z')
    - 'prefix_YYYYMMDDTHHMMSS_suffix' (Local time in 'America/Los_Angeles')

    Parameters:
    filename (str): A string containing a timestamp in one of the expected formats.

    Returns:
    datetime.date: The extracted and converted local date.
    str: "Invalid filename format" if the filename format does not match expectations.
    """
    seattle_tz = pytz.timezone('America/Los_Angeles')

    # Extract the timestamp part
    timestamp_part = filename.split("_")[1]

    try:
        if "Z" in timestamp_part:  # Case: UTC timestamp
            dt_utc = datetime.strptime(timestamp_part, "%Y-%m-%dT%H%M%SZ")
            dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(seattle_tz)
        else:  # Case: Already local time
            dt_local = datetime.strptime(timestamp_part, "%Y%m%dT%H%M%S")
            dt_local = seattle_tz.localize(dt_local)
        return dt_local.date()
    except ValueError:
        return "Invalid filename format"

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
        try:
            version = Version(self._data["config"].streams[self.tasklogic].data["version"])
        except KeyError:
            version = Version("0.0.0")
        
        if version >= Version("0.5.1"):
            self.environment = "environment"
            self.reward_specification = "reward_specification"
            self.odor_specifications = "odor_specification"
            self.odor_index = "index"
            
        elif (
            "environment_statistics" in self._data["config"].streams[self.tasklogic].data or 
            "task_parameters" in self._data["config"].streams[self.tasklogic].data
        ):

            self.environment = "environment_statistics"
            self.reward_specification = "reward_specification"
            self.odor_specifications = "odor_specification"
            self.odor_index = "index"

        else:
            print('here')
            self.environment = "environmentStatistics"
            self.reward_specification = "rewardSpecifications"
            self.odor_specifications = "odorSpecifications"
            self.odor_index = "odorIndex"

        if "task_parameters" in self._data["config"].streams[self.tasklogic].data:
            if 'blocks' in self._data["config"].streams[self.tasklogic].data["task_parameters"][self.environment].keys():
                patches = []
                for i, blocks in enumerate(self._data["config"].streams['tasklogic_input'].data["task_parameters"]['environment']['blocks']):
                    if len(blocks['environment_statistics']['patches']) > 0:
                        patches.extend(blocks['environment_statistics']['patches'])
                    else:
                        patches.extend(blocks['environment_statistics']['patches'][i])
                self.patches = patches
                
            else:
                self.patches = (
                    self._data["config"].streams[self.tasklogic].data["task_parameters"][self.environment]["patches"]
                )
        else:
            self.patches = self._data["config"].streams[self.tasklogic].data[self.environment]["patches"]
            
class ContinuousData:
    def __init__(self, data, load_continuous: bool = True):

        self.data = data

        self.data["harp_behavior"].streams.OutputSet.load_from_file()
        self.data["harp_behavior"].streams.PulseSupplyPort0.load_from_file()  # Duration of each pulse
        self.data["harp_behavior"].streams.DigitalInputState.load_from_file()
        if "rig_input" in self.data["config"].streams.keys():
            self.rig = "rig_input"
        else:
            self.rig = "Rig"
        self.data["config"].streams[self.rig].load_from_file()

        if "schema_version" in self.data["config"].streams[self.rig].data:
            self.current_version = Version(self.data["config"].streams[self.rig].data["schema_version"])
        elif "version" in self.data["config"].streams[self.rig].data:
            self.current_version = Version(self.data["config"].streams[self.rig].data["version"])
        else:
            self.current_version = Version("0.0.0")

        if load_continuous == True:
            self.encoder_data = self.encoder_loading()
            self.choice_feedback = self.choice_feedback_loading()
            self.lick_onset, self.lick_offset = self.lick_onset_loading()
            self.give_reward, self.pulse_duration = self.water_valve_loading()
            # self.succesful_wait = self.succesfull_wait_loading()
            self.sniff_data_loading()
            self.position_loading()
            self.odor_triggers = odor_data_harp_olfactometer(self.data)

    def position_loading(self):
        position = self.data['operation_control'].streams.CurrentPosition.data
        self.position_data = position
        
        return self.position_data
        
    def encoder_loading(self, parser: str = "filter"):
        ## Load data from encoder efficiently
        if self.current_version >= Version("0.4.0"):
            self.data["harp_treadmill"].streams.SensorData.load_from_file()
            sensor_data = self.data["harp_treadmill"].streams.SensorData.data

            wheel_size = self.data["config"].streams[self.rig].data["harp_treadmill"]["calibration"]['output']["wheel_diameter"]
            PPR = self.data["config"].streams[self.rig].data["harp_treadmill"]["calibration"]['output']["pulses_per_revolution"]
            invert_direction = (
                self.data["config"].streams[self.rig].data["harp_treadmill"]["calibration"]['output']["invert_direction"]
            )

            converter = wheel_size * np.pi / PPR * (-1 if invert_direction else 1)
            sensor_data["Encoder"] = sensor_data.Encoder.diff()
            dispatch = 250
            
        elif self.current_version >= Version("0.3.0") and self.current_version < Version("0.4.0"):
            self.data["harp_treadmill"].streams.SensorData.load_from_file()
            sensor_data = self.data["harp_treadmill"].streams.SensorData.data

            wheel_size = self.data["config"].streams[self.rig].data["harp_treadmill"]["calibration"]["wheel_diameter"]
            PPR = self.data["config"].streams[self.rig].data["harp_treadmill"]["calibration"]["pulses_per_revolution"]
            invert_direction = (
                self.data["config"].streams[self.rig].data["harp_treadmill"]["calibration"]["invert_direction"]
            )

            sensor_data["Encoder"] = sensor_data.Encoder.diff()
            dispatch = 250

        else:
            self.data["harp_behavior"].streams.AnalogData.load_from_file()
            sensor_data = self.data["harp_behavior"].streams.AnalogData.data
            if "settings" in self.data["config"].streams[self.rig].data["treadmill"].keys():
                wheel_size = self.data["config"].streams[self.rig].data["treadmill"]["settings"]["wheel_diameter"]
                PPR = self.data["config"].streams[self.rig].data["treadmill"]["settings"]["pulses_per_revolution"]
                invert_direction = (
                    self.data["config"].streams[self.rig].data["treadmill"]["settings"]["invert_direction"]
                )
            else:
                if "wheel_diameter" in self.data["config"].streams[self.rig].data["treadmill"].keys():
                    wheel_diameter = "wheel_diameter"
                    pulses = "pulses_per_revolution"
                    invert = "invert_direction"
                else:
                    wheel_diameter = "wheelDiameter"
                    pulses = "pulsesPerRevolution"
                    invert = "invertDirection"

                wheel_size = self.data["config"].streams[self.rig].data["treadmill"][wheel_diameter]
                PPR = self.data["config"].streams[self.rig].data["treadmill"][pulses]
                invert_direction = self.data["config"].streams[self.rig].data["treadmill"][invert]

            dispatch = 1000
        
        converter = wheel_size * np.pi / PPR * (-1 if invert_direction else 1)
        if parser == "filter":
            sensor_data["velocity"] = (
                sensor_data["Encoder"] * converter
            ) * dispatch  # To be replaced by dispatch rate whe it works
            sensor_data["distance"] = sensor_data["Encoder"] * converter
            sensor_data = processing.fir_filter(sensor_data, "velocity", 50)
            encoder = sensor_data[["filtered_velocity"]]

        elif parser == "resampling":
            encoder = sensor_data["Encoder"]
            encoder = encoder.apply(lambda x: x * converter)
            encoder.index = pd.to_datetime(encoder.index, unit="s")
            encoder = encoder.resample("33ms").sum().interpolate(method="linear") / 0.033
            encoder.index = encoder.index - pd.to_datetime(0)
            encoder.index = encoder.index.total_seconds()
            encoder = encoder.to_frame()
            encoder.rename(columns={"Encoder": "filtered_velocity"}, inplace=True)

        self.encoder_data = encoder

        return self.encoder_data

    def torque_loading(self, parser: str = "filter"):
        ## Load data from encoder efficiently
        if self.current_version >= Version("0.3.0"):
            self.data["harp_treadmill"].streams.SensorData.load_from_file()
            torque_data = self.data["harp_treadmill"].streams.SensorData.data[["Torque", "TorqueLoadCurrent"]]

            self.data["harp_treadmill"].streams.BrakeCurrentSetPoint.load_from_file()
            brake_data = self.data["harp_treadmill"].streams.BrakeCurrentSetPoint.data
        return torque_data, brake_data

    def choice_feedback_loading(self):
        self.data["harp_behavior"].streams.PwmStart.load_from_file()
        if self.current_version < Version("0.3.0"):
            # Find responses to Reward site
            choice_feedback = self.data["harp_behavior"].streams.PwmStart.data.loc[
                self.data["harp_behavior"].streams.PwmStart.data["PwmDO1"] == True
            ]
        else:
            choice_feedback = self.data["harp_behavior"].streams.PwmStart.data.loc[
                self.data["harp_behavior"].streams.PwmStart.data["PwmDO2"] == True
            ]
        return choice_feedback

    def lick_onset_loading(self):
        if "harp_lickometer" in self.data:
            self.data["harp_lickometer"].streams.LickState.load_from_file()
            licks = self.data["harp_lickometer"].streams.LickState.data["Channel0"] == True
            lick_onset = licks.loc[licks == True]
            lick_offset = licks.loc[licks == False]

        else:
            di_state = self.data["harp_behavior"].streams.DigitalInputState.data["DIPort0"]
            lick_onset = di_state.loc[di_state == True]
            lick_offset = di_state.loc[di_state == False]
        return lick_onset, lick_offset

    def water_valve_loading(self):
        # Find hardware reward events
        give_reward = self.data["harp_behavior"].streams.OutputSet.data[["SupplyPort0"]]
        self.give_reward = give_reward.loc[give_reward.SupplyPort0 == True]

        # Pulses delivered for water
        self.pulse_duration = self.data["harp_behavior"].streams.PulseSupplyPort0.data["PulseSupplyPort0"]

        return self.give_reward, self.pulse_duration

    def sniff_data_loading(self):
        if "harp_sniffsensor" in self.data:
            self.data["harp_sniffsensor"].streams.RawVoltage.load_from_file()
            self.breathing = pd.DataFrame(
                index=self.data["harp_sniffsensor"].streams.RawVoltage.data["RawVoltage"].index,
                columns=["data"],
            )
            self.breathing["data"] = self.data["harp_sniffsensor"].streams.RawVoltage.data["RawVoltage"].values

        else:
            ## Breathing
            self.breathing = pd.DataFrame(
                index=self.data["harp_behavior"].streams.AnalogData.data["AnalogInput0"].index,
                columns=["data"],
            )
            self.breathing["data"] = self.data["harp_behavior"].streams.AnalogData.data["AnalogInput0"].values
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
            if row["patch_number"] != previous_patch:
                previous_patch = row["patch_number"]
                cumulative_rewards = 0

            self.reward_sites.loc[index, "cumulative_rewards"] = cumulative_rewards

            if row["is_reward"] != 0:
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
            if "reward_function" not in patches[self.schema_properties.reward_specification]:
                dict_odor[patches["label"]] = np.repeat(
                    patches[self.schema_properties.reward_specification]["amount"], 500
                )
                update = "site_number"
            else:
                function_type = patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["function_type"]
                if (
                    function_type
                    == "ConstantFunction"
                ):
                    odor_label = patches["label"]
                    y = np.repeat(
                        patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["value"],
                        500,
                    )
                elif(
                    function_type
                    == 'LookupTableFunction' ):
                    odor_label = patches["label"]
                    y = np.array(
                        patches[self.schema_properties.reward_specification]['reward_function']['amount']['lut_values']
                    )
                elif(
                    function_type
                    == 'PowerFunction' ):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["b"]
                    c = -patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["c"]
                    d = patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["d"]

                    # Generate x values
                    y = a * pow(b, -c * x) + d
                elif(
                    function_type == 'LinearFunction'):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["b"]
                    y = b + a * x
                    
                dict_odor[odor_label] = y

                depletion_rule = patches[self.schema_properties.reward_specification]['reward_function']['depletion_rule']
                if depletion_rule == 'OnChoice':
                    update = "site_number"
                elif depletion_rule == 'OnReward':
                    update = "cumulative_rewards"
                    
        for index, row in self.reward_sites.iterrows():           
            self.reward_sites.at[index, "reward_amount"] = np.around(
                dict_odor[row["patch_label"]][int(row[update])], 3
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
            if "reward_function" not in patches[self.schema_properties.reward_specification]:
                dict_odor[patches["label"]] = np.repeat(
                    patches[self.schema_properties.reward_specification]["probability"],
                    500,
                )
                update = 'site_number'
            else:
                function_type = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["function_type"]
                if (
                    function_type
                    == "ConstantFunction"
                ):
                    odor_label = patches["label"]
                    y = np.repeat(
                        patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["value"],
                        500,
                    )
                elif(
                    function_type
                    == 'LookupTableFunction'):
                    odor_label = patches["label"]
                    y = np.array(
                        patches[self.schema_properties.reward_specification]['reward_function']['probability']['lut_values']
                    )
                elif(
                    function_type
                    == 'PowerFunction' ):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["b"]
                    c = -patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["c"]
                    d = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["d"]

                    # Generate x values
                    y = a * pow(b, -c * x) + d
                elif(
                    function_type == 'LinearFunction'):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["b"]
                    y = b + a * x
                
                dict_odor[odor_label] = y

                depletion_rule = patches[self.schema_properties.reward_specification]['reward_function']['depletion_rule']
                if depletion_rule == 'OnChoice':
                    update = "site_number"
                elif depletion_rule == 'OnReward':
                    update = "cumulative_rewards"
            
        #### ----------- Need to add the modification for On Choice, right now specific for OnReward
        for index, row in self.reward_sites.iterrows():
            self.reward_sites.at[index, "reward_probability"] = np.around(
                dict_odor[row["patch_label"]][int(row[update])], 3
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
            if "reward_function" not in patches[self.schema_properties.reward_specification]:
                if patches["patchRewardFunction"]["initialRewardAmount"] >= 100:
                    dict_odor[patches["label"]] = np.repeat(100, 500)
                else:
                    odor_label = patches["label"]
                    initial = patches["patchRewardFunction"]["initialRewardAmount"]
                    amount = patches[self.schema_properties.reward_specification]["amount"]
                    y = initial - amount * x
                    dict_odor[odor_label] = y
                update = "site_number"
            else:
                function_type = patches[self.schema_properties.reward_specification]["reward_function"]["available"][
                    "function_type"]
                if (
                    function_type == "ConstantFunction"
                ):
                    odor_label = patches["label"]
                    y = np.repeat(
                        patches[self.schema_properties.reward_specification]["reward_function"]['available']["value"],
                        500,
                    )
                elif(
                    function_type
                    == 'LookupTableFunction' ):
                    odor_label = patches["label"]
                    y = np.array(
                        patches[self.schema_properties.reward_specification]['reward_function']['available']['lut_values']
                    )
                elif(
                    function_type
                    == 'PowerFunction'):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["b"]
                    c = -patches[self.schema_properties.reward_specification]["reward_function"]["available"]["c"]
                    d = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["d"]

                    # Generate x values
                    y = a * pow(b, -c * x) + d

                elif(
                    function_type == 'LinearFunction'):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["b"]
                    y = b + a * x
                    
                dict_odor[odor_label] = y
                
                depletion_rule = patches[self.schema_properties.reward_specification]['reward_function']['depletion_rule']
                if depletion_rule == 'OnChoice':
                    update = "site_number"
                elif depletion_rule == 'OnReward':
                    update = "cumulative_rewards"
                
        for index, row in self.reward_sites.iterrows():
            self.reward_sites.at[index, "reward_available"] = np.around(
                dict_odor[row["patch_label"]][int(row[update])], 3)

        return self.reward_sites

def read_harp_bin(file):
    """
    Reads binary data from a HARP file and returns it as a pandas DataFrame.
    Parameters:
    file (str or file-like object): The path to the HARP binary file or a file-like object.

    Returns:
    pd.DataFrame or None: A DataFrame containing the parsed data with 'Seconds' as the index.
                          Returns None if the file is empty.
    Notes:
    - The function assumes a specific binary format for the HARP file.
    - The payload type and size are determined based on the data in the file.
    - The DataFrame's index is in seconds, calculated from ticks and seconds in the file.
    """

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

def load_device_with_fallback(local_dir: Path, filename: str, url: str):
    local_path = local_dir / "behavior" / filename / "device.yml"
    if local_path.exists():
        return data_io.reader_from_yml(local_path)
    return data_io.reader_from_url(url)


def load_session_data(
    session_path: str | PathLike,
) -> Dict[str, data_io.DataStreamSource]:
    _out_dict = {}
    session_path = Path(session_path)

    session_path_behavior = session_path
    session_path_config = session_path
    suffix = "Register__"

    # Work around the change in the folder structure
    if "behavior" in os.listdir(session_path):
        session_path_behavior = session_path / "behavior"
        suffix = None
    else:
        session_path_behavior = session_path
        
    if "other" in os.listdir(session_path):
        session_path_config = session_path / "other"
    else:
        session_path_config = session_path

    if "Behavior.harp" in os.listdir(session_path_behavior):
        HarpBehavior = load_device_with_fallback(session_path, "Behavior.harp", r"https://raw.githubusercontent.com/harp-tech/device.behavior/main/device.yml")
        _out_dict["harp_behavior"] = data_io.HarpSource(
            device=HarpBehavior,
            path=session_path_behavior / "Behavior.harp",
            name="behavior",
            autoload=False,
            remove_suffix=suffix,
        )
    elif "Behavior" in os.listdir(session_path_behavior):
        print("Old behavior loading")
        HarpBehavior = load_device_with_fallback(session_path, "Behavior.harp", r"https://raw.githubusercontent.com/harp-tech/device.behavior/main/device.yml")

        _out_dict["harp_behavior"] = data_io.HarpSource(
            device=HarpBehavior,
            path=session_path_behavior / "Behavior",
            name="behavior",
            autoload=False,
            remove_suffix=suffix,
        )

    if "Olfactometer.harp" in os.listdir(session_path_behavior):
        HarpOlfactometer = load_device_with_fallback(session_path, "Olfactometer.harp", r"https://raw.githubusercontent.com/harp-tech/device.olfactometer/7f38395f95c164bb55821139f752cd579da8d4af/device.yml")
        _out_dict["harp_olfactometer"] = data_io.HarpSource(
            device=HarpOlfactometer,
            path=session_path_behavior / "Olfactometer.harp",
            name="olfactometer",
            autoload=False,
            remove_suffix=suffix,
        )

    if "Lickometer.harp" in os.listdir(session_path_behavior):
        HarpLickometer = load_device_with_fallback(session_path, "Lickometer.harp", r"https://raw.githubusercontent.com/AllenNeuralDynamics/harp.device.lickety-split/main/device.yml")
        _out_dict["harp_lickometer"] = data_io.HarpSource(
            device=HarpLickometer,
            path=session_path_behavior / "Lickometer.harp",
            name="lickometer",
            autoload=False,
            remove_suffix=suffix,
        )

    if "SniffDetector.harp" in os.listdir(session_path_behavior):
        HarpSniffsensor = load_device_with_fallback(session_path, "SniffDetector.harp", r"https://raw.githubusercontent.com/AllenNeuralDynamics/harp.device.sniff-detector/main/device.yml")
        _out_dict["harp_sniffsensor"] = data_io.HarpSource(
            device=HarpSniffsensor,
            path=session_path_behavior / "SniffDetector.harp",
            name="sniffdetector",
            autoload=False,
            remove_suffix=suffix,
        )

    if "StepperDriver.harp" in os.listdir(session_path_behavior):
        HarpStepperDriver = load_device_with_fallback(session_path, "StepperDriver.harp", r"https://raw.githubusercontent.com/harp-tech/device.stepperdriver/main/device.yml")
        
        _out_dict["harp_stepperdriver"] = data_io.HarpSource(
            device=HarpStepperDriver,
            path=session_path_behavior / "StepperDriver.harp",
            name="stepper_driver",
            autoload=False,
            remove_suffix=suffix,
        )

    if "ClockGenerator.harp" in os.listdir(session_path_behavior):
        HarpClock = load_device_with_fallback(session_path, "ClockGenerator.harp", r"https://raw.githubusercontent.com/harp-tech/device.clocksynchronizer/main/device.yml")
        _out_dict["harp_clock"] = data_io.HarpSource(
            device=HarpClock,
            path=session_path_behavior / "ClockGenerator.harp",
            name="clock",
            autoload=False,
            remove_suffix=suffix,
        )

    if "Treadmill.harp" in os.listdir(session_path_behavior):
        HarpTreadmill = load_device_with_fallback(session_path, "Treadmill.harp", r"https://raw.githubusercontent.com/AllenNeuralDynamics/harp.device.treadmill-driver/main/device.yml")
        
        _out_dict["harp_treadmill"] = data_io.HarpSource(
            device=HarpTreadmill, path=session_path_behavior / "Treadmill.harp", name="treadmill", autoload=False
        )
        
    if "EnvironmentSensor.harp" in os.listdir(session_path_behavior):
        HarpEnvironmentSensor = load_device_with_fallback(session_path, "EnvironmentSensor.harp", r"https://raw.githubusercontent.com/AllenNeuralDynamics/harp.device.environment-sensor/refs/heads/main/device.yml")
        
        _out_dict["harp_environment_sensor"] = data_io.HarpSource(
            device=HarpEnvironmentSensor, path=session_path_behavior / "EnvironmentSensor.harp", name="environment_sensor", autoload=False
        )

    if "AnalogInput.harp" in os.listdir(session_path_behavior):
        HarpAnalogInput = load_device_with_fallback(session_path, "AnalogInput.harp", r"https://raw.githubusercontent.com/harp-tech/device.analoginput/main/device.yml")
    
        _out_dict["harp_analog"] = data_io.HarpSource(
            device=HarpAnalogInput,
            path=session_path_behavior / "AnalogInput.harp",
            name="analog_input",
            autoload=False,
            remove_suffix=suffix,
        )

    if "OperationControl" in os.listdir(session_path_behavior):
        _out_dict["operation_control"] = data_io.OperationControlSource(
            path=session_path_behavior / "OperationControl", name="operation_control", autoload=True
        )

    if "UpdaterEvents" in os.listdir(session_path_behavior):
        _out_dict["updater_events"] = data_io.SoftwareEventSource(
            path=session_path_behavior / "UpdaterEvents", name="updater_events", autoload=True
        )

    if "SoftwareEvents" in os.listdir(session_path_behavior):
        _out_dict["software_events"] = data_io.SoftwareEventSource(
            path=session_path_behavior / "SoftwareEvents", name="software_events", autoload=True
        )
    # Load config old version
    if "config.json" in os.listdir(session_path_config):
        with open(str(session_path_config) + r"\config.json", "r") as json_file:
            _out_dict["config"] = json.load(json_file)
    elif "Logs" in os.listdir(session_path_behavior):
        try:
            _out_dict["config"] = data_io.ConfigSource(path=session_path_behavior / "Logs", name="config", autoload=True)
        except json.JSONDecodeError:
            print("Config file is empty or corrupted, loading manually")
            _out_dict["config"] = data_io.ConfigSource(path=session_path_behavior / "Logs", name="config", autoload=False)
            try:
                for item in ['rig_input', 'tasklogic_input', 'session_input']:
                    _out_dict['config'].streams[item].load_from_file()
            except KeyError:
                for item in ['Rig', 'TaskLogic']:
                    _out_dict['config'].streams[item].load_from_file()
    return _out_dict

## ------------------------------------------------------------------------- ##
def odor_data_harp_olfactometer(data):
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

    # Selecting which odor valve is open before the end valves are opened
    OdorValveState = pd.DataFrame()
    OdorValveState["time"] = data["harp_olfactometer"].streams.OdorValveState.data.index.values
    data["harp_olfactometer"].streams.OdorValveState.data["Valve0"] = np.where(
        data["harp_olfactometer"].streams.OdorValveState.data["Valve0"] == True,
        "0",
        False,
    )
    data["harp_olfactometer"].streams.OdorValveState.data["Valve1"] = np.where(
        data["harp_olfactometer"].streams.OdorValveState.data["Valve1"] == True,
        "1",
        False,
    )
    data["harp_olfactometer"].streams.OdorValveState.data["Valve2"] = np.where(
        data["harp_olfactometer"].streams.OdorValveState.data["Valve2"] == True,
        "2",
        False,
    )

    # Create a new dataframe to store the results
    OdorValveState = pd.DataFrame(columns=["time", "condition"])

    # Loop through each row and find the values that are not False
    for index, row in data["harp_olfactometer"].streams.OdorValveState.data[["Valve0", "Valve1", "Valve2"]].iterrows():
        non_false_values = row[row != "False"].tolist()
        if non_false_values:  # Check if there are any non-False values
            OdorValveState = pd.concat(
                [
                    OdorValveState,
                    pd.DataFrame([[index, non_false_values[0]]], columns=["time", "condition"]),
                ]
            )

    EndValveState = pd.DataFrame()
    EndValve = data["harp_olfactometer"].streams.EndValveState.data
    EndValve = EndValve[EndValve["MessageType"] == "WRITE"]
    EndValveState["time"] = EndValve.index.values
    EndValveState["condition"] = np.where(
        EndValve["EndValve0"] == True,
        "EndValveOn",
        "EndValveOff",
    )

    odor_updates = pd.concat([EndValveState[["time", "condition"]], OdorValveState[["time", "condition"]]])
    odor_updates = odor_updates.sort_values(by="time")
    odor_updates = odor_updates[odor_updates["condition"] != False]

    odor_triggers = pd.DataFrame(columns=["odor_onset", "odor_offset", "patch_type"])
    onset = np.nan
    offset = np.nan
    opened = np.nan
    condition = "EndValveOff"
    for i, row in odor_updates.iterrows():
        if (row["condition"] != "EndValveOn") and (row["condition"] != "EndValveOff"):
            condition = row["condition"]
            opened = True

        elif row["condition"] == "EndValveOn":
            onset = row["time"]
            opened = True

        elif row["condition"] == "EndValveOff" and opened:
            offset = row["time"]
            opened = False

        if opened == False:
            new_row = {
                "odor_onset": onset,
                "odor_offset": offset,
                "patch_type": condition,
            }
            odor_triggers.loc[len(odor_triggers)] = new_row
            condition = row["condition"]

    if row["condition"] == "EndValveOn":
        new_row = {"odor_onset": onset, "odor_offset": np.nan, "patch_type": condition}
        odor_triggers.loc[len(odor_triggers)] = new_row

    # Assign odor labels to odor indexes
    odor0 = False
    odor1 = False
    odor2 = False

    data["config"].streams[schema_properties.tasklogic].load_from_file()

    for patches in schema_properties.patches:
        if patches[schema_properties.odor_specifications][schema_properties.odor_index] == 0:
            odor0 = patches["label"]
        elif patches[schema_properties.odor_specifications][schema_properties.odor_index] == 1:
            odor1 = patches["label"]
        else:
            odor2 = patches["label"]

    odor_triggers["patch_type"] = np.where(odor_triggers["patch_type"] == "0", odor0, odor_triggers["patch_type"])
    odor_triggers["patch_type"] = np.where(odor_triggers["patch_type"] == "1", odor1, odor_triggers["patch_type"])
    odor_triggers["patch_type"] = np.where(odor_triggers["patch_type"] == "2", odor2, odor_triggers["patch_type"])

    # return reward_sites  ## ------------------------------------------------------------------------- ##
    return odor_triggers

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
        encoder_data["Encoder"] = read_harp_bin(path + r"\Behavior\Register__44" + ".bin")[1]

    try:
        # Open and read the JSON file
        if 'Config' in os.listdir(path + r"\behavior"):
            with open(str(path) + r"\behavior\Config\TaskLogic.json", "r") as json_file:
                config = json.load(json_file)
        else:
            with open(str(path) + r"\behavior\Logs\TaskLogic.json", "r") as json_file:
                config = json.load(json_file)

    except:
        with open(str(path) + r"\config.json", "r") as json_file:
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

    active_site["label"] = np.where(active_site["label"] == "Reward", "OdorSite", active_site["label"])
    active_site.rename(columns={"startPosition": "start_position"}, inplace=True)
    # Rename columns

    active_site = active_site[["label", "start_position", "length"]]
    reward_sites = active_site[active_site["label"] == "OdorSite"]

    data["software_events"].streams.GiveReward.load_from_file()
    reward = data["software_events"].streams.GiveReward.data
    reward.fillna(0, inplace=True)

    try:
        data["software_events"].streams.ActivePatch.load_from_file()
        patches = data["software_events"].streams.ActivePatch.data

    except:
        patches = active_site.loc[active_site["label"] == "InterPatch"]
        patches.rename(columns={"label": "name"}, inplace=True)
        patches["name"] = np.where(patches["name"] == "InterPatch", "ActivePatch", patches["name"])

    try:
        # Old way of obtaining the reward amount
        reward_available = event[1]["data"]["patchRewardFunction"]["initialRewardAmount"]
    except:
        reward_available = config["environmentStatistics"]["patches"][0]["patchRewardFunction"]["initialRewardAmount"]

    reward_updates = pd.concat([patches, reward])
    reward_updates.sort_index(inplace=True)
    reward_updates["current_reward"] = np.nan

    for event in reward_updates.iterrows():
        if event[1]["name"] == "GiveReward":  # update reward
            reward_available -= event[1]["data"]
        elif event[1]["name"] == "ActivePatch":  # reset reward
            try:
                # Old way of obtaining the reward amount
                reward_available = event[1]["data"]["patchRewardFunction"]["initialRewardAmount"]
            except:
                reward_available = config["environmentStatistics"]["patches"][0]["patchRewardFunction"][
                    "initialRewardAmount"
                ]
        else:
            raise ValueError("Unknown event type")
        reward_updates.at[event[0], "current_reward"] = reward_available

    for site in reward_sites.itertuples():
        arg_min, val_min = processing.find_closest(site.Index, reward_updates.index.values, mode="below_zero")
        try:
            reward_sites.loc[site.Index, "reward_available"] = reward_updates["current_reward"].iloc[arg_min]
        except:
            reward_sites.loc[site.Index, "reward_available"] = reward_updates["current_reward"].iloc[arg_min]

    # Find responses to Reward site
    data["software_events"].streams.ChoiceFeedback.load_from_file()
    choiceFeedback = data["software_events"].streams.ChoiceFeedback.data

    reward_sites.loc[:, "patch_number"] = -1
    reward_sites.loc[:, "site_number"] = -1
    reward_sites.loc[:, "is_choice"] = False
    reward_sites.loc[:, "is_reward"] = 0
    reward_sites.loc[:, "past_no_reward_count"] = 0
    past_no_reward_counter = 0
    current_patch_idx = -1

    site_number = 0
    for idx, event in enumerate(reward_sites.iterrows()):
        arg_min, val_min = processing.find_closest(event[0], patches.index.values, mode="below_zero")
        if not (np.isnan(arg_min)):
            reward_sites.loc[event[0], "patch_number"] = arg_min
        if current_patch_idx != arg_min:
            current_patch_idx = arg_min
            site_number = 0
        else:
            site_number += 1
        reward_sites.loc[event[0], "site_number"] = site_number

        if idx < len(reward_sites) - 1:
            choice = choiceFeedback.loc[
                (choiceFeedback.index >= reward_sites.index[idx]) & (choiceFeedback.index < reward_sites.index[idx + 1])
            ]
            reward_in_site = reward.loc[
                (reward.index >= reward_sites.index[idx]) & (reward.index < reward_sites.index[idx + 1])
            ]
        else:
            choice = choiceFeedback.loc[(choiceFeedback.index >= reward_sites.index[idx])]
            reward_in_site = reward.loc[(reward.index >= reward_sites.index[idx])]
        reward_sites.loc[event[0], "is_choice"] = len(choice) > 0
        reward_sites.loc[event[0], "is_reward"] = (
            reward_in_site.iloc[0]["data"] if len(reward_in_site) > 0 else 0
        )
        reward_sites.loc[event[0], "past_no_reward_count"] = past_no_reward_counter
        if reward_sites.loc[event[0], "is_reward"] == 0 and reward_sites.loc[event[0], "is_choice"] == 1:
            past_no_reward_counter += 1
        else:
            past_no_reward_counter = 0
    try:
        df_patch = pd.json_normalize(patches["data"])
        df_patch.reset_index(inplace=True)
        df_patch.rename(
            columns={
                "index": "patch_number",
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
        df_patch = pd.DataFrame(columns=["patch_number", "odor_label", "amount"])
        df_patch["patch_number"] = np.arange(len(patches))
        df_patch["odor_label"] = config["environmentStatistics"]["patches"][0]["label"]
        df_patch["amount"] = config["environmentStatistics"]["patches"][0]["rewardSpecifications"]["amount"]

    reward_sites = pd.merge(
        reward_sites.reset_index(),
        df_patch[["odor_label", "patch_number", "amount"]],
        on="patch_number",
    )

    # Create new column for adjusted seconds to start of session
    reward_sites["adj_seconds"] = reward_sites["Seconds"] - start_time
    reward_sites.index = reward_sites["Seconds"]
    reward_sites.drop(columns=["Seconds"], inplace=True)

    # ---------------- Add water triggers times ---------------- #
    data["harp_behavior"].streams.OutputSet.load_from_file()
    water = data["harp_behavior"].streams.OutputSet.data[["SupplyPort0"]]
    reward_sites["next_index"] = reward_sites.index.to_series().shift(-1)
    reward_sites["reward_onset"] = None

    # Iterate through the actual index of df1
    for value in water.index:
        # Check if the value is between 'Start' and 'End' in df2
        matching_row = reward_sites[(reward_sites.index <= value) & (reward_sites["next_index"].values >= value)]

        # If a matching row is found, update the corresponding row in water with the index value
        if not matching_row.empty:
            matching_index = matching_row.index[0]  # Assuming there's at most one matching row
            reward_sites.at[matching_index, "reward_onset"] = value

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

    odor_df = pd.concat([odor_0[["Time", "odor_onset"]], odor_1[["Time", "odor_onset"]]])
    odor_df.sort_index(inplace=True)
    odor_df.dropna(inplace=True)

    odor_df["time_diff"] = odor_df["Time"].diff()
    odor_df = odor_df.drop(index=odor_df.loc[(odor_df["time_diff"] < 1) & (odor_df.index > 0)].index)

    try:
        reward_sites["odor_onset"] = odor_df["Time"].values
    except:
        pass

    # ---------------- Add stop triggers times ---------------- #
    reward_sites["stop_time"] = None

    # Iterate through the actual index of df1
    for value in choiceFeedback.index:
        # Check if the value is between 'Start' and 'End' in df2
        matching_row = reward_sites[(reward_sites.index <= value) & (reward_sites["next_index"].values >= value)]

        # If a matching row is found, update the corresponding row in water with the index value
        if not matching_row.empty:
            matching_index = matching_row.index[0]  # Assuming there's at most one matching row
            reward_sites.at[matching_index, "stop_time"] = value

    reward_sites.drop(columns=["next_index"], inplace=True)
    # ---------------------------------------------------- #

    # Add colum for site number
    reward_sites.loc[:, "odor_sites"] = np.arange(len(reward_sites))
    reward_sites.loc[:, "depleted"] = np.where(reward_sites["reward_available"] == 0, 1, 0)
    reward_sites.loc[:, "collected"] = np.where((reward_sites["is_reward"] != 0), 1, 0)

    # reward_sites['next_site_number'] = reward_sites['site_number'].shift(-2)
    # reward_sites['last_visit'] = np.where(reward_sites['next_site_number']==0, 1, 0)
    # reward_sites.drop(columns=['next_site_number'], inplace=True)

    # reward_sites['last_site'] = reward_sites['site_number'].shift(-1)
    # reward_sites['last_site'] = np.where(reward_sites['last_site'] == 0, 1,0)

    # reward_sites['next_patch'] = reward_sites['patch_number'].shift(1)
    # reward_sites['next_odor'] = reward_sites['odor_label'].shift(1)
    # reward_sites['same_patch'] = np.where((reward_sites['next_patch'] != reward_sites['patch_number'])&(reward_sites['odor_label'] == reward_sites['next_odor'] ), 1, 0)
    # reward_sites.drop(columns=['next_patch', 'next_odor'], inplace=True)

    encoder_data = processing.fir_filter(encoder_data, "velocity", cutoff_hz=5)

    if reward_sites.reward_available.max() >= 100:
        reward_sites["reward_available"] = 100

    return reward_sites, active_site, encoder_data, config

## ------------------------------------------------------------------------- ##
def parse_dataframe(data: dict) -> pd.DataFrame:
    """
    Parse the data from the session and return the reward sites, active sites and encoder data

    Inputs:
    data: dict
        Data from the session

    Returns:
    all_epochs: pd.DataFrame
        DataFrame containing the  active sites

    """
    data["software_events"].streams.ActiveSite.load_from_file()
    active_site_temp = data["software_events"].streams.ActiveSite.data

    # Use json_normalize to create a new DataFrame from the 'data' column
    active_site = pd.json_normalize(active_site_temp["data"])
    active_site.index = active_site_temp.index

    # Add the postpatch label
    active_site["previous_epoch"] = active_site["label"].shift(-1)
    active_site["label"] = np.where(
        active_site["label"] == active_site["previous_epoch"], "PostPatch", active_site["label"]
    )
    active_site.drop(columns=["previous_epoch"], inplace=True)

    active_site["label"] = active_site["label"].replace("Reward", "OdorSite")
    active_site["label"] = active_site["label"].replace("RewardSite", "OdorSite")
    
    if "treadmill_specification.friction.distribution_parameters.value" in active_site.columns:
        active_site.rename(
            columns={
                "startPosition": "start_position",
                "treadmill_specification.friction.distribution_parameters.value": "friction",
            },
            inplace=True,
        )
        # Crop and rename columns
        active_site = active_site[["label", "start_position", "length", "friction"]]

    else:
        active_site.rename(columns={"startPosition": "start_position"}, inplace=True)
        active_site = active_site[["label", "start_position", "length"]]

    # Add patch_number column
    group = (active_site["label"] == "InterPatch").cumsum()
    active_site["patch_number"] = group - 1

    # Patch initialization
    data["software_events"].streams.ActivePatch.load_from_file()
    patches = data["software_events"].streams.ActivePatch.data

    # Instances where a patch gets defined but it's not really used. Happens during block transitions. 
    patches['real_diff'] = patches.index.to_series().diff().shift(-1).fillna(0.1)
    patches = patches[patches.real_diff >= 0.09]

    df_patch = pd.json_normalize(patches["data"])
    df_patch.index = patches.index

    df_patch["patch_number"] = np.arange(len(df_patch))
    if "odor_specification.index" in df_patch.columns:
        df_patch.rename(columns={"label": "patch_label", "odor_specification.index": "odor_label"}, inplace=True)
        df_patch = df_patch[["patch_label", "patch_number", "odor_label"]]
    else: 
        df_patch.rename(columns={"label": "odor_label"}, inplace=True)
        df_patch = df_patch[["patch_number", "odor_label"]]
        df_patch["patch_label"] = df_patch["odor_label"]
        
    all_epochs = pd.merge(active_site, df_patch, on="patch_number", how="left")
    all_epochs.index = active_site.index
    
# ------------
    try:
        if 'calibration' in data["config"].streams.rig_input.data["harp_olfactometer"]:
            if data["config"].streams.rig_input.data["harp_olfactometer"]["calibration"] is not None:
                # Create a mapping dictionary from the nested structure
                mapping = {i: data["config"].streams.rig_input.data["harp_olfactometer"]["calibration"]['input']['channel_config'][str(i)]['odorant'] for i in range(0, 3)}

                # Replace numbers in the dataframe column with the corresponding odorant values
                all_epochs['odor_label'] = all_epochs['odor_label'].replace(mapping)
                
            else:
                all_epochs["odor_label"] = all_epochs['patch_label']   
        else:
            all_epochs["odor_label"] = all_epochs['patch_label']
    except:
        all_epochs["odor_label"] = all_epochs['patch_label']
# ----------------

    # Count 'OdorSite' occurrences within each group
    all_epochs["site_number"] = all_epochs[all_epochs["label"] == "OdorSite"].groupby(group).cumcount()
    all_epochs["stop_time"] = all_epochs.index.to_series().shift(-1)
    all_epochs.index.name = "start_time"
    
    ## Add last timestamp
    try:
        data["config"].streams.endsession.load_from_file()
        all_epochs.stop_time.iloc[-1] = data['config'].streams.endsession.data['timestamp']    
    except json.JSONDecodeError:
        print('Removing last epoch because of empty endsession file')
        all_epochs = all_epochs.iloc[:-1]
    except AttributeError:
        print('Removing last epoch because of empty endsession file')
        all_epochs = all_epochs.iloc[:-1]
        
    # Recover tones
    choiceFeedback = ContinuousData(data, load_continuous=False).choice_feedback_loading()

    # Recover water delivery
    water = ContinuousData(data, load_continuous=False).water_valve_loading()[0]

    if "WaitRewardOutcome" in data["software_events"].streams:
        # Successfull waits
        data["software_events"].streams.WaitRewardOutcome.load_from_file()
        succesfull_wait = pd.DataFrame(
            index=data["software_events"].streams.WaitRewardOutcome.data.index,
            columns=["data"],
        )

        new_data = pd.json_normalize(data["software_events"].streams.WaitRewardOutcome.data["data"])["IsSuccessfulWait"]
        succesfull_wait["data"] = new_data.values
        succesfull_wait = succesfull_wait[succesfull_wait["data"] == True]
    else:
        succesfull_wait = pd.Series([])

    # Create an empty list to hold the results
    stop_cues = []
    reward_onsets = []
    successful_waits = []
    reward_sites = all_epochs[all_epochs["label"] == "OdorSite"]
    if reward_sites.empty:
        print("No reward sites found")
        return all_epochs
    
    # Loop over the reward_sites
    for current_idx, row in reward_sites.iterrows():
        # Define the current and next reward site index
        next_idx = row.stop_time
        
        # Find slices based on the current and next indices
        choice = choiceFeedback[(choiceFeedback.index >= current_idx) & (choiceFeedback.index < next_idx)]
        reward_in_site = water[(water.index >= current_idx) & (water.index < next_idx)]
        waits = succesfull_wait[(succesfull_wait.index >= current_idx) & (succesfull_wait.index < next_idx)]

        # Store the first relevant index or NaN
        stop_cues.append(choice.index[0] if len(choice) > 0 else np.nan)
        reward_onsets.append(reward_in_site.index[0] if len(reward_in_site) > 0 else np.nan)
        successful_waits.append(waits.index[0] if len(waits) > 0 else np.nan)

    # Assign the results to the DataFrame
    reward_sites["choice_cue_time"] = stop_cues
    reward_sites["reward_onset_time"] = reward_onsets
    reward_sites["succesful_wait_time"] = successful_waits
    reward_sites["succesful_wait_time"] = reward_sites["reward_onset_time"].combine_first(reward_sites["succesful_wait_time"])

    # Add the new columns for choice and reward delivered
    reward_sites["is_choice"] = reward_sites["choice_cue_time"].notnull().astype(bool)
    reward_sites["is_reward"] = reward_sites["reward_onset_time"].notnull().astype(bool)
    
    # try:
    print("Reward functions from software events")
    merged = reward_sites.copy()
    
    if  "PatchStateAtReward" in data['software_events'].streams:
        # print("Using PatchStateAtReward stream for reward characteristics")
        patch_stats = pd.json_normalize(data['software_events'].streams.PatchStateAtReward.data['data'])
        patch_stats.index = data['software_events'].streams.PatchStateAtReward.data.index
        patch_stats.drop(columns=['PatchId'], inplace=True)
        patch_stats.rename(columns={'Amount':'reward_amount', 'Available':'reward_available', 'Probability':'reward_probability'}, inplace=True)
        
        # Make sure both DataFrames are sorted by index
        reward_sites = reward_sites.sort_index()
        patch_stats = patch_stats.sort_index()

        stopped_sites = reward_sites.loc[reward_sites['is_choice']==True]
        # Perform merge_asof on the index
        merged = pd.merge_asof(
            stopped_sites,
            patch_stats,
            left_index=True,
            right_index=True,
            direction='forward'
        )
        
        merged = pd.concat([reward_sites.loc[reward_sites['is_choice']==False], merged], axis=0).sort_index()
        
    if "GlobalPatchState" in data['software_events'].streams:

            gs = pd.json_normalize(
                data['software_events'].streams.GlobalPatchState.data['data']
            )
            gs.index = data['software_events'].streams.GlobalPatchState.data.index
            gs = gs.drop(columns=['PatchId']).rename(
                columns={
                    'Amount': 'reward_amount',
                    'Available': 'reward_available',
                    'Probability': 'reward_probability',
                }
            )

            merged_gs = pd.merge_asof(
                merged.sort_index(),
                gs.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward',
                suffixes=('', '_global')
            )

            for col in ['reward_amount', 'reward_available', 'reward_probability']:
                merged[col] = merged[col].combine_first(
                    merged_gs[f"{col}_global"]
                )
                
    else:
        # Add the reward characteristics columns
        patch_stats = pd.DataFrame()
        patch_stats.index = data['software_events'].streams.PatchRewardProbability.data.index
        patch_stats['reward_amount'] = data['software_events'].streams.PatchRewardAmount.data['data'].values
        patch_stats['reward_available'] = data['software_events'].streams.PatchRewardAvailable.data['data'].values
        patch_stats['reward_probability'] = data['software_events'].streams.PatchRewardProbability.data['data'].values
        patch_stats['reward_probability'] = patch_stats['reward_probability'].round(3)

        patch_stats['real_diff'] = patch_stats.index.to_series().diff().shift(-1).fillna(5)
        patch_stats = patch_stats[patch_stats.real_diff >= 0.05]
        patch_stats.drop(columns=['real_diff'], inplace=True)
        
        # Make sure both DataFrames are sorted by index
        reward_sites = reward_sites.sort_index()
        patch_stats = patch_stats.sort_index()

        # Perform merge_asof on the index
        merged = pd.merge_asof(
            reward_sites,
            patch_stats,
            left_index=True,
            right_index=True,
            direction='backward'
        )

    assert len(merged) == len(reward_sites), "Length mismatch after merge"

    # Concatenate the results to all_epochs
    all_epochs = pd.concat([all_epochs.loc[all_epochs.label != 'OdorSite'], merged], axis=0).sort_index()
    
    # except (KeyError, AttributeError):
    #     reward_sites = RewardFunctions(data, reward_sites).calculate_reward_functions()
    #     all_epochs = pd.concat([all_epochs.loc[all_epochs.label != 'OdorSite'], reward_sites], axis=0).sort_index()
    #     all_epochs.loc[all_epochs.label == 'Gap', 'label'] = 'InterSite'
        
    return all_epochs

## ------------------------------------------------------------------------- ##
