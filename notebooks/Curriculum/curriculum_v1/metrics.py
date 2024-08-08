# IPython magig  tools
%load_ext autoreload
%autoreload 2

import os
from typing import Dict
from os import PathLike
from pathlib import Path
from pydantic import Field

from aind_behavior_curriculum import (
    Metrics,
)

from aind_vr_foraging_analysis.utils import parse

import pandas as pd
import numpy as np

class MetricsVrForaging:
    def __init__(self, session_path: PathLike):
        self.session_path = Path(session_path)
        self.data = parse.load_session_data(self.session_path)
        self.reward_sites, self.active_site, self.config = parse.parse_dataframe(self.data)
        self.df = self.retrieve_metrics()

    def retrieve_metrics(self) -> pd.DataFrame:
        reward_sites = self.reward_sites
        active_site = self.active_site
        data = self.data

        df = pd.DataFrame()
        # Summary of different relevants aspects -------------------------------------------------

        unrewarded_stops = reward_sites.loc[reward_sites.reward_delivered==0]['reward_amount'].count()
        rewarded_stops = reward_sites.loc[reward_sites.reward_delivered==1]['reward_amount'].count()
        water_collected = reward_sites.loc[(reward_sites['reward_delivered']==1)]['reward_amount'].sum()
        total_stops = reward_sites.loc[(reward_sites['has_choice']==True)]['reward_amount'].count()

        print('Total sites: ' ,len(reward_sites), ' | ', 'Total rewarded stops: ',rewarded_stops, '(',  np.round((rewarded_stops/total_stops)*100,2),'%) | ', 
            'Total unrewarded stops: ',unrewarded_stops,'(',  np.round((unrewarded_stops/total_stops)*100,2),'%) | ','Water consumed: ', water_collected, 'ul')

        print('Total travelled m: ', np.round(active_site.start_position.max()/100,2), ', current position (cm): ', data['operation_control'].streams.CurrentPosition.data.max()[0]
        )

        for odor_label in reward_sites.odor_label.unique():
            values = reward_sites.loc[(reward_sites['odor_label']==odor_label)&(reward_sites['reward_delivered']==1)]['reward_amount'].sum()
            print(f'{odor_label} {values} ul')
            
        df.at[0,'odor_sites_travelled'] = int(len(reward_sites))
        df.at[0,'distance_m'] = data['operation_control'].streams.CurrentPosition.data.max()[0]/100
        df.at[0,'water_collected_ul'] = water_collected
        df.at[0,'rewarded_stops'] = int(rewarded_stops)
        df.at[0,'total_stops'] = int(total_stops)
        df.at[0,'session_duration_min'] = (reward_sites.index[-1] - reward_sites.index[0])/60
        df.at[0, 'total_patches_visited'] = reward_sites.loc[reward_sites['visit_number'] >= 1].active_patch.nunique()
        
        # Initialize a pointer for the data values
        data_pointer = 0

        # Save the updater values
        stop_duration = data['updater_events'].streams.UpdaterStopDurationOffset.data['data']
        stop_duration.reset_index(drop=True, inplace=True)
        delay = data['updater_events'].streams.UpdaterRewardDelayOffset.data['data']
        delay.reset_index(drop=True, inplace=True)
        velocity_threshold = data['updater_events'].streams.UpdaterStopVelocityThreshold.data['data']
        velocity_threshold.reset_index(drop=True, inplace=True)
        
        # Create a new column in reward_sites to store the updated values
        reward_sites['delay_s'] = None
        reward_sites['velocity_threshold_cms'] = None
        reward_sites['stop_duration_s'] = None

        try:
            # Iterate through each row of reward_sites
            for index, row in reward_sites.iterrows():
                if row['reward_delivered'] == 1:
                    # Copy the next available value from data and move the pointer
                    reward_sites.at[index, 'delay_s'] = delay[data_pointer]
                    reward_sites.at[index, 'velocity_threshold_cms'] = velocity_threshold[data_pointer]
                    reward_sites.at[index, 'stop_duration_s'] = stop_duration[data_pointer]
                    data_pointer += 1
                else:
                    # Copy the same value without moving the pointer
                    reward_sites.at[index, 'delay_s'] = delay[data_pointer]
                    reward_sites.at[index, 'velocity_threshold_cms'] = velocity_threshold[data_pointer]
                    reward_sites.at[index, 'stop_duration_s'] = stop_duration[data_pointer]
        except KeyError:
                reward_sites.at[index, 'delay_s'] = max(delay)
                reward_sites.at[index, 'velocity_threshold_cms'] = max(velocity_threshold)
                reward_sites.at[index, 'stop_duration_s'] = max(stop_duration)

        # Summary of the training metrics
        reward_sites['odor_sites'] = np.arange(1, len(reward_sites)+1)
        df.at[0,'start_delay'] = reward_sites['delay_s'].min()
        df.at[0,'end_delay'] = reward_sites['delay_s'].max()
        df.at[0, 'sites_to_max_delay'] = reward_sites[reward_sites['delay_s'] == reward_sites['delay_s'].max()].iloc[0]['odor_sites']
        df.at[0,'start_stop_duration'] = reward_sites['stop_duration_s'].min()
        df.at[0,'end_stop_duration'] = reward_sites['stop_duration_s'].max()
        df.at[0, 'sites_to_max_stop_duration'] = reward_sites[reward_sites['stop_duration_s'] == reward_sites['stop_duration_s'].max()].iloc[0]['odor_sites']
        df.at[0, 'rewarded_sites_in_max_stop'] = int(reward_sites[(reward_sites['stop_duration_s'] == reward_sites['stop_duration_s'].max())&(reward_sites.has_choice == 1)]['odor_sites'].nunique())

        df.at[0,'start_velocity_threshold'] = reward_sites['velocity_threshold_cms'].min()
        df.at[0,'end_velocity_threshold'] = reward_sites['velocity_threshold_cms'].max()
        df.at[0, 'sites_to_min_velocity'] = reward_sites[reward_sites['velocity_threshold_cms'] == reward_sites['velocity_threshold_cms'].min()].iloc[0]['odor_sites']
            
        return df

    def get_metrics(self):
        return self.df


class ExampleMetrics(Metrics):
    """
  Parameters
    ----------
    Pending
    """
    odor_sites: float = Field(default=0)
    # rewarded_sites: int = Field(default=0)
    rewarded_sites_max: float = Field(default=0)
    visited_patches: float = Field(default=0)
    water: float = Field(default=0)
    

session_path = r'Z:\scratch\vr-foraging\data\716455\20240413T111724'
df = MetricsVrForaging(session_path).get_metrics()
ExampleMetrics(odor_sites=df.odor_sites_travelled, rewarded_sites_max=df.rewarded_sites_in_max_stop, visited_patches=df.total_patches_visited)

