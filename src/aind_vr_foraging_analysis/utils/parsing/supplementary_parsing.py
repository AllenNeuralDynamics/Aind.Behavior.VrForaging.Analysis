import numpy as np
import pandas as pd


def assign_odor_triggers(reward_sites, odor_triggers):
    reward_sites["odor_onset"] = np.nan
    reward_sites["odor_offset"] = np.nan
    reward_sites["all_odor_onsets"] = reward_sites.apply(lambda x: [], axis=1)
    reward_sites["all_odor_offsets"] = reward_sites.apply(lambda x: [], axis=1)

    for i, (index, row) in enumerate(reward_sites.iterrows()):
        if i < len(reward_sites) - 1:
            next_index = reward_sites.index[i + 1]
            # Find odor triggers within the range
            mask = (odor_triggers.odor_onset >= index) & (odor_triggers.odor_offset < next_index)
        if i == len(reward_sites)-1:
            # Handle the last row separately if needed
            last_index = reward_sites.index[-1]
            mask = odor_triggers.odor_onset >= last_index
            
        onsets_within_range = odor_triggers.loc[mask, "odor_onset"].values
        offsets_within_range = odor_triggers.loc[mask, "odor_offset"].values

        # Assign the first onset to the new column
        if len(onsets_within_range) > 0:
            reward_sites.at[index, "odor_onset"] = onsets_within_range[0]
            if len(onsets_within_range) > 1:
                reward_sites.at[index, "all_odor_onsets"] = list(onsets_within_range[:])
            else:
                reward_sites.at[index, "all_odor_onsets"] = np.nan

        # Assign the last offset to the new column
        if len(offsets_within_range) > 0:
            reward_sites.at[index, "odor_offset"] = offsets_within_range[-1]
            if len(offsets_within_range) > 1:
                reward_sites.at[index, "all_odor_offsets"] = list(offsets_within_range)
            else:
                reward_sites.at[index, "all_odor_offsets"] = np.nan

    reward_sites['odor_duration'] = reward_sites['odor_offset'] - reward_sites['odor_onset']
    reward_sites = reward_sites.dropna(axis=1, how='all')
    return reward_sites
    
class AddExtraColumns:
    def __init__(self, all_epochs, run_on_init=True):
        self.all_epochs = all_epochs.copy()

        if run_on_init:
            self.add_main_info()
            self.cumulative_consecutive()
            self.patch_time_entry()
            self.skipped_sites()

    def get_odor_sites(self):
        odor_sites = self.all_epochs.loc[self.all_epochs.label == "OdorSite"]
        return odor_sites

    def get_all_epochs(self):
        return self.all_epochs
        
    def patch_time_entry(self):
        self.all_epochs['duration_epoch'] = self.all_epochs['stop_time'] - self.all_epochs.index
                
        patch_number = -1
        first_entry = True
        patch_onset = pd.DataFrame()
        for index, row in self.all_epochs.iterrows():
            if row['label'] == 'InterSite' and patch_number == row['patch_number'] and first_entry:
                new_rows = pd.DataFrame([
                {'patch_number': row['patch_number'], 'patch_onset': row.name}])
                patch_onset = pd.concat([patch_onset, new_rows])
                first_entry = False
                
            if patch_number != row['patch_number']:
                patch_number = row['patch_number']
                first_entry = True
        
        merged_df = pd.merge(self.all_epochs, patch_onset, on='patch_number', how='left')
        self.all_epochs['patch_onset'] = merged_df['patch_onset'].values
        self.all_epochs['time_since_entry'] = self.all_epochs.index - self.all_epochs['patch_onset']
        self.all_epochs['exit_epoch'] = self.all_epochs['time_since_entry'] + self.all_epochs['duration_epoch']

    def cumulative_consecutive(self):
        
        odor_sites = self.all_epochs.loc[self.all_epochs.label == "OdorSite"]
        
        previous_patch = -1
        cumulative_rewards = 0
        consecutive_rewards = 0
        cumulative_failures = 0
        consecutive_failures = 0
        after_choice_cumulative_rewards = 0

        for index, row in odor_sites.iterrows():
            # Total number of rewards in the current patch ( accumulated)
            if row["patch_number"] != previous_patch:
                previous_patch = row["patch_number"]
                cumulative_rewards = 0
                cumulative_failures = 0
                consecutive_failures = 0
                consecutive_rewards = 0
                after_choice_cumulative_rewards = 0

            odor_sites.loc[index, "cumulative_rewards"] = cumulative_rewards
            odor_sites.loc[index, "consecutive_rewards"] = consecutive_rewards
            odor_sites.loc[index, "cumulative_failures"] = cumulative_failures
            odor_sites.loc[index, "consecutive_failures"] = consecutive_failures

            if row["is_reward"] != 0:
                cumulative_rewards += 1
                consecutive_rewards += 1
                consecutive_failures = 0
                after_choice_cumulative_rewards += 1

            odor_sites.loc[index, "after_choice_cumulative_rewards"] = (
                after_choice_cumulative_rewards
            )

            if row["is_reward"] == 0 and row["is_choice"] == True:
                cumulative_failures += 1
                consecutive_failures += 1
                consecutive_rewards = 0

        self.all_epochs = pd.concat([self.all_epochs.loc[self.all_epochs.label != 'OdorSite'], odor_sites], axis=0)
        self.all_epochs = self.all_epochs.sort_index()
        
    def skipped_sites(self):
        odor_sites = self.all_epochs.loc[self.all_epochs.label == "OdorSite"]
        
        skipped_count = 0

        for index, row in odor_sites.iterrows():
            # Number of first sites without stopping - useful for filtering disengagement
            if (row["is_choice"] == False and row["site_number"] == 0):
                skipped_count += 1
            # elif row["is_choice"] == False and row["site_number"] == 1:
            #     skipped_count += 1
            elif row["is_choice"] == True:
                skipped_count = 0
            odor_sites.loc[index, "skipped_count"] = skipped_count
            
        self.all_epochs = pd.concat([self.all_epochs.loc[self.all_epochs.label != 'OdorSite'], odor_sites], axis=0)
        self.all_epochs = self.all_epochs.sort_index()
        
    def add_main_info(self):
        odor_sites = self.all_epochs.loc[
            self.all_epochs.label == "OdorSite"
        ].copy()
        
        # Add column for site number
        odor_sites["odor_sites"] = np.arange(len(odor_sites))

        odor_sites["collected"] = (
            odor_sites["is_reward"] * odor_sites["reward_amount"]
        )

        odor_sites["depleted"] = np.where(
            odor_sites["reward_available"] == 0, 1, 0
        )

        odor_sites["next_site_number"] = odor_sites[
            "site_number"
        ].shift(-2)
        odor_sites["last_visit"] = np.where(
            (odor_sites["next_site_number"] == 0)
            & (odor_sites["is_choice"] == True),
            1,
            0,
        )
        odor_sites.drop(columns=["next_site_number"], inplace=True)

        odor_sites["last_site"] = odor_sites["site_number"].shift(-1)
        odor_sites["last_site"] = np.where(
            odor_sites["last_site"] == 0, 1, 0
        )

        self.all_epochs = pd.concat(
            [self.all_epochs.loc[self.all_epochs.label != "OdorSite"], odor_sites], axis=0
        )
        self.all_epochs = self.all_epochs.sort_index()
        
    def add_time_previous_intersite_interpatch(self):
        self.all_epochs.loc[:, "total_sites"] = 0

        patch_number = -1
        total_sites = -1
        time_interpatch = 0
        time_intersite = 0
        for i, row in self.all_epochs.iterrows():
            if row["label"] == "InterPatch":
                patch_number += 1
                time_interpatch = i
                self.all_epochs.at[i, "patch_number"] = patch_number
            if row["label"] == "InterSite":
                total_sites += 1
                time_intersite = i
                self.all_epochs.at[i, "patch_number"] = patch_number
                self.all_epochs.at[i, "total_sites"] = total_sites
            if row["label"] == "OdorSite":
                if row["site_number"] == 0:
                    self.all_epochs.at[i, "previous_interpatch"] = time_interpatch
                    self.all_epochs.at[i, "previous_intersite"] = time_intersite
                else:
                    self.all_epochs.at[i, "previous_intersite"] = time_intersite

        self.all_epochs["total_sites"] = np.where(
            self.all_epochs["total_sites"] == -1, 0, self.all_epochs["total_sites"]
        )
    
    def add_previous_odor_info(self):
        odor_sites = self.all_epochs.loc[
            self.all_epochs.label == "OdorSite"
        ]
        # -------------------------------- Add previous and next site information ---------------------
        index = odor_sites.index[1:].tolist()
        index.append(0)
        odor_sites["next_odor"] = index

        index = odor_sites["odor_offset"].iloc[:-1].tolist()
        index.insert(0, 0)
        odor_sites["previous_odor"] = index

        self.all_epochs = pd.concat(
            [self.all_epochs.loc[self.all_epochs.label != "OdorSite"], odor_sites], axis=0
        )
        self.all_epochs = self.all_epochs.sort_index()
        
    def add_previous_patch_info(self):
        odor_sites = self.all_epochs.loc[
            self.all_epochs.label == "OdorSite"
        ]
       
        odor_sites["next_patch"] = odor_sites["patch_number"].shift(1)
        odor_sites["next_odor"] = odor_sites["odor_label"].shift(1)
        odor_sites["same_patch"] = np.where(
            (odor_sites["next_patch"] != odor_sites["patch_number"])
            & (odor_sites["odor_label"] == odor_sites["next_odor"]),
            1,
            0,
        )
        odor_sites.drop(columns=["next_patch", "next_odor"], inplace=True)
        
        self.all_epochs = pd.concat(
            [self.all_epochs.loc[self.all_epochs.label != "OdorSite"], odor_sites], axis=0
        )
        self.all_epochs = self.all_epochs.sort_index()