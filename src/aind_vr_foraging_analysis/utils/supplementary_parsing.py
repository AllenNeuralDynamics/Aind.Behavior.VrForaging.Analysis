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
        self.reward_sites = all_epochs.loc[all_epochs.label == "RewardSite"]
        self.all_epochs = all_epochs
        self.run_on_init = run_on_init

        if self.run_on_init:
            self.add_main_info()
            self.skipped_sites()
            self.cumulative_consecutive()
            self.add_time_previous_intersite_interpatch()
            self.add_previous_patch_info()

    def cumulative_consecutive(self):
        previous_patch = -1
        cumulative_rewards = 0
        consecutive_rewards = 0
        cumulative_failures = 0
        consecutive_failures = 0
        after_choice_cumulative_rewards = 0

        for index, row in self.reward_sites.iterrows():
            # Total number of rewards in the current patch ( accumulated)
            if row["active_patch"] != previous_patch:
                previous_patch = row["active_patch"]
                cumulative_rewards = 0
                cumulative_failures = 0
                consecutive_failures = 0
                consecutive_rewards = 0
                after_choice_cumulative_rewards = 0

            self.reward_sites.loc[index, "cumulative_rewards"] = cumulative_rewards
            self.reward_sites.loc[index, "consecutive_rewards"] = consecutive_rewards
            self.reward_sites.loc[index, "cumulative_failures"] = cumulative_failures
            self.reward_sites.loc[index, "consecutive_failures"] = consecutive_failures

            if row["reward_delivered"] != 0:
                cumulative_rewards += 1
                consecutive_rewards += 1
                consecutive_failures = 0
                after_choice_cumulative_rewards += 1

            self.reward_sites.loc[index, "after_choice_cumulative_rewards"] = (
                after_choice_cumulative_rewards
            )

            if row["reward_delivered"] == 0 and row["has_choice"] == True:
                cumulative_failures += 1
                consecutive_failures += 1
                consecutive_rewards = 0

    def skipped_sites(self):
        skipped_count = 0

        for index, row in self.reward_sites.iterrows():
            # Number of first sites without stopping - useful for filtering disengagement
            if row["has_choice"] == False and row["visit_number"] == 0:
                skipped_count += 1
            elif row["has_choice"] == True:
                skipped_count = 0
            self.reward_sites.loc[index, "skipped_count"] = skipped_count

        return self.reward_sites

    def add_main_info(self):

        # Add column for site number
        self.reward_sites.loc[:, "odor_sites"] = np.arange(len(self.reward_sites))

        self.reward_sites["collected"] = (
            self.reward_sites["reward_delivered"] * self.reward_sites["reward_amount"]
        )

        self.reward_sites.loc[:, "depleted"] = np.where(
            self.reward_sites["reward_available"] == 0, 1, 0
        )

        self.reward_sites["next_visit_number"] = self.reward_sites[
            "visit_number"
        ].shift(-2)
        self.reward_sites["last_visit"] = np.where(
            (self.reward_sites["next_visit_number"] == 0)
            & (self.reward_sites["has_choice"] == True),
            1,
            0,
        )
        self.reward_sites.drop(columns=["next_visit_number"], inplace=True)

        self.reward_sites["last_site"] = self.reward_sites["visit_number"].shift(-1)
        self.reward_sites["last_site"] = np.where(
            self.reward_sites["last_site"] == 0, 1, 0
        )

        return self.reward_sites

    def add_time_previous_intersite_interpatch(self):
        all_epochs = self.all_epochs
        all_epochs.loc[:, "total_sites"] = 0

        active_patch = -1
        total_sites = -1
        time_interpatch = 0
        time_intersite = 0
        for i, row in all_epochs.iterrows():
            if row["label"] == "InterPatch":
                active_patch += 1
                time_interpatch = i
                all_epochs.at[i, "active_patch"] = active_patch
            if row["label"] == "InterSite":
                total_sites += 1
                time_intersite = i
                all_epochs.at[i, "active_patch"] = active_patch
                all_epochs.at[i, "total_sites"] = total_sites
            if row["label"] == "RewardSite":
                if row["visit_number"] == 0:
                    all_epochs.at[i, "previous_interpatch"] = time_interpatch
                    all_epochs.at[i, "previous_intersite"] = time_intersite
                else:
                    all_epochs.at[i, "previous_intersite"] = time_intersite

        all_epochs["total_sites"] = np.where(
            all_epochs["total_sites"] == -1, 0, all_epochs["total_sites"]
        )

        self.total_epochs = all_epochs.copy()
        self.reward_sites = self.total_epochs.loc[
            self.total_epochs.label == "RewardSite"
        ]

        return self.total_epochs
    
    def add_previous_odor_info(self):
        # -------------------------------- Add previous and next site information ---------------------
        index = self.reward_sites.index[1:].tolist()
        index.append(0)
        self.reward_sites["next_odor"] = index

        index = self.reward_sites["odor_offset"].iloc[:-1].tolist()
        index.insert(0, 0)
        self.reward_sites["previous_odor"] = index

        return self.reward_sites

    def add_previous_patch_info(self):
        self.reward_sites["next_patch"] = self.reward_sites["active_patch"].shift(1)
        self.reward_sites["next_odor"] = self.reward_sites["odor_label"].shift(1)
        self.reward_sites["same_patch"] = np.where(
            (self.reward_sites["next_patch"] != self.reward_sites["active_patch"])
            & (self.reward_sites["odor_label"] == self.reward_sites["next_odor"]),
            1,
            0,
        )
        self.reward_sites.drop(columns=["next_patch", "next_odor"], inplace=True)

        return self.reward_sites
