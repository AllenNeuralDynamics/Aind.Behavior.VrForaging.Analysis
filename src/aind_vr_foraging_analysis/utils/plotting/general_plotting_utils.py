# %%
import sys

sys.path.append("../src/")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
# Data processing toold
import pandas as pd
# Plotting libraries
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.lines import Line2D

def format_func(value, tick_number):
    return f"{value:.0f}"


sns.set_context("talk")

import warnings

pd.options.mode.chained_assignment = None  # Ignore SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)

def choose_palette(trial_summary: pd.DataFrame, condition: str = "reward_available"):
    """
    This function assigns a color palette to unique values in a specified column of a DataFrame.

    Parameters:
    trial_summary (pd.DataFrame): The DataFrame to process.
    condition (str, optional): The column in the DataFrame to assign colors to. Defaults to 'reward_available'.

    Returns:
    dict: A dictionary where keys are unique values from the 'condition' column and values are corresponding colors.
    """

    # Get unique values from the 'condition' column, sorted in ascending order
    strings = sorted(trial_summary[condition].unique(), reverse=False)

    # Create a color palette with as many colors as there are unique values in the 'condition' column
    palette = sns.color_palette("RdYlBu", n_colors=trial_summary[condition].nunique())

    # Set the first color to '#bc1626' and the last color to '#3a53a3'
    palette[0] = "#bc1626"
    palette[-1] = "#3a53a3"

    def assign_colors_sequential(strings, palette):
        """
        This function assigns colors to each unique value in 'strings' in a sequential manner.

        Parameters:
        strings (list): The list of unique values.
        palette (list): The list of colors.

        Returns:
        dict: A dictionary where keys are unique values from 'strings' and values are corresponding colors.
        """
        n = len(palette)
        return {string: palette[i % n] for i, string in enumerate(strings)}

    # Assign colors to unique values
    assigned_colors = assign_colors_sequential(strings, palette)

    return assigned_colors

def speed_traces_epochs(
    reward_sites,
    inter_site,
    inter_patch,
    encoder_data,
    mean: bool = False,
    single: bool = True,
    patch: int = 4,
    save=False,
):
    window = [-0.1, 1]
    colors_reward = ["#d73027", "#fdae61", "#abd9e9", "#4575b4"]
    n_col = 3

    trial_summary = pd.DataFrame()
    fig, ax = plt.subplots(1, n_col, figsize=(n_col * 4, 5), sharey=True)
    for j, dataframe in enumerate([inter_patch, inter_site, reward_sites]):
        for start_reward, row in dataframe.iterrows():
            trial_average = pd.DataFrame()
            if dataframe["label"].values[0] == "OdorSite":
                trial = encoder_data.loc[
                    start_reward + -0.9 : start_reward + 2, "filtered_velocity"
                ]
            else:
                trial = encoder_data.loc[
                    start_reward + window[0] : start_reward + window[1],
                    "filtered_velocity",
                ]

            trial.index -= start_reward

            trial_average["speed"] = trial.values
            trial_average["times"] = np.around(trial.index, 3)

            for column in dataframe.columns:
                trial_average[column] = np.repeat(row[column], len(trial.values))

            trial_summary = pd.concat([trial_summary, trial_average], ignore_index=True)

            if single:
                ax[j].plot(
                    trial.index,
                    trial.values,
                    color=colors_reward[int(row["reward_available"])],
                    linewidth=0.5,
                    alpha=0.5,
                )

        trial_summary["mouse"] = reward_sites["mouse"].values[0]
        trial_summary["session"] = reward_sites["session"].values[0]

        if mean:
            sns.lineplot(
                data=trial_summary.loc[
                    trial_summary.label == dataframe.label.unique()[0]
                ],
                hue="reward_available",
                x="times",
                y="speed",
                ax=ax[j],
                legend=False,
                ci=None,
                palette=colors_reward,
                linewidth=2,
            )

        ax[j].vlines(0, -15, 70, color="black", linestyle="solid", linewidth=0.5)

        ax[j].set_ylim(-15, 70)
        if dataframe["label"].values[0] == "Gap":
            ax[j].set_title("InterSite")
            ax[j].set_xlabel("Time after entering InterSite (s)")
            ax[j].hlines(
                5,
                window[0],
                window[1],
                color="black",
                linestyle="dashed",
                linewidth=0.5,
            )

        elif dataframe["label"].values[0] == "InterPatch":
            ax[j].set_title("InterPatch")
            ax[j].set_xlabel("Time after entering InterPatch (s)")
            ax[j].hlines(
                5,
                window[0],
                window[1],
                color="black",
                linestyle="dashed",
                linewidth=0.5,
            )

        else:
            ax[j].set_title("Site")
            ax[j].hlines(5, -1, 2, color="black", linestyle="dashed", linewidth=0.5)
            ax[j].set_xlabel("Time after odor onset (s)")

    ax[0].set_ylabel("Velocity (cm/s)")
    sns.despine()
    handles = [mpatches.Patch(color=colors_reward[i], label=f"{i}") for i in range(4)]

    ax[0].legend(
        handles=handles,
        ncol=2,
        title="Reward remaining \n in patch",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.5),
    )
    plt.tight_layout()
    if save:
        save.figure(fig)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def segmented_raster_vertical_reversal(reward_sites: pd.DataFrame,
    color_dict_label: dict = {
        "Ethyl Butyrate": "#d95f02",
        "Alpha-pinene": "#1b9e77",
        "Amyl Acetate": "#7570b3"
    }, 
    save = False):
    
    # Define markers and offsets

    available_markers = ['^', 's', 'X', 'P', 'D', '*', 'o', 'v', '>']  # Add more if needed
    available_offsets = [-0.25, -0.5, -0.75, -1.0, -1.25]

    unique_labels = reward_sites["odor_label"].unique()

    # Dynamically assign markers and y-offsets
    marker_dict_label = {}
    y_offset_dict = {}

    for i, label in enumerate(unique_labels):
        marker_dict_label[label] = available_markers[i % len(available_markers)]
        y_offset_dict[label] = available_offsets[i % len(available_offsets)]

    patch_number = len(reward_sites.patch_number.unique())
    number_odors = len(reward_sites["odor_label"].unique())

    # Make second row proportional to the number of odors
    list_odors = []
    for odor in reward_sites.odor_label.unique():
        list_odors.append(
            reward_sites.loc[reward_sites.odor_label == odor].patch_number.nunique()
        )
    grid = (np.array(list_odors) / patch_number) * number_odors
    width = patch_number/8
    if width < 6:
        width = 14
    fig = plt.figure(figsize=(width, 8))
    gs = GridSpec(2, number_odors, width_ratios=grid)

    for index, row in reward_sites.iterrows():
        ax1 = plt.subplot(gs[0, 0:number_odors])
        if row["is_reward"] == 1 and row["is_choice"] == True:
            color = "steelblue"
        elif row["is_reward"] == 0 and row["is_choice"]:
            if row['patch_label'] == 'N':
                color = "crimson"
            elif row['patch_label'] == 'D' and row['site_number'] >= 3:
                color = "crimson"
            elif row['patch_label'] == 'S' and row['site_number'] >= 1:
                color = "crimson"
            else:
                color = "pink"
        else:
            if row['patch_label'] in ['N', 'D', 'S']:
                if row['patch_label'] == 'N':
                    color = "black"
                elif row['patch_label'] == 'D' and row['site_number'] >= 3:
                    color = "black"
                elif row['patch_label'] == 'S' and row['site_number'] >= 1:
                    color = "black"
                else:
                    color = "lightgrey"
            else:
                color = "lightgrey"

        # ax1.barh(int(row['patch_number']), left=row['site_number'], height=1, width=1, color=color, edgecolor='darkgrey', linewidth=0.5)
        ax1.bar(
            int(row["patch_number"]),
            bottom=row["site_number"],
            height=1,
            width=1,
            color=color,
            edgecolor="darkgrey",
            linewidth=0.5,
        )
        ax1.set_xlim(-1, max(reward_sites.patch_number) + 1)          
        ax1.set_xlabel("Patch number")
        ax1.set_ylabel("Site number")

        # ax1.bar(int(row['patch_number']), bottom = -1, height=0.5, width = 1, color=patch_color, edgecolor='black', linewidth=0.5)
        label = row["patch_label"]
        odor = row["odor_label"]
        marker = marker_dict_label.get(odor, 'o')
        y_offset = y_offset_dict.get(odor, -0.25)

        ax1.scatter(
            row["patch_number"],
            y_offset,
            color=color_dict_label[label],
            marker=marker,
            s=35,
            edgecolor="black",
            linewidth=0.0,
        )

    if reward_sites.block.nunique() > 1:
        change_indices = reward_sites.loc[reward_sites.block.diff() != 0, 'patch_number'].values[1:]
        for change_index in change_indices:
            ax1.axvline(change_index - 0.5, color='black', linestyle='--', label='Change')

    # Get unique patch_label and odor_label values
    unique_patch_labels = reward_sites["patch_label"].unique()
    unique_odor_labels = reward_sites["odor_label"].unique()

    # Marker shape for odor_label (black)
    odor_handles = [
        Line2D(
            [0], [0],
            marker=marker_dict_label.get(odor, 'o'),
            linestyle='None',
            color='black',
            markerfacecolor='black',
            markeredgecolor='black',
            label=odor,
            markersize=8
        )
        for odor in unique_odor_labels
    ]

    # Marker color for patch_label (S/N/D) — fixed shape (e.g., 's')
    patch_label_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            color='w',
            markerfacecolor=color_dict_label[label],
            markeredgecolor='black',
            label=label,
            markersize=8
        )
        for label in ['S', 'N', 'D'] if label in unique_patch_labels
    ]

    # Bar color legend (reward/choice states)
    bar_handles = [
        mpatches.Patch(color="steelblue", label="Harvest, rewarded"),
        mpatches.Patch(color="crimson", label="Harvest, no reward, depleted"),
        mpatches.Patch(color="pink", label="Harvest, no reward, probabilistic"),
        mpatches.Patch(color="lightgrey", label="Leave, not depleted"),
        mpatches.Patch(color="black", label="Leave, depleted"),
    ]

    # Combine everything
    legend_handles = odor_handles + patch_label_handles + bar_handles

    # Create subplots dynamically in the bottom row (row index 1)
    unique_patches = reward_sites["odor_label"].unique()
    axes = [plt.subplot(gs[1, i]) for i in range(number_odors)]

    # Loop over each odor and axis
    for ax, odor_label in zip(axes, unique_patches):
        selected_sites = reward_sites[reward_sites.odor_label == odor_label]
        previous_patch = None
        value = 0  # running x-axis index

        # Keep track of block number and where each patch is on the x-axis
        patch_positions = []  # (value, block)
        for patch_number in selected_sites['patch_number'].unique():
            block = selected_sites[selected_sites['patch_number'] == patch_number]['block'].iloc[0]
            patch_positions.append((value, block))
            value += 1

        # Draw vertical lines between blocks
        for i in range(1, len(patch_positions)):
            val_prev, block_prev = patch_positions[i - 1]
            val_curr, block_curr = patch_positions[i]
            if block_curr != block_prev:
                # Add line at transition
                ax.axvline(val_curr + 0.5, color='black', linestyle='--',
                        label='Block Change' if i == 1 else None)

                # Add labels for each side
                for offset, block in zip([-1, 1], [block_prev, block_curr]):
                    label_data = selected_sites[selected_sites['block'] == block]['patch_label'].unique()
                    if len(label_data):
                        ax.text(val_curr + offset, y=6, s=label_data[0], color=color_dict_label[label_data[0]],
                                ha='right' if offset < 0 else 'left')

        # Plot bars per site
        value = 0
        previous_patch = None
        for _, row in selected_sites.iterrows():
            if row['patch_number'] != previous_patch:
                value += 1
                previous_patch = row['patch_number']

            # Choose bar color
            if row["is_reward"] == 1 and row["is_choice"]:
                color = "steelblue"
            elif row["is_reward"] == 0 and row["is_choice"]:
                if row['patch_label'] == 'N':
                    color = "crimson"
                elif row['patch_label'] == 'D' and row['site_number'] >= 3:
                    color = "crimson"
                elif row['patch_label'] == 'S' and row['site_number'] >= 1:
                    color = "crimson"
                else:
                    color = "pink"
            else:
                if row['patch_label'] in ['N', 'D', 'S']:
                    if row['patch_label'] == 'N':
                        color = "black"
                    elif row['patch_label'] == 'D' and row['site_number'] >= 3:
                        color = "black"
                    elif row['patch_label'] == 'S' and row['site_number'] >= 1:
                        color = "black"
                    else:
                        color = "lightgrey"
                else:
                    color = "lightgrey"

            # Plot bar
            ax.bar(
                value,
                bottom=row["site_number"],
                height=1,
                width=1,
                color=color,
                edgecolor="darkgrey",
                linewidth=0.5,
            )

        # Formatting
        ax.set_xlim(-1, value + 2)
        ax.set_ylim(-0.5, reward_sites.site_number.max() + 1)
        ax.set_ylabel("Site number")
        ax.set_xlabel("Patch number")
        # ax.set_title(odor_label)
        

    plt.legend(handles=legend_handles, loc='best', bbox_to_anchor=(1.05, 1), fontsize=12, ncol=1)
    # plt.suptitle(reward_sites.session.unique()[0])

    fig.tight_layout()
    sns.despine()
    
    if save:
        save.savefig(fig)
        plt.close(fig)
    else:
        plt.show()

def segmented_raster_vertical(
    reward_sites: pd.DataFrame,
    save: bool = False,
    color_dict_label: dict = {
        "Ethyl Butyrate": "#d95f02",
        "Alpha-pinene": "#1b9e77",
        "Amyl Acetate": "#7570b3",
    },
):

    patch_number = len(reward_sites.patch_number.unique())
    number_odors = len(reward_sites["patch_label"].unique())

    # Make second row proportional to the number of odors
    list_odors = []
    for odor in reward_sites.patch_label.unique():
        list_odors.append(
            reward_sites.loc[reward_sites.patch_label == odor].patch_number.nunique()
        )
    grid = (np.array(list_odors) / patch_number) * number_odors

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, number_odors, width_ratios=grid)

    for index, row in reward_sites.iterrows():
        ax1 = plt.subplot(gs[0, 0:number_odors])
        if row["is_reward"] == 1 and row["is_choice"] == True:
            color = "steelblue"
        elif row["is_reward"] == 0 and row["is_choice"] == True:
            color = "pink"
            if row["reward_available"] == 0 or row["reward_probability"] <= 0:
                color = "crimson"
        else:
            if row["reward_available"] == 0 or row["reward_probability"] <= 0:
                color = "black"
            else:
                color = "lightgrey"

        # ax1.barh(int(row['patch_number']), left=row['site_number'], height=1, width=1, color=color, edgecolor='darkgrey', linewidth=0.5)
        ax1.bar(
            int(row["patch_number"]),
            bottom=row["site_number"],
            height=1,
            width=1,
            color=color,
            edgecolor="darkgrey",
            linewidth=0.5,
        )
        ax1.set_xlim(-1, max(reward_sites.patch_number) + 1)          
        ax1.set_xlabel("Patch number")
        ax1.set_ylabel("Site number")

        # ax1.bar(int(row['patch_number']), bottom = -1, height=0.5, width = 1, color=patch_color, edgecolor='black', linewidth=0.5)
        ax1.scatter(
            row["patch_number"],
            -0.25,
            color=color_dict_label[row["patch_label"]],
            marker="^",
            s=35,
            edgecolor="black",
            linewidth=0.0,
        )

    if reward_sites.block.nunique() > 1:
        change_index = reward_sites[reward_sites.block.diff() != 0]['patch_number'].values[1]
        ax1.axvline(change_index-0.5, color='black', linestyle='--', label='Change')
            
    odors = []
    for odor in reward_sites["patch_label"].unique():
        odors.append(
            mpatches.Patch(
                color=color_dict_label[odor],
                label=(
                    str(odor)
                ),
            )
        )

    label_2 = mpatches.Patch(color="steelblue", label="Harvest, rewarded")
    label_3 = mpatches.Patch(color="crimson", label="Harvest, no reward, depleted")
    label_4 = mpatches.Patch(color="lightgrey", label="Leave, not depleted")
    label_5 = mpatches.Patch(color="black", label="Leave, depleted")
    label_6 = mpatches.Patch(color="pink", label="Harvest, no reward, probabilitic")

    odors.extend([label_2, label_3, label_6, label_4, label_5])
    ax1.set_ylim(-1, max(reward_sites.site_number) + 1)

    # Create subplots dynamically in the bottom row (row index 1)
    unique_patches = reward_sites["patch_label"].unique()
    axes = [plt.subplot(gs[1, i]) for i in range(number_odors)]

    # Now loop over each odor and axis
    for ax, patch_label in zip(axes, unique_patches):
        selected_sites = reward_sites.loc[reward_sites.patch_label == patch_label]
        previous_active = 0
        value = 0
        if selected_sites.block.nunique() > 1:
            change_index = selected_sites[selected_sites.block == selected_sites.block.unique()[0]].patch_number.nunique()
            ax.axvline(change_index, color='black', linestyle='--', label='Change')
        for index, row in selected_sites.iterrows():
            # Choose the color of the site
            if row["is_reward"] == 1 and row["is_choice"] == True:
                color = "steelblue"
            elif row["is_reward"] == 0 and row["is_choice"] == True:
                color = "pink"
                if row["reward_available"] == 0 or row["reward_probability"] <= 0:
                    color = "crimson"
            else:
                if row["reward_available"] == 0 or row["reward_probability"] <= 0:
                    color = "black"
                else:
                    color = "lightgrey"

            ax.set_title(patch_label, color=color_dict_label[row["patch_label"]])

            if row["patch_number"] != previous_active:
                value += 1
                previous_active = row["patch_number"]
            ax.bar(
                value,
                bottom=row["site_number"],
                height=1,
                width=1,
                color=color,
                edgecolor="darkgrey",
                linewidth=0.5,
            )
            ax.set_xlim(-1, selected_sites.patch_number.nunique() + 1)
            ax.set_ylim(-0.5, reward_sites.site_number.max() + 1)
            ax.set_ylabel("Site number")
            ax.set_xlabel("Patch number")

    fig.tight_layout()
    plt.legend(handles=odors, loc='best', bbox_to_anchor=(0.75, 1), fontsize=12, ncol=1)

    sns.despine()

    if save != False:
        save.savefig(fig, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

def speed_traces_value(
    trial_summary: pd.DataFrame,
    mouse: str,
    session: str,
    condition: str = "reward_available",
    window=(-1, 3),
    save=False,
):

    n_odors = len(trial_summary.patch_label.unique())
    fig, ax = plt.subplots(
        n_odors, 5, figsize=(18, n_odors * 5), sharex=True, sharey=True
    )
    colors = ["crimson", "darkgreen"]
    for j in range(n_odors):
        if n_odors == 1:
            ax1 = ax
        else:
            ax1 = ax[j]
        ax1[0].set_ylabel("Velocity (cm/s)")

        for i in [0, 1, 2, 3, 4]:
            if n_odors == 1:
                ax[i].set_xlabel("Time after odor onset (s)")
            else:
                ax[len(trial_summary.patch_label.unique()) - 1][j].set_xlabel(
                    "Time after odor onset (s)"
                )

            ax1[i].set_ylim(-10, 80)
            ax1[i].set_xlim(window)
            ax1[i].hlines(
                5, window[0], window[1], color="black", linewidth=1, linestyles=":"
            )
            ax1[i].fill_betweenx(
                np.arange(-10, 80, 0.1),
                window[0],
                0,
                color="#808080",
                alpha=0.5,
                linewidth=0,
            )

    for i in trial_summary.is_choice.unique():
        assigned_colors = choose_palette(trial_summary, condition=condition)
        i = int(i)
        for j, patch_label in enumerate(trial_summary.patch_label.unique()):
            if n_odors == 1:
                ax1 = ax
            else:
                ax1 = ax[j]

            df_results = (
                trial_summary.loc[
                    (trial_summary.patch_label == patch_label)
                    & (trial_summary.is_choice == i)
                ]
                .groupby([condition, "odor_sites", "times"])[["speed"]]
                .mean()
                .reset_index()
            )

            if df_results.empty:
                continue

            sns.lineplot(
                x="times",
                y="speed",
                data=df_results,
                hue=condition,
                palette=assigned_colors,
                ci=None,
                legend=False,
                ax=ax1[i + 2],
            )
            for site in df_results.odor_sites.unique():
                plot_df = df_results.loc[df_results.odor_sites == site]
                sns.lineplot(
                    x="times",
                    y="speed",
                    data=plot_df,
                    color=assigned_colors[plot_df["reward_probability"].unique()[0]],
                    legend=False,
                    linewidth=0.5,
                    alpha=0.5,
                    ax=ax1[i],
                )

            sns.lineplot(
                x="times",
                y="speed",
                data=df_results,
                color=colors[i],
                palette=assigned_colors,
                ci=("sd"),
                ax=ax1[4],
            )

            ax1[2].set_title(f"Patch {patch_label} ")

            if i == 1:
                ax1[i].text(1.2, 75, "Stopped", fontsize=12)
                ax1[i + 2].text(1.2, 75, "Stopped", fontsize=12)
            else:
                ax1[i].text(1.2, 75, "Not stopped", fontsize=12)
                ax1[i + 2].text(1.2, 75, "Not stopped", fontsize=12)

    sns.despine()
    plt.suptitle(str(mouse) + "_" + str(session))
    plt.tight_layout()

    if save != False:
        save.savefig(fig)
        plt.close(fig)


def speed_traces_efficient(
    trial_summary: pd.DataFrame,
    mouse,
    session,
    patch: str = "all",
    window: tuple = (-1, 3),
    save=False,
):
    """Plots the speed traces for each stopping condition"""

    if patch != "all":
        reward_sites = reward_sites.loc[reward_sites["patch_label"] == patch]

    colors = ["crimson", "darkgreen"]
    fig, ax = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

    for j in [0, 1, 2]:
        ax[j][0].set_ylabel("Velocity (cm/s)")
        ax[2][j].set_xlabel("Time after odor onset (s)")

        for i in [0, 1]:
            ax[j][i].set_ylim(-10, 80)
            ax[j][i].set_xlim(window)

            ax[j][i].hlines(
                5, window[0], window[1], color="black", linewidth=1, linestyles=":"
            )
            ax[j][i].fill_betweenx(
                np.arange(-10, 80, 0.1), -2, 0, color="#808080", alpha=0.5, linewidth=0
            )

            ax[j][2].fill_betweenx(
                np.arange(-10, 80, 0.1), -2, 0, color="#808080", alpha=0.5, linewidth=0
            )
            ax[j][2].hlines(
                5, window[0], window[1], color="black", linewidth=1, linestyles=":"
            )

    for collected_label in [0, 1]:
        for i, selected_trials in trial_summary.loc[
            trial_summary.is_choice == collected_label
        ].groupby("odor_sites"):
            ax[0][collected_label].plot(
                "times",
                "speed",
                data=selected_trials,
                color="black",
                alpha=0.3,
                linewidth=0.5,
            )

        selected_trials = (
            trial_summary.loc[trial_summary.is_choice == collected_label]
            .groupby("times")["speed"]
            .mean()
            .reset_index()
        )
        ax[0][collected_label].plot(
            "times", "speed", data=selected_trials, color="black", linewidth=3
        )

        if collected_label == 1:
            ax[0][collected_label].set_title(" Stopped", color=colors[collected_label])
        else:
            ax[0][collected_label].set_title(
                " Not Stopped", color=colors[collected_label]
            )

        sns.lineplot(
            x="times",
            y="speed",
            data=trial_summary.loc[(trial_summary.is_choice == collected_label)],
            color=colors[collected_label],
            ax=ax[0][2],
            errorbar=("sd"),
            linewidth=2,
        )

        # ---------- water collection or not
        plot_df = trial_summary.loc[
            (trial_summary.is_reward == collected_label)
            & (trial_summary.is_choice == 1)
        ]
        for i, selected_trials in plot_df.groupby("odor_sites"):
            ax[1][collected_label].plot(
                "times",
                "speed",
                data=selected_trials,
                color="black",
                alpha=0.3,
                linewidth=0.5,
            )

        selected_trials = plot_df.groupby("times")["speed"].mean().reset_index()
        ax[1][collected_label].plot(
            "times", "speed", data=selected_trials, color="black", linewidth=3
        )

        if collected_label == 1:
            ax[1][collected_label].set_title(
                " Rewarded stop", color=colors[collected_label]
            )
        else:
            ax[1][collected_label].set_title(
                " No reward stop", color=colors[collected_label]
            )

        sns.lineplot(
            x="times",
            y="speed",
            data=plot_df,
            color=colors[collected_label],
            ax=ax[1][2],
            errorbar=("sd"),
            linewidth=2,
        )

        # ------------ water depletion or not
        plot_df = trial_summary.loc[
            (trial_summary.depleted == collected_label)
            & (trial_summary.is_choice == 1)
            & (trial_summary.is_reward == 0)
        ]
        for i, selected_trials in plot_df.groupby("odor_sites"):
            ax[2][collected_label].plot(
                "times",
                "speed",
                data=selected_trials,
                color="black",
                alpha=0.3,
                linewidth=0.5,
            )

        selected_trials = plot_df.groupby("times")["speed"].mean().reset_index()
        ax[2][collected_label].plot(
            "times", "speed", data=selected_trials, color="black", linewidth=3
        )

        if collected_label == 1:
            ax[2][collected_label].set_title(
                " No reward - depleted", color=colors[collected_label]
            )
        else:
            ax[2][collected_label].set_title(
                " No reward - not depleted", color=colors[collected_label]
            )

        sns.lineplot(
            x="times",
            y="speed",
            data=plot_df,
            color=colors[collected_label],
            ax=ax[2][2],
            errorbar=("sd"),
            linewidth=2,
        )

    sns.despine()
    plt.suptitle(str(mouse) + "_" + str(session) + "_" + patch)
    plt.tight_layout()

    if save != False:
        save.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def preward_estimates(
    reward_sites,
    minimum_size: int = 2,
    color_dict_label: dict = {
        "Ethyl Butyrate": "#d95f02",
        "Alpha-pinene": "#1b9e77",
        "Amyl Acetate": "#7570b3",
    },
    save: bool = False,
):

    summary = reward_sites.groupby(["patch_number", "patch_label"]).agg(
        {
            "is_reward": "sum",
            "site_number": "count",
            "reward_probability": "min",
        }
    )

    summary = summary.loc[summary.site_number > minimum_size]
    summary.reset_index(inplace=True)

    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 3, 1)
    sns.boxplot(
        x="patch_label",
        y="is_reward",
        hue="patch_label",
        order = summary.patch_label.unique(),
        palette=color_dict_label,
        data=summary,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        x="patch_label",
        y="is_reward",
        hue="patch_label",
        palette=["black", "black", "black"],
        data=summary,
        ax=ax,
        order=summary.patch_label.unique(),
        linewidth=0.2,
        edgecolor="black",
        jitter=0.25,
    )
    plt.xlabel("Odor")
    plt.ylabel("Total reward \n collected")
    plt.xticks(rotation=45)
    plt.xlabel("Initial P(reward)")
    ax.get_legend().remove()
    
    ax = plt.subplot(1, 3, 2)
    sns.boxplot(
        x="patch_label",
        y="site_number",
        hue="patch_label",
        palette=color_dict_label,
        order=summary.patch_label.unique(),
        data=summary,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        x="patch_label",
        y="site_number",
        hue="patch_label",
        palette=["black", "black", "black"],
        data=summary,
        order=summary.patch_label.unique(),
        ax=ax,
        linewidth=0.2,
        edgecolor="black",
        legend=False,
        jitter=0.25,
    )

    plt.xlabel("Odor")
    plt.ylabel("Total stops")
    plt.xticks(rotation=45)
    plt.xlabel("Initial P(reward)")
    ax.get_legend().remove()
    
    ax = plt.subplot(1, 3, 3)
    sns.boxplot(
        x="patch_label",
        y="reward_probability",
        hue="patch_label",
        palette=color_dict_label,
        order=summary.patch_label.unique(),
        data=summary,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        x="patch_label",
        y="reward_probability",
        hue="patch_label",
        palette=["black", "black", "black"],
        data=summary,
        ax=ax,
        order=summary.patch_label.unique(),
        linewidth=0.2,
        edgecolor="black",
        legend=False,
        jitter=0.25,
    )

    plt.xlabel("Odor")
    plt.xticks(rotation=45)
    plt.ylabel("P(reward) when leaving")
    plt.xlabel("Initial P(reward)")
    ax.get_legend().remove()

    sns.despine()
    plt.tight_layout()

    if save != False:
        save.savefig(fig)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def velocity_traces_odor_entry(
    trial_summary,
    window: tuple = (-0.5, 2),
    max_range: int = 60,
    color_dict_label: dict = {
        "Ethyl Butyrate": "#d95f02",
        "Alpha-pinene": "#1b9e77",
        "Amyl Acetate": "#7570b3",
    },
    mean: bool = False,
    save: bool = False,
):
    """Plots the speed traces for each odor label condition"""
    n_odors = trial_summary.patch_label.unique()
    
    fig, ax1 = plt.subplots(
        1, len(n_odors), figsize=(len(n_odors) * 3.5, 4), sharex=True, sharey=True
    )

    for j, patch_label in enumerate(n_odors):
        if len(n_odors) != 1:
            ax = ax1[j]
            ax1[0].set_ylabel("Velocity (cm/s)")
        else:
            ax = ax1
            ax.set_ylabel("Velocity (cm/s)")

        ax.set_xlabel("Time after odor onset (s)")
        ax.set_title(f"Patch {patch_label}")
        ax.set_ylim(-13, max_range)
        ax.set_xlim(window)
        ax.hlines(
            5, window[0], window[1], color="black", linewidth=1, linestyles="dashed"
        )
        ax.fill_betweenx(
            np.arange(-20, max_range, 0.1),
            0,
            window[1],
            color=color_dict_label[patch_label],
            alpha=0.5,
            linewidth=0,
        )
        ax.fill_betweenx(
            np.arange(-20, max_range, 0.1),
            window[0],
            0,
            color="grey",
            alpha=0.3,
            linewidth=0,
        )

        df_results = (
            trial_summary.loc[
                (trial_summary.patch_label == patch_label)
                & (trial_summary.site_number == 0)
            ]
            .groupby(["odor_sites", "times", "patch_label"])[["speed"]]
            .median()
            .reset_index()
        )

        if df_results.empty:
            continue

        sns.lineplot(
            x="times",
            y="speed",
            data=df_results,
            hue="odor_sites",
            palette=["black"] * df_results["odor_sites"].nunique(),
            legend=False,
            linewidth=0.4,
            alpha=0.4,
            ax=ax,
        )

        if mean:
            sns.lineplot(
                x="times",
                y="speed",
                data=df_results,
                color="black",
                ci=None,
                legend=False,
                linewidth=2,
                ax=ax,
            )

    sns.despine()
    plt.tight_layout()
    if save != False:
        save.savefig(fig)
    else:
        plt.show()

    plt.close(fig)

def summary_withinsession_values(reward_sites, 
                                 color_dict_label = {'Ethyl Butyrate': '#d95f02', 'Alpha-pinene': '#1b9e77', 'Amyl Acetate': '#7570b3',
                                                     '2-Heptanone' : '#1b9e77', 'Methyl Acetate': '#d95f02', 'Fenchone': '#7570b3', '2,3-Butanedione': '#e7298a'}, 
                                 save=None):

    fig, ax = plt.subplots(3,2,figsize=(16,10), sharex=True)

    df = reward_sites.groupby(['patch_number', 'patch_label']).agg({'reward_probability':'min','site_number':'max', "is_choice": 'sum', 'is_reward': 'sum', 'consecutive_rewards': 'max', 'cumulative_failures': 'max', 'consecutive_failures': 'max'}).reset_index()

    ax[0][0].set_ylabel('P(reward) \n when leaving')            
    ax[0][0].set_ylim(-0.1,1.1)

    # df = df.groupby(['patch_number','odor_label']).agg({'site_number':'sum', 'reward_probability':'mean'}).reset_index()      
    sns.scatterplot(df, x='patch_number', size="site_number", hue='patch_label', sizes=(30, 500), y='reward_probability', ax=ax[0][0], palette=color_dict_label,  legend=False)
    ax[0][0].set_ylabel('P(reward) \n when leaving')            
    ax[0][0].set_ylim(-0.1,1.1)

    sns.scatterplot(df, x='patch_number', hue='patch_label', sizes=(30, 500), y='is_choice', ax=ax[0][1], palette=color_dict_label,  legend=False)
    ax[0][1].set_ylabel('Total stops')            

    sns.scatterplot(df, x='patch_number', size="site_number", hue='patch_label', sizes=(30, 500), y='consecutive_rewards', ax=ax[1][0], palette=color_dict_label,  legend=False)
    ax[1][0].set_ylabel('Consecutive rewards')            

    sns.scatterplot(df, x='patch_number', size="site_number", hue='patch_label', sizes=(30, 500), y='is_reward', ax=ax[1][1], palette=color_dict_label,  legend=False)
    ax[1][1].set_ylabel('Cumulative rewards')            

    sns.scatterplot(df, x='patch_number', size="site_number", hue='patch_label', sizes=(30, 500), y='cumulative_failures', ax=ax[2][0], palette=color_dict_label,  legend=False)
    ax[2][0].set_ylabel('Cumulative failures')            

    sns.scatterplot(df, x='patch_number', size="site_number", hue='patch_label', sizes=(30, 500), y='consecutive_failures', ax=ax[2][1], palette=color_dict_label,  legend=False)
    ax[2][1].set_ylabel('Consecutive failures')            

    ax[2][0].set_xlabel('Patch number')
    ax[2][1].set_xlabel('Patch number')

    plt.tight_layout()
    sns.despine()
    if save:
        save.savefig(fig)
        plt.close(fig)
    else:
        plt.show()
        
def trial_collection(
    reward_sites: pd.DataFrame,
    continuous_data: pd.DataFrame,
    aligned: str = 'index',
    cropped_to_length: str = 'window',
    window: list = [-0.5, 2],
    taken_col: str = "filtered_velocity",
    continuous: bool = True,
):
    """
    Crop the snippets of speed traces that are aligned to different epochs

    Parameters
    ----------
    reward_sites : pd.DataFrame
        DataFrame containing the reward sites information (odor sites)
    continuous_data : pd.DataFrame
        DataFrame containing the continuous data (encoder, sniffing, etc)
    mouse : str
        Mouse name
    session : str
        Session name
    aligned : str
        Column name to align the snippets
    window : tuple
        Time window to crop the snippets
    taken_col: string
        name of the column that you want to segment the data from. Default is 'filtered_velocity'

    Returns
    -------
    trial_summary : pd.DataFrame
        DataFrame containing the snippets of speed traces aligned to different epochs

    """
    trial_summary = pd.DataFrame()
    samples_per_second = np.around(np.mean(continuous_data.index.diff().dropna()), 3)
    
    # Iterate through reward sites and align the continuous data to whatever value was chosen. If aligned is used, it will align to any of the columns with time values.
    # If align is empty, it will align to the index, which in the case of the standard reward sites is the start of the odor site.
    for start_reward, row in reward_sites.iloc[:-1].iterrows():
        if cropped_to_length == 'sniff':
            # window[0] = -1
            # window[1] = row['odor_duration']
            window[0] = 0
            window[1] = row['next_index'] - start_reward   
        elif cropped_to_length == 'patch':    
            window[0] = row['time_since_entry']
            window[1] = row['exit_epoch']
        elif cropped_to_length == 'epoch':
            window[0] = -2
            window[1] = row['duration_epoch']
        else:
            pass
            
        trial_average = pd.DataFrame()
        if aligned != 'index':
            trial = continuous_data[(continuous_data.index >= row[aligned] + window[0]) & (continuous_data.index < row[aligned] + window[1])][taken_col]
            trial.index -= row[aligned]
            time_reference = row[aligned]
        else:
            trial = continuous_data.loc[
                start_reward + window[0] : start_reward + window[1], taken_col
            ]

            trial.index -= start_reward
            time_reference = start_reward
            
        if continuous == True:
            # Assuming trial.values, window, and samples_per_second are defined
            # Calculate the maximum number of intervals that can fit within the available data points
            max_intervals = len(trial.values) * samples_per_second

            # Calculate the actual stop value based on the maximum possible intervals
            actual_stop = min(window[1], window[0] + max_intervals)

            # Generate the time range with the adjusted stop value
            times = np.arange(window[0], actual_stop, samples_per_second)
            if len(times) != len(trial.values):
                # print('Different timing than values, ', len(times), len(trial.values))
                trial = trial.values[:len(times)]
            else:
                trial = trial.values
            trial_average["times"] = times

        else:
            trial_average["times"] = trial.index     
            trial = trial.values    
             
        
        trial_average['time_reference'] = time_reference
        
        if len(trial) == len(trial_average["times"]):
            if "filtered_velocity" == taken_col:
                trial_average["speed"] = trial
            else:
                trial_average[taken_col] = trial
        else:
            continue
            
        # Rewrites all the columns in the reward_sites to be able to segment the chosen traces in different values
        for column in reward_sites.columns:
            trial_average[column] = np.repeat(row[column], len(trial))
        
        trial_summary = pd.concat([trial_summary, trial_average], ignore_index=True)

    return trial_summary


def trial_collection_distance(
    reward_sites: pd.DataFrame,
    continuous_data: pd.DataFrame,
    aligned: str = 'index',
    cropped_to_length: str = 'window',
    window: list = [-0.5, 2],
    taken_col: str = "filtered_velocity",
    continuous: bool = True,
):
    """
    Crop the snippets of speed traces that are aligned to different epochs

    Parameters
    ----------
    reward_sites : pd.DataFrame
        DataFrame containing the reward sites information (odor sites)
    continuous_data : pd.DataFrame
        DataFrame containing the continuous data (encoder, sniffing, etc)
    mouse : str
        Mouse name
    session : str
        Session name
    aligned : str
        Column name to align the snippets
    window : tuple
        Time window to crop the snippets
    taken_col: string
        name of the column that you want to segment the data from. Default is 'filtered_velocity'

    Returns
    -------
    trial_summary : pd.DataFrame
        DataFrame containing the snippets of speed traces aligned to different epochs

    """
    trial_summary = pd.DataFrame()
    samples_per_second = np.around(np.mean(continuous_data.index.diff().dropna()), 3)
    
    # Iterate through reward sites and align the continuous data to whatever value was chosen. If aligned is used, it will align to any of the columns with time values.
    # If align is empty, it will align to the index, which in the case of the standard reward sites is the start of the odor site.
    for start_reward, row in reward_sites.iloc[:-1].iterrows():
        if cropped_to_length == 'sniff':
            # window[0] = -1
            # window[1] = row['odor_duration']
            window[0] = 0
            window[1] = row['next_index'] - start_reward   
        elif cropped_to_length == 'patch':    
            window[0] = row['time_since_entry']
            window[1] = row['exit_epoch']
        elif cropped_to_length == 'epoch':
            window[0] = -2
            window[1] = row['duration_epoch']
        else:
            pass
            
        trial_average = pd.DataFrame()
        if aligned != 'index':
            trial = continuous_data[(continuous_data['Position'] >= row[aligned] + window[0]) & (continuous_data['Position'] < row[aligned] + window[1])][taken_col]
            trial.index -= row[aligned]
            time_reference = row[aligned]
            print(trial)
        else:
            trial = continuous_data.loc[
                start_reward + window[0] : start_reward + window[1], taken_col
            ]

            trial.index -= start_reward
            time_reference = start_reward
            
        if continuous == True:
            # Assuming trial.values, window, and samples_per_second are defined
            # Calculate the maximum number of intervals that can fit within the available data points
            max_intervals = len(trial.values) * samples_per_second

            # Calculate the actual stop value based on the maximum possible intervals
            actual_stop = min(window[1], window[0] + max_intervals)

            # Generate the time range with the adjusted stop value
            times = np.arange(window[0], actual_stop, samples_per_second)
            if len(times) != len(trial.values):
                # print('Different timing than values, ', len(times), len(trial.values))
                trial = trial.values[:len(times)]
            else:
                trial = trial.values
            trial_average["times"] = times
        elif continuous == 'distance':
            trial_average["times"] = trial['Position']
            trial = trial.values
        else:
            trial_average["times"] = trial.index     
            trial = trial.values    
             
        
        trial_average['time_reference'] = time_reference
        
        if len(trial) == len(trial_average["times"]):
            if "filtered_velocity" == taken_col:
                trial_average["filtered_velocity"] = trial
            else:
                trial_average[taken_col] = trial
        else:
            continue
            
        # Rewrites all the columns in the reward_sites to be able to segment the chosen traces in different values
        for column in reward_sites.columns:
            trial_average[column] = np.repeat(row[column], len(trial))
        
        trial_summary = pd.concat([trial_summary, trial_average], ignore_index=True)

    return trial_summary


def raster_with_velocity(
    active_site: pd.DataFrame,
    stream_data: pd.DataFrame,
    save = False,
    color_dict_label: dict = {
        "Ethyl Butyrate": "#d95f02",
        "Alpha-pinene": "#1b9e77",
        "Amyl Acetate": "#7570b3",
    },
    with_velocity: bool = True,
    barplots: bool = True,
):
            
    test_df = active_site.groupby('patch_number').agg({'time_since_entry': 'min', 'patch_onset': 'mean','exit_epoch' : 'max'})
    test_df.reset_index(inplace=True)
    test_df.fillna(15, inplace=True)   
    
    if len(test_df) <= 1:
        print('Not enough patches to plot raster with velocity')
        return
    
    trial_summary = trial_collection(test_df, stream_data.encoder_data, aligned='patch_onset', cropped_to_length='patch')

    n_patches = active_site.patch_number.nunique()
    n_max_stops = active_site.site_number.max() + 1
    fig, ax1 = plt.subplots(figsize=(15+ n_max_stops/2, n_patches/2))
    ax2 = ax1.twinx()

    max_speed = np.quantile(trial_summary['speed'],0.99)
    for index, row in active_site.iterrows():
        if row['label'] == 'InterPatch':
            color = '#b3b3b3'
        elif row['label'] == 'InterSite':
            color = '#808080'
        elif row['label'] == 'PostPatch':
            color = '#b3b3b3'
            
        if row['label'] == 'OdorSite':
            if row['site_number'] == 0:
                ax1.scatter(0, row.patch_number, color=color_dict_label[row.patch_label], marker='s', s=60, edgecolor='black', linewidth=0.0)

            if row["is_reward"] == 1 and row["is_choice"] == True:
                color = "steelblue"
            elif row["is_reward"] == 0 and row["is_choice"] == True:
                color = "pink"
            else:
                color = 'yellow'
                
        if barplots:
            ax1.barh(int(row['patch_number']), left=row.time_since_entry, height=0.85, width=row.duration_epoch, color=color,  linewidth=0.5)
        
        if with_velocity:
            if row['time_since_entry'] <0:
                current_trial = trial_summary[trial_summary['patch_number'] == row['patch_number']]

                ax2.plot(current_trial['times'], current_trial['speed']+(max_speed*(row['patch_number']))+max_speed/1.8, color='black', linewidth=0.8, alpha=0.8)
                ax2.set_ylim(0, max_speed*(active_site['patch_number'].max()+2))

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Patch number")
    sns.despine()
    ax1.set_ylim(-1, max(active_site.patch_number) + 1)
    
    if active_site.groupby('patch_number').time_since_entry.min().min() < -50:
        time_left = -50
    else:
        time_left = active_site.groupby('patch_number').time_since_entry.min().min()
    
    if active_site.groupby('patch_number').time_since_entry.max().max() > 300:
        time_right = 250
    else:
        time_right = active_site.groupby('patch_number').time_since_entry.max().max()
      
    ax1.set_xlim(time_left, time_right)
    
    # Create legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    if save:
        save.savefig(fig)
        plt.close(fig)
    else:
        plt.show()
        return fig


def pstay_past_no_rewards(reward_sites, config, save=False, summary: bool = False):
    odor_label_list = reward_sites["odor_label"].unique()
    df_results_summary = pd.DataFrame()

    fig, ax1 = plt.subplots(
        1, len(odor_label_list), figsize=(4 * len(odor_label_list), 4)
    )
    for i, odor_label in enumerate(odor_label_list):
        if len(odor_label_list) != 1:
            ax = ax1[i]
        else:
            ax = ax1

        df_results = (
            reward_sites.loc[reward_sites["odor_label"] == odor_label]
            .groupby("past_no_reward_count")["site_number"]
            .count()
            .reset_index()
        )
        df_results["p(Stay)"] = (
            reward_sites.loc[reward_sites["odor_label"] == odor_label]
            .groupby("past_no_reward_count")["is_choice"]
            .mean()
        )
        df_results = df_results.loc[df_results["site_number"] >= 3]

        sufficient_sites = df_results.site_number.unique()

        df_results["odor_label"] = odor_label
        df_results["amount"] = reward_sites.loc[
            reward_sites["odor_label"] == odor_label
        ]["amount"].unique()[0]
        df_results_summary = pd.concat([df_results, df_results_summary])

        ax.set_title(
            odor_label
            + "_"
            + reward_sites.loc[reward_sites["odor_label"] == odor_label][
                "is_reward"
            ]
            .max()
            .astype(str)
        )
        sns.lineplot(
            x="past_no_reward_count",
            y="p(Stay)",
            data=df_results,
            color="k",
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("Previous unrewarded")
        ax.set_ylabel("P(Stay)")
        ax.set_ylim([-0.05, 1.05])
        ax.set_title(
            odor_label
            + "_"
            + reward_sites.loc[reward_sites["odor_label"] == odor_label][
                "is_reward"
            ]
            .max()
            .astype(str)
        )

        # ax[i].xaxis.set_major_formatter(FuncFormatter(format_func))
        # ax[i].xaxis.set_major_locator(FixedLocator(sufficient_sites))

        for j in range(len(df_results)):
            if df_results["p(Stay)"].values[j] < 0.1:
                ax.text(
                    df_results["past_no_reward_count"].values[j],
                    df_results["p(Stay)"].values[j] + 0.1,
                    str(df_results["site_number"].values[j]),
                    ha="center",
                    size=10,
                    color="red",
                )
            else:
                ax.text(
                    df_results["past_no_reward_count"].values[j],
                    df_results["p(Stay)"].values[j] - 0.1,
                    str(df_results["site_number"].values[j]),
                    ha="center",
                    size=10,
                    color="red",
                )

        ax.set_xticks(df_results["past_no_reward_count"].values)

    sns.despine()
    plt.tight_layout()
    if save != False:
        save.savefig(fig)
    else:
        plt.show()
    plt.close(fig)

    if summary == True:
        return df_results_summary


def pstay_site_number(reward_sites, config, save=False, summary: bool = False):
    odor_label_list = reward_sites["odor_label"].unique()
    df_results_summary = pd.DataFrame()

    fig, ax1 = plt.subplots(
        1, len(odor_label_list), figsize=(4 * len(odor_label_list), 4)
    )
    for i, odor_label in enumerate(odor_label_list):
        if len(odor_label_list) != 1:
            ax = ax1[i]
        else:
            ax = ax1

        df_results = (
            reward_sites.loc[reward_sites["odor_label"] == odor_label]
            .groupby("site_number")["label"]
            .count()
            .reset_index()
        )
        df_results.rename(columns={"label": "total_trials"}, inplace=True)
        df_results["p(Stay)"] = (
            reward_sites.loc[reward_sites["odor_label"] == odor_label]
            .groupby("site_number")["is_choice"]
            .mean()
        )
        df_results = df_results.loc[df_results["total_trials"] >= 3]
        sufficient_sites = df_results.site_number.unique()

        df_results["odor_label"] = odor_label
        df_results["amount"] = reward_sites.loc[
            reward_sites["odor_label"] == odor_label
        ]["amount"].unique()[0]
        df_results_summary = pd.concat([df_results, df_results_summary])

        ax.set_title(
            odor_label
            + "_"
            + reward_sites.loc[reward_sites["odor_label"] == odor_label][
                "is_reward"
            ]
            .max()
            .astype(str)
        )
        sns.lineplot(
            x="site_number", y="p(Stay)", data=df_results, color="k", marker="o", ax=ax
        )
        ax.set_xlabel("Visit number")
        ax.set_ylabel("P(Stay)")
        ax.set_ylim([-0.05, 1.05])

        for j in range(len(df_results)):
            if df_results["p(Stay)"].values[j] < 0.1:
                ax.text(
                    df_results["site_number"].values[j],
                    df_results["p(Stay)"].values[j] + 0.1,
                    str(df_results["total_trials"].values[j]),
                    ha="center",
                    size=10,
                    color="red",
                )
            else:
                ax.text(
                    df_results["site_number"].values[j],
                    df_results["p(Stay)"].values[j] - 0.1,
                    str(df_results["total_trials"].values[j]),
                    ha="center",
                    size=10,
                    color="red",
                )

        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        ax.xaxis.set_major_locator(FixedLocator(sufficient_sites))
        ax.set_ylim(0, 1)

    sns.despine(trim=True, offset=4)
    plt.tight_layout()
    if save != False:
        save.savefig(fig)
    else:
        plt.show()
    plt.close(fig)

    if summary == True:
        return df_results_summary


def length_distributions(
    active_site: pd.DataFrame, data, delay: bool = False, save=False
):

    def larger_value(value1, value2):
        if value1 < value2:
            return value1
        else:
            return value2

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    if "Gap" in active_site["label"].values:
        gap_size = active_site.loc[active_site["label"] == "Gap"].length.values
    else:
        gap_size = active_site.loc[active_site["label"] == "InterSite"].length.values
    # print('Gap: ', np.round(np.mean(gap_size),3))
    x = ax[0].hist(gap_size, bins=20, color="black")
    ax[0].vlines(np.mean(gap_size), 0, max(x[0]), color="red", linewidth=2)
    ax[0].set_title("Gap")
    ax[0].set_xlabel("Distance (mm)")

    intersite_size = active_site.loc[active_site["label"] == "InterPatch"].length.values
    # print('InterPatch: ', np.round(np.mean(intersite_size),3))
    x = ax[1].hist(intersite_size, bins=20, color="black")
    ax[1].vlines(np.mean(intersite_size), 0, max(x[0]), color="red", linewidth=2)
    ax[1].set_title("InterPatch")
    ax[1].set_xlabel("Distance (mm)")

    if delay == True:
        waitReward = data["software_events"].streams.ChoiceFeedback.data.index
        waitLick = data["software_events"].streams.GiveReward.data.index
        if len(waitLick) == len(waitReward):
            delay = waitLick - waitReward
        else:
            result = larger_value(len(waitLick), len(waitReward))
            delay = waitLick[:result] - waitReward[:result]

        # print('Delay: ', np.round(np.mean(delay),3))
        x = ax[2].hist(delay, bins=20, range=(0, 2), color="black")
        ax[2].vlines(np.median(delay), 0, max(x[0]), color="red", linewidth=2)
        ax[2].set_title("Delay")
        ax[2].set_xlabel("Time (s)")

    sns.despine()

    if save != False:
        save.savefig(fig)
        plt.close(fig)
    else:
        plt.show()


def raster_plot(x_start, pdf):
    fig, axs = plt.subplots(1, 1, figsize=(20, 4))

    _legend = {}
    for idx, site in enumerate(sites.iloc[:-1].iterrows()):
        site_label = site[1]["data"]["label"]
        if site_label == "Reward":
            site_label = f"Odor {site[1]['data']['odor']['index']+1}"
            facecolor = label_dict[site_label]
        elif site_label == "OdorSite":
            site_label = f"Odor {site[1]['data']['odor_specification']['index']+1}"
            facecolor = label_dict[site_label]
        elif site_label == "InterPatch":
            facecolor = label_dict[site_label]
        else:
            site_label = "InterSite"
            facecolor = label_dict["InterSite"]

        p = Rectangle(
            (sites.index[idx] - zero_index, -2),
            sites.index[idx + 1] - sites.index[idx],
            8,
            linewidth=0,
            facecolor=facecolor,
            alpha=0.5,
        )
        _legend[site_label] = p
        axs.add_patch(p)

    s, lw = 400, 2
    # Plotting raster
    y_idx = -0.4
    _legend["Choice Tone"] = axs.scatter(
        choice_feedback.index - zero_index + 0.2,
        choice_feedback.index * 0 + y_idx,
        marker="s",
        s=100,
        lw=lw,
        c="darkblue",
        label="Choice Tone",
    )
    y_idx += 1
    _legend["Lick"] = axs.scatter(
        lick_onset.index - zero_index,
        lick_onset.index * 0 + y_idx,
        marker="|",
        s=s,
        lw=lw,
        c="k",
        label="Lick",
    )
    _legend["Reward"] = axs.scatter(
        give_reward.index - zero_index,
        give_reward.index * 0 + y_idx,
        marker=".",
        s=s,
        lw=lw,
        c="deepskyblue",
        label="Reward",
    )
    _legend["Waits"] = axs.scatter(
        succesfullwait.index - zero_index,
        succesfullwait.index * 0 + 1.2,
        marker=".",
        s=s,
        lw=lw,
        c="green",
        label="Reward",
    )

    # _legend["Odor_on"] = axs.scatter(odor_on - zero_index,
    #     odor_on*0 + 2.5,
    #     marker="|", s=s, lw=lw, c='pink',
    #     label="ON")

    # _legend["Odor_off"] = axs.scatter(odor_off - zero_index,
    #     odor_off*0 + 2.5,
    #     marker="|", s=s, lw=lw, c='purple',
    #     label="ON")

    y_idx += 1

    # ax.set_xticks(np.arange(0, sites.index[-1] - zero_index, 10))
    axs.set_yticklabels([])
    axs.set_xlabel("Time(s)")
    axs.set_ylim(bottom=-1, top=3)
    axs.grid(False)
    plt.gca().yaxis.set_visible(False)

    ax2 = axs.twinx()
    _legend["Velocity"] = ax2.plot(
        encoder_data.index - zero_index,
        encoder_data.filtered_velocity,
        c="k",
        label="Encoder",
        alpha=0.8,
    )[0]
    try:
        v_thr = config.streams.TaskLogic.data["operationControl"]["positionControl"][
            "stopResponseConfig"
        ]["velocityThreshold"]
    except:
        v_thr = 8
    _legend["Stop Threshold"] = ax2.plot(
        ax2.get_xlim(), (v_thr, v_thr), c="k", label="Encoder", alpha=0.5, lw=2, ls="--"
    )[0]
    ax2.grid(False)
    ax2.set_ylim((-5, 70))
    ax2.set_ylabel("Velocity (cm/s)")

    axs.legend(
        _legend.values(),
        _legend.keys(),
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        borderaxespad=0.0,
    )
    axs.set_xlabel("Time(s)")
    axs.grid(False)
    axs.set_ylim(bottom=-1, top=4)
    axs.set_yticks([0, 3])
    axs.yaxis.tick_right()
    axs.set_xlim([x_start, x_start + 80])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def update_values(reward_sites, save = False):
    fig, ax = plt.subplots(3,1, figsize=(10,10), sharex=True)
    try:
        sns.lineplot(data=reward_sites, x='odor_sites', y='velocity_threshold_cms', color='black', ax=ax[0])
        ax[0].set_ylabel('Velocity \n threshold (cm/s)')
    except:
        pass
    
    try:
        sns.lineplot(data=reward_sites, x='odor_sites', y='delay_s', color='black', ax=ax[1])
        ax[1].set_ylabel('Delay (s)')
    except:
        pass
    
    try:
        sns.lineplot(data=reward_sites, x='odor_sites', y='stop_duration_s', color='black', ax=ax[2])
        ax[2].set_ylabel('Stop duration (s)')
        ax[2].set_xlabel('Odor sites')
    except:
        pass
    
    sns.despine()
    plt.tight_layout()
    
    if save:
        save.savefig(fig)
    else:
        plt.show()
    plt.close(fig)