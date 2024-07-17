# %%
import sys
from math import e

sys.path.append("../src/")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
# Data processing toold
import pandas as pd
# Plotting libraries
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


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
            if dataframe["label"].values[0] == "RewardSite":
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


def segmented_raster_vertical(
    reward_sites: pd.DataFrame,
    config,
    save: bool = False,
    color_dict_label: dict = {
        "Ethyl Butyrate": "#d95f02",
        "Alpha-pinene": "#1b9e77",
        "Amyl Acetate": "#7570b3",
    },
):

    patch_number = len(reward_sites.active_patch.unique())
    number_odors = len(reward_sites["odor_label"].unique())

    # Make second row proportional to the number of odors
    list_odors = []
    for odor in reward_sites.odor_label.unique():
        list_odors.append(
            reward_sites.loc[reward_sites.odor_label == odor].active_patch.nunique()
        )
    grid = (np.array(list_odors) / patch_number) * number_odors

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, number_odors, width_ratios=grid)

    for index, row in reward_sites.iterrows():
        ax1 = plt.subplot(gs[0, 0:number_odors])
        if row["reward_delivered"] == 1 and row["has_choice"] == True:
            color = "steelblue"
        elif row["reward_delivered"] == 0 and row["has_choice"] == True:
            color = "pink"
            if row["reward_available"] == 0:
                color = "crimson"
        else:
            if row["reward_available"] == 0:
                color = "black"
            else:
                color = "lightgrey"

        # ax1.barh(int(row['active_patch']), left=row['visit_number'], height=1, width=1, color=color, edgecolor='darkgrey', linewidth=0.5)
        ax1.bar(
            int(row["active_patch"]),
            bottom=row["visit_number"],
            height=1,
            width=1,
            color=color,
            edgecolor="darkgrey",
            linewidth=0.5,
        )
        ax1.set_xlim(-1, max(reward_sites.active_patch) + 1)
        ax1.set_xlabel("Patch number")
        ax1.set_ylabel("Site number")

        # ax1.bar(int(row['active_patch']), bottom = -1, height=0.5, width = 1, color=patch_color, edgecolor='black', linewidth=0.5)
        ax1.scatter(
            row["active_patch"],
            -0.25,
            color=color_dict_label[row["odor_label"]],
            marker="^",
            s=35,
            edgecolor="black",
            linewidth=0.0,
        )

    odors = []
    for odor in reward_sites["odor_label"].unique():
        odors.append(
            mpatches.Patch(
                color=color_dict_label[odor],
                label=(
                    str(odor)
                    + "_"
                    + str(
                        reward_sites.loc[
                            reward_sites.odor_label == odor
                        ].reward_delivered.max()
                    )
                ),
            )
        )

    label_2 = mpatches.Patch(color="steelblue", label="Harvest, rewarded")
    label_3 = mpatches.Patch(color="crimson", label="Harvest, no reward, depleted")
    label_4 = mpatches.Patch(color="lightgrey", label="Leave, not depleted")
    label_5 = mpatches.Patch(color="black", label="Leave, depleted")
    label_6 = mpatches.Patch(color="pink", label="Harvest, no reward, probabilitic")

    odors.extend([label_2, label_3, label_6, label_4, label_5])
    ax1.set_ylim(-1, max(reward_sites.visit_number) + 1)

    if len(reward_sites["odor_label"].unique()) != 1:
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[1, 1])
        ax4 = plt.subplot(gs[1, 2])
        for ax, odor_label in zip([ax2, ax3, ax4], reward_sites.odor_label.unique()):
            selected_sites = reward_sites.loc[reward_sites.odor_label == odor_label]
            previous_active = 0
            value = 0
            for index, row in selected_sites.iterrows():
                # Choose the color of the site
                if row["reward_delivered"] == 1 and row["has_choice"] == True:
                    color = "steelblue"
                elif row["reward_delivered"] == 0 and row["has_choice"] == True:
                    color = "pink"
                    if row["reward_available"] == 0:
                        color = "crimson"
                else:
                    if row["reward_available"] == 0:
                        color = "black"
                    else:
                        color = "lightgrey"

                ax.set_title(odor_label, color=color_dict_label[row["odor_label"]])

                if row["active_patch"] != previous_active:
                    value += 1
                    previous_active = row["active_patch"]
                ax.bar(
                    value,
                    bottom=row["visit_number"],
                    height=1,
                    width=1,
                    color=color,
                    edgecolor="darkgrey",
                    linewidth=0.5,
                )
                ax.set_xlim(-1, selected_sites.active_patch.nunique() + 1)
                ax.set_ylim(-0.5, reward_sites.visit_number.max() + 1)
                ax.set_ylabel("Site number")
                ax.set_xlabel("Patch number")

    fig.tight_layout()
    plt.legend(
        handles=odors,
        loc="lower right",
        bbox_to_anchor=(0.95, -1.25),
        fontsize=12,
        ncol=3,
    )
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

    n_odors = len(trial_summary.odor_label.unique())
    fig, ax = plt.subplots(
        n_odors, 5, figsize=(18, n_odors * 4), sharex=True, sharey=True
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
                ax[len(trial_summary.odor_label.unique()) - 1][j].set_xlabel(
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

    for i in trial_summary.has_choice.unique():
        assigned_colors = choose_palette(trial_summary, condition=condition)

        i = int(i)
        for j, odor_label in enumerate(trial_summary.odor_label.unique()):
            if n_odors == 1:
                ax1 = ax
            else:
                ax1 = ax[j]

            df_results = (
                trial_summary.loc[
                    (trial_summary.odor_label == odor_label)
                    & (trial_summary.has_choice == i)
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

            ax1[2].set_title(f"Odor {odor_label} ")

            if i == 1:
                ax1[i].text(1.2, 75, f"Stopped", fontsize=12)
                ax1[i + 2].text(1.2, 75, f"Stopped", fontsize=12)
            else:
                ax1[i].text(1.2, 75, f"Not stopped", fontsize=12)
                ax1[i + 2].text(1.2, 75, f"Not stopped", fontsize=12)

    sns.despine()
    plt.suptitle(mouse + "_" + session)
    plt.tight_layout()

    if save != False:
        save.savefig(fig)
        plt.close(fig)


def speed_traces_efficient(
    trial_summary: pd.DataFrame,
    mouse,
    session,
    odor: str = "all",
    window: tuple = (-1, 3),
    save=False,
):
    """Plots the speed traces for each stopping condition"""

    if odor != "all":
        reward_sites = reward_sites.loc[reward_sites["odor_label"] == odor]

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
            trial_summary.has_choice == collected_label
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
            trial_summary.loc[trial_summary.has_choice == collected_label]
            .groupby("times")["speed"]
            .mean()
            .reset_index()
        )
        ax[0][collected_label].plot(
            "times", "speed", data=selected_trials, color="black", linewidth=3
        )

        if collected_label == 1:
            ax[0][collected_label].set_title(f" Stopped", color=colors[collected_label])
        else:
            ax[0][collected_label].set_title(
                f" Not Stopped", color=colors[collected_label]
            )

        sns.lineplot(
            x="times",
            y="speed",
            data=trial_summary.loc[(trial_summary.has_choice == collected_label)],
            color=colors[collected_label],
            ax=ax[0][2],
            errorbar=("sd"),
            linewidth=2,
        )

        # ---------- water collection or not
        plot_df = trial_summary.loc[
            (trial_summary.reward_delivered == collected_label)
            & (trial_summary.has_choice == 1)
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
                f" Rewarded stop", color=colors[collected_label]
            )
        else:
            ax[1][collected_label].set_title(
                f" No reward stop", color=colors[collected_label]
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
            & (trial_summary.has_choice == 1)
            & (trial_summary.reward_delivered == 0)
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
                f" No reward - depleted", color=colors[collected_label]
            )
        else:
            ax[2][collected_label].set_title(
                f" No reward - not depleted", color=colors[collected_label]
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
    plt.suptitle(mouse + "_" + session + "_" + odor)
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

    summary = reward_sites.groupby(["active_patch", "odor_label"]).agg(
        {
            "reward_delivered": "sum",
            "visit_number": "count",
            "reward_probability": "min",
        }
    )
    summary = summary.loc[summary.visit_number > minimum_size]
    summary.reset_index(inplace=True)

    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 3, 1)
    sns.boxplot(
        x="odor_label",
        y="reward_delivered",
        hue="odor_label",
        palette=color_dict_label,
        order=[list(color_dict_label.keys())[0], list(color_dict_label.keys())[1]],
        data=summary,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        x="odor_label",
        y="reward_delivered",
        hue="odor_label",
        palette=["black", "black", "black"],
        order=[list(color_dict_label.keys())[0], list(color_dict_label.keys())[1]],
        data=summary,
        ax=ax,
        linewidth=0.2,
        edgecolor="black",
        jitter=0.25,
    )
    plt.xlabel("Odor")
    plt.ylabel("Total reward \n collected")
    plt.xticks([0, 1], [0.9, 0.6])
    plt.xlabel("Initial P(reward)")

    ax = plt.subplot(1, 3, 2)
    sns.boxplot(
        x="odor_label",
        y="visit_number",
        hue="odor_label",
        palette=color_dict_label,
        order=[list(color_dict_label.keys())[0], list(color_dict_label.keys())[1]],
        data=summary,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        x="odor_label",
        y="visit_number",
        hue="odor_label",
        palette=["black", "black", "black"],
        order=[list(color_dict_label.keys())[0], list(color_dict_label.keys())[1]],
        data=summary,
        ax=ax,
        linewidth=0.2,
        edgecolor="black",
        jitter=0.25,
    )

    plt.xlabel("Odor")
    plt.ylabel("Total stops")
    plt.xticks([0, 1], [0.9, 0.6])
    plt.xlabel("Initial P(reward)")

    ax = plt.subplot(1, 3, 3)
    sns.boxplot(
        x="odor_label",
        y="reward_probability",
        hue="odor_label",
        palette=color_dict_label,
        order=[list(color_dict_label.keys())[0], list(color_dict_label.keys())[1]],
        data=summary,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        x="odor_label",
        y="reward_probability",
        hue="odor_label",
        palette=["black", "black", "black"],
        order=[list(color_dict_label.keys())[0], list(color_dict_label.keys())[1]],
        data=summary,
        ax=ax,
        linewidth=0.2,
        edgecolor="black",
        jitter=0.25,
    )

    plt.xlabel("Odor")
    plt.ylabel("P(reward) when leaving")
    plt.xticks([0, 1], [0.9, 0.6])
    plt.xlabel("Initial P(reward)")

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
    n_odors = [
        list(color_dict_label.keys())[0],
        list(color_dict_label.keys())[1],
        list(color_dict_label.keys())[2],
        list(color_dict_label.keys())[3],
    ]

    fig, ax1 = plt.subplots(
        1, len(n_odors), figsize=(len(n_odors) * 3.5, 4), sharex=True, sharey=True
    )

    for j, odor_label in enumerate(n_odors):
        if len(n_odors) != 1:
            ax = ax1[j]
            ax1[0].set_ylabel("Velocity (cm/s)")
        else:
            ax = ax1
            ax.set_ylabel("Velocity (cm/s)")

        ax.set_xlabel("Time after odor onset (s)")
        ax.set_ylim(-13, max_range)
        ax.set_xlim(window)
        ax.hlines(
            5, window[0], window[1], color="black", linewidth=1, linestyles="dashed"
        )
        ax.fill_betweenx(
            np.arange(-20, max_range, 0.1),
            0,
            window[1],
            color=color_dict_label[odor_label],
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
                (trial_summary.odor_label == odor_label)
                & (trial_summary.visit_number == 0)
            ]
            .groupby(["odor_sites", "times", "odor_label"])[["speed"]]
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


def trial_collection(
    reward_sites: pd.DataFrame,
    continuous_data: pd.DataFrame,
    mouse: str,
    session: str,
    aligned: str = None,
    window: tuple = (-0.5, 2),
    taken_col: str = "filtered_velocity",
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

    # Iterate through reward sites and align the continuous data to whatever value was chosen. If aligned is used, it will align to any of the columns with time values.
    # If align is empty, it will align to the index, which in the case of the standard reward sites is the start of the odor site.
    for start_reward, row in reward_sites.iterrows():
        trial_average = pd.DataFrame()
        if aligned is not None:
            trial = continuous_data[(continuous_data.index >= row[aligned] + window[0]) & (continuous_data.index <= row[aligned] + window[1])][taken_col]
            trial.index -= row[aligned]
        else:
            trial = continuous_data.loc[
                start_reward + window[0] : start_reward + window[1], taken_col
            ]
            trial.index -= start_reward

        if "filtered_velocity" == taken_col:
            trial_average["speed"] = trial.values
        else:
            trial_average[taken_col] = trial.values

        trial_average["times"] = np.around(trial.index, 3)

        # Rewrites all the columns in the reward_sites to be able to segment the chosen traces in different values
        for column in reward_sites.columns:
            trial_average[column] = np.repeat(row[column], len(trial.values))

        trial_summary = pd.concat([trial_summary, trial_average], ignore_index=True)

    trial_summary["mouse"] = mouse
    trial_summary["session"] = session
    return trial_summary


# def session_raster_segmented(reward_sites,config, save=False):
#     # Create a figure with a 2x2 grid
#     fig = plt.figure(figsize=(12, 16))
#     gs = GridSpec(3, 2, width_ratios=[1, 1])

#     df_skip = pd.DataFrame()
#     for index, row in reward_sites.iterrows():
#         ax1 = plt.subplot(gs[0:3, 0])
#         if row['collected'] == 1 and row['has_choice'] == True:
#             color='steelblue'
#         elif row['collected'] == 0 and row['has_choice'] == True:
#             color='pink'
#             if row['reward_available'] == 0:
#                 color='crimson'
#         else:
#             if  row['reward_available'] == 0:
#                 color='black'
#             else:
#                 color='lightgrey'

#         ax1.barh(int(row['active_patch']), left=row['visit_number'], height=1, width=1, color=color, edgecolor='darkgrey', linewidth=0.5)

#         ax1.set_xlim(-1,max(reward_sites.visit_number)+1)
#         ax1.set_ylabel('Patch number')
#         ax1.set_xlabel('Site number')

#         if row['odor_label'] == reward_sites['odor_label'].unique()[0]:
#             patch_color='orange'
#         elif row['odor_label'] == reward_sites['odor_label'].unique()[1]:
#             patch_color='darkgreen'
#         else:
#             patch_color='yellow'

#         ax1.barh(int(row['active_patch']), left = -0.8, height=1, width =0.5, color=patch_color, edgecolor='black', linewidth=0.5)

#     odor_list_color = ['orange', 'indigo', 'darkgreen']
#     odors = []
#     for index, odor in enumerate(reward_sites['odor_label'].unique()):
#         odors.append(mpatches.Patch(color=odor_list_color[index], label=(str(odor) + '_' + str(reward_sites.loc[reward_sites.odor_label == odor].reward_delivered.max()))))

#     label_2 = mpatches.Patch(color='steelblue', label='Harvested')
#     label_3 = mpatches.Patch(color='crimson', label='No reward - depleted')
#     label_4 = mpatches.Patch(color='lightgrey', label='Skipped - not depleted')
#     label_5 = mpatches.Patch(color='black', label='Skipped - depleted')
#     label_6 = mpatches.Patch(color='pink', label='No reward - not depleted')
#     odors.extend([label_2, label_3,label_4,label_5,label_6])
#     ax1.legend(handles=odors, loc='upper left', bbox_to_anchor=(0.8, 1), fontsize=8)
#     ax1.set_ylim(-1,max(reward_sites.active_patch)+1)
#     plt.tight_layout()
#     sns.despine()

#     if len(reward_sites['odor_label'].unique()) != 1:
#         ax2 = plt.subplot(gs[0, 1])
#         ax3 = plt.subplot(gs[1, 1])
#         ax4 = plt.subplot(gs[2, 1])
#         for ax, odor_label in zip([ax2, ax3, ax4], reward_sites.odor_label.unique()):
#             selected_sites = reward_sites.loc[reward_sites.odor_label == odor_label]
#             previous_active = 0
#             value = 0
#             for index, row in selected_sites.iterrows():
#                 # Choose the color of the site
#                 if row['collected'] == 1 and row['has_choice'] == True:
#                     color='steelblue'
#                 elif row['collected'] == 0 and row['has_choice'] == True:
#                     color='pink'
#                     if row['reward_available'] == 0:
#                         color='crimson'
#                 else:
#                     if  row['reward_available'] == 0:
#                         color='black'
#                     else:
#                         color='lightgrey'

#                 if row['odor_label'] == reward_sites['odor_label'].unique()[0]:
#                     patch_color='orange'
#                 elif row['odor_label'] == reward_sites['odor_label'].unique()[1]:
#                     patch_color='darkgreen'
#                 else:
#                     patch_color='black'

#                 ax.set_title(odor_label, color=patch_color)

#                 if row['active_patch'] != previous_active:
#                     value+=1
#                     previous_active = row['active_patch']
#                 ax.barh(value, left=row['visit_number'], height=1, width=1, color=color, edgecolor='darkgrey', linewidth=0.5)
#                 ax.set_xlim(0,max(reward_sites.visit_number)+1)
#         ax4.set_xlabel('Site number')

#     # Set the maximum number of ticks on the x-axis
#     max_ticks = 5  # Replace this with the desired number of ticks
#     plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
#     plt.tight_layout()
#     sns.despine()
#     if save != False:
#         save.savefig(fig, bbox_inches='tight')
#         plt.close(fig)


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
            .groupby("past_no_reward_count")["visit_number"]
            .count()
            .reset_index()
        )
        df_results["p(Stay)"] = (
            reward_sites.loc[reward_sites["odor_label"] == odor_label]
            .groupby("past_no_reward_count")["has_choice"]
            .mean()
        )
        df_results = df_results.loc[df_results["visit_number"] >= 3]

        sufficient_sites = df_results.visit_number.unique()

        df_results["odor_label"] = odor_label
        df_results["amount"] = reward_sites.loc[
            reward_sites["odor_label"] == odor_label
        ]["amount"].unique()[0]
        df_results_summary = pd.concat([df_results, df_results_summary])

        ax.set_title(
            odor_label
            + "_"
            + reward_sites.loc[reward_sites["odor_label"] == odor_label][
                "reward_delivered"
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
                "reward_delivered"
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
                    str(df_results["visit_number"].values[j]),
                    ha="center",
                    size=10,
                    color="red",
                )
            else:
                ax.text(
                    df_results["past_no_reward_count"].values[j],
                    df_results["p(Stay)"].values[j] - 0.1,
                    str(df_results["visit_number"].values[j]),
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


def pstay_visit_number(reward_sites, config, save=False, summary: bool = False):
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
            .groupby("visit_number")["label"]
            .count()
            .reset_index()
        )
        df_results.rename(columns={"label": "total_trials"}, inplace=True)
        df_results["p(Stay)"] = (
            reward_sites.loc[reward_sites["odor_label"] == odor_label]
            .groupby("visit_number")["has_choice"]
            .mean()
        )
        df_results = df_results.loc[df_results["total_trials"] >= 3]
        sufficient_sites = df_results.visit_number.unique()

        df_results["odor_label"] = odor_label
        df_results["amount"] = reward_sites.loc[
            reward_sites["odor_label"] == odor_label
        ]["amount"].unique()[0]
        df_results_summary = pd.concat([df_results, df_results_summary])

        ax.set_title(
            odor_label
            + "_"
            + reward_sites.loc[reward_sites["odor_label"] == odor_label][
                "reward_delivered"
            ]
            .max()
            .astype(str)
        )
        sns.lineplot(
            x="visit_number", y="p(Stay)", data=df_results, color="k", marker="o", ax=ax
        )
        ax.set_xlabel("Visit number")
        ax.set_ylabel("P(Stay)")
        ax.set_ylim([-0.05, 1.05])

        for j in range(len(df_results)):
            if df_results["p(Stay)"].values[j] < 0.1:
                ax.text(
                    df_results["visit_number"].values[j],
                    df_results["p(Stay)"].values[j] + 0.1,
                    str(df_results["total_trials"].values[j]),
                    ha="center",
                    size=10,
                    color="red",
                )
            else:
                ax.text(
                    df_results["visit_number"].values[j],
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
        elif site_label == "RewardSite":
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
