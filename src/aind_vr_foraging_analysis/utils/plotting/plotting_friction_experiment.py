import sys
sys.path.append('../../../src/')

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math 
from scipy.stats import ttest_rel

sns.set_context('talk')

import warnings
pd.options.mode.chained_assignment = None  # Ignore SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.stats import ttest_ind
from matplotlib.ticker import MultipleLocator, MaxNLocator

pdf_path = r'Z:\scratch\vr-foraging\sessions'
base_path = r'Z:\scratch\vr-foraging\data'
data_path = r'../../../data/'

color1='#d95f02'
color2='#1b9e77'
color3='#7570b3'
color4='#e7298a'
odor_list_color = [color1, color2, color3]
color_dict = {0: color1, 1: color2, 2: color3}
color_dict_label = {'Ethyl Butyrate': color1, 'Alpha-pinene': color2, 'Alpha pinene': color2, 'Amyl Acetate': color3, 
                    '2-Heptanone' : color2, 'Methyl Acetate': color1, 'Fenchone': color3, '2,3-Butanedione': color4,
                    'Methyl Butyrate': color1, 
                    '90': color1, '60': color2, '0': color3, 
                    'odor_90': color1, 'odor_60': color2, 'odor_0': color3, 'odor_slow': color1, 'odor_fast': color2}

results_path = r'C:\Users\tiffany.ona\OneDrive - Allen Institute\Documents\VR foraging\experiments\batch 4 - manipulating cost of travelling and global statistics\results'

def set_clean_yaxis(ax, data, variable, annotation_top=None, n_ticks=4):

    ymin = 0
    ymax_data = np.nanmax(data[variable])
    ymax = max(ymax_data, annotation_top or 0)

    # Case 1: data is decimal (e.g., in [0, 1])
    if ymax <= 1.0:
        n_ticks = 5
        step = 0.75 / (n_ticks - 1)
        yticks = np.linspace(0.0, 0.75, n_ticks)
        ax.set_ylim(0.0, 0.75)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.1f}" for y in yticks])
    else:
        # Case 2: data is integer-like
        rough_range = ymax - ymin

        # Choose a sensible step size
        if rough_range < 6:
            step = 2
        elif rough_range < 10:
            step = 5
            ymax = 10
        elif rough_range < 20:
            step = 5
        elif rough_range < 50:
            step = 10
        else:
            step = 20

        ymax_rounded = int(np.ceil(ymax / step)) * step
        
        yticks = np.arange(ymin, ymax_rounded + 1, step)
        
        ax.set_ylim(ymin, ymax_rounded+0.1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(t) for t in yticks])
        
def solve_quadratic(y, a, b, c):
    # Adjust c for y
    c -= y
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return "No real solutions"
    elif discriminant == 0:
        # One solution
        x = -b / (2 * a)
        return [x]
    else:
        # Two solutions
        x1 = (-b + math.sqrt(discriminant)) / (2 * a)
        x2 = (-b - math.sqrt(discriminant)) / (2 * a)
        return [x1, x2]
    
# Define a quadratic model function to fit
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# Define exponential function
def exponential_func(x, a, b):
    return a * np.exp(b * x)

def format_func(value, tick_number):
    return f"{value:.0f}"

def plot_lines(data: pd.DataFrame, ax, variable='total_rewards', group='patch_label', one_line='mouse', order=None):
    """
    Plots individual lines for a specified variable across groups for each unique entity in the data.
    Args:
        data (pd.DataFrame): The input DataFrame containing the data to plot.
        ax (matplotlib.axes.Axes): The matplotlib axes object on which to plot.
        variable (str, optional): The column name in `data` to use for the y-axis values. Defaults to 'total_rewards'.
        group (str, optional): The column name in `data` to use for grouping on the x-axis. Defaults to 'patch_label'.
        one_line (str, optional): The column name in `data` representing individual entities (e.g., 'mouse'), each of which will have a separate line. Defaults to 'mouse'.
        order (list, optional): The order of group labels to use on the x-axis. If None, uses sorted unique values from the `group` column.
    Notes:
        - Each unique value in `one_line` will be plotted as a separate line.
        - X-axis positions are mapped from group labels according to the `order` parameter.
        - Lines are plotted with black color, partial transparency, and no markers.
    """
    if order is None:
        order = sorted(data['patch_label'].unique())

    # Map string patch_labels to seaborn's internal numeric positions (0, 1, ...)
    x_map = {label: i for i, label in enumerate(order)}

    for value in data[one_line].unique():
        df_subset = data[data[one_line] == value]
        y = df_subset[variable].values
        x_labels = df_subset[group].values
        x = [x_map[label] for label in x_labels]
        ax.plot(x, y, marker='', linestyle='-', color='black', alpha=0.4, linewidth=1)

def plot_significance(general_df: pd.DataFrame, axes, variable = 'total_rewards', group='patch_label', conditions = ['90', '60']):
    """
    Performs a statistical significance test between two groups in a DataFrame and annotates the result on a given matplotlib axes.
    Parameters
    ----------
    general_df : pd.DataFrame
        The DataFrame containing the data to be compared.
    axes : matplotlib.axes.Axes
        The axes object on which to plot the significance annotation.
    variable : str, optional
        The column name in `general_df` to compare between groups (default is 'total_rewards').
    group : str, optional
        The column name in `general_df` that defines the grouping variable (default is 'patch_label').
    conditions : list of str, optional
        The two group labels to compare (default is ['90', '60']).
    Returns
    -------
    float
        The y-coordinate at the top of the significance annotation.
    Notes
    -----
    - Performs a paired t-test (`ttest_rel`) by default. If this fails, falls back to an independent t-test (`ttest_ind`).
    - Annotates the axes with the significance level ('ns', '*', '**', or '***') based on the p-value.
    - Adjusts annotation position for the 'reward_probability' variable.
    """
        # Perform statistical test and add significance annotations
    group1 = general_df.loc[general_df[group]== conditions[0], variable]
    group2 = general_df.loc[general_df[group] == conditions[1], variable]
    
    # Perform t-test
    try:
        t_stat, p_value = ttest_rel(group1, group2, nan_policy='omit')
    except:
        print('Error in t-test paired, running independent t-test')
        t_stat, p_value = ttest_ind(group1, group2, nan_policy='omit')
    
    print(f'{variable} p-value: {p_value}')
    # Add significance annotation
    x1, x2 = 0, 1  # x-coordinates of the groups
    y, h, col = general_df[variable].max() + 1, 0.5, 'k'  # y-coord, line height, color
    if variable == 'reward_probability':
        y = 0.7
        h=0.025
        
    if p_value < 0.001:
        significance = "***" 
    elif p_value < 0.01:
        significance = "**" 
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"
    
    axes.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    axes.text((x1 + x2) * 0.5, y + h, significance, ha='center', va='bottom', color=col)
    return y + h  # Return the top of the annotation

def across_sessions_one_plot(summary_df, variable, save=False):
    experiments = summary_df['experiment'].unique()
    palette = sns.color_palette("tab20", len(experiments))
    color_dict_experiment = dict(zip(experiments, palette))

    # Create a style dictionary for each odor label
    odor_labels = summary_df['odor_label'].unique()
    styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    style_dict_odor_label = dict(zip(odor_labels, styles))
    
    for i, mouse in enumerate(summary_df.mouse.unique()):
        fig = plt.figure(figsize=(20,6))
        sns.scatterplot(summary_df.loc[(summary_df.mouse == mouse)], x='session_number', size="site_number", hue='experiment', style='odor_label', sizes=(30, 500), y=variable, 
                        palette=color_dict_experiment,  alpha=0.7,
                        markers=style_dict_odor_label)

        plt.xlabel('')
        plt.title(f'{mouse}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title='Experiment')
        sns.despine()
        plt.tight_layout()
        plt.show()
        if save:
            fig.savefig(save, format='pdf')
        plt.close()
            
    
def summary_main_variables(general_df, 
                           experiment: str = None, 
                           condition = 'mouse', 
                           odor_labels = ['Methyl Butyrate', 'Alpha-pinene'],
                           save=False):
    """
    Generates a summary plot of main behavioral variables for a given experiment and grouping condition.
    
    Parameters:
    general_df (pd.DataFrame): DataFrame containing the behavioral data.
    experiment (str): The name of the experiment to filter the data.
    condition (str, optional): The condition to group the data by. Default is 'mouse'.
    save (bool or str, optional): If False, the plot is displayed. If True, the plot is saved as 'summary_mouse.pdf'.
                                    If a string is provided, the plot is saved to the specified path.
    Returns:
    None
    """
            
    fig,ax = plt.subplots(2,3, figsize=(9,8))
    if condition == 'session_n':
        plt.suptitle(f'{general_df.mouse.iloc[0]} {experiment}')
    else:
        plt.suptitle(experiment)
        
    general_df = general_df.loc[(general_df.patch_label != '0')]

    if experiment:
        if 'experiment' in general_df.columns:
            general_df = general_df.loc[general_df.experiment == experiment]
        else:
            print(general_df.stage.unique())
            general_df = general_df.loc[general_df.stage == experiment]

    axes = ax[0][0]
    variable = 'total_rewards'
    sns.boxplot(x='patch_label', y=variable,  palette = color_dict_label, data=general_df, order=odor_labels, zorder=10, width =0.7, ax=axes, fliersize=0)
    plot_lines(data = general_df, ax = axes, variable = variable, one_line = condition, order=odor_labels)
    annotation_top = plot_significance(general_df, axes, variable, conditions=odor_labels)
    set_clean_yaxis(axes, general_df, variable, annotation_top=annotation_top)

    axes.set_ylabel('Rewards collected')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')

    axes = ax[0][1]
    variable = 'reward_probability'
    sns.boxplot(x='patch_label', y=variable,  palette = color_dict_label, data=general_df, order=odor_labels,  zorder=10, width =0.7, ax=axes, fliersize=0)

    plot_lines(data = general_df, ax = axes, variable = variable, one_line = condition, order=odor_labels)
    annotation_top = plot_significance(general_df, axes, variable, conditions=odor_labels)
    set_clean_yaxis(axes, general_df, variable, annotation_top=annotation_top)

    axes.set_ylabel('P(reward) at leaving')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')

    # Stops --------------------------------
    axes = ax[0][2]
    variable = 'stops'
    sns.boxplot(x='patch_label', y=variable,  palette = color_dict_label, data=general_df, order=odor_labels, zorder=10, width =0.7, ax=axes, fliersize=0)
    plot_lines(data = general_df, ax = axes, variable = variable, one_line = condition, order=odor_labels)
    annotation_top = plot_significance(general_df, axes, variable, conditions=odor_labels)
    set_clean_yaxis(axes, general_df, variable, annotation_top=annotation_top)

    axes.set_ylabel('Stops')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')
    
    # Total failures
    axes = ax[1][0]
    variable = 'total_failures'
    sns.boxplot(x='patch_label', y=variable,  palette = color_dict_label, data=general_df, order=odor_labels, zorder=10, width =0.7, ax=axes, fliersize=0)
    plot_lines(data = general_df, ax = axes, variable = variable, one_line = condition, order=odor_labels)
    annotation_top = plot_significance(general_df, axes, variable, conditions=odor_labels)
    set_clean_yaxis(axes, general_df, variable, annotation_top=annotation_top)

    axes.set_ylabel('Total failures')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')
    
    # Consecutive failures
    axes = ax[1][1]
    variable = 'consecutive_failures'
    sns.boxplot(x='patch_label', y=variable, palette = color_dict_label, data=general_df, order=odor_labels, zorder=10, width =0.7, ax=axes, fliersize=0)
    plot_lines(data = general_df, ax = axes, variable = variable, one_line = condition, order=odor_labels)
    annotation_top = plot_significance(general_df, axes, variable, conditions=odor_labels)
    set_clean_yaxis(axes, general_df, variable, annotation_top=annotation_top)
    axes.set_ylabel('Consecutive failures')
    axes.set_xticks([0, 1])
    axes.set_xticklabels(['Odor 1', 'Odor 2'])
    axes.set_xlabel('')

    # Duration epoch
    axes = ax[1][2]
    # variable = 'duration_epoch'
    # sns.boxplot(x='odor_label', y=variable, hue='odor_label', palette = color_dict_label, data=general_df, order=['Methyl Butyrate', 'Alpha-pinene'],legend=False, zorder=10, width =0.7, ax=axes)
    # plot_lines(general_df, axes, variable, condition)
    # axes.set_ylabel('Duration odor sites (s)')
    # axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    # axes.set_xlabel('')
    
    # Total patches
    # axes = ax[2][0]
    variable = 'patch_number'
    sns.boxplot(x='patch_label', y=variable, palette = color_dict_label, data=general_df, order=odor_labels, zorder=10, width =0.7, ax=axes, fliersize=0)
    plot_lines(data = general_df, ax = axes, variable = variable, one_line = condition, order=odor_labels)
    annotation_top = plot_significance(general_df, axes, variable, conditions=odor_labels)
    set_clean_yaxis(axes, general_df, variable, annotation_top=annotation_top)
    axes.set_ylabel('# patches')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')
    
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(save, format='pdf')
    else:
        plt.show()
        plt.close()

experiments_palette = {
    'data_collection': '#1f77b4',  # Blue
    'control': "#bebebe",  # Red
    'friction_high': '#6a51a3',  # Purple
    'friction_med': '#807dba',  # Lighter Purple
    'friction_low': '#9e9ac8',  # Lightest Purple
    'distance_extra_short': 'crimson',  # Blue
    'distance_short': 'pink',  # Lighter Blue
    'distance_extra_long': '#fd8d3c',  # Yellow
    'distance_long': '#fdae6b',  # Lighter Yellow
    'odor_60': '#1b9e77',  # Green
    'odor_90': '#d95f02',  # Orange
}

def across_sessions_one_plot(summary_df, variable, save=False):
    """
    Plots a scatterplot of a specified variable across sessions for each mouse, 
    with points colored by experiment and styled by patch label.
    Parameters
    ----------
    summary_df : pandas.DataFrame
        DataFrame containing summary data with at least the following columns:
        'experiment', 'patch_label', 'mouse', 'session_n', 'site_number', and the specified variable.
    variable : str
        The name of the column in `summary_df` to plot on the y-axis.
    save : str or bool, optional
        If a string (file path) is provided, saves each figure as a PDF to the specified path.
        If False (default), figures are not saved.
    Returns
    -------
    None
        Displays the plots for each mouse. Optionally saves the plots as PDF files.
    """
    experiments = summary_df['experiment'].unique()
    palette = sns.color_palette("tab10", len(experiments))
    color_dict_experiment = dict(zip(experiments, experiments_palette))

    # Create a style dictionary for each odor label
    odor_labels = summary_df['patch_label'].unique()
    styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    style_dict_odor_label = dict(zip(odor_labels, styles))
    
    min_value = summary_df[variable].min()
    max_value = summary_df[variable].max()
    for i, mouse in enumerate(summary_df.mouse.unique()):
        fig = plt.figure(figsize=(20,6))
        sns.scatterplot(summary_df.loc[(summary_df.mouse == mouse)], x='session_n', size="site_number", hue='experiment', style='patch_label', sizes=(30, 500), y=variable, 
                        palette=experiments_palette,  alpha=0.7,
                        markers=style_dict_odor_label)

        plt.xlabel('')
        plt.title(f'{mouse}')
        plt.ylim(min_value, max_value)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title='Experiment')
        sns.despine()
        plt.tight_layout()
        plt.show()
        if save:
            fig.savefig(save, format='pdf')
            
def across_sessions_multi_plot(summary_df, variable, condition: str = 'None', save=False):
    """
    Plots data across multiple experimental sessions, visualizing a specified variable for each unique experiment.
    Parameters
    ----------
    summary_df : pandas.DataFrame
        DataFrame containing summary data for plotting. Must include columns: 'experiment', 'within_session_n', 'site_number', 'odor_label', and optionally 'mouse'.
    variable : str
        The name of the column in `summary_df` to plot on the y-axis.
    condition : str, optional
        Condition for plot customization. If set to 'mouse', the plot will be titled with the mouse identifier. Default is 'None'.
    save : bool or str, optional
        If False, the plot is not saved. If a string (filepath), the plot is saved as a PDF to the specified path. Default is False.
    Returns
    -------
    None
        Displays the generated plots and optionally saves them to a file.
    Notes
    -----
    - Uses seaborn for scatter and line plots.
    - Each subplot corresponds to a unique experiment in `summary_df`.
    - The function expects a global variable `color_dict_label` for color mapping.
    """

    fig = plt.figure(figsize=(18,10))
    
    if condition == 'mouse':
        plt.suptitle(f'{summary_df.mouse.iloc[0]}')
        
    for i, experiment in enumerate(summary_df.experiment.unique()):
        ax = plt.subplot(2, 4, i + 1)
            
        sns.scatterplot(summary_df.loc[(summary_df.experiment == experiment)], x='within_session_n', size="site_number", hue='odor_label', sizes=(30, 500), y=variable, palette=color_dict_label, ax=ax, legend=False, alpha=0.7)

        sns.lineplot(x='within_session_n', y=variable, hue='odor_label', palette = color_dict_label,  legend=False,  data=summary_df.loc[(summary_df.experiment == experiment)], marker='', ax=ax)

        plt.title(f'{experiment}')
        sns.despine()
        
    plt.tight_layout()
    if save:
        fig.savefig(save, format='pdf')
    plt.show()

def plot_velocity_across_sessions(cum_velocity, save=False, xlim = [-1, 2]):
    """
    Plots velocity traces across multiple experimental sessions.
    This function generates a multi-panel figure showing the velocity (speed) of subjects over time,
    aligned to the start of inter-patch intervals, for different experiments and sessions. The first subplot
    shows all experiments together, while subsequent subplots show individual experiments with sessions colored.
    Parameters
    ----------
    cum_velocity : pandas.DataFrame
        DataFrame containing velocity data with at least the following columns:
        'times' (time from inter-patch start), 'speed' (velocity), 'experiment' (experiment label),
        'cropped' (boolean for valid data), and 'within_session_n' (session number).
    save : bool, optional
        If True, saves the generated figure using the provided save function. Default is False.
    xlim : list of float, optional
        Limits for the x-axis (time), specified as [xmin, xmax]. Default is [-1, 2].
    Returns
    -------
    None
        The function displays the plot and optionally saves it if `save` is True.
    Notes
    -----
    - Requires matplotlib.pyplot as plt and seaborn as sns to be imported.
    - Assumes `color1` is defined in the global scope for shading.
    - The function uses hardcoded color palettes for up to 8 experiments.
    """
    fig = plt.figure(figsize=(12,22))

    fig.add_subplot(5,2,1)
    sns.lineplot(data=cum_velocity.loc[cum_velocity.cropped==True], x='times', y='speed', hue='experiment',  errorbar=None, legend=True)
    plt.xlim(xlim[0], max(cum_velocity.loc[cum_velocity.cropped==True].times))
    plt.ylim(0, 50)
    plt.fill_betweenx([-5, 50], -1, 0, color=color1, alpha=0.2)
    plt.fill_betweenx([-5, 50],0, 15, color='grey', alpha=0.2)
    plt.xlabel('Time from inter-patch start (s)')

    i=0
    for experiment, colors in zip(cum_velocity.experiment.unique(), ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'Purples', 'Greys', 'inferno']):
        i+=1
        fig.add_subplot(5,2,1+i)
        sns.lineplot(data=cum_velocity.loc[(cum_velocity.cropped==True)&(cum_velocity.experiment==experiment)], x='times', y='speed', 
                    hue='within_session_n', palette=colors, errorbar=None, alpha=0.8)
        plt.xlim(xlim[0], max(cum_velocity.loc[cum_velocity.cropped==True].times))
        plt.ylim(0, 50)
        plt.fill_betweenx([-5, 50], -1, 0, color=color1, alpha=0.2)
        plt.fill_betweenx([-5, 50],0, 15, color='grey', alpha=0.2)
        plt.xlabel('Time from inter-patch start (s)')
        plt.ylabel('Velocity (cm/s)')
        plt.title(experiment)
        plt.legend(borderaxespad=0., title='Session')
        
    plt.tight_layout()
    sns.despine()
    plt.show()
    if save:
        save.savefig(fig)
        
def torque_plots(cum_torque, limits: list = [1500, 2400], save= False):
    """
    Plots torque data aligned to event onset and offset for friction experiments.
    Parameters
    ----------
    cum_torque : pandas.DataFrame
        DataFrame containing torque measurements with columns including 'align', 'times', 'Torque', and 'experiment'.
        The 'align' column should specify alignment type ('onset' or 'offset').
    limits : list, optional
        Y-axis limits for the torque plots, specified as [min, max]. Default is [1500, 2400].
    save : bool, optional
        If True, saves the generated figure using the provided save function. Default is False.
    Notes
    -----
    - The function creates two subplots: one aligned to event onset and one to event offset.
    - Shaded regions are added to indicate different experimental phases.
    - Requires seaborn (sns) and matplotlib.pyplot (plt) to be imported, as well as color1 to be defined in the scope.
    """
    fig = plt.figure(figsize=(12,4))
    fig.add_subplot(121)

    sns.lineplot(data=cum_torque.loc[cum_torque['align'] =='onset'], x='times', y='Torque', hue='experiment', errorbar=None, legend=False, alpha=0.7)
    plt.xlim(-1, 15)
    plt.ylim(limits)
    sns.despine()
    plt.fill_betweenx(limits, -1, 0, color=color1, alpha=0.2)
    plt.fill_betweenx(limits,0, 15, color='grey', alpha=0.2)
    plt.xlabel('Time from inter-patch start (s)')

    fig.add_subplot(122)
    sns.lineplot(data=cum_torque.loc[cum_torque['align'] =='offset'], x='times', y='Torque',  hue='experiment', errorbar=None, alpha=0.7)
    plt.xlim(-5, 2)
    plt.ylim(limits)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.fill_betweenx(limits, -15, 0, color='grey', alpha=0.2)
    plt.fill_betweenx(limits, 0, 2, color=color1, alpha=0.2)
    plt.xlabel('Time from interpatch end (s)')
    sns.despine()
    plt.tight_layout()
    plt.show()
    if save:
        save.savefig(fig)
        
        
def plot_experiments_comparison(ax, summary_df, variable: str = 'reward_probability', 
                                experiments= ['control', 'friction_low', 'friction_med', 'friction_high', 'distance_short', 'distance_long', 'distance_extra_short', 'distance_extra_long']):
        """
        Plots a comparison of experimental conditions using boxplots and swarmplots for a specified variable.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object on which to plot.
        summary_df : pandas.DataFrame
            DataFrame containing experimental data. Must include 'experiment', 'mouse', 'patch_label', and the specified variable columns.
        variable : str, optional
            The variable to plot on the y-axis (default is 'reward_probability').
        experiments : list of str, optional
            List of experiment names to include and order on the x-axis (default is 
            ['control', 'friction_low', 'friction_med', 'friction_high', 'distance_short', 'distance_long', 'distance_extra_short', 'distance_extra_long']).
        Notes
        -----
        - Excludes rows where 'patch_label' is 'Amyl Acetate'.
        - Aggregates data by experiment and mouse, taking the mean of the specified variable.
        - Uses seaborn to plot boxplots and swarmplots for each experiment.
        - Customizes x-tick labels, y-axis limits, and labels based on the variable.
        - Draws a horizontal dashed line at the mean value of the control group.
        - Removes plot spines and applies tight layout for better appearance.
        """
        labels_dict = {'control': 'Control', 'friction_low': 'Friction low', 'friction_med': 'Friction med', 'friction_high': 'Friction high', 'distance_short': 'Dis. short', 'distance_long': 'Dis. long', 'distance_extra_short': 'Ext. short', 'distance_extra_long': 'Ext. long', 'odor_60': 'Odor 60', 'odor_90': 'Odor 90'}

        
        summary_df = summary_df.loc[summary_df.patch_label != 'Amyl Acetate']
        summary_df = summary_df.groupby(['experiment', 'mouse']).agg({variable:'mean'}).reset_index()
        # Adjust the linewidth or dodge parameter based on experiment_width
        sns.boxplot(x='experiment', y=variable,
                    data=summary_df, palette=experiments_palette,
                    showfliers=False, ax=ax,  order= experiments)  # Example adjustment
        sns.swarmplot(x='experiment', y=variable, legend=False, color='black', data=summary_df, dodge=True, ax=ax, order=experiments)
        
        for experiment in experiments:
            labels_plot = [labels_dict[experiment] for experiment in experiments]
            
        ax.set_xticks(np.arange(len(experiments)), labels_plot, rotation = 45, ha='right')
        ax.set_ylabel(variable)
        # Additional plot adjustments
        ax.set_title('All')
        ax.hlines(summary_df.loc[(summary_df.experiment == 'control')][variable].mean(), -1, len(experiments), linestyles='dashed', alpha=0.5, color='grey')
        ax.set_xlabel('')
        if variable == 'reward_probability':
            ax.set_yticks([0, 0.2, 0.4, 0.6])
            ax.set_ylim(0, 0.6)
            ax.set_ylabel('P(reward) when leaving')
        if variable == 'stops':
            ax.set_ylim(0, 20)
            ax.set_ylabel('Stops')
        sns.despine()
        plt.tight_layout()
        

def plot_experiments_comparison_with_odors(ax, summary_df, variable, 
                                           experiments= ['control', 'friction_low', 'friction_med', 'friction_high', 'distance_short', 'distance_long', 'distance_extra_short', 'distance_extra_long'], 
                                           mean_line = 'control'):
    """
    Plots boxplots and swarmplots for a given variable across different experimental conditions and odor patch labels.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object on which to plot.
    summary_df : pandas.DataFrame
        DataFrame containing summary statistics for each trial, including columns 'experiment', 'patch_label', and the variable to plot.
    variable : str
        The name of the column in `summary_df` to plot on the y-axis.
    experiments : list of str, optional
        List of experiment names to include and their order on the x-axis. Defaults to a predefined list of experiment conditions.
    Notes
    -----
    - Uses seaborn to plot boxplots and swarmplots, colored by 'patch_label'.
    - Adjusts x-tick labels, y-axis limits, and labels based on the variable.
    - Draws a horizontal dashed line at the mean value of the control experiment for reference.
    - Assumes `color_dict_label` is defined in the global scope for color mapping.
    - Filters out rows where 'patch_label' is '0'.
    """
    
    labels_dict = {'data_collection': 'Data collection', 'control': 'Control', 'friction_low': 'Friction low', 'friction_med': 'Friction med', 'friction_high': 'Friction high', 'distance_short': 'Dis. short', 'distance_long': 'Dis. long', 'distance_extra_short': 'Ext. short', 'distance_extra_long': 'Ext. long', 'odor_60': 'Odor 60', 'odor_90': 'Odor 90', 'odor_0': 'Odor 0', "rewards_offset": "R_offset", "rewards_rate": "R_rate", "stops_offset": "S_offset", "stops_rate": "S_rate"}
    
    
    summary_df = summary_df.loc[summary_df.patch_label != 'odor_0']

    # Adjust the linewidth or dodge parameter based on experiment_width
    sns.boxplot(x='experiment', y=variable, hue='patch_label',
                palette=color_dict_label, data=summary_df,
                showfliers=False, ax=ax,  order=experiments)  # Example adjustment
    sns.swarmplot(x='experiment', y=variable, hue='patch_label', legend=False, data=summary_df, dodge=True, palette=color_dict_label, ax=ax, order=experiments)
    # Additional plot adjustments
    ax.set_title('All')
    for experiment in experiments:
        labels_plot = [labels_dict[experiment] for experiment in experiments]
        
    ax.set_xticks(np.arange(len(experiments)), labels_plot, rotation = 45, ha='right')
    ax.hlines(summary_df.loc[(summary_df.experiment == mean_line)][variable].mean(), -0.5, len(experiments)-0.5, linestyles='dashed', alpha=0.5, color='grey')
    ax.set_xlabel('')
    if variable == 'reward_probability':
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_ylim(0, 0.8)
        ax.set_ylabel('P(reward) when leaving')
    if variable == 'stops':
        ax.set_ylim(0, 20)
    sns.despine()
    plt.tight_layout()
