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

pdf_path = r'Z:\scratch\vr-foraging\sessions'
base_path = r'Z:\scratch\vr-foraging\data'
data_path = r'../../../data/'

color1='#d95f02'
color2='#1b9e77'
color3='#7570b3'
color4='#e7298a'
odor_list_color = [color1, color2, color3]
color_dict = {0: color1, 1: color2, 2: color3}
color_dict_label = {'Ethyl Butyrate': color1, 'Alpha-pinene': color2, 'Amyl Acetate': color3, 
                    '2-Heptanone' : color2, 'Methyl Acetate': color1, 'Fenchone': color3, '2,3-Butanedione': color4,
                    'Methyl Butyrate': color1}

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

results_path = r'C:\Users\tiffany.ona\OneDrive - Allen Institute\Documents\VR foraging\experiments\batch 4 - manipulating cost of travelling and global statistics\results'

def plot_lines(data: pd.DataFrame, ax, variable = 'total_rewards', condition =  'mouse'):
    for value in data[condition].unique():
        y = data.loc[(data[condition] == value)][variable].values
        x = data.loc[(data[condition] == value)].odor_label.values
        ax.plot(x, y, marker='', linestyle='-', color='black', alpha=0.4, linewidth=1)

def plot_significance(general_df: pd.DataFrame, axes, variable = 'total_rewards'):
        # Perform statistical test and add significance annotations
    group1 = general_df.loc[general_df.odor_label == 'Methyl Butyrate', variable]
    group2 = general_df.loc[general_df.odor_label == 'Alpha-pinene', variable]
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
        y = 0.6
        h=0.05
        
    if p_value < 0.001:
        significance = "***" 
    elif p_value < 0.01:
        significance = "**" 
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"
    
    print(significance)
    axes.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    axes.text((x1 + x2) * 0.5, y + h, significance, ha='center', va='bottom', color=col)

def summary_main_variables(general_df, 
                           experiment, 
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
            
    fig,ax = plt.subplots(2,3, figsize=(9,9))
    if condition == 'session_n':
        plt.suptitle(f'{general_df.mouse.iloc[0]} {experiment}')
    else:
        plt.suptitle(experiment)
        
    general_df = general_df.loc[(general_df.odor_label != 'Amyl Acetate')&(general_df.odor_label != 'Fenchone')]
    general_df = general_df.loc[general_df.experiment == experiment]
    
    axes = ax[0][0]
    variable = 'total_rewards'
    sns.boxplot(x='odor_label', y=variable, hue='odor_label', palette = color_dict_label, data=general_df, order=odor_labels,legend=False, zorder=10, width =0.7, ax=axes)
    plot_lines(general_df, axes, variable, condition)
    plot_significance(general_df, axes, variable)
    
    axes.set_ylabel('Rewards collected')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')
    axes.set_ylim(0,15)

    axes = ax[0][1]
    variable = 'reward_probability'
    sns.boxplot(x='odor_label', y=variable, hue='odor_label', palette = color_dict_label, data=general_df, order=odor_labels, legend=False, zorder=10, width =0.7, ax=axes)

    plot_lines(general_df, axes, variable, condition)
    plot_significance(general_df, axes, variable)

    axes.set_ylabel('p(reward) upon leaving')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')
    axes.set_ylim(0.1,0.8)

    # Stops --------------------------------
    axes = ax[0][2]
    variable = 'stops'
    sns.boxplot(x='odor_label', y=variable, hue='odor_label', palette = color_dict_label, data=general_df, order=odor_labels,legend=False, zorder=10, width =0.7, ax=axes)
    plot_lines(general_df, axes, variable, condition)
    plot_significance(general_df, axes, variable)

    axes.set_ylabel('Stops')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')
    
    # Total failures
    axes = ax[1][0]
    variable = 'total_failures'
    sns.boxplot(x='odor_label', y=variable, hue='odor_label', palette = color_dict_label, data=general_df, order=odor_labels,legend=False, zorder=10, width =0.7, ax=axes)
    plot_lines(general_df, axes, variable, condition)
    plot_significance(general_df, axes, variable)

    axes.set_ylabel('Total failures')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')
    
    # Consecutive failures
    axes = ax[1][1]
    variable = 'consecutive_failures'
    sns.boxplot(x='odor_label', y=variable, hue='odor_label', palette = color_dict_label, data=general_df, order=odor_labels,legend=False, zorder=10, width =0.7, ax=axes)
    plot_lines(general_df, axes, variable, condition)
    plot_significance(general_df, axes, variable)

    axes.set_ylabel('Consecutive failures')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
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
    variable = 'active_patch'
    sns.boxplot(x='odor_label', y=variable, hue='odor_label', palette = color_dict_label, data=general_df, order=['Methyl Butyrate', 'Alpha-pinene'],legend=False, zorder=10, width =0.7, ax=axes)
    plot_lines(general_df, axes, variable, condition)
    plot_significance(general_df, axes, variable)

    axes.set_ylabel('# patches')
    axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
    axes.set_xlabel('')
    
    sns.despine()
    plt.tight_layout()
    if save:
        fig.savefig(save, format='pdf')
    else:
        plt.savefig(results_path+'/summary_mouse.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def across_sessions_one_plot(summary_df, variable, save=False):
    experiments = summary_df['experiment'].unique()
    palette = sns.color_palette("tab10", len(experiments))
    color_dict_experiment = dict(zip(experiments, palette))

    # Create a style dictionary for each odor label
    odor_labels = summary_df['odor_label'].unique()
    styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    style_dict_odor_label = dict(zip(odor_labels, styles))
    
    min_value = summary_df[variable].min()
    max_value = summary_df[variable].max()
    for i, mouse in enumerate(summary_df.mouse.unique()):
        fig = plt.figure(figsize=(20,6))
        sns.scatterplot(summary_df.loc[(summary_df.mouse == mouse)], x='session_n', size="visit_number", hue='experiment', style='odor_label', sizes=(30, 500), y=variable, 
                        palette=color_dict_experiment,  alpha=0.7,
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

    fig = plt.figure(figsize=(18,10))
    
    if condition == 'mouse':
        plt.suptitle(f'{summary_df.mouse.iloc[0]}')
        
    for i, experiment in enumerate(summary_df.experiment.unique()):
        ax = plt.subplot(2, 4, i + 1)
            
        sns.scatterplot(summary_df.loc[(summary_df.experiment == experiment)], x='within_session_n', size="visit_number", hue='odor_label', sizes=(30, 500), y=variable, palette=color_dict_label, ax=ax, legend=False, alpha=0.7)

        sns.lineplot(x='within_session_n', y=variable, hue='odor_label', palette = color_dict_label,  legend=False,  data=summary_df.loc[(summary_df.experiment == experiment)], marker='', ax=ax)

        plt.title(f'{experiment}')
        sns.despine()
        
    plt.tight_layout()
    if save:
        fig.savefig(save, format='pdf')
    plt.show()

def plot_velocity_across_sessions(cum_velocity, save=False, xlim = [-1, 2]):
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