# %%
import sys
sys.path.append('../src/')

from utils import breathing_signal as lib
from utils import analysis_utils as analysis
from utils import processing

# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator

# Data processing toold
import pandas as pd
import numpy as np

def format_func(value, tick_number):
    return f"{value:.0f}"

sns.set_context('talk')

import warnings
pd.options.mode.chained_assignment = None  # Ignore SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)

def SessionRasterSegmented(reward_sites,config, save=False):
    # Create a figure with a 2x2 grid
    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(3, 2, width_ratios=[1, 1])

    df_skip = pd.DataFrame()
    for index, row in reward_sites.iterrows():
        ax1 = plt.subplot(gs[0:3, 0])
        if row['collected'] == 1 and row['has_choice'] == True:
            color='steelblue'
        elif row['collected'] == 0 and row['has_choice'] == True:
            color='pink'
            if row['reward_available'] == 0:
                color='crimson'
        else:
            if  row['reward_available'] == 0:
                color='black'
            else:
                color='lightgrey'
            
        ax1.barh(int(row['active_patch']), left=row['visit_number'], height=1, width=1, color=color, edgecolor='darkgrey', linewidth=0.5)
        
        ax1.set_xlim(-1,max(reward_sites.visit_number)+1)
        ax1.set_ylabel('Patch number')
        ax1.set_xlabel('Site number')
        
        if row['odor_label'] == reward_sites['odor_label'].unique()[0]:
            patch_color='orange'
        elif row['odor_label'] == reward_sites['odor_label'].unique()[1]:
            patch_color='darkgreen'
        else:
            patch_color='yellow'
        
        ax1.barh(int(row['active_patch']), left = -0.8, height=1, width =0.5, color=patch_color, edgecolor='black', linewidth=0.5)

    odor_list_color = ['orange', 'indigo', 'darkgreen']
    odors = []
    for index, odor in enumerate(reward_sites['odor_label'].unique()):
        odors.append(mpatches.Patch(color=odor_list_color[index], label=(str(odor) + '_' + str(reward_sites.loc[reward_sites.odor_label == odor].reward_delivered.max()))))
    
    label_2 = mpatches.Patch(color='steelblue', label='Harvested')
    label_3 = mpatches.Patch(color='crimson', label='No reward - depleted')
    label_4 = mpatches.Patch(color='lightgrey', label='Skipped - not depleted')
    label_5 = mpatches.Patch(color='black', label='Skipped - depleted')
    label_6 = mpatches.Patch(color='pink', label='No reward - not depleted')
    odors.extend([label_2, label_3,label_4,label_5,label_6])
    ax1.legend(handles=odors, loc='upper left', bbox_to_anchor=(0.8, 1), fontsize=8)
    ax1.set_ylim(-1,max(reward_sites.active_patch)+1)
    ax1.set_title(mouse + '_' + session)
    plt.tight_layout()
    sns.despine()

    if len(reward_sites['odor_label'].unique()) != 1:
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 1])
        ax4 = plt.subplot(gs[2, 1])
        for ax, odor_label in zip([ax2, ax3, ax4], reward_sites.odor_label.unique()):
            selected_sites = reward_sites.loc[reward_sites.odor_label == odor_label]
            previous_active = 0
            value = 0
            for index, row in selected_sites.iterrows():
                # Choose the color of the site
                if row['collected'] == 1 and row['has_choice'] == True:
                    color='steelblue'
                elif row['collected'] == 0 and row['has_choice'] == True:
                    color='pink'
                    if row['reward_available'] == 0:
                        color='crimson'
                else:
                    if  row['reward_available'] == 0:
                        color='black'
                    else:
                        color='lightgrey'
                
                if row['odor_label'] == reward_sites['odor_label'].unique()[0]:
                    patch_color='orange'
                elif row['odor_label'] == reward_sites['odor_label'].unique()[1]:
                    patch_color='darkgreen'
                else:
                    patch_color='black'
                
                ax.set_title(odor_label, color=patch_color)
                        
                if row['active_patch'] != previous_active:
                    value+=1
                    previous_active = row['active_patch']
                ax.barh(value, left=row['visit_number'], height=1, width=1, color=color, edgecolor='darkgrey', linewidth=0.5)
                ax.set_xlim(0,max(reward_sites.visit_number)+1)
        ax4.set_xlabel('Site number')
        
    # Set the maximum number of ticks on the x-axis
    max_ticks = 5  # Replace this with the desired number of ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
    plt.tight_layout()
    sns.despine()
    if save != False:
        save.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def PStayPastNoRewards(reward_sites, config, save=False):
    odor_label_list = reward_sites['odor_label'].unique()
    fig, ax1 = plt.subplots(1, len(odor_label_list), figsize=(4*len(odor_label_list),4))        
    for i, odor_label in enumerate(odor_label_list):
        if len(odor_label_list) != 1:
            ax = ax1[i]
        else:
            ax = ax1
            
        df_results = reward_sites.loc[reward_sites['odor_label'] == odor_label].groupby('past_no_reward_count')['visit_number'].count().reset_index()
        df_results['p(Stay)'] = reward_sites.loc[reward_sites['odor_label'] == odor_label].groupby('past_no_reward_count')['has_choice'].mean()
        df_results = df_results.loc[df_results['visit_number'] >= 3]
        
        sufficient_sites = df_results.visit_number.unique()
        
        ax.set_title(odor_label + '_' + reward_sites.loc[reward_sites['odor_label'] == odor_label]['reward_delivered'].max().astype(str))
        sns.lineplot(x='past_no_reward_count', y='p(Stay)', data=df_results, color = 'k', marker = "o", ax=ax)
        ax.set_xlabel("Previous unrewarded")
        ax.set_ylabel("P(Stay)")
        ax.set_ylim([-0.05, 1.05])
        ax.set_title(odor_label + '_' + reward_sites.loc[reward_sites['odor_label'] == odor_label]['reward_delivered'].max().astype(str))
        
        # ax[i].xaxis.set_major_formatter(FuncFormatter(format_func))    
        # ax[i].xaxis.set_major_locator(FixedLocator(sufficient_sites))
        
        for j in range(len(df_results)):
            if df_results['p(Stay)'].values[j] <0.1:
                ax.text(df_results['past_no_reward_count'].values[j], df_results['p(Stay)'].values[j]+0.1, str(df_results['visit_number'].values[j]), ha='center', size=10, color='red')
            else:
                ax.text(df_results['past_no_reward_count'].values[j], df_results['p(Stay)'].values[j]-0.1, str(df_results['visit_number'].values[j]), ha='center', size=10, color='red')
        
    sns.despine()
    plt.tight_layout()
    if save != False:
        save.savefig(fig)
        plt.close(fig)

def PStayVisitNumber(reward_sites, config, save=False):
    odor_label_list = reward_sites['odor_label'].unique()
    fig, ax1 = plt.subplots(1, len(odor_label_list), figsize=(4*len(odor_label_list),4))        
    for i, odor_label in enumerate(odor_label_list):
        if len(odor_label_list) != 1:
            ax = ax1[i]
        else:
            ax = ax1
            
        df_results = reward_sites.loc[reward_sites['odor_label'] == odor_label].groupby('visit_number')['label'].count().reset_index()
        df_results.rename(columns={'label':'total_trials'}, inplace=True)
        df_results['p(Stay)'] = reward_sites.loc[reward_sites['odor_label'] == odor_label].groupby('visit_number')['has_choice'].mean()
        # df_results = df_results.loc[df_results['total_trials'] >= 3]
        sufficient_sites = df_results.visit_number.unique()
        
        ax.set_title(odor_label + '_' + reward_sites.loc[reward_sites['odor_label'] == odor_label]['reward_delivered'].max().astype(str))
        sns.lineplot(x='visit_number', y='p(Stay)', data=df_results, color = 'k', marker = "o", ax=ax)
        ax.set_xlabel("Visit number")
        ax.set_ylabel("P(Stay)")
        ax.set_ylim([-0.05, 1.05])

        for j in range(len(df_results)):
            if df_results['p(Stay)'].values[j] <0.1:
                ax.text(df_results['visit_number'].values[j], df_results['p(Stay)'].values[j]+0.1, str(df_results['total_trials'].values[j]), ha='center', size=10, color='red')
            else:
                ax.text(df_results['visit_number'].values[j], df_results['p(Stay)'].values[j]-0.1, str(df_results['total_trials'].values[j]), ha='center', size=10, color='red')

        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        ax.xaxis.set_major_locator(FixedLocator(sufficient_sites))
        ax.set_ylim(0,1)
        
    sns.despine(trim=True, offset=4)
    plt.tight_layout()
    if save != False:
        save.savefig(fig)
        plt.close(fig)
        
def length_distributions(active_site: pd.DataFrame, delay: bool=False, save: bool=False):
    
    def larger_value(value1, value2):
        if value1 < value2:
            return value1
        else:
            return value2

    fig,ax = plt.subplots(1,3, figsize=(12,4))
    gap_size = active_site.loc[active_site['label'] == 'Gap'].length.values
    # print('Gap: ', np.round(np.mean(gap_size),3))
    ax[0].hist(gap_size, bins=20, color='black')
    ax[0].vlines(np.mean(gap_size), 0, 50, color='red', linewidth=2)
    ax[0].set_title('Gap')

    intersite_size = active_site.loc[active_site['label'] == 'InterPatch'].length.values
    # print('InterPatch: ', np.round(np.mean(intersite_size),3))
    ax[1].hist(intersite_size, bins=20, color='black')
    ax[1].vlines(np.mean(intersite_size), 0, 50, color='red', linewidth=2)
    ax[1].set_title('InterPatch')

    if delay == True:
        waitReward = data['software_events'].streams.ChoiceFeedback.data['frameTimestamp'].values
        waitLick = data['software_events'].streams.GiveReward.data['frameTimestamp'].values[1:]
        if len(waitLick) == len(waitReward):
            delay = waitLick-waitReward
        else:
            result = larger_value(len(waitLick), len(waitReward))
            delay = waitLick[:result]-waitReward[:result]
            
        # print('Delay: ', np.round(np.mean(delay),3))
        ax[2].hist(delay, bins=20, range=(0,2), color='black')
        ax[2].vlines(np.mean(delay), 0, 50, color='red', linewidth=2)
        ax[2].set_title('Delay')

    sns.despine()
    
    if save != False:
        save.savefig(fig)
        plt.close(fig)