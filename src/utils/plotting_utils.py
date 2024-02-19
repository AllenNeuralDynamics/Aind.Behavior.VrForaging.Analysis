# %%
from math import e
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


def speed_traces_available(trial_summary, mouse, session, config, save=False):
    n_odors = len(trial_summary.odor_label.unique())
    fig, ax = plt.subplots(n_odors,5, figsize=(18, len(trial_summary.odor_label.unique())*4), sharex=True, sharey=True)
    window = (-0.5, 2)
    colors = ['crimson','darkgreen']
    for j in range(len(trial_summary.odor_label.unique())):
        if n_odors == 1:
            ax1 = ax
        else:
            ax1 = ax[j]
        ax1[0].set_ylabel('Velocity (cm/s)')

        for i in [0,1,2,3,4]:
            if n_odors == 1:
                ax[i].set_xlabel('Time after odor onset (s)')
            else:
                ax[len(trial_summary.odor_label.unique())-1][j].set_xlabel('Time after odor onset (s)')

            ax1[i].set_ylim(-10,80)
            ax1[i].set_xlim(window)
            ax1[i].hlines(5, window[0], window[1], color='black', linewidth=1, linestyles=':')
            ax1[i].fill_betweenx(np.arange(-10,80,0.1), -0.5,0, color='#808080', alpha=.5, linewidth=0)

    for i in trial_summary.has_choice.unique():
        i = int(i)
        for j, odor_label in enumerate(trial_summary.odor_label.unique()):
            if n_odors == 1:
                ax1 = ax
            else:
                ax1 = ax[j]
                
            palette_dict = choose_palette(odor_label, trial_summary.loc[(trial_summary.odor_label == odor_label)], config)

            df_results = (trial_summary.loc[(trial_summary.odor_label == odor_label)&(trial_summary.has_choice == i)]
                        .groupby(['reward_available','total_sites','times','amount'])[['speed']].mean().reset_index())
            
            if df_results.empty:
                continue
                        
            sns.lineplot(x='times', y='speed', data=df_results, hue='reward_available',  palette=palette_dict, ci=None,legend=False, ax= ax1[i+2])   
            sns.lineplot(x='times', y='speed', data=df_results, color=colors[i], palette=palette_dict, ci=('sd'), ax= ax1[4])      
            
            for site in df_results.total_sites.unique():
                plot_df = df_results.loc[df_results.total_sites==site]
                sns.lineplot(x='times', y='speed', data=plot_df, color=palette_dict[plot_df['reward_available'].unique()[0]], legend=False, linewidth=0.5, alpha=0.5, ax=ax1[i])  

            ax1[2].set_title(f'Odor {odor_label} ')
            
            if i == 1:
                ax1[i].text(1.2, 75, f'Stopped', fontsize=12)
                ax1[i+2].text(1.2, 75, f'Stopped', fontsize=12)
            else:
                ax1[i].text(1.2, 75, f'Not stopped', fontsize=12)
                ax1[i+2].text(1.2, 75, f'Not stopped', fontsize=12)

    sns.despine()     
    plt.suptitle(mouse +'_' + session)       
    plt.tight_layout()

    if save != False:
        save.savefig(fig)
        plt.close(fig)

def speed_traces_efficient(trial_summary: pd.DataFrame, mouse, session, save=False, odor: str = 'all'):
    ''' Plots the speed traces for each stopping condition '''
    odor='all'
    if odor != 'all':
        reward_sites = reward_sites.loc[reward_sites['odor_label'] == odor]

    window = (-0.5, 2)
    colors = ['crimson', 'darkgreen']
    fig, ax= plt.subplots(3,3, figsize=(12,12), sharex=True, sharey=True)

    for j in [0,1,2]:
        ax[j][0].set_ylabel('Velocity (cm/s)')
        for i in [0,1]:
            ax[j][i].set_ylim(-10,80)
            ax[j][i].set_xlim(window)

            ax[j][i].hlines(5, window[0], window[1], color='black', linewidth=1, linestyles=':')
            ax[j][i].fill_betweenx(np.arange(-10,80,0.1), -2,0, color='#808080', alpha=.5, linewidth=0)

            ax[j][2].fill_betweenx(np.arange(-10,80,0.1), -2,0, color='#808080', alpha=.5, linewidth=0)
            ax[j][2].hlines(5, window[0], window[1], color='black', linewidth=1, linestyles=':')
            ax[len(trial_summary.odor_label.unique())-1][i].set_xlabel('Time after odor onset (s)')

    for collected_label in [0, 1]:
        for i, selected_trials in trial_summary.loc[trial_summary.has_choice == collected_label].groupby('total_sites'):
            ax[0][collected_label].plot('times','speed', data=selected_trials, color='black', alpha=0.3, linewidth=0.5)

        selected_trials = trial_summary.loc[trial_summary.has_choice == collected_label].groupby('times')['speed'].mean().reset_index()
        ax[0][collected_label].plot('times','speed', data=selected_trials, color='black', linewidth=3)
            
        if collected_label == 1:
            ax[0][collected_label].set_title(f' Stopped', color=colors[collected_label])
        else:
            ax[0][collected_label].set_title(f' Not Stopped', color=colors[collected_label])
        
        sns.lineplot(x='times', y='speed', data=trial_summary.loc[(trial_summary.has_choice == collected_label)],  color=colors[collected_label], ax=ax[0][2], errorbar=('sd'), linewidth=2)
        
    # ---------- water collection or not
        plot_df = trial_summary.loc[(trial_summary.collected == collected_label)&(trial_summary.has_choice == 1)]
        for i, selected_trials in plot_df.groupby('total_sites'):
            ax[1][collected_label].plot('times','speed', data=selected_trials, color='black', alpha=0.3, linewidth=0.5)

        selected_trials = plot_df.groupby('times')['speed'].mean().reset_index()
        ax[1][collected_label].plot('times','speed', data=selected_trials, color='black', linewidth=3)
            
        if collected_label == 1:
            ax[1][collected_label].set_title(f' Rewarded stop', color=colors[collected_label])
        else:
            ax[1][collected_label].set_title(f' No reward stop', color=colors[collected_label])
        
        sns.lineplot(x='times', y='speed', data=plot_df,  color=colors[collected_label], ax=ax[1][2], errorbar=('sd'), linewidth=2)
        
    # ------------ water depletion or not
        plot_df = trial_summary.loc[(trial_summary.depleted == collected_label)&(trial_summary.has_choice == 1)&(trial_summary.collected == 0)]
        for i, selected_trials in plot_df.groupby('total_sites'):
            ax[2][collected_label].plot('times','speed', data=selected_trials, color='black', alpha=0.3, linewidth=0.5)

        selected_trials = plot_df.groupby('times')['speed'].mean().reset_index()
        ax[2][collected_label].plot('times','speed', data=selected_trials, color='black', linewidth=3)
            
        if collected_label == 1:
            ax[2][collected_label].set_title(f' No reward - depleted', color=colors[collected_label])
        else:
            ax[2][collected_label].set_title(f' No rward - not depleted', color=colors[collected_label])
        
        sns.lineplot(x='times', y='speed', data=plot_df,  color=colors[collected_label], ax=ax[2][2], errorbar=('sd'), linewidth=2)

    sns.despine()
    plt.suptitle(mouse +'_' + session +'_' + odor)
    plt.tight_layout()
    
    if save != False:
        save.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
def choose_palette(odor_label, data, config):
    if data.amount.iloc[0] == 7:
        if int(data.reward_available.max()) == 25:  
            colors_reward=sns.color_palette("RdYlBu", n_colors=5)
            palette_dict = dict(zip(np.arange(4,32,7), colors_reward))
            palette_dict[0] = '#d73027'
        else:
            colors_reward=['#d73027','#fdae61','#abd9e9','#4575b4']        
            palette_dict = dict(zip(np.arange(0,28,7), colors_reward))
    elif data.amount.iloc[0] == 1:
        if int(data.reward_available.max()) == 3:  
            colors_reward=['#d73027','#fdae61','#abd9e9','#4575b4']
            palette_dict = dict(zip(np.arange(0,4,1), colors_reward))

        elif int(data.reward_available.max()) == 10:  
            colors_reward=sns.color_palette("RdYlBu", n_colors=11)
            palette_dict = dict(zip(np.arange(0,11,1), colors_reward))
            
        elif int(data.reward_available.max()) == 100:
            palette_dict = {100: '#4575b4'}
            
        elif int(data.reward_available.max()) == 0:
            palette_dict = {0: '#d73027'}
            
        else:
            colors_reward=sns.color_palette("RdYlBu", n_colors=int(data.reward_available.max())+1)
            palette_dict = dict(zip(np.arange(0,config['environmentStatistics']['patches'][0]['patchRewardFunction']['initialRewardAmount']+1,1), colors_reward))

    elif data.amount.iloc[0] == 5:
        if data.reward_available.max() == 100:
            colors_reward=['#4575b4']
            palette_dict = {100: '#4575b4'}
        else:
            colors_reward=sns.color_palette("RdYlBu", n_colors=6)
            palette_dict = dict(zip(np.arange(0,30,5), colors_reward))
        
    elif data.amount.iloc[0] == 3:     
        if int(data.reward_available.max()) == 25:  
            colors_reward=sns.color_palette("RdYlBu", n_colors=9)
            palette_dict = dict(zip(np.arange(1,28,3), colors_reward))
            palette_dict[0] = '#d73027'
        else:
            colors_reward=sns.color_palette("RdYlBu", n_colors=9)
            palette_dict = dict(zip(np.arange(0,24,3), colors_reward))
        
    elif data.amount.iloc[0] == 4:  
        if int(data.reward_available.max()) == 25: 
            colors_reward=sns.color_palette("RdYlBu", n_colors=8)
            palette_dict = dict(zip(np.arange(1,29,4), colors_reward))
            palette_dict[0] = '#d73027'
        elif int(data.reward_available.max()) == 50: 
            colors_reward=sns.color_palette("RdYlBu", n_colors=14)
            palette_dict = dict(zip(np.arange(2,54,4), colors_reward))
            palette_dict[0] = '#d73027'

    elif data.amount.iloc[0] == 0:  
        palette_dict = {0: '#d73027'}

    return palette_dict

def velocity_traces_odor_summary(trial_summary, config, mouse, session, window: tuple = (-0.5, 2), max_range: int = 60, mean: bool = False, save=False):
    
    ''' Plots the speed traces for each odor label condition '''
    n_odors = trial_summary.odor_label.unique()
    
    fig, ax1 = plt.subplots(1,len(n_odors), figsize=(len(n_odors)*4, 5), sharex=True, sharey=True)
    colors = ['crimson','darkgreen']
    colors_odors = ['orange', 'yellow', 'darkgreen']
    for j, odor_label in enumerate(n_odors):
        if len(n_odors) != 1:
            ax = ax1[j]
            ax1[0].set_ylabel('Velocity (cm/s)')
        else:
            ax = ax1        
            ax.set_ylabel('Velocity (cm/s)')

        ax.set_xlabel('Time after odor onset (s)')
        ax.set_ylim(-10,max_range)
        ax.set_xlim(window)
        ax.hlines(5, window[0], window[1], color='black', linewidth=1, linestyles='dashed')
        ax.fill_betweenx(np.arange(-10,max_range,0.1), 0, window[1], color=colors_odors[j], alpha=.3, linewidth=0)
        
        df_results = (trial_summary.loc[(trial_summary.odor_label == odor_label)&(trial_summary.visit_number == 0)]
                    .groupby(['reward_available','total_sites','times','amount'])[['speed']].mean().reset_index())
        
        if df_results.empty:
            continue
        
        for site in df_results.total_sites.unique():
            plot_df = df_results.loc[df_results.total_sites==site]
            sns.lineplot(x='times', y='speed', data=plot_df, color='black', legend=False, linewidth=0.5, alpha=0.5, ax=ax)  
        
        if mean:
            sns.lineplot(x='times', y='speed', data=df_results, color='black', ci=('sd'), legend=False, linewidth=2, ax=ax)  

        ax.set_title(f'Odor {odor_label} ')

    sns.despine()     
    plt.suptitle(mouse +'_' + session+'_' + 'First visit speed traces')       
    plt.tight_layout()
    
    if save != False:
        save.savefig(fig)
        plt.close(fig)

def trial_collection(reward_sites: pd.DataFrame, encoder_data: pd.DataFrame, mouse: str, session: str, aligned: str=None, window: tuple=(-0.5, 2)):
    '''
    Crop the snippets of speed traces that are aligned to different epochs
    
    Parameters
    ----------
    reward_sites : pd.DataFrame
        DataFrame containing the reward sites information
    encoder_data : pd.DataFrame
        DataFrame containing the encoder data
    mouse : str
        Mouse name
    session : str
        Session name
    aligned : str
        Column name to align the snippets
    window : tuple
        Time window to crop the snippets
        
    Returns
    -------
    trial_summary : pd.DataFrame
        DataFrame containing the snippets of speed traces aligned to different epochs
        
        '''
    trial_summary = pd.DataFrame()
        
    for start_reward, row in reward_sites.iterrows():
        trial_average = pd.DataFrame()
        if aligned is not None:
            trial = encoder_data.loc[row[aligned] + window[0]: row[aligned] + window[1], 'filtered_velocity']
            trial.index -=  row[aligned]
        else:
            trial = encoder_data.loc[start_reward + window[0]: start_reward + window[1], 'filtered_velocity']
            trial.index -=  start_reward
        trial_average['speed'] = trial.values
        trial_average['times'] = np.around(trial.index,3)
        
        for column in reward_sites.columns:
            trial_average[column] = np.repeat(row[column], len(trial.values))

        trial_summary = pd.concat([trial_summary, trial_average], ignore_index=True)
    trial_summary['mouse'] = mouse
    trial_summary['session'] = session
    return trial_summary

def session_raster_segmented(reward_sites,config, save=False):
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

def pstay_past_no_rewards(reward_sites, config, save=False, summary: bool=False):
    odor_label_list = reward_sites['odor_label'].unique()
    df_results_summary = pd.DataFrame()

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
        
        df_results['odor_label'] = odor_label
        df_results['amount'] = reward_sites.loc[reward_sites['odor_label'] == odor_label]['amount'].unique()[0]
        df_results_summary = pd.concat([df_results, df_results_summary])
        
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
        
        ax.set_xticks(df_results['past_no_reward_count'].values)

    sns.despine()
    plt.tight_layout()
    if save != False:
        save.savefig(fig)
    else:
        plt.show()
    plt.close(fig)
        
    if summary == True:
        return df_results_summary

def pstay_visit_number(reward_sites, config, save=False, summary: bool=False):
    odor_label_list = reward_sites['odor_label'].unique()
    df_results_summary = pd.DataFrame()
    
    fig, ax1 = plt.subplots(1, len(odor_label_list), figsize=(4*len(odor_label_list),4))        
    for i, odor_label in enumerate(odor_label_list):
        if len(odor_label_list) != 1:
            ax = ax1[i]
        else:
            ax = ax1
            
        df_results = reward_sites.loc[reward_sites['odor_label'] == odor_label].groupby('visit_number')['label'].count().reset_index()
        df_results.rename(columns={'label':'total_trials'}, inplace=True)
        df_results['p(Stay)'] = reward_sites.loc[reward_sites['odor_label'] == odor_label].groupby('visit_number')['has_choice'].mean()
        df_results = df_results.loc[df_results['total_trials'] >= 3]
        sufficient_sites = df_results.visit_number.unique()
        
        df_results['odor_label'] = odor_label
        df_results['amount'] = reward_sites.loc[reward_sites['odor_label'] == odor_label]['amount'].unique()[0]
        df_results_summary = pd.concat([df_results, df_results_summary])
        
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
    else:
        plt.show()
    plt.close(fig)
        
    if summary == True:
        return df_results_summary
        
def length_distributions(active_site: pd.DataFrame, delay: bool=False, save =False):
    
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
    else:
        plt.show()
        
    plt.close(fig)