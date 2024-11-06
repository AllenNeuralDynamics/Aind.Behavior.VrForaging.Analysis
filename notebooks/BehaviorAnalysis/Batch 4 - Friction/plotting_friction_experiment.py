from matplotlib import patches as mpatches
import sys
sys.path.append('../../../src/')

import os

from aind_vr_foraging_analysis.utils import parse, processing, plotting_utils as plotting, AddExtraColumns

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
from scipy.optimize import curve_fit
import math 
from os import PathLike
from pathlib import Path
from scipy.stats import pearsonr, ttest_rel

sns.set_context('talk')

import warnings
pd.options.mode.chained_assignment = None  # Ignore SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

dict_odor = {}
rate = -0.12
offset = 0.6
dict_odor['Methyl Butyrate'] = {'rate':rate, 'offset':0.9, 'color': '#d95f02'}
dict_odor['Alpha-pinene'] = {'rate':rate, 'offset':offset, 'color': '#1b9e77'}
dict_odor['Amyl Acetate'] = {'rate':rate, 'offset':offset, 'color': '#7570b3'}

# Define exponential function
def exponential_func(x, a, b):
    return a * np.exp(b * x)

def format_func(value, tick_number):
    return f"{value:.0f}"

results_path = r'C:\Users\tiffany.ona\OneDrive - Allen Institute\Documents\VR foraging\experiments\batch 4 - manipulating cost of travelling and global statistics\results'

class SingleMouseResults(mouse: str,
                        summary_df: pd.Dataframe(), 
                        velocity_df: pd.Dataframe(),
                        torque_df: pd.Dataframe(),
                         ):
    
    self.summary_df = summary_df
    self.velocity_df = velocity_df
    self.torque_df = torque_df
    self.mouse
    
    groups = ['session','mouse','active_patch','odor_label','experiment']
    summary = summary_df.loc[(summary_df.visit_number != 0)&(summary_df.has_choice ==True)].groupby(groups).agg({'reward_delivered':'sum','visit_number':'count'}).reset_index()
    summary = summary.loc[summary.mouse == mouse]
    summary = summary.groupby(['session','odor_label','experiment']).agg({'reward_delivered':'mean','visit_number':'mean'})
    summary = summary.loc[(summary.odor_label != 'Amyl Acetate')&(summary.odor_label != 'Fenchone')]
    summary.reset_index(inplace=True)
        
    def run_summary(self):
        with PdfPages(os.path.join(results_path, f'{self.mouse_id}_batch4_experiments.pdf')) as pdf:
            self.plot_summary(pdf)
            self.plot_velocity(pdf)
            self.plot_torque(pdf)
    
    def general_parameter_summary(self, pdf):
        summary_df = self.summary_df
        mouse = self.mouse
        
        fig,ax = plt.subplots(1,3, figsize=(9,4.5))

        axes = ax[0]
        sns.boxplot(x='odor_label', y='reward_delivered', hue='odor_label', palette = color_dict_label, data=summary.loc[summary['experiment']==experiment], order=['Methyl Butyrate', 'Alpha-pinene'],legend=False, zorder=10, width =0.7, ax=axes, fliersize=0)

        for session in summary.session.unique():
            y = summary.loc[(summary.session == session)&(summary['experiment']==experiment)].reward_delivered.values
            x = summary.loc[(summary.session == session)&(summary['experiment']==experiment)].odor_label.values
            axes.plot(x, y, marker='', linestyle='-', color='black', alpha=0.4, linewidth=1)

        axes.set_ylabel('Rewards collected')
        axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
        axes.set_xlabel('')
        axes.set_ylim(0,15)

        summary = summary_df.loc[(summary_df.has_choice ==True)].groupby(['session','mouse','active_patch','odor_label', 'experiment']).agg({'collected':'sum','visit_number':'count', 'reward_probability':'min'}).reset_index()
        summary = summary.loc[(summary.visit_number > 1)]
        summary = summary.loc[summary.mouse == mouse]

        summary = summary.groupby(['session','mouse','odor_label','experiment']).agg({'collected':'mean','reward_probability':'median', 'active_patch': 'nunique'}).reset_index()
        summary = summary.loc[(summary.odor_label != 'Amyl Acetate')&(summary.odor_label != 'Fenchone')]

        axes = ax[1]
        sns.boxplot(x='odor_label', y='reward_probability', hue='odor_label', palette = color_dict_label, data=summary.loc[summary['experiment']==experiment], order=['Methyl Butyrate', 'Alpha-pinene'],legend=False, zorder=10, width =0.7, ax=axes, fliersize=0)

        for session in summary.session.unique():
            y = summary.loc[(summary.session == session)&(summary['experiment']==experiment)].reward_probability.values
            x = summary.loc[(summary.session == session)&(summary['experiment']==experiment)].odor_label.values
            axes.plot(x, y, marker='', linestyle='-', color='black', alpha=0.4, linewidth=1)

        axes.set_ylabel('p(reward) when leaving')
        axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
        axes.set_xlabel('')
        axes.set_ylim(0.2,1)

        # Stops --------------------------------
        summary = summary_df.loc[(summary_df.visit_number != 0)&(summary_df.has_choice ==True)].groupby(['session','mouse','active_patch','odor_label','experiment']).agg({'reward_delivered':'sum','visit_number':'count'})
        summary = summary.groupby(['session','mouse','odor_label','experiment']).agg({'visit_number':'mean'})
        
        summary.reset_index(inplace=True)
        summary = summary.loc[summary.mouse == mouse]
        summary = summary.loc[(summary.odor_label != 'Amyl Acetate')&(summary.odor_label != 'Fenchone')]

        axes = ax[2]
        sns.boxplot(x='odor_label', y='visit_number', hue='odor_label', palette = color_dict_label, data=summary.loc[summary['experiment']==experiment], order=['Methyl Butyrate', 'Alpha-pinene'],legend=False, zorder=10, width =0.7, ax=axes, fliersize=0)

        for session in summary.session.unique():
            y = summary.loc[(summary.session == session)&(summary['experiment']==experiment)].visit_number.values
            x = summary.loc[(summary.session == session)&(summary['experiment']==experiment)].odor_label.values
            axes.plot(x, y, marker='', linestyle='-', color='black', alpha=0.4, linewidth=1)

        axes.set_ylabel('Stops')
        axes.set_xticks([0,1], ['Odor 1', 'Odor 2'])
        axes.set_xlabel('')

        sns.despine()
        plt.suptitle(mouse)
        plt.tight_layout()
        
