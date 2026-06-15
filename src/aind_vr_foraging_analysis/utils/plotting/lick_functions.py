
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
import matplotlib.patches as mpatches

def compute_lick_rate(
    df,
    group_var,
    group_labels=None,
    time_col='times',
    bin_size=0.05,
    window=5,
    t_start=None,
    t_end=None,
    return_mice=False,
    return_sessions=False,
):
    """
    Compute smoothed lick rate (licks/sec) per group using uniform filter,
    averaged hierarchically: trial → session → mouse.

    Parameters
    ----------
    return_mice     : also store per-mouse traces in results
    return_sessions : also store per-session traces in results
    """
    from scipy.ndimage import uniform_filter1d

    if group_labels is None:
        group_labels = sorted(df[group_var].unique())

    t_start = t_start or df[time_col].min()
    t_end   = t_end   or df[time_col].max()
    bins    = np.arange(t_start, t_end + bin_size, bin_size)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    results = {}
    for label in group_labels:
        group_df = df[df[group_var] == label]

        mouse_traces       = []
        all_session_traces = []

        for mouse, mdf in group_df.groupby('mouse'):

            session_traces = []
            for session, sdf in mdf.groupby('session'):

                trial_traces = []
                for trial, tdf in sdf.groupby('odor_sites'):
                    counts, _ = np.histogram(tdf[time_col], bins=bins)
                    rate = uniform_filter1d(counts.astype(float), size=window)
                    rate = rate / bin_size
                    trial_traces.append(rate)

                session_mean = np.mean(trial_traces, axis=0)
                session_traces.append(session_mean)
                all_session_traces.append(session_mean)

            mouse_traces.append(np.mean(session_traces, axis=0))

        mouse_traces       = np.array(mouse_traces)
        all_session_traces = np.array(all_session_traces)

        results[label] = {
            'mean':        mouse_traces.mean(axis=0),
            'sem':         mouse_traces.std(axis=0) / np.sqrt(len(mouse_traces)),
            'bin_centers': bin_centers,
        }
        if return_mice:
            results[label]['mouse_traces']   = mouse_traces
        if return_sessions:
            results[label]['session_traces'] = all_session_traces

    return results


from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

def plot_lick_rate(
    results,
    group_labels=None,
    palette=None,
    ax=None,
    plot_type='mean_sem',
    level='mouse',
    stats=False,
    stats_alpha=0.05,
    correction='fdr_bh',  # 'fdr_bh', 'bonferroni', etc.
    sig_y=None,           # y position of significance markers (None = auto)
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    if group_labels is None:
        group_labels = sorted(results.keys())
        
        
    if len(group_labels) == 1:
        # no grouping → all black regardless of palette
        colors = ["black"] * len(group_labels)
    elif palette is None:
        # grouping but no palette → default Set1
        colors = sns.color_palette('Set1', len(group_labels))
    elif isinstance(palette, str):
        colors = sns.color_palette(palette, len(group_labels))
    elif isinstance(palette, dict):
        colors = [palette[label] for label in group_labels]
    else:
        colors = palette

    trace_key = 'mouse_traces' if level == 'mouse' else 'session_traces'

    for label, color in zip(group_labels, colors):
        r = results[label]

        traces = r.get(trace_key)
        if traces is None:
            raise ValueError(
                f"'{trace_key}' not found in results — "
                f"pass return_mice=True or return_sessions=True in compute_lick_rate"
            )

        mean = traces.mean(axis=0)
        sem  = traces.std(axis=0) / np.sqrt(len(traces))

        if plot_type in ('individual', 'mean_individual'):
            for trace in traces:
                ax.plot(r['bin_centers'], trace,
                        color=color, alpha=0.15, linewidth=0.8)

        if plot_type in ('mean_sem', 'mean_individual'):
            ax.plot(r['bin_centers'], mean,
                    color=color, label=label, linewidth=1.5)
            ax.fill_between(r['bin_centers'], mean - sem, mean + sem,
                            color=color, alpha=0.2, linewidth=0)

    # --- Significance bar ---
    if stats:

        t1 = results[group_labels[0]].get(trace_key)
        t2 = results[group_labels[-1]].get(trace_key)

        if t1 is None or t2 is None:
            raise ValueError(
                f"'{trace_key}' not found — pass return_mice=True or return_sessions=True"
            )

        # paired t-test requires same n — trim to smallest
        n = min(len(t1), len(t2))
        t1, t2 = t1[:n], t2[:n]

        # bin-by-bin paired t-test
        if len(t1) != len(t2):
            from scipy.stats import ttest_ind
            p_vals = np.array([
                ttest_ind(t1[:, b], t2[:, b]).pvalue
                for b in range(t1.shape[1])
            ])
        else:
            p_vals = np.array([
                ttest_rel(t1[:, b], t2[:, b]).pvalue
                for b in range(t1.shape[1])
            ])

        # multiple comparisons correction
        reject, _, _, _ = multipletests(p_vals, alpha=stats_alpha, method=correction)

        # auto y position just above the highest trace + SEM
        bin_centers = results[group_labels[0]]['bin_centers']
        y_max = max(
            results[l]['mean'].max() + results[l][trace_key][:n].std(axis=0).max()
            for l in group_labels
        )
        y_sig = sig_y if sig_y is not None else y_max * 1.08

        # draw a solid bar over consecutive significant bins
        sig_x = bin_centers[reject]
        if len(sig_x) > 0:
            # group consecutive bins into spans for a cleaner bar
            bin_size = np.diff(bin_centers).mean()
            breaks   = np.where(np.diff(sig_x) > bin_size * 1.5)[0] + 1
            spans    = np.split(sig_x, breaks)
            for span in spans:
                ax.plot([span[0], span[-1]], [y_sig, y_sig],
                        color='black', linewidth=2.5, solid_capstyle='butt')
            ax.text(bin_centers[-1], y_sig, f' p<{stats_alpha}\n ({correction})',
                    va='center', ha='left', fontsize=7, color='black')

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Time from reward (s)')
    ax.set_ylabel('Lick rate (licks/s)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize =8)
    sns.despine(ax=ax)
    

def plot_lick_count_by_condition(
    plot,
    group_var: str = 'patch_label',
    group_labels: list = None,
    palette='Set1',
    condition: str = 'session',
    save=False,
    ax=None,
    window: tuple = None,
    show_legend=True,
):
    if group_labels is None:
        group_labels = sorted(plot[group_var].unique())

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.25, 4.2))

    plot = plot.copy()
    if window is not None:
        window_length = window[1] - window[0]
        plot['lick_count'] = plot['lick_count'] / window_length
        y_label = 'Lick rate (licks/s)'
    else:
        y_label = 'Lick count'

    variable = 'lick_count'

    # --- Boxplot ---
    sns.boxplot(
        x=group_var, y=variable,
        data=plot, order=group_labels, zorder=10,
        width=0.5, ax=ax, fliersize=0,
        linewidth=0,
        palette=palette,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1),
        capprops=dict(color='k', linewidth=1),
        flierprops=dict(marker=''),
    )

    # --- Guard against empty/invalid data ---
    y_top = plot[variable].max()
    if not np.isfinite(y_top) or y_top == 0:
        ax.set_ylabel(y_label)
        ax.set_title(ax.get_title() + '\n(no data)')
        return

    # --- Connecting lines per condition ---
    x_map = {label: i for i, label in enumerate(group_labels)}
    for _, sdf in plot.groupby(condition):
        pts = sdf.set_index(group_var)[variable].reindex(group_labels).dropna()
        if len(pts) >= 2:
            ax.plot(
                [x_map[l] for l in pts.index], pts.values,
                color='black', alpha=0.4, linewidth=1, marker='',
            )

    # --- Mean line ---
    group_means = (
        plot.groupby(group_var)[variable]
        .mean()
        .reindex(group_labels)
        .dropna()
    )
    ax.plot(
        [x_map[l] for l in group_means.index],
        group_means.values,
        color='black', linewidth=2.5, marker='o',
        markersize=5, zorder=20,
    )
    # --- Significance ---
    g1 = plot.loc[plot[group_var] == group_labels[0], variable].dropna()
    g2 = plot.loc[plot[group_var] == group_labels[-1], variable].dropna()
    try:
        _, p = ttest_rel(g1, g2, nan_policy='omit')
    except:
        _, p = ttest_ind(g1, g2, nan_policy='omit')

    if p < 0.001:  sig = '***'
    elif p < 0.01: sig = '**'
    elif p < 0.05: sig = '*'
    else:          sig = 'ns'
    print(f'{group_labels[0]} vs {group_labels[-1]}: p={p:.3g} ({sig})')
    y_bar, h = y_top * 1.05, y_top * 0.025
    x_left  = 0
    x_right = len(group_labels) - 1
    ax.plot([x_left, x_left, x_right, x_right],
            [y_bar, y_bar + h, y_bar + h, y_bar], lw=1.2, color='black')
    ax.text((x_left + x_right) / 2, y_bar + h, sig,
            ha='center', va='bottom', fontsize=10)

    # --- Formatting ---
    ax.set_ylabel(y_label)
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_xticklabels([])
    sns.despine(ax=ax, bottom=True)
    
# --- Legend ---
    if show_legend:
        if isinstance(palette, dict):
            colors = [palette[label] for label in group_labels]
        else:
            colors = sns.color_palette(palette, n_colors=len(group_labels))
        handles = [mpatches.Patch(color=colors[i], label=group_labels[i])
                   for i in range(len(group_labels))]
        ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    if ax.get_figure() is not None and save:
        ax.get_figure().savefig(save, format='pdf')
    elif not save and ax is None:
        plt.show()
        plt.close()

def apply_filters(df, filters):
    """
    filters: dict where values can be:
        - a scalar         → equality (col == val)
        - a tuple (op, val) → comparison, e.g. ('>', 0.3), ('>=', 0.3), ('<', 5)
    
    Example:
        filters = {
            'is_choice': 1,
            'is_reward': 1,
            'reward_probability': ('>=', 0.3),
            'site_number': ('<', 3),
        }
    """
    ops = {
        '==': lambda col, val: col == val,
        '!=': lambda col, val: col != val,
        '>':  lambda col, val: col >  val,
        '>=': lambda col, val: col >= val,
        '<':  lambda col, val: col <  val,
        '<=': lambda col, val: col <= val,
    }

    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if isinstance(val, tuple):
            op, threshold = val
            mask &= ops[op](df[col], threshold)
        else:
            mask &= (df[col] == val)
    return df.loc[mask]


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate_window(df, filters, variable):
    """Aggregate lick counts within a specified time window and conditions."""
    
    per_trial = (
        df.loc[
          filters
        ]
        .groupby(['mouse', 'session', 'odor_sites', variable])['Channel0']
        .sum()
        .reset_index(name='lick_count')
    )
    per_session = (
        per_trial.groupby(['mouse', 'session', variable])['lick_count']
        .mean()
        .reset_index()
    )
    per_mouse = (
        per_session.groupby(['mouse', variable])['lick_count']
        .mean()
        .reset_index()
    )
    return per_trial, per_mouse, per_session

def collect_lick_trials(odor_sites, lick_onsets, aligned):
    """
    For each trial (row in odor_sites), collect lick timestamps relative to
    `aligned` event, keeping only licks that fall within [start_time, stop_time].

    Parameters
    ----------
    odor_sites   : DataFrame with columns `aligned`, 'start_time', 'stop_time',
                   and any metadata columns you want to carry forward.
    lick_onsets  : array-like of lick timestamps (seconds, absolute time).
    aligned      : str — column name of the alignment event (e.g. 'choice_cue_time')

    Returns
    -------
    DataFrame, one row per lick, with columns:
        ['trial_idx', 'rel_time', + metadata cols]
    """

    licks = np.asarray(lick_onsets).ravel()

    meta_cols = ['session', 'patch_number', 'site_number', 'patch_label', 'consecutive_failures', 'cumulative_rewards',
                 'is_reward', 'previous_outcome', 'is_choice', 'reward_probability', 'consecutive_rewards',
                 'trials_to_leave', 'total_sites', 'odor_sites']
    
    meta_cols = [c for c in meta_cols if c in odor_sites.columns]

    rows = []
    for trial_idx, trial in odor_sites.iterrows():
        t0 = trial[aligned]
        if pd.isna(t0):
            continue

        # Keep only licks within the epoch window
        mask = (licks >= trial['start_time']) & (licks <= trial['stop_time'])
        trial_licks = licks[mask]

        # Express each lick relative to the alignment event
        rel_licks = trial_licks - t0

        meta = {c: trial[c] for c in meta_cols}
        for rl in rel_licks:
            rows.append({'trial': trial_idx, 'times': rl, **meta})

    return pd.DataFrame(rows)