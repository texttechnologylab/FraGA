import helper as hdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind
from typing import Dict, List, Tuple

def compute_path_stats(pos_list: List[dict]) -> dict:
    """
    Compute basic movement statistics from a sequence of 3D positions.

    Parameters:
    pos_list : List[dict]
        A list of dictionaries containing 3D coordinates with keys {'x', 'y', 'z'}.
        Example: [{'x':1,'y':2,'z':3}, {'x':2,'y':4,'z':6}, ...]

    Returns:
        A dictionary with the following keys:
        - 'total_distance' : float
            Total traveled distance across the entire path (sum of step distances).
        - 'n_steps' : int
            Number of steps (i.e., number of consecutive position differences).
        - 'step_distances' : np.ndarray
            Array of Euclidean distances between consecutive positions.
    """
    if not pos_list or len(pos_list) < 2:
        return {'total_distance': 0.0, 'n_steps': 0, 'step_distances': np.array([])}
    coord_array   = np.array([[p['x'], p['y'], p['z']] for p in pos_list])
    coord_diffs   = np.diff(coord_array, axis=0)
    step_distances = np.linalg.norm(coord_diffs, axis=1)
    return {
        'total_distance': step_distances.sum(),
        'n_steps':        len(step_distances),
        'step_distances': step_distances
    }

def analyze_all(loopvar_root: str, all_token_timestamps: Dict[str, List[Tuple[float, float]]],
                player_dict: Dict[str, str], simulation_root: str, include_dom: bool = True) -> pd.DataFrame:
    """
    Analyze hand movement data for all players across different experimental phases.
    This function combines loopVarIds, token timestamps, and
    3D tracking data (left/right hand) to compute path statistics for each player.
    The analysis is performed separately for the phases 'before', 'during', and 'after'.

    Parameters:
    loopvar_root : str
        Root directory containing per-player loopVarID folders.
    all_token_timestamps : Dict[str, List[Tuple[float, float]]]
        Dictionary mapping player_id → list of (start_time, end_time) tuples.
        These define the time segments used for extracting tracking data.
    player_dict : Dict[str, str]
        Dictionary mapping player_id → dominant hand ('left' or 'right').
    simulation_root : str
        Root directory containing simulation folders with JSON tracking data.
        Folder names are expected to contain the player_id as the 4th part.
    include_dom : bool, optional (default=True)
        Whether to include dominant hand information in the output.
        If True, entries without dominant-hand info are skipped.

    Returns:
    pd.DataFrame
        A DataFrame containing per-player statistics with columns:
        - 'player_id'      : str, unique player identifier
        - 'hand'           : str, 'left' or 'right'
        - 'phase'          : str, 'before', 'during', or 'after'
        - 'total_distance' : float, total traveled distance across the phase
        - 'mean_step'      : float, mean step length during the phase
        - 'dominant'       : str, dominant hand (if include_dom=True)

    Notes:
    - Players are skipped if any of the required data (timestamps, loopVarIDs,
      simulation folder, or tracking JSON) is missing.
    - Path statistics are computed using `compute_path_stats`, which measures
      step distances as Euclidean norms in 3D.
    - If `n_steps = 0`, the mean step length is set to 0.0.
    - The function prints a summary of skipped player IDs for debugging.
    """
    # Map simulation folders by player_id:
    sim_map: Dict[str,str] = {}
    for sim_folder in os.listdir(simulation_root):
        simpath = os.path.join(simulation_root, sim_folder)
        if not os.path.isdir(simpath):
            continue
        parts = sim_folder.split('_')
        if len(parts) >= 4:
            pid = parts[3]
            sim_map[pid] = sim_folder

    skipped_ids: List[str] = []
    records: List[dict] = []

    for sub in os.listdir(loopvar_root):
        if sub == os.path.basename(simulation_root):
            continue
        subpath = os.path.join(loopvar_root, sub)
        if not os.path.isdir(subpath):
            continue
        parts = sub.split('_')
        if len(parts) < 4:
            continue
        player_id = parts[3]

        # Dominant Hand
        dom = player_dict.get(player_id)
        if include_dom and dom is None:
            print(f"Warning: No dominant Hand for {player_id}")
            skipped_ids.append(player_id)
            continue
        if not include_dom:
            dom = 'all'

        # Timestamps:
        times = all_token_timestamps.get(player_id)
        if times is None:
            print(f"Warning: No Timestamps for {player_id}")
            skipped_ids.append(player_id)
            continue

        # Load LoopVarIDs: 
        loopvar_path = os.path.join(subpath, "loopVarIDs.txt")
        try:
            loopvar_ids = hdb.load_loopvarids(loopvar_path)
        except FileNotFoundError:
            print(f"Warning: No loopVarIDs for {player_id}")
            skipped_ids.append(player_id)
            continue

        # Check JSON-Folder
        sim_folder = sim_map.get(player_id)
        if sim_folder is None:
            print(f"Warning: No JSON-Folder for {player_id}")
            skipped_ids.append(player_id)
            continue
        simpath = os.path.join(simulation_root, sim_folder)

        # Load Tracking-Data
        try:
            lefthand  = hdb.load_tracking_data(os.path.join(simpath, "handleft.json"))
            righthand = hdb.load_tracking_data(os.path.join(simpath, "handright.json"))
        except FileNotFoundError:
            print(f"Warning: Missing Hand-data for {player_id}")
            skipped_ids.append(player_id)
            continue

        # Analysis per Hand & Phase
        for hand_name, data in [('left', lefthand), ('right', righthand)]:
            phase_acc = {phase: {'distance':0.0, 'steps':0} for phase in ['before','during','after']}
            for phase in phase_acc:
                for ts in times:
                    try:
                        seg = hdb.get_tracking_data_for_timesegment([ts], loopvar_ids, data, timestamp=phase)
                    except IndexError:
                        print(f"Warning: Timesegment out of range for {player_id}, Phase {phase}, Timestamp {ts}")
                        continue
                    seg_data = seg.get('botPos', [])
                    stats    = compute_path_stats(seg_data)
                    phase_acc[phase]['distance'] += stats['total_distance']
                    phase_acc[phase]['steps']    += stats['n_steps']

            for phase, acc in phase_acc.items():
                total_dist = acc['distance']
                n_steps    = acc['steps']
                mean_step  = total_dist / n_steps if n_steps > 0 else 0.0
                rec = {
                    'player_id':      player_id,
                    'hand':           hand_name,
                    'phase':          phase,
                    'total_distance': total_dist,
                    'mean_step':      mean_step
                }
                if include_dom:
                    rec['dominant'] = dom
                records.append(rec)

    unique_skipped = sorted(set(skipped_ids))
    print("Skipped Player-IDs:", unique_skipped)
    print("Amount of skipped IDs: ", len(unique_skipped))

    df = pd.DataFrame(records)
    if not include_dom:
        df.drop(columns=['dominant'], inplace=True, errors='ignore')
    return df

def aggregate(df: pd.DataFrame, include_dom: bool):
    """Aggregates mean step lengths across hand and phase (and optionally dominant hand) 
    by computing group-wise averages and standard errors.
    """
    if include_dom:
        idx = ['dominant','hand','phase']
    else:
        idx = ['hand','phase']
    agg   = df.groupby(idx)['mean_step']
    means = agg.mean().unstack(idx[:-1])
    sems  = agg.sem().unstack(idx[:-1])
    return means, sems

def plot_bars(means, sems, var, ylabel, df_all, mean_step_ref=None, include_dom=True):
    """Create a grouped bar plot with error bars and significance testing for hand movement data.
    
    Parameters:
    means : pandas.DataFrame
        DataFrame containing mean values of the variable of interest, grouped by conditions.
        Columns correspond to grouping factors (hands and optionally dominance).
        Rows correspond to phases (before, during, after).
    sems : pandas.DataFrame
        DataFrame containing the standard errors of the mean (SEM), in the same structure as `means`.
    var : str
        The column name in `df_all` that specifies which variable is being compared (e.g., "mean_step").
    ylabel : str
        Label for the y-axis of the plot.
    df_all : pandas.DataFrame
        Original dataset containing the raw values, including columns:
        - "phase": experiment phase (before/during/after),
        - "hand": "left" or "right",
        - plus the variable indicated by `var`.
    mean_step_ref : dict, optional (default=None)
        Dictionary with reference mean values per hand, e.g. {"left": 0.123, "right": 0.145}.
        If provided, horizontal reference lines will be drawn for comparison.
    include_dom : bool, optional (default=True)
        If True, bars are grouped by dominance + hand.
        If False, bars are grouped by hand only.

    Behavior:
    - Bars are drawn for each phase (before, during, after), grouped either by hand
      or by dominance+hand depending on `include_dom`.
    - Error bars represent SEMs.
    - A paired t-test is performed between left and right hand values for each phase,
      and the result is annotated with either "*" (p < 0.05) or "n.s." (not significant).
    - If `mean_step_ref` is provided, dashed horizontal reference lines are added
      with labels for left and right hands.

    Output:
    - Displays the plot.
    - Saves the figure as a PDF to the path ---> SET PATH!!!
    """
    order = ['before','during','after'] # ensure order (x-axis)
    means = means.reindex(order)
    sems  = sems.reindex(order)
    phases = means.index
    x = np.arange(len(phases))

    # Original-columns
    orig_labels = list(means.columns)
    if include_dom:
        width   = 0.2
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
        display = [str(lbl) for lbl in orig_labels]
    else:
        width   = 0.3
        offsets = np.array([-0.5, 0.5]) * width
        display = ['Left Hand', 'Right Hand']

    fig, ax = plt.subplots()
    for off, orig, disp in zip(offsets, orig_labels, display):
        vals = means[orig]
        errs = sems[orig]
        ax.bar(x + off, vals, width, yerr=errs, capsize=5, label=disp)

    ax.set_xticks(x)
    ax.set_xticklabels([ph.capitalize() for ph in phases])
    ax.set_ylabel(ylabel)
    ax.set_title(f"Mean Distance per Frame - Token")
    ax.legend(loc='upper right')

    for i, phase in enumerate(phases):
        left_vals  = df_all[(df_all.phase==phase)&(df_all.hand=='left')][var]
        right_vals = df_all[(df_all.phase==phase)&(df_all.hand=='right')][var]
        t, p = ttest_rel(left_vals, right_vals, nan_policy='omit')
        x0, x1 = i + offsets[0], i + offsets[-1]
        y0 = means.iloc[i, 0] + sems.iloc[i, 0]
        y1 = means.iloc[i, -1] + sems.iloc[i, -1]
        y  = max(y0, y1) * 1.1
        ax.plot([x0, x1], [y, y], 'k-', lw=1.5)
        sig = 'n.s.' if p>=0.05 else '*'

        ylim = ax.get_ylim()
        y_span = ylim[1] - ylim[0]
        y_text_left  = y + 0.02 * y_span
        y_text_right = y - 0.02 * y_span
        ax.text((x0+x1)/2, y_text_left if include_dom else y_text_left,
                sig, ha='center', va='bottom')

    # Reference-line
    if mean_step_ref:
        ylim = ax.get_ylim()
        y_span = ylim[1] - ylim[0]
        for idx, hand in enumerate(['left','right']):
            ref_val = mean_step_ref.get(hand)
            if ref_val is not None:
                ax.axhline(y=ref_val, linestyle='--', linewidth=1.2, color='black')
                offset = (0.02 if hand=='left' else -0.02) * y_span
                ax.text(len(phases)-0.5, ref_val + offset,
                        f"{hand.capitalize()}: {ref_val:.3f}",
                        va='center', ha='left', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(r"...\Github\Results\Handmovement.pdf")                                                                  ### SET PATH ###
    plt.show()

def cohen_d_paired(x, y):
    """
    Compute Cohen's d effect size for paired samples.
    """
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

def perform_stat_tests(df: pd.DataFrame, var_list: List[str], label: str):
    """
    Perform paired t-tests on hand movement data and compute Cohen's d effect sizes.

    Two types of comparisons are performed for each variable in `var_list`:
    1. Left vs. Right hand for each phase ("before", "during", "after").
    2. Before vs. After phase for each hand separately ("left" and "right").
    """
    for var in var_list:
        print(f"\n=== Paired t-Tests Left vs Right ({var}) – {label} ===")
        for phase in ['before','during','after']:
            # Linke vs. rechte Hand gepaart
            left = df[(df.phase==phase)&(df.hand=='left')].set_index('player_id')[var]
            right= df[(df.phase==phase)&(df.hand=='right')].set_index('player_id')[var]
            common = left.index.intersection(right.index)
            n = len(common)
            if n < 2:
                print(f"{phase.capitalize()}: n={n}, too few for test")
                continue
            l = left.loc[common].values
            r = right.loc[common].values
            t, p = ttest_rel(l, r, nan_policy='omit')
            dfree = n - 1
            d = cohen_d_paired(l, r)
            print(f"{phase.capitalize()}: t({dfree}) = {t:.2f}, p = {p:.4f}, d = {d:.2f}")

        print(f"\n=== Paired t-Test Before vs After ({var}) – {label} ===")
        for hand in ['left','right']:
            b = df[(df.phase=='before')&(df.hand==hand)].set_index('player_id')[var]
            a = df[(df.phase=='after') &(df.hand==hand)].set_index('player_id')[var]
            common2 = b.index.intersection(a.index)
            n2 = len(common2)
            if n2 < 2:
                print(f"{hand.capitalize()} Hand Before vs After: n={n2}, too few for test")
                continue
            bv = b.loc[common2].values
            av = a.loc[common2].values
            t2, p2 = ttest_rel(bv, av, nan_policy='omit')
            df2 = n2 - 1
            d2 = cohen_d_paired(bv, av)
            print(f"{hand.capitalize()} Hand Before vs After: t({df2}) = {t2:.2f}, p = {p2:.4f}, d = {d2:.2f}")


def perform_ref_vs_token_tests(df_token: pd.DataFrame, df_dialogue: pd.DataFrame, var: str = 'mean_step',label: str = ''):
    """
    Perform paired t-tests comparing the 'During' phase of dialogue data (reference) 
    against all three phases ('before', 'during', 'after') of token data, separately for each hand.

    Only subjects present in both DataFrames for the same hand and phase are included in each test.
    """
    print(f"\n=== Paired t-Tests: Dialogue 'During' vs Token Phases – {label} ===")
    for hand in ['left','right']:
        dlg = df_dialogue[(df_dialogue.phase=='during') & (df_dialogue.hand==hand)]
        dlg = dlg.set_index('player_id')[var]
        for phase in ['before','during','after']:
            tok = df_token[(df_token.phase==phase) & (df_token.hand==hand)]
            tok = tok.set_index('player_id')[var]
            common = dlg.index.intersection(tok.index)
            n = len(common)
            if n < 2:
                print(f"{hand.capitalize()}, {phase}: n={n}, too few for test")
                continue
            x = dlg.loc[common].values
            y = tok.loc[common].values
            t, p = ttest_rel(x, y, nan_policy='omit')
            dfree = n - 1
            d = cohen_d_paired(x, y)
            print(f"{hand.capitalize()}, {phase}: t({dfree}) = {t:.2f}, p = {p:.4f}, d = {d:.2f}")

if __name__ == "__main__":
    # Parameter: If True, also differentiate by dominant hand; if False, group only by hand side (left vs. right)
    include_dom = False

    # Load Player-Dict  --> Dict: {ID : Dominant Hand, ...}
    excel_path = r"...\Github\FraGa_allPlayer_playerID.xlsx"                                            ### SET PATH!!! ###
    player_dict = hdb.dominant_hand_dict(excel_path)
    
    folder_dialog   = r"...\Github\dialogue"                                                            ### SET PATH!!! ###
    folder_json     = r"...\Github\crisper_whisper_json"                                                ### SET PATH!!! ###
    
    # Extract Timestamps:
    all_dialogue_ts = hdb.extract_dialogue_timestamps(folder_dialog)                                    # Dialogue-Timestamps --> All sentences
    all_bracketed_ts= hdb.extract_bracketed_timestamps(folder_json, output_path="no", export=False)     # Bracketed Timestamps --> Hesitations e.g. "[UM]"
    #all_aussage_ts  = hdb.extract_speechact_timestamps(folder_dialog, speechact_type="Aussage")        # Timestamps for all Sentences with a "." at the end.
    #all_frage_ts    = hdb.extract_speechact_timestamps(folder_dialog, speechact_type="Frage")          # Timestamps for all Sentences with a "?" at the end.

    # Folder Path --> Trackingdaten (Json) und loopvarids
    loopvar_root    = r"...\Github\tracking_json_zip"                          #LoopVarIDs                        ### SET PATH!!! ###
    simulation_root = r"...\Github\tracking_json_zip\SimulationJsonFiles"      #Trackingdaten JSON                ### SET PATH!!! ###

    # Analysis:
    df_bracketed = analyze_all(loopvar_root, all_bracketed_ts, player_dict, simulation_root, include_dom)
    df_dialogue  = analyze_all(loopvar_root, all_dialogue_ts,  player_dict, simulation_root, include_dom)

    # Aggregation:
    means_br, sems_br       = aggregate(df_bracketed, include_dom)
    means_dlg, sems_dlg     = aggregate(df_dialogue,  include_dom)

    # Reference Values: During-Phase
    mean_step_ref_dialog = (df_dialogue[df_dialogue.phase == 'during'].groupby('hand')['mean_step'].mean().to_dict())
    mean_step_ref_bracketed = (df_bracketed[df_bracketed.phase == 'during'].groupby('hand')['mean_step'].mean().to_dict())

    # Plots:
    plot_bars(means_br, sems_br, 'mean_step', 'Distance', df_bracketed, mean_step_ref_dialog, include_dom)              # Dialog as reference line
    #plot_bars(means_dlg, sems_dlg, 'mean_step', 'Distance', df_dialogue,  mean_step_ref_bracketed, include_dom)        # Hesitations as reference line

    # Statistical Tests:
    print("=== Bracketed ===")
    perform_stat_tests(df_bracketed, ['mean_step'], 'Bracketed')
    #print("=== Dialogue ===")
    #perform_stat_tests(df_dialogue,  ['mean_step'], 'Dialogue')

    # Paired t-Tests: Dialogue 'During' vs. Token Phases
    perform_ref_vs_token_tests(df_bracketed, df_dialogue, var='mean_step', label='Token vs. Dialogue-During')
    
    # Return Means of generated plot:
    print("\n===Mittelwerte===")
    print(means_br)
