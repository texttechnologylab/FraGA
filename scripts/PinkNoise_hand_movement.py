import helper as hdb                    # custom helper module with user-defined functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import nolds                            # nonlinear time series analysis

mpl.rcParams.update({                   # global plot settings (style, font size, colors)
    'font.size':        11,
    'font.weight':      'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'legend.fontsize':  'small',
    'xtick.color':      'black',
    'ytick.color':      'black',})

def positions_to_time_series(pos_list, method='step'):
    """Converts a list of {'x','y','z'} positions into a 1D time series.
    - 'magnitude': Euclidean distance from the origin per frame
    - 'step': Euclidean distance between consecutive frames"""
    coords = np.array([[p['x'], p['y'], p['z']] for p in pos_list])
    if method == 'magnitude':
        return np.linalg.norm(coords, axis=1)
    elif method == 'step':
        diffs = np.diff(coords, axis=0)
        return np.linalg.norm(diffs, axis=1)
    else:
        raise ValueError("method must be 'magnitude' or 'step'")

def compute_alpha(time_series):
    """Compute the DFA (Detrended Fluctuation Analysis) exponent alpha 
    for a given 1D time series using nolds.dfa()."""
    return nolds.dfa(time_series)

def generate_alpha_dataframe(root, sim_root, all_token_timestamps, alpha_threshold=None, processed_ids=None):
    """
    Generate a DataFrame of DFA exponents (alpha) for each subject, hand, 
    and experimental phase based on tracking and timestamp data.
    
    This function processes motion-tracking data (hand positions) from 
    JSON files and computes the DFA exponent alpha using :func:`compute_alpha`. 
    For each subject, results are aggregated across the left and right hand, 
    and the phases "before", "during", and "after". 
    
    Parameters:
    root : str
        Path to the main experiment directory containing per-subject folders with LoopVarIds.
    sim_root : str
        Path to the directory containing "SimulationJsonFiles" (subfolders with tracking JSONs).
    all_token_timestamps : dict
        Dictionary mapping ``player_id`` to a list of timestamps used to 
        extract segments from the tracking data.
    alpha_threshold : float, optional
        If provided, alpha values larger than this threshold will be ignored 
        (e.g., to filter out extreme/unrealistic values). Default is None.
    processed_ids : list, optional
        If provided, this list will be cleared at the start and populated with 
        all ``player_id`` values for which at least one valid alpha mean 
        was successfully computed.
    
    Returns:
    pandas.DataFrame
        A DataFrame with the columns:
        - ``subject`` : str, unique player ID
        - ``hand`` : {"left", "right"}
        - ``phase`` : {"before", "during", "after"}
        - ``alpha`` : float, mean DFA exponent alpha for that condition
    """
    temp_results = []
    if processed_ids is not None:
        processed_ids.clear()

    for sub in os.listdir(root):
        if sub == "SimulationJsonFiles":
            continue
        subpath = os.path.join(root, sub)
        if not os.path.isdir(subpath):
            continue

        player_id = sub.split('_')[3]
        times = all_token_timestamps.get(player_id)
        if times is None:                                   # If there are no timestamps for this playerid --> Continue with next playerid
            continue

        loopvar_ids = hdb.load_loopvarids(os.path.join(subpath, "loopVarIDs.txt"))      # Loading Loopvarid for current playerid
        sim_folder = next((f for f in os.listdir(sim_root) if player_id in f), None)
        if sim_folder is None:
            continue
        sim_path = os.path.join(sim_root, sim_folder)

        left_file = os.path.join(sim_path, "handleft.json")                             # Loading Hand Tracking-data for current playerid
        right_file = os.path.join(sim_path, "handright.json")
        if not os.path.isfile(left_file) or not os.path.isfile(right_file):
            continue

        lefthand = hdb.load_tracking_data(left_file)
        righthand = hdb.load_tracking_data(right_file)
        segments = {'before':'before', 'during':'during', 'after':'after'}

        sub_has_result = False                                                          # Flag: Player has at least one alpha_mean?
        for hand_side, data in [('left',lefthand), ('right',righthand)]:
            for phase, ts_label in segments.items():
                alphas = []
                for ts in times:
                    try:
                        seg = hdb.get_tracking_data_for_timesegment(
                            [ts], loopvar_ids, data, timestamp=ts_label
                        )['botPos']
                    except Exception:
                        continue
                    if len(seg) < 2:
                        continue
                    ts_series = positions_to_time_series(seg, method='step')
                    if len(ts_series) < 3:
                        continue
                    try:
                        alpha = compute_alpha(ts_series)
                    except ValueError:
                        continue
                    if alpha_threshold is not None and alpha > alpha_threshold:
                        continue
                    alphas.append(alpha)
                    temp_results.append({'subject':player_id, 'hand':hand_side,
                                          'phase':phase, 'alpha_ts':alpha})

                mean_alpha = float(np.mean(alphas)) if alphas else np.nan
                temp_results.append({'subject':player_id, 'hand':hand_side,
                                     'phase':phase, 'alpha_mean':mean_alpha})
                if not np.isnan(mean_alpha):
                    sub_has_result = True                                               

        # If this player has a valid average in at least one phase/hand, add them to the list.
        if processed_ids is not None and sub_has_result and player_id not in processed_ids:
            processed_ids.append(player_id)

    df_temp = pd.DataFrame(temp_results)
    df = df_temp.dropna(subset=['alpha_mean'])[['subject','hand','phase','alpha_mean']]
    return df.rename(columns={'alpha_mean':'alpha'})


if __name__ == "__main__":
    # SET PATH to Trackingdata:
    root = r"...\Github\tracking_json_zip"                           # Analysis with 146 Subjects --> 72 processed                ### SET PATH!!! ###
    sim_root = os.path.join(root, "SimulationJsonFiles")
    
    folder_dialog = r"...\Github\dialogue"                           # Combined Transkriptions --> 1 File but 2 Subjects (Chat)   ### SET PATH!!! ###
    folder_json   = r"...\Github\crisper_whisper_json"               # Word Timestamps --> 1 File for each Subject                ### SET PATH!!! ###
    
    all_token_timestamps = hdb.extract_dialogue_timestamps(folder_dialog)                                       # Dialogue-Timestamps --> All sentences
    #all_token_timestamps = hdb.extract_bracketed_timestamps(folder_json, output_path="no", export=False)       # Bracketed Timestamps --> Hesitations e.g. "[UM]"

    #Analysis:
    # Generate DataFrame und process IDs
    alpha_threshold = 10.0                                                                                      # Filtering out extreme alpha values
    processed_ids = []
    df = generate_alpha_dataframe(root, sim_root, all_token_timestamps,alpha_threshold, processed_ids)

    # Output processed Player-IDs
    print("\nVerarbeitete Player-IDs:", processed_ids)
    print("Anzahl: ", len(processed_ids))

    # Pivot-Table
    print("\nMittelwert des DFA-Exponenten (α) pro Subjekt, Hand & Phase:")
    df_pivot = df.pivot_table(index='subject', columns=['hand','phase'], values='alpha')
    print(df_pivot)
    df_pivot.to_excel(r"...\Github\Results\Pinknoise Dialog.xlsx")   # Export: Excel Table with alpha Values for all Phases and both Hands.            ### SET PATH!!! ###

    # Statistics
    phase_stats = df.groupby(['hand','phase'])['alpha'].agg(['mean','median']).unstack(level='phase')
    print("\nMean & Median pro Hand & Phase:")
    print(phase_stats)

    # Scatter-Plots --> used for plot Generation
    df_during = df[df['phase']=='during'].copy()
    subjects = sorted(df['subject'].unique())
    idx_map = {s:i+1 for i,s in enumerate(subjects)}
    df_during['sub_index'] = df_during['subject'].map(idx_map)

    ########################### left Hand Plot: ###############################
    left   = df_during[df_during['hand']=='left']
    l_mean, l_med = left['alpha'].mean(), left['alpha'].median()
    N = len(subjects)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.scatter(left['sub_index'], left['alpha'], s=50, alpha=0.8)
    ax.axhline(l_mean, linestyle='--', label=f'Mean {l_mean:.2f}')
    ax.axhline(l_med,  linestyle=':',  label=f'Median {l_med:.2f}')

    ax.grid(axis='y', linestyle='--', alpha=0.3)                        # style

    step = 4
    ticks = list(range(1, N+1, step))                                   # ticks every 4, but always include 1 and N
    if ticks[0] != 1:
        ticks.insert(0, 1)
    if ticks[-1] != N:
        ticks.append(N)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=0)

    ax.set_xlim(0.5, N + 0.5)                                           # leave a margin of 0.5 on each side so point 1 isn't hidden
    ax.set_xlabel('Subject (Index)', labelpad=12)                       # move x-label 12 points down
    ax.set_ylabel('Alpha-Value',  labelpad=12)
    ax.set_title('Pink Noise – Dialogue – Left Hand')
    ax.legend(fontsize='small', loc='upper right')                      # legend in upper-right
    
    fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.18)   # tighten margins: reduce bottom/left whitespace
    plt.savefig(r"...\Github\Results\PinkNoise_LeftHand.pdf")           # PDF-Format                                                    ### SET PATH!!! ###

    
    ################################ Right Hand Plot: ##################################
    right = df_during[df_during['hand']=='right']
    r_mean, r_med = right['alpha'].mean(), right['alpha'].median()
    N = len(subjects)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.scatter(right['sub_index'], right['alpha'], s=50, alpha=0.8)
    ax.axhline(r_mean, linestyle='--', label=f'Mean {r_mean:.2f}')         # Reference line for the Mean.
    ax.axhline(r_med,  linestyle=':',  label=f'Median {r_med:.2f}')        # Reference line for the Median.

    for spine in ['left','right','top','bottom']:                           #full bounding box around the plot
        ax.spines[spine].set_visible(True)

    ax.grid(axis='y', linestyle='--', alpha=0.3)                            # style

    step = 4
    ticks = list(range(1, N+1, step))                                       # ticks every 4, but always include 1 and N
    if ticks[0] != 1:
        ticks.insert(0, 1)
    if ticks[-1] != N:
        ticks.append(N)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=0)

    ax.set_xlim(0.5, N + 0.5)                                               # leave a margin of 0.5 on each side so point 1 isn't hidden
    ax.set_xlabel('Subject (Index)', labelpad=12)                           # move x-label 12 points down
    ax.set_ylabel('Alpha-Value',  labelpad=12)
    ax.set_title('Pink Noise – Dialogue – Right Hand')
    ax.legend(fontsize='small', loc='upper right')                          # legend in upper-right

    fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.18)       # tighten margins: reduce bottom/left whitespace
    plt.savefig(r"...\Github\Results\Pinknoise_RightHand.pdf")              # PDF-Format                                                ### SET PATH!!! ###

    # tighten margins: reduce bottom/left whitespace
    fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.18)

    plt.savefig(r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\Results\Pinknoise_RightHand.pdf")             #PDF
    #plt.savefig(r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\Results\PinkNoise_RightHand.png")            #PNG
