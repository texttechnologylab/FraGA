import helper as hdb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nolds
import matplotlib as mpl

mpl.rcParams.update({
    'font.size':        11,
    'font.weight':      'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'legend.fontsize':  'small',
    'xtick.color':      'black',
    'ytick.color':      'black',
})

def positions_to_time_series(pos_list, method='step'):
    """Wandelt eine Liste von {'x','y','z'} in eine 1D-Zeitreihe um.
    - 'magnitude': euklidischer Abstand zum Ursprung pro Frame
    - 'step': Frame-zu-Frame-Distanzen"""
    coords = np.array([[p['x'], p['y'], p['z']] for p in pos_list])
    if method == 'magnitude':
        return np.linalg.norm(coords, axis=1)
    elif method == 'step':
        diffs = np.diff(coords, axis=0)
        return np.linalg.norm(diffs, axis=1)
    else:
        raise ValueError("method must be 'magnitude' or 'step'")

def compute_alpha(time_series):
    """Berechnet den DFA-Exponent alpha mit nolds.dfa()."""
    return nolds.dfa(time_series)

def generate_alpha_dataframe(root, sim_root, all_token_timestamps, alpha_threshold=None, processed_ids=None):
    """
    Verarbeitet die Tracking- und Timestamp-Daten und gibt ein DataFrame mit dem DFA-Exponent α
    pro Subjekt, Hand und Phase zurück. Optional kann ein Schwellenwert gesetzt werden,
    um Alpha-Werte oberhalb dieses Wertes zu ignorieren.

    Parameters:
    - root: Pfad zum Hauptordner mit Experimentdaten
    - sim_root: Pfad zum Ordner mit SimulationJsonFiles (innerhalb von root)
    - all_token_timestamps: Dict mit Token-Timestamps
    - alpha_threshold: float, optionaler Schwellenwert zum Filtern extremer Alpha-Werte
    - processed_ids: Liste, in die verarbeitete player_ids eingefügt werden

    Returns:
        pandas DataFrame mit Spalten ['subject','hand','phase','alpha']
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
        if times is None:
            continue

        loopvar_ids = hdb.load_loopvarids(os.path.join(subpath, "loopVarIDs.txt"))
        sim_folder = next((f for f in os.listdir(sim_root) if player_id in f), None)
        if sim_folder is None:
            continue
        sim_path = os.path.join(sim_root, sim_folder)

        left_file = os.path.join(sim_path, "handleft.json")
        right_file = os.path.join(sim_path, "handright.json")
        if not os.path.isfile(left_file) or not os.path.isfile(right_file):
            continue

        lefthand = hdb.load_tracking_data(left_file)
        righthand = hdb.load_tracking_data(right_file)
        segments = {'before':'before', 'during':'during', 'after':'after'}

        # Flag, ob dieser Spieler mindestens einen alpha_mean erhalten hat
        sub_has_result = False

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

        # Wenn dieser Spieler in mind. einer Phase/Hand einen validen Mittelwert hat, zur Liste hinzufügen
        if processed_ids is not None and sub_has_result and player_id not in processed_ids:
            processed_ids.append(player_id)

    df_temp = pd.DataFrame(temp_results)
    df = df_temp.dropna(subset=['alpha_mean'])[['subject','hand','phase','alpha_mean']]
    return df.rename(columns={'alpha_mean':'alpha'})



if __name__ == "__main__":
    # Pfade setzen
    root = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\ExperimentExport59"                          #Analyse mit 59 Probanden  --> 59 kommen durch
    #root = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\ExperimentExport146"                        #Analyse mit 146 Probanden --> 72 kommen durch
    sim_root = os.path.join(root, "SimulationJsonFiles")
    
    folder_dialog = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\dialogue"                           #Zusammengefügte Transkriptionen --> 1 Datei aber 2 Probanden 
    folder_json   = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\crisper_whisper_json"               #Timestamps auf Wortebene -->für jeden Probanden eine Datei
    
    all_token_timestamps = hdb.extract_dialogue_timestamps(folder_dialog)                                       ###Dialog-Timestamps --> All sentences
    #all_token_timestamps = hdb.extract_bracketed_timestamps(folder_json, output_path="no", export=False)       ###Bracketed Timestamps --> "[UM]"

    #Analysis:
    # Generiere DataFrame und verarbeite IDs
    alpha_threshold = 10.0                                                                                      #Extreme Alpha-Werte Filtern
    processed_ids = []
    df = generate_alpha_dataframe(root, sim_root, all_token_timestamps,alpha_threshold, processed_ids)

    # Ausgabe der verarbeiteten Player-IDs
    print("\nVerarbeitete Player-IDs:", processed_ids)
    print("Anzahl: ", len(processed_ids))

    # Pivot-Tabelle
    print("\nMittelwert des DFA-Exponenten (α) pro Subjekt, Hand & Phase:")
    df_pivot = df.pivot_table(index='subject', columns=['hand','phase'], values='alpha')
    print(df_pivot)
    df_pivot.to_excel(r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\Results\Pinknoise Dialog.xlsx")   #Excel Table with alpha Values for all Phases and both Hands

    # Statistics
    phase_stats = df.groupby(['hand','phase'])['alpha'].agg(['mean','median']).unstack(level='phase')
    print("\nMean & Median pro Hand & Phase:")
    print(phase_stats)

    overall_stats = df.groupby('hand')['alpha'].agg(overall_mean='mean', overall_median='median')
    print("\nOverall Mean & Median pro Hand:")
    print(overall_stats)

    # Scatter-Plots
    df_during = df[df['phase']=='during'].copy()
    subjects = sorted(df['subject'].unique())
    idx_map = {s:i+1 for i,s in enumerate(subjects)}
    df_during['sub_index'] = df_during['subject'].map(idx_map)

    
    ########################### left Hand Plot: ###############################
    left   = df_during[df_during['hand']=='left']
    l_mean = left['alpha'].mean()
    l_med  = left['alpha'].median()
    N = len(subjects)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.scatter(left['sub_index'], left['alpha'], s=50, alpha=0.8)
    ax.axhline(l_mean, linestyle='--', label=f'Mean {l_mean:.2f}')
    ax.axhline(l_med,  linestyle=':',  label=f'Median {l_med:.2f}')

    # style
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # ticks every 5, but always include 1 and N
    step = 4
    ticks = list(range(1, N+1, step))
    if ticks[0] != 1:
        ticks.insert(0, 1)
    if ticks[-1] != N:
        ticks.append(N)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=0)

    # leave a margin of 0.5 on each side so point 1 isn't hidden
    ax.set_xlim(0.5, N + 0.5)
    ax.set_xlabel('Subject (Index)', labelpad=12)   # move x-label 12 points down
    ax.set_ylabel('Alpha-Value',  labelpad=12)
    ax.set_title('Pink Noise – Dialogue – Left Hand')
    
    # legend in upper-right
    ax.legend(fontsize='small', loc='upper right')
    fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.18)
    
    plt.savefig(r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\Results\PinkNoise_LeftHand.pdf")      #PDF
    #plt.savefig(r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\Results\PinkNoise_LeftHand.png")     #PNG

    
    ################################ Right Hand Plot: ##################################
    right = df_during[df_during['hand']=='right']
    r_mean, r_med = right['alpha'].mean(), right['alpha'].median()
    N = len(subjects)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.scatter(right['sub_index'], right['alpha'], s=50, alpha=0.8)

    ax.axhline(r_mean, linestyle='--', label=f'Mean {r_mean:.2f}')
    ax.axhline(r_med,  linestyle=':',  label=f'Median {r_med:.2f}')

    # full box
    for spine in ['left','right','top','bottom']:
        ax.spines[spine].set_visible(True)

    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # ticks every 5, but always include 1 and N
    step = 4
    ticks = list(range(1, N+1, step))
    if ticks[0] != 1:
        ticks.insert(0, 1)
    if ticks[-1] != N:
        ticks.append(N)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=0)

    # leave a margin of 0.5 on each side so point 1 isn't hidden
    ax.set_xlim(0.5, N + 0.5)

    ax.set_xlabel('Subject (Index)', labelpad=12)   # move x-label 12 points down
    ax.set_ylabel('Alpha-Value',  labelpad=12)
    ax.set_title('Pink Noise – Dialogue – Right Hand')
    
    # legend in upper-right
    ax.legend(fontsize='small', loc='upper right')

    # tighten margins: reduce bottom/left whitespace
    fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.18)

    plt.savefig(r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\Results\Pinknoise_RightHand.pdf")             #PDF
    #plt.savefig(r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\Results\PinkNoise_RightHand.png")            #PNG