import helper as hdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind
from typing import Dict, List, Tuple

def compute_path_stats(pos_list: List[dict]) -> dict:
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

def analyze_all(loopvar_root: str,
                all_token_timestamps: Dict[str, List[Tuple[float, float]]],
                player_dict: Dict[str, str],
                simulation_root: str,
                include_dom: bool = True) -> pd.DataFrame:
    # Map simulation folders by player_id
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

        # Dominante Hand
        dom = player_dict.get(player_id)
        if include_dom and dom is None:
            print(f"Warnung: Keine dominante Hand für {player_id}")
            skipped_ids.append(player_id)
            continue
        if not include_dom:
            dom = 'all'

        # Zeitstempel
        times = all_token_timestamps.get(player_id)
        if times is None:
            print(f"Warnung: keine Zeitstempel für {player_id}")
            skipped_ids.append(player_id)
            continue

        # LoopVarIDs laden
        loopvar_path = os.path.join(subpath, "loopVarIDs.txt")
        try:
            loopvar_ids = hdb.load_loopvarids(loopvar_path)
        except FileNotFoundError:
            print(f"Warnung: Keine loopVarIDs für {player_id}")
            skipped_ids.append(player_id)
            continue

        # JSON-Ordner prüfen
        sim_folder = sim_map.get(player_id)
        if sim_folder is None:
            print(f"Warnung: Keine JSON-Ordner für {player_id}")
            skipped_ids.append(player_id)
            continue
        simpath = os.path.join(simulation_root, sim_folder)

        # Tracking-Daten laden
        try:
            lefthand  = hdb.load_tracking_data(os.path.join(simpath, "handleft.json"))
            righthand = hdb.load_tracking_data(os.path.join(simpath, "handright.json"))
        except FileNotFoundError:
            print(f"Warnung: Fehlende Handdaten für {player_id}")
            skipped_ids.append(player_id)
            continue

        # Analyse pro Hand & Phase
        for hand_name, data in [('left', lefthand), ('right', righthand)]:
            phase_acc = {phase: {'distance':0.0, 'steps':0} for phase in ['before','during','after']}
            for phase in phase_acc:
                for ts in times:
                    try:
                        seg = hdb.get_tracking_data_for_timesegment([ts], loopvar_ids, data, timestamp=phase)
                    except IndexError:
                        print(f"Warnung: Zeitsegment außerhalb für {player_id}, Phase {phase}, Timestamp {ts}")
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
    print("Übersprungene Player-IDs:", unique_skipped)
    print("Anzahl übersprungener IDs: ", len(unique_skipped))

    df = pd.DataFrame(records)
    if not include_dom:
        df.drop(columns=['dominant'], inplace=True, errors='ignore')
    return df

def aggregate(df: pd.DataFrame, include_dom: bool):
    if include_dom:
        idx = ['dominant','hand','phase']
    else:
        idx = ['hand','phase']
    agg   = df.groupby(idx)['mean_step']
    means = agg.mean().unstack(idx[:-1])
    sems  = agg.sem().unstack(idx[:-1])
    return means, sems

def plot_bars(means, sems, var, ylabel, df_all, mean_step_ref=None, include_dom=True):
    # Reihenfolge sicherstellen
    order = ['before','during','after']
    means = means.reindex(order)
    sems  = sems.reindex(order)
    phases = means.index
    x = np.arange(len(phases))

    # Original-Spalten
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

    # Signifikanz-Linie: Left vs. Right (paired)
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
        # Y-Offset zur Vermeidung von Überlappung
        ylim = ax.get_ylim()
        y_span = ylim[1] - ylim[0]
        y_text_left  = y + 0.02 * y_span
        y_text_right = y - 0.02 * y_span
        ax.text((x0+x1)/2, y_text_left if include_dom else y_text_left,
                sig, ha='center', va='bottom')

    # Referenzlinien
    if mean_step_ref:
        ylim = ax.get_ylim()
        y_span = ylim[1] - ylim[0]
        for idx, hand in enumerate(['left','right']):
            ref_val = mean_step_ref.get(hand)
            if ref_val is not None:
                ax.axhline(y=ref_val, linestyle='--', linewidth=1.2, color='black')
                # leichten Y-Versatz pro Hand
                offset = (0.02 if hand=='left' else -0.02) * y_span
                ax.text(len(phases)-0.5, ref_val + offset,
                        f"{hand.capitalize()}: {ref_val:.3f}",
                        va='center', ha='left', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\Results\Handmovement.pdf")
    plt.show()

def cohen_d_paired(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

def perform_stat_tests(df: pd.DataFrame, var_list: List[str], label: str):
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


def perform_ref_vs_token_tests(df_token: pd.DataFrame,
                               df_dialogue: pd.DataFrame,
                               var: str = 'mean_step',
                               label: str = ''):
    """
    Vergleicht für jede Hand, ob der Mittelwert der 'During'-Phase aus den
    Dialogue-Daten (Referenzlinie) signifikant unterschiedlich ist von den
    drei Phasen ('before', 'during', 'after') aus den Token-Daten.
    Es wird jeweils ein gepaarter t-Test durchgeführt, wobei nur jene Player-IDs
    berücksichtigt werden, die in beiden DataFrames in der entsprechenden Hand-Phase
    vorkommen.
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
    # Parameter: True = zusätzlich nach dominanter Hand differenzieren; False = nur nach Hand (Links vs. Rechts)
    include_dom = False

    # Player-Dict laden --> Dict mit {ID : Dominant Hand, ...}
    excel_path = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\EX3_allPlayer_playerID.xlsx"
    player_dict = hdb.dominant_hand_dict(excel_path)
    
    folder_dialog   = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\dialogue"
    folder_json     = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\crisper_whisper_json"
    
    # Timestamps extrahieren:
    
    all_dialogue_ts = hdb.extract_dialogue_timestamps(folder_dialog)                                    # Timestamps für alle Sätze der Dialogdatei
    all_bracketed_ts= hdb.extract_bracketed_timestamps(folder_json, output_path="no", export=False)     #Token Timestamps: bsp. [UH]
    
    #all_aussage_ts  = hdb.extract_speechact_timestamps(folder_dialog, speechact_type="Aussage")
    #all_frage_ts    = hdb.extract_speechact_timestamps(folder_dialog, speechact_type="Frage")

    # Ordnerpfade --> Trackingdaten (Json) und loopvarids (146 Probanden)
    loopvar_root    = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\ExperimentExport146"                          #LoopVarIDs
    simulation_root = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\ExperimentExport146\SimulationJsonFiles"    #Trackingdaten JSON

    # Analysen
    df_bracketed = analyze_all(loopvar_root, all_bracketed_ts, player_dict, simulation_root, include_dom)
    df_dialogue  = analyze_all(loopvar_root, all_dialogue_ts,  player_dict, simulation_root, include_dom)
    #df_aussage   = analyze_all(loopvar_root, all_aussage_ts, player_dict, simulation_root, include_dom)
    #df_frage     = analyze_all(loopvar_root, all_frage_ts,   player_dict, simulation_root, include_dom)

    # Aggregation
    means_br, sems_br       = aggregate(df_bracketed, include_dom)
    means_dlg, sems_dlg     = aggregate(df_dialogue,  include_dom)
    #means_auss, sems_auss   = aggregate(df_aussage,   include_dom)
    #means_fr, sems_fr       = aggregate(df_frage,     include_dom)

    # Referenzwerte: During-Phase
    mean_step_ref_dialog = (df_dialogue[df_dialogue.phase == 'during'].groupby('hand')['mean_step'].mean().to_dict())
    mean_step_ref_bracketed = (df_bracketed[df_bracketed.phase == 'during'].groupby('hand')['mean_step'].mean().to_dict())
    #mean_step_ref_aussage = (df_aussage[df_aussage.phase == 'during'].groupby('hand')['mean_step'].mean().to_dict())
    #mean_step_ref_frage = (df_frage[df_frage.phase == 'during'].groupby('hand')['mean_step'].mean().to_dict())

    # Plots
    plot_bars(means_br, sems_br, 'mean_step', 'Distance', df_bracketed, mean_step_ref_dialog, include_dom)
    #plot_bars(means_dlg, sems_dlg, 'mean_step', 'Distance', df_dialogue,  mean_step_ref_bracketed, include_dom)
    #plot_bars(means_auss, sems_auss,'mean_step', 'Distance', df_aussage,   mean_step_ref_frage, include_dom)

    # Statistische Tests
    print("=== Bracketed ===")
    perform_stat_tests(df_bracketed, ['mean_step'], 'Bracketed')
    #print("=== Dialogue ===")
    #perform_stat_tests(df_dialogue,  ['mean_step'], 'Dialogue')
    #print("=== Aussage ===")
    #perform_stat_tests(df_aussage,   ['mean_step'], 'Aussage')
    #Paired t-Tests: Dialogue 'During' vs. Token Phases
    
    perform_ref_vs_token_tests(df_bracketed, df_dialogue, var='mean_step', label='Token vs. Dialogue-During')
    
    
    # #Mittelwerte des plots Ausgeben:
    print("\n=== Mittelwerte Bracketed ===")
    print(means_br)
    