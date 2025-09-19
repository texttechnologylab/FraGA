import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import re
from collections import defaultdict
import ast
import pandas as pd

def get_tracking_data_for_timesegment(timestamps: list, loopvar_ids: list, tracking_data: dict, frames: int = 32, timestamp: str = "no") -> dict:
    """
    Get tracking data for a specific time segment.
    Returns a dict with keys 'botPos' and 'botRot'.
    """
    time_per_frame = 1 / frames
    merged_dict = {}

    for start_offset, end_offset in timestamps:
        if timestamp == "before":
            start = start_offset - 1
            stop  = start_offset
        elif timestamp == "during":
            start = start_offset
            stop  = end_offset
        elif timestamp == "after":
            start = end_offset
            stop  = end_offset + 1
        else:
            raise ValueError(f"Unbekannter timestamp mode: {timestamp}")

        start_frame = round(start / time_per_frame)
        stop_frame  = round(stop  / time_per_frame)
        start_loopvarid = loopvar_ids[start_frame] - 1
        stop_loopvarid  = loopvar_ids[stop_frame]  + 1

        dict1 = {
            'botPos': tracking_data['botPos'][start_loopvarid:stop_loopvarid + 1],
            'botRot': tracking_data['botRot'][start_loopvarid:stop_loopvarid + 1],
        }
        
        merged_dict = merge_ordered_dicts(merged_dict, dict1)

    return merged_dict

def load_loopvarids(path: str) -> list:
    """
    Load loopvarids from a file.
    :param path: Path to the loopvarids file.
    :return: List of loopvarids.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        loopvarids = [int(line.strip()) for line in f.readlines()]

    return loopvarids

def load_tracking_data(path: str) -> dict:
    """
    Load tracking data from a JSON file. No preprocessing is done here.
    :param path: Path to the tracking data file.
    :return: Dictionary containing the tracking data.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        trackingdata = json.load(f)

    return trackingdata

def merge_ordered_dicts(dict1, dict2):
    merged = {}
    all_keys = dict1.keys() | dict2.keys()  # Alle Schlüssel aus beiden Dictionaries

    for key in all_keys:
        if key in dict1 and key in dict2:
            # Falls beide Dictionaries den Schlüssel haben, Listen verketten
            if isinstance(dict1[key], list) and isinstance(dict2[key], list):
                merged[key] = dict1[key] + dict2[key]  # Reihenfolge bleibt erhalten
            else:
                merged[key] = dict1[key]  # Falls es kein Listenwert ist, übernehme einen Wert
        elif key in dict1:
            merged[key] = dict1[key]
        else:
            merged[key] = dict2[key]
    
    return merged

def dominant_hand_dict(excel_path: str) -> Dict[str, str]:
    """Liest eine Excel-Datei ein und erstellt ein Dictionary mit playerId → Dominante Hand.
    
    Erwartetes Format der Spalte 'playerId': Python-Listen wie ['id1', 'id2']
    Spalte 'Dominant Hand' enthält den Handtyp (z.B. 'right hand', 'left hand')"""
    df = pd.read_excel(excel_path)
    player_dict: Dict[str, str] = {}

    for pid_cell, hand in zip(df["playerId"], df["Dominant Hand"]):
        try:
            ids = ast.literal_eval(pid_cell)
            for pid in ids:
                player_dict[pid.strip()] = hand
        except Exception as e:
            print(f"Fehler beim Parsen von {pid_cell}: {e}")

    return player_dict

def extract_speechact_timestamps(folder_path: str, speechact_type: str = "Aussage") -> Dict[str, List[Tuple[float, float]]]:
    """
    Liest alle .txt-Dateien ein und extrahiert nur die time stamps des gewünschten Speechacts (Aussage oder Frage).
    
    Parameter:
    - folder_path: Pfad zum Ordner mit den Dialog Transkriptionen (.txt-Dateien)
    - speechact_type: "Aussage" oder "Frage"

    Rückgabe:
    Dict[player_id, List[(start, end)]] für den gewählten Speechact
    """
    if speechact_type not in {"Aussage", "Frage"}:
        raise ValueError("speechact_type muss 'Aussage' oder 'Frage' sein.")

    line_pattern = re.compile(
        r'^Player\s*(?P<num>[12]):\s*\[(?P<start_min>\d+):(?P<start_sec>\d+\.\d+)\s*-->\s*'
        r'(?P<end_min>\d+):(?P<end_sec>\d+\.\d+)\]'
    )
    speechacts: Dict[str, List[Tuple[float, float]]] = {}
    folder = Path(folder_path)

    for txt_file in folder.glob("*.txt"):
        stem = txt_file.stem
        if not stem.endswith("_dialogue"):
            continue
        ids_part = stem[:-len("_dialogue")]
        try:
            raw_id1, raw_id2 = ids_part.split("+")
        except ValueError:
            continue
        id1, id2 = raw_id1.strip(), raw_id2.strip()
        for pid in (id1, id2):
            if pid not in speechacts:
                speechacts[pid] = []
        player_map = {"1": id1, "2": id2}

        with txt_file.open(encoding="utf-8") as f:
            for line in f:
                match = line_pattern.match(line)
                if not match:
                    continue
                num = match.group("num")
                start_total = int(match.group("start_min")) * 60 + float(match.group("start_sec"))
                end_total = int(match.group("end_min")) * 60 + float(match.group("end_sec"))
                rest = line[match.end():].strip()
                if rest.endswith("?"):
                    act = "Frage"
                elif rest.endswith("."):
                    act = "Aussage"
                else:
                    act = "Aussage"
                if act == speechact_type:
                    player_id = player_map[num]
                    speechacts[player_id].append((start_total, end_total))

    return speechacts


def extract_bracketed_timestamps(folder_path: str, output_path, export=False) -> Dict[str, List[Tuple[float, float]]]:
    """
    Liest alle JSON-Dateien in `folder_path` ein und extrahiert nur die Wort-Timestamps,
    deren "text" in eckigen Klammern steht (z.B. "[UM]", "[UH]" etc.).
    Die playerid wird aus dem Dateinamen im Format
    ..._YYYYMMDD_HHMMSS_<playerid>_words.json entnommen.

    Rückgabe:
        Dict[playerid, List[(start, end), ...]]
    """
    timestamps_by_player: Dict[str, List[Tuple[float, float]]] = {}
    folder = Path(folder_path)
    
    # Alle .json-Dateien im Verzeichnis
    for json_file in folder.glob("*.json"):
        parts = json_file.stem.split("_")
        if len(parts) < 4:
            continue
        player_id = parts[3]
        
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        words = data.get("words", [])
        if not isinstance(words, list):
            continue
        
        if player_id not in timestamps_by_player:
            timestamps_by_player[player_id] = []
        
        # Nur Wörter mit Text in eckigen Klammern auswählen
        for w in words:
            text = w.get("text", "")
            if isinstance(text, str) and text.startswith("[") and text.endswith("]"):
                ts = w.get("timestamp")
                if (
                    isinstance(ts, list)
                    and len(ts) == 2
                    and all(isinstance(x, (int, float)) for x in ts)
                ):
                    start, end = float(ts[0]), float(ts[1])
                    timestamps_by_player[player_id].append((start, end))
    
    # JSON schreiben
    if export == True:
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(timestamps_by_player, out_f, ensure_ascii=False, indent=2)
        print(f"Ergebnisse wurden nach '{output_path}' geschrieben.")
    
    return timestamps_by_player

def extract_dialogue_timestamps(folder_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Liest alle .txt-Dateien in `folder_path` ein, deren Namen das Format
    <player1_id>+<player2_id>_dialogue.txt haben.
    Extrahiert aus jeder Zeile:
      Player N: [MM:SS.mmm --> MM:SS.mmm]  Text
    die Start- und Endzeit, konvertiert beides in Sekunden und sammelt sie
    pro player_id.

    Rückgabe:
        Dict[player_id, List[(start_seconds, end_seconds), ...]]
    """
    # Regex zum Parsen der Timestamp-Zeile
    line_pattern = re.compile(
        r'^Player\s*(?P<num>[12]):\s*\[(?P<start_min>\d+):(?P<start_sec>\d+\.\d+)\s*-->\s*'
        r'(?P<end_min>\d+):(?P<end_sec>\d+\.\d+)\]'
    )

    timestamps_by_player: Dict[str, List[Tuple[float, float]]] = {}
    folder = Path(folder_path)

    for txt_file in folder.glob("*.txt"):
        stem = txt_file.stem  # e.g. '0a0b9bcf-2f6a498c+e05b7223-d1c3a586_dialogue'
        if not stem.endswith("_dialogue"):
            continue
        # IDs aus Dateinamen extrahieren und säubern
        ids_part = stem[:-len("_dialogue")]
        try:
            raw_id1, raw_id2 = ids_part.split("+")
        except ValueError:
            # Unerwartetes Format überspringen
            continue
        # Whitespace entfernen
        id1 = raw_id1.strip()
        id2 = raw_id2.strip()

        # Initialisiere Listen
        timestamps_by_player.setdefault(id1, [])
        timestamps_by_player.setdefault(id2, [])

        # Mapping von Player-Nummer zur ID
        player_map = {"1": id1, "2": id2}

        # Datei zeilenweise lesen
        with txt_file.open(encoding="utf-8") as f:
            for line in f:
                match = line_pattern.match(line)
                if not match:
                    continue
                num = match.group("num")
                # Umwandlung in Sekunden
                start_min = int(match.group("start_min"))
                start_sec = float(match.group("start_sec"))
                end_min = int(match.group("end_min"))
                end_sec = float(match.group("end_sec"))

                start_total = start_min * 60 + start_sec
                end_total = end_min * 60 + end_sec

                player_id = player_map[num]
                timestamps_by_player[player_id].append((start_total, end_total))
    
    return timestamps_by_player


if __name__ == "__main__":
    
    folder_dialog = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\dialogue"
    folder_json     = r"C:\Users\Danie\Desktop\Studentische Hilfskraft\Github\crisper_whisper_json"
    
    #Timestamps für alle Sätze:
    print("All Sentence Timestamps for 'ab133da0-f27e51e0':")
    dialog_timestamps = extract_dialogue_timestamps(folder_dialog)
    single_dialog_timestamps = dialog_timestamps["ab133da0-f27e51e0"]
    print(single_dialog_timestamps)
    
    #Timestamps für Aussagen oder Fragen:
    print("All Speechact Timestamps for 'ab133da0-f27e51e0':")
    all_speechact_timestamps = extract_speechact_timestamps(folder_dialog, speechact_type="Frage")
    single_speecact_timestamps= all_speechact_timestamps["ab133da0-f27e51e0"]
    print(single_speecact_timestamps)

    #Timestamps für alle Hesitations (stottern) z.b. [UH]
    print("All Hesitations for 'ab133da0-f27e51e0':")
    all_token_timestamps = extract_bracketed_timestamps(folder_json, output_path="no", export=False)
    single_token_timestamps = all_token_timestamps["ab133da0-f27e51e0"]
    print(single_token_timestamps)

    #Timestamps für alle turnover (Backchannelling, Turn transition) -->AUSSTEHEND