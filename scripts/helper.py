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
            raise ValueError(f"Unknown timestamp mode: {timestamp}")

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
    """
    Merge two dictionaries while preserving order and handling list concatenation.
    The function returns a new dictionary containing all keys from both inputs.
    """
    merged = {}
    all_keys = dict1.keys() | dict2.keys()  # All Keys from both Dictionaries

    for key in all_keys:
        if key in dict1 and key in dict2:
            # If both dictionaries contain the key, concatenate the lists.
            if isinstance(dict1[key], list) and isinstance(dict2[key], list):
                merged[key] = dict1[key] + dict2[key]  # The order remains unchanged
            else:
                merged[key] = dict1[key]  
        elif key in dict1:
            merged[key] = dict1[key]
        else:
            merged[key] = dict2[key]
    
    return merged

def dominant_hand_dict(excel_path: str) -> Dict[str, str]:
    """Reads an Excel file and creates a dictionary mapping playerId → dominant hand.
    Expected format of the column 'playerId': Python lists stored as strings, e.g. ['id1', 'id2']
    Column 'Dominant Hand' contains the hand type (e.g. 'right hand', 'left hand').
    """
    df = pd.read_excel(excel_path)
    player_dict: Dict[str, str] = {}

    for pid_cell, hand in zip(df["playerId"], df["Dominant Hand"]):
        try:
            ids = ast.literal_eval(pid_cell)
            for pid in ids:
                player_dict[pid.strip()] = hand
        except Exception as e:
            print(f"Error while Parsing {pid_cell}: {e}")

    return player_dict

def extract_speechact_timestamps(folder_path: str, speechact_type: str = "Aussage") -> Dict[str, List[Tuple[float, float]]]:
    """
    Reads all `.txt` transcription files in a folder and extracts only the timestamps 
    of the requested speech act type ("Aussage" = statement or "Frage" = question).
    
    Parameter:
    - folder_path: Path to the folder containing dialogue transcription files (`.txt`).
        The file names are expected to follow the pattern: <id1>+<id2>_dialogue.txt
    - Which type of speech act to extract. Must be either:
        - "Aussage" → statements (lines ending with a period `.` or otherwise defaulting to statement)
        - "Frage"   → questions (lines ending with a question mark `?`)
        Default is "Aussage".

    Rückgabe:
    A dictionary mapping each `player_id` to a list of `(start_time, end_time)` tuples, 
        where times are expressed in seconds (floats) --> Dict[player_id, List[(start, end)]]
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
    Reads all JSON files in a given folder and extracts word-level timestamps 
    where the `"text"` field is enclosed in square brackets (e.g., "[UM]", "[UH]").

    The `player_id` is inferred from the filename, which must follow the format:
    ..._YYYYMMDD_HHMMSS_<playerid>_words.json

    Returns:
        Dict[playerid, List[(start, end), ...]]
    """
    timestamps_by_player: Dict[str, List[Tuple[float, float]]] = {}
    folder = Path(folder_path)
    
    # All json-Files in the Repository
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
        
        # Only select words containing text enclosed in square brackets.
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
    
    # Write JSON: 
    if export == True:
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(timestamps_by_player, out_f, ensure_ascii=False, indent=2)
        print(f"Ergebnisse wurden nach '{output_path}' geschrieben.")
    
    return timestamps_by_player

def extract_dialogue_timestamps(folder_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Reads all `.txt` files in the given folder and extracts dialogue timestamps 
    for each player based on the expected filename and line format.

    Expected filename format:
    <player1_id>+<player2_id>_dialogue.txt
    Example: "0a0b9bcf-2f6a498c+e05b7223-d1c3a586_dialogue.txt"

    Expected line format inside the file:
    Example: Player 1: [00:05.230 --> 00:07.580]  Hello, how are you?

    Parsing logic:
    - Uses a regular expression to capture:
        * Player number (1 or 2)
        * Start time (minutes, seconds.milliseconds)
        * End time (minutes, seconds.milliseconds)
    - Converts times into total seconds as floats.
    - Maps the extracted times to the correct `player_id` from the filename.

    Parameters:
    folder_path : Path to the folder containing dialogue transcription `.txt` files.

    Returns:
    Dict[str, List[Tuple[float, float]]]
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
        # Remove Whitespace 
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
    # SET PATH to word_timestamps_json-Folder and Dialogue-Folder
    folder_dialog = r"...\Github\dialogue"                                                                          ### SET PATH!!! ###
    folder_json     = r"...\Github\crisper_whisper_json"                                                            ### SET PATH!!! ###
    
    #Timestamps for all Sentences for player "ab133da0-f27e51e0":
    print("All Sentence Timestamps for 'ab133da0-f27e51e0':")
    dialog_timestamps = extract_dialogue_timestamps(folder_dialog)                                                  # folder_dialog
    single_dialog_timestamps = dialog_timestamps["ab133da0-f27e51e0"]
    print(single_dialog_timestamps)

    #Timestamps for all Hesitations (stottern) e.g. [UH] for player "ab133da0-f27e51e0":
    print("All Hesitations for 'ab133da0-f27e51e0':")
    all_token_timestamps = extract_bracketed_timestamps(folder_json, output_path="no", export=False)                # folder_json
    single_token_timestamps = all_token_timestamps["ab133da0-f27e51e0"]
    print(single_token_timestamps)

    #Timestamps for Statements or Questions for player "ab133da0-f27e51e0":
    print("All Speechact Timestamps for 'ab133da0-f27e51e0':")
    all_speechact_timestamps = extract_speechact_timestamps(folder_dialog, speechact_type="Frage")                  # folder_dialog
    single_speecact_timestamps= all_speechact_timestamps["ab133da0-f27e51e0"]
    print(single_speecact_timestamps)
    
    #Timestamps for all turnover (Backchannelling, Turn transition) --> TO BE CONTINUED...
