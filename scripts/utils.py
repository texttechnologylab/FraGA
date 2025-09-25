import os
import json
import pandas as pd
import settings as st


def get_data_for_body(tracking_data: dict, start_loopvarid: int, stop_loopvarid: int):
    return {
        'botPos': tracking_data['botPos'][start_loopvarid:stop_loopvarid + 1],
        'botRot': tracking_data['botRot'][start_loopvarid:stop_loopvarid + 1],
        'startTime': tracking_data['startTime'],
        'endTime': tracking_data['endTime'],
        'segmentLength': tracking_data['segmentLength'],
    }


def get_data_for_eye(tracking_data: dict, start_loopvarid: int, stop_loopvarid: int):
    return {
        'leftEye': {
            'position': tracking_data['leftEye']['position'][start_loopvarid:stop_loopvarid + 1],
            'rotation': tracking_data['leftEye']['rotation'][start_loopvarid:stop_loopvarid + 1],
            'isValid': tracking_data['leftEye']['isValid'][start_loopvarid:stop_loopvarid + 1],
            'confidence': tracking_data['leftEye']['confidence'][start_loopvarid:stop_loopvarid + 1],
    },
        'rightEye': {
            'position': tracking_data['rightEye']['position'][start_loopvarid:stop_loopvarid + 1],
            'rotation': tracking_data['rightEye']['rotation'][start_loopvarid:stop_loopvarid + 1],
            'isValid': tracking_data['rightEye']['isValid'][start_loopvarid:stop_loopvarid + 1],
            'confidence': tracking_data['rightEye']['confidence'][start_loopvarid:stop_loopvarid + 1],
    },
        'nullOrEmpty': tracking_data['nullOrEmpty'][start_loopvarid:stop_loopvarid + 1],
    }


def get_data_for_hand(tracking_data: dict, start_loopvarid: int, stop_loopvarid: int):
    return {
        'botPos': tracking_data['botPos'][start_loopvarid:stop_loopvarid + 1],
        'botRot': tracking_data['botRot'][start_loopvarid:stop_loopvarid + 1],
        'messageId': tracking_data['messageId'][start_loopvarid:stop_loopvarid + 1],
        'localTime': tracking_data['localTime'][start_loopvarid:stop_loopvarid + 1],
    }


def get_data_for_head(tracking_data: dict, start_loopvarid: int, stop_loopvarid: int):
    return {
        'botPos': tracking_data['botPos'][start_loopvarid:stop_loopvarid + 1],
        'botRot': tracking_data['botRot'][start_loopvarid:stop_loopvarid + 1],
    }


def get_tracking_data_for_timesegment(start: float, stop: float, loopvar_ids: list, tracking_data: dict, data_name: str, frames: int = 32):
    """
    Get tracking data for a specific time segment.
    :param start:
    :param stop:
    :param loopvar_ids:
    :param tracking_data:
    :param data_name:
    :param frames:
    :return:
    """

    time_per_frame = 1 / frames
    start_frame = round(start / time_per_frame)
    stop_frame = round(stop / time_per_frame)

    # Ensure the start and stop frames are within the bounds of loopvarids
    start_loopvarid = loopvar_ids[start_frame] - 1
    stop_loopvarid = loopvar_ids[stop_frame] + 1

    # Get the tracking data for the specified time segment
    if data_name == 'body':
        return get_data_for_body(tracking_data, start_loopvarid, stop_loopvarid)
    elif data_name == 'eye':
        return get_data_for_eye(tracking_data, start_loopvarid, stop_loopvarid)
    elif data_name == 'hand':
        return get_data_for_hand(tracking_data, start_loopvarid, stop_loopvarid)
    elif data_name == 'head':
        return get_data_for_head(tracking_data, start_loopvarid, stop_loopvarid)
    else:
        print('Data name not recognized')



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


def excel_to_dict(excel_path: str = st.excel_data, json_output: str = st.player_json, sheet_name: str = 0, export: bool = False, language_filter: bool = False) -> dict:
    """
    Liest eine Excel-Datei ein und erstellt ein verschachteltes Dictionary:
    {'YYYY-MM-DD HH:MM:SS': {
            Person1: [...playerIds...],
            Person2: [...playerIds...],
            ...},...}
    Optional kann das Ergebnis als JSON-Datei exportiert werden.

    Args:
        excel_path: Pfad zur Excel-Datei.
        json_output: Pfad zur JSON-Datei.
        sheet_name: Name oder Index des zu lesenden Sheets (default: erstes Sheet).
        export: Wenn True, wird das resultierende Dict als 'output.json' gespeichert.
        language_filter: Wenn True, werden nur deutschsprachige Experimente gespeichert.

    Returns:
        dict der Form {startzeit_str: {person: [playerId, ...], ...}, ...}
    """

    # Excel einlesen
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Deutsche Datumsangaben umwandeln
    df['Startzeit'] = pd.to_datetime(df['Startzeit'], dayfirst=True)

    # Falls playerId als String-Literal vorliegt, in Listen umwandeln
    if df['playerId'].dtype == object and df['playerId'].apply(lambda x: isinstance(x, str)).all():
        df['playerId'] = df['playerId'].apply(eval)

    result = {}
    for _, row in df.iterrows():
        # Timestamp in formatierten String umwandeln
        start_str = row['Startzeit'].strftime('%Y-%m-%d %H:%M:%S')
        person    = row['Person']
        players   = row['playerId']
        start     = row['Start']
        end       = row['Ende']
        language  = row['language']

        if not pd.isna(language) and language_filter is True:
            continue
        if start_str not in result:
            result[start_str] = {}
        result[start_str][person] = players
        if start not in result and not pd.isna(start):
            result[start_str]["start"] = start
        if end not in result and not pd.isna(end):
            result[start_str]["end"] = end

    # Optionaler JSON-Export
    if export:
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    return result




