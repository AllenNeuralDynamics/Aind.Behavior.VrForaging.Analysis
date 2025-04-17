from datetime import datetime
import pytz
import os 
from pathlib import Path
from typing import List, Literal

import time

from aind_vr_foraging_analysis.utils.parsing import parse, AddExtraColumns

def parse_user_date(user_date_str):
    """
    Parses a user-provided date string in the format 'YYYY-MM-DD' and returns a datetime.date object.

    Parameters:
    user_date_str (str): A string representing a date in the format 'YYYY-MM-DD'.

    Returns:
    datetime.date: The parsed date if the format is valid.
    None: If the input format is incorrect.
    """
    try:
        return datetime.strptime(user_date_str, "%Y-%m-%d").date()  # Convert user input to date
    except ValueError:
        return "Invalid date format"  # Return None if the format is incorrect

def extract_and_convert_time(filename):
    """
    Extracts a timestamp from a filename and converts it to a local date in the 'America/Los_Angeles' timezone.

    The filename must follow one of these formats:
    - 'prefix_YYYY-MM-DDTHHMMSSZ_suffix' (UTC timestamp, indicated by 'Z')
    - 'prefix_YYYYMMDDTHHMMSS_suffix' (Local time in 'America/Los_Angeles')

    Parameters:
    filename (str): A string containing a timestamp in one of the expected formats.

    Returns:
    datetime.date: The extracted and converted local date.
    str: "Invalid filename format" if the filename format does not match expectations.
    """
    seattle_tz = pytz.timezone('America/Los_Angeles')

    # Extract the timestamp part
    timestamp_part = filename.split("_")[1]

    try:
        if "Z" in timestamp_part:  # Case: UTC timestamp
            dt_utc = datetime.strptime(timestamp_part, "%Y-%m-%dT%H%M%SZ")
            dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(seattle_tz)
        else:  # Case: Already local time
            dt_local = datetime.strptime(timestamp_part, "%Y%m%dT%H%M%S")
            dt_local = seattle_tz.localize(dt_local)
        return dt_local.date()
    except ValueError:
        return "Invalid filename format"

def find_sessions_relative_to_date(mouse: str,
                                   date_string: str,
                                   base_path: str = 'Z:/scratch/vr-foraging/data/',
                                   when: Literal['before', 'after', 'on_or_before', 'on_or_after', 'on'] = 'on_or_before'
                                  ) -> List[Path]:
    """
    Returns a list of session paths for a given mouse that are before, after, or on a specified date.
    
    Parameters:
    - mouse: The mouse name/id.
    - date_string: The reference date (e.g. '2024-12-01').
    - base_path: Base path to the data folder.
    - when: One of ['before', 'after', 'on_or_before', 'on_or_after', 'on'] to filter sessions.
    
    Returns:
    - List of Path objects corresponding to matching sessions.
    """
    
    target_date = parse_user_date(date_string)
    directory = os.path.join(base_path, mouse)
    files = os.listdir(directory)
    sorted_files = sorted(files, key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=False)

    matching_sessions = []
    
    for file_name in sorted_files:
        session_date = extract_and_convert_time(file_name)
        compare = (session_date < target_date if when == 'before' else
                    session_date > target_date if when == 'after' else
                    session_date <= target_date if when == 'on_or_before' else
                    session_date >= target_date if when == 'on_or_after' else
                    session_date == target_date)
        if compare:
            full_path = Path(directory) / file_name
            matching_sessions.append(full_path)
    
    if len(matching_sessions) == 0:
        print(f"No sessions found for mouse {mouse} on {date_string}")
        
    return matching_sessions


def load_session(session_path: Path):
    """
    Loads and processes a behavioral session from a given path.
    
    Parameters:
    ----------
    session_path : Path
        Full path to the session directory containing the raw data.

    Returns:
    -------
    all_epochs : pd.DataFrame
        A DataFrame of parsed and enriched behavioral epochs from the session.
        
    stream_data : object
        An object containing continuous data streams (e.g., analog signals, encoder).
        
    data : dict or object
        The raw session data as returned by `parse.load_session_data()`.
    """
    
    data = parse.load_session_data(session_path)

    all_epochs = parse.parse_dataframe(data)

    extra_columns = AddExtraColumns(all_epochs, run_on_init=True)
    all_epochs = extra_columns.get_all_epochs()

    stream_data = parse.ContinuousData(data)
    
    return all_epochs, stream_data, data
