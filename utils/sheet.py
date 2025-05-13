import gspread
from google.oauth2.service_account import Credentials
from typing import List
from .helpers import *
from .data_io import *
from .operate import get_zone_pre

def read_sheet_info() -> List[List[str]]:
    """
    Read the TPU information from the Google Sheet.
    Return: a dictionary of dictionaries with TPU information.
    Keys: TPU full name
    Values: a dictionary with keys ['zone', 'pre', 'belong', 'running_status', 'user', 'user_note', 'script_note', 'alias', 'version', 'type', 'line']
    Logic: Read the lines that COL B starts with 'v'.
    """
    data = read_data()

    secret_path = SECRET_PATH
    sheet_id  = "1MFtgLx7uzBFdiPxrIqck00ilrSslZU2w2jRwriVpKMw"
    sheet_name  = "ka[experimental]"

    # 1. authenticate
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds  = Credentials.from_service_account_file(secret_path, scopes=scopes)
    client = gspread.authorize(creds)

    # 2. open the sheet
    ws = client.open_by_key(sheet_id).worksheet(sheet_name)

    # 3. find the number of rows
    last_row = 0
    for sentinel_col in range(1, 10):
        col_values = ws.col_values(sentinel_col)
        last_row = max(last_row, len(col_values))

    # 4. get the data
    table = ws.get(f"A1:Z{last_row}")      # gspread.values_get -> List[List[str]]
    
    tpu_information = {}
    for i, row in enumerate(table):
        if len(row) > 1 and row[1].startswith('v'):
            assert len(row) >= 7, f"line {i+1} is too short: {row}"
            _, tpu, belong, running_status, user, user_note, script_note, env = row[:8]
            zone, pre, full_name = get_zone_pre(tpu)
            
            assert zone is not None, f"line {i+1} tpu {tpu} not found in zone"
            assert zone.startswith(env), f"line {i+1} zone {zone} does not start with env {env}"

            assert running_status in ['running', 'reserved', '闲的'], f"line {i+1} running status {running_status} cannot be recognized"
            if running_status == '闲的':
                running_status = 'free'

            if user == '闲的':
                user = 'free'

            tpu_type = None

            if 'v2' in full_name: 
                tpu_version = 'v2'
                if 'v2-8' in full_name: tpu_type = 'v2-8'
                elif 'v2-32' in full_name: tpu_type = 'v2-32'
                elif 'v2-64' in full_name: tpu_type = 'v2-64'
                elif 'v2-128' in full_name: tpu_type = 'v2-128'
                else: raise ValueError(f"line {i+1} tpu {tpu} type cannot be recognized")
            elif 'v3' in full_name:
                tpu_version = 'v3'
                if 'v3-8' in full_name: tpu_type = 'v3-8'
                elif 'v3-32' in full_name: tpu_type = 'v3-32'
                elif 'v3-64' in full_name: tpu_type = 'v3-64'
                elif 'v3-128' in full_name: tpu_type = 'v3-128'
                else: raise ValueError(f"line {i+1} tpu {tpu} type cannot be recognized")
            elif 'v4' in full_name:
                tpu_version = 'v4'
                if 'v4-8' in full_name: tpu_type = 'v4-8'
                elif 'v4-32' in full_name: tpu_type = 'v4-32'
                elif 'v4-64' in full_name: tpu_type = 'v4-64'
                elif 'v4-128' in full_name: tpu_type = 'v4-128'
                else: raise ValueError(f"line {i+1} tpu {tpu} type cannot be recognized")

            assert tpu_version is not None, f"line {i+1} tpu {tpu} type cannot be recognized"

            tpu_information[full_name] = {
                'zone': zone,
                'pre': pre,
                'belong': belong,
                'version': tpu_version,
                'type': tpu_type,
                'running_status': running_status,
                'user': user,
                'user_note': user_note,
                'script_note': script_note,
                'alias': tpu,
                'line': i + 1
            }

    return tpu_information

def read_tpu_info_from_type(args):
    """
    Read the TPU information from specific args.
    Supported args: ['v2', 'v3', 'v4', 'v23', 'v24', 'v34', 'v234'/'-a'/'-all'/'--all'/'v*', 'v2-8', 'v2-32', 'v2-64', 'v2-128', 'v3-8', 'v3-32', 'v3-64', 'v3-128', 'v4-8', 'v4-32', 'v4-64', 'v4-128', '-p'/'-pre', '-n'/'-norm']
    """
    v2_list = ['v2-8', 'v2-32', 'v2-64', 'v2-128']
    v3_list = ['v3-8', 'v3-32', 'v3-64', 'v3-128']
    v4_list = ['v4-8', 'v4-32', 'v4-64', 'v4-128']
    type_list = []
    pre_filter = None
    for arg in args:
        if arg == 'v2': type_list += v2_list
        elif arg == 'v3': type_list += v3_list
        elif arg == 'v4': type_list += v4_list
        elif arg == 'v23': type_list += v2_list + v3_list
        elif arg == 'v24': type_list += v2_list + v4_list
        elif arg == 'v34': type_list += v3_list + v4_list
        elif arg in ['v234', '-a', '-all', '--all', 'v*']: type_list += v2_list + v3_list + v4_list
        elif arg in ['-p', '-pre']: pre_filter = True
        elif arg in ['-n', '-norm']: pre_filter = False
    tpu_information = read_sheet_info()

    if pre_filter is not None:
        filtered_tpu_information = filter_tpu_information(tpu_information, type=type_list, pre=pre_filter)
    else:
        filtered_tpu_information = filter_tpu_information(tpu_information, type=type_list)

    return filtered_tpu_information

def find_tpu_from_type(args):   
    """
    Find the TPU information from specific args, and display it.
    Supported args: ['v2', 'v3', 'v4', 'v23', 'v24', 'v34', 'v234'/'-a'/'-all'/'--all'/'v*', 'v2-8', 'v2-32', 'v2-64', 'v2-128', 'v3-8', 'v3-32', 'v3-64', 'v3-128', 'v4-8', 'v4-32', 'v4-64', 'v4-128', '-p'/'-pre', '-n'/'-norm']
    Display Style: ['full', 'category', 'category_note'(default)]
    """
    style = None
    for arg in args:
        if arg.startswith('style='):
            style = arg.split('=')[1]
            break
    information = read_tpu_info_from_type(args)
    return display_tpu_information(information, style=style)

def get_tpu_info_sheet(tpu):
    """
    Get the information of a specific TPU from the Google Sheet.
    Args: TPU name or alias
    Return: a dictionary with keys ['zone', 'pre', 'belong', 'running_status', 'user', 'user_note', 'script_note', 'alias', 'version', 'type', 'line']
    """
    tpu_information = read_sheet_info()
    _, _, full_name = get_zone_pre(tpu)
    if full_name in tpu_information:
        return tpu_information[full_name]
    else:
        print(f"{FAIL} TPU {tpu} not found in the sheet")
        return None




