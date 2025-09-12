import gspread
from google.oauth2.service_account import Credentials
from typing import List
from .helpers import *
from .constants import *
from .data_io import *

def read_sheet_info() -> dict:
    """
    Read the TPU information from the Google Sheet.
    Return: a dictionary of dictionaries with TPU information.
    Keys: TPU full name
    Values: a dictionary with keys ['zone', 'pre', 'belong', 'running_status', 'user', 'user_note', 'script_note', 'alias', 'version', 'type', 'other_note', 'env', 'line']
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
    for sentinel_col in range(2, 4): # COL B, C (ka, belonging)
        col_values = ws.col_values(sentinel_col)
        last_row = max(last_row, len(col_values))

    # 4. get the data
    table = ws.get(f"A1:Z{last_row}")      # gspread.values_get -> List[List[str]]
    
    tpu_information = {}
    for i, row in enumerate(table):
        if len(row) > 1 and row[1].startswith('v'):
            assert len(row) >= 7, f"line {i+1} is too short: {row}"
            _, tpu, belong, running_status, user, user_note, script_note, env, other = row[:9]
            zone, pre, spot, full_name = get_zone_pre_spot(tpu)
            
            assert zone is not None, f"line {i+1} tpu {tpu} not found in zone"
            assert zone.startswith(env), f"line {i+1} zone {zone} does not start with env {env}"

            # print(running_status, user, user_note, script_note)
            assert running_status in ['running', 'reserved', 'reserved(error)', '闲的', '没了!'], f"line {i+1} running status {running_status} cannot be recognized"
            if running_status == '闲的':
                running_status = 'free'

            if user == '闲的':
                user = 'free'

            tpu_type, tpu_version = None, None
            for key in NAME_TO_VER:
                if key in full_name:
                    tpu_version = NAME_TO_VER[key]

            for key in NAME_TO_TYPE:
                if key in full_name:
                    tpu_type = NAME_TO_TYPE[key]

            assert (tpu_version is not None) and (tpu_type is not None), f"line {i+1} tpu {tpu} name cannot be recognized"

            tpu_information[full_name] = {
                'zone': zone,
                'pre': pre,
                'belong': belong,
                'version': tpu_version,
                'type': tpu_type,
                'running_status': running_status,
                'user': user,
                'env': env,
                'user_note': user_note,
                'script_note': script_note,
                'alias': tpu,
                'other_note': other,
                'line': i + 1
            }

    return tpu_information

def write_sheet_info(info_to_write):
    """
    Write the tpu information to the Google Sheet.
    Args: a dictionary of a specific TPU information, with keys ['zone', 'pre', 'belong', 'running_status', 'user', 'user_note', 'script_note', 'alias', 'version', 'type', 'other_note', 'line']
    Only updating belong, running_status, user, user_note, script_note
    """
    secret_path = SECRET_PATH
    sheet_id  = "1MFtgLx7uzBFdiPxrIqck00ilrSslZU2w2jRwriVpKMw"
    sheet_name  = "ka[experimental]"

    # 1. authenticate
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds  = Credentials.from_service_account_file(secret_path, scopes=scopes)
    client = gspread.authorize(creds)

    # 2. open the sheet
    ws = client.open_by_key(sheet_id).worksheet(sheet_name)

    # 3. write the data
    row = info_to_write['line']
    col = 1
    transform_dict = {'free': '闲的'}

    ws.update(f"C{row}:I{row}", [
        [
            transform_dict.get(info_to_write['belong'], info_to_write['belong']),
            transform_dict.get(info_to_write['running_status'], info_to_write['running_status']),
            transform_dict.get(info_to_write['user'], info_to_write['user']),
            transform_dict.get(info_to_write['user_note'], info_to_write['user_note']),
            transform_dict.get(info_to_write['script_note'], info_to_write['script_note']),            
            transform_dict.get(info_to_write['env'], info_to_write['env']),
            transform_dict.get(info_to_write['other_note'], info_to_write['other_note']),
        ]
    ], value_input_option='USER_ENTERED')

    # print("" + str(resp))  # Debugging output to see the response from the update

    # print(f"{INFO} update row {row} in the sheet with TPU information: {info_to_write}")

    print(f"{INFO} write_sheet_info: TPU {info_to_write['alias']} information updated in the sheet")
    return True

def read_tpu_info_from_type(args):
    """
    Read the TPU information from specific args.
    Supported args: ['v<num>', 'v<num>+', 'v<num>-<num>', 'v*/-a/--all', '-p'/'-pre', '-n'/'-norm']
    Return: a dictionary of dictionaries with TPU information.
    """
    type_list = []
    pre_filter = None

    for arg in args:
        if arg in ARG_TO_LIST: 
            if isinstance(ARG_TO_LIST[arg], list):
                type_list += ARG_TO_LIST[arg]
            else:
                type_list.append(ARG_TO_LIST[arg])
        if arg in ['-p', '-pre']: pre_filter = True
        if arg in ['-n', '-norm']: pre_filter = False

    if len(type_list) == 0:
        type_list = all_type_list

    tpu_information = read_sheet_info()

    if pre_filter is not None:
        filtered_tpu_information = filter_tpu_information(tpu_information, type=type_list, pre=pre_filter)
    else:
        filtered_tpu_information = filter_tpu_information(tpu_information, type=type_list)

    return filtered_tpu_information

def find_tpu_from_type(args):   
    """
    Find the TPU information from specific args, and display it.
    Supported args: ['v<num>', 'v<num>-<num>', 'v*', 'v<num>+', '-p'/'-pre', '-n'/'-norm']
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
    _, _, _, full_name = get_zone_pre_spot(tpu)
    if full_name in tpu_information:
        return tpu_information[full_name]
    else:
        print(f"{FAIL} TPU {tpu} not found in the sheet")
        return None

def release_tpu(args):
    """
    Change the running status to '闲的', user to '闲的', and user_note to be empty.
    Args: TPU name or alias
    """
    tpu, user = None, None
    data = read_data()
    assert len(args) in [1, 2], f"release_tpu: {args} is not valid"
    if len(args) == 1:
        tpu = args[0]
    elif len(args) == 2:
        tpu = args[0]
        user = args[1]
    spreadsheet_name = None
    if user is not None:
        spreadsheet_name = data['users'][user]['spreadsheet_name']
    
    tpu_information = get_tpu_info_sheet(tpu)
    if tpu_information is not None:
        if user is not None:
            assert tpu_information['user'] == spreadsheet_name, f"TPU {tpu} is not used by {user}, but by {tpu_information['user']}"
        tpu_information['running_status'] = '闲的'
        tpu_information['user'] = '闲的'
        tpu_information['user_note'] = ''
        write_sheet_info(tpu_information)
        print(f"{GOOD} TPU {tpu} released")
    else:
        print(f"{FAIL} TPU {tpu} not found in the sheet")

def set_spreadsheet_notes(tpu, notes):
    """
    Set the spreadsheet notes for a specific TPU.
    Args: TPU name or alias, notes
    """
    tpu_information = get_tpu_info_sheet(tpu)
    if tpu_information is not None:
        tpu_information['user_note'] = notes
        write_sheet_info(tpu_information)
        print(f"{INFO} TPU {tpu} notes updated")
    else:
        print(f"{FAIL} TPU {tpu} not found in the sheet")

def add_spreadsheet_notes(tpu, notes):
    """
    Set the spreadsheet notes for a specific TPU.
    Args: TPU name or alias, notes
    """
    tpu_information = get_tpu_info_sheet(tpu)
    if tpu_information is not None:
        tpu_information['user_note'] += notes
        write_sheet_info(tpu_information)
        print(f"{INFO} TPU {tpu} notes updated")
    else:
        print(f"{FAIL} TPU {tpu} not found in the sheet")



