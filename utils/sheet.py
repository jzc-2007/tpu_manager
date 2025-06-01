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
    for sentinel_col in range(2, 4): # COL B, C (ka, belonging)
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

            # print(running_status, user, user_note, script_note)
            assert running_status in ['running', 'reserved', '闲的', '没了!'], f"line {i+1} running status {running_status} cannot be recognized"
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

def write_sheet_info(info_to_write):
    """
    Write the tpu information to the Google Sheet.
    Args: a dictionary of a specific TPU information, with keys ['zone', 'pre', 'belong', 'running_status', 'user', 'user_note', 'script_note', 'alias', 'version', 'type', 'line']
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

    ws.update(f"C{row}:G{row}", [
        [
            transform_dict.get(info_to_write['belong'], info_to_write['belong']),
            transform_dict.get(info_to_write['running_status'], info_to_write['running_status']),
            transform_dict.get(info_to_write['user'], info_to_write['user']),
            transform_dict.get(info_to_write['user_note'], info_to_write['user_note']),
            transform_dict.get(info_to_write['script_note'], info_to_write['script_note']),
        ]
    ], value_input_option='USER_ENTERED')

    print(f"{INFO} write_sheet_info: TPU {info_to_write['alias']} information updated in the sheet")
    return True

def read_tpu_info_from_type(args):
    """
    Read the TPU information from specific args.
    Supported args: ['v2', 'v3', 'v4', 'v23', 'v24', 'v34', 'v234'/'-a'/'-all'/'--all'/'v*', 'v2-8',  'v2-16', 'v2-32', 'v2-64', 'v2-128', 'v3-8',  'v4-16', 'v3-32', 'v3-64', 'v3-128', 'v4-8', 'v4-16', 'v4-32', 'v4-64', 'v4-128', 'v2/3/4+', 'v2/3/4-', '-p'/'-pre', '-n'/'-norm']
    Return: a dictionary of dictionaries with TPU information.
    """
    v2_list = ['v2-8', 'v2-16', 'v2-32', 'v2-64', 'v2-128']
    v3_list = ['v3-8', 'v3-16', 'v3-32', 'v3-64', 'v3-128']
    v4_list = ['v4-8', 'v4-16', 'v4-32', 'v4-64', 'v4-128']
    all_type_list = v2_list + v3_list + v4_list
    type_list = []
    pre_filter = None

    for arg in args:
        if arg in ['v2', 'v2-']: type_list += v2_list
        elif arg in ['v3']: type_list += v3_list
        elif arg in ['v4', 'v4+']: type_list += v4_list
        elif arg in ['v23', 'v3-']: type_list += v2_list + v3_list
        elif arg in ['v24']: type_list += v2_list + v4_list
        elif arg in ['v34', 'v3+']: type_list += v3_list + v4_list
        elif arg in ['v234', '-a', '-all', '--all', 'v*', 'v2+', 'v4-']: type_list += all_type_list
        elif arg in all_type_list: type_list.append(arg)
        elif arg in ['-p', '-pre']: pre_filter = True
        elif arg in ['-n', '-norm']: pre_filter = False

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



