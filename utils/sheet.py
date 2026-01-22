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
    for sentinel_col in range(2, 7): # COL B, C (ka, belonging)
        col_values = ws.col_values(sentinel_col)
        last_row = max(last_row, len(col_values))

    # 4. get the data
    table = ws.get(f"A1:Z{last_row}")      # gspread.values_get -> List[List[str]]
    
    tpu_information = {}
    for i, row in enumerate(table):
        if len(row) > 1 and (row[1].startswith('v') or row[1].startswith('kmh-tpuvm-')):
            assert len(row) >= 7, f"line {i+1} is too short: {row}"
            _, tpu, belong, running_status, user, user_note, script_note, env, other = row[:9]
            zone, pre, spot, full_name = get_zone_pre_spot(tpu)
            
            assert zone is not None, f"line {i+1} tpu {tpu} not found in zone"
            assert zone.startswith(env), f"line {i+1} zone {zone} does not start with env {env}, for tpu {tpu}"

            # print(running_status, user, user_note, script_note)
            assert running_status in ['running', 'reserved', 'reserved(error)', '闲的', '没了!'], f"line {i+1} running status {running_status} cannot be recognized"
            if running_status == '闲的':
                running_status = 'free'

            if running_status == 'reserved(error)':
                running_status = 'reserved'

            if user == '闲的':
                user = 'free'

            tpu_type, tpu_version = None, None
            for key in NAME_TO_VER:
                if key in full_name:
                    tpu_version = NAME_TO_VER[key]

            if full_name == 'kmh-tpuvm-v6e-spot-301': tpu_type = 'v6e-64'
            for key in NAME_TO_TYPE:
                if key in full_name:
                    tpu_type = NAME_TO_TYPE[key]

            assert (tpu_version is not None) and (tpu_type is not None), f"line {i+1} tpu {tpu} name cannot be recognized: {tpu_version}, {tpu_type}, {full_name}"

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

    # assert 'kmh-tpuvm-v6e-64-spot-keya-su30sn' in tpu_information, f"kmh-tpuvm-v6e-64-spot-keya-su30sn not found in tpu_information"

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

def get_tpu_usage_by_zone_and_type():
    """
    Use gcloud list command to get TPU usage statistics for each zone and type.
    Returns: dict with keys like 'v6(us-central1-b)' and values as the number of chips used.
    """
    import subprocess
    import re
    from .constants import PROJECT, ZONE_DICT, NAME_TO_TYPE
    
    usage_stats = {}
    
    # Get all zones
    all_zones = ZONE_DICT['all']
    
    for zone in all_zones:
        # List all TPUs in this zone (including creating state)
        # Use CSV format for easier parsing
        cmd = f"gcloud compute tpus tpu-vm list --zone={zone} --project={PROJECT} --format='csv(name,acceleratorType,state)'"
        try:
            result = subprocess.run(cmd, shell=True, timeout=60, check=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"{WARNING} Failed to list TPUs in zone {zone}: {result.stderr}")
                continue
            
            lines = result.stdout.strip().splitlines()
            if len(lines) < 2:  # Header + no data
                continue
            
            # Parse CSV output (skip header line)
            import csv
            from io import StringIO
            
            csv_reader = csv.reader(StringIO('\n'.join(lines[1:])))
            for row in csv_reader:
                if len(row) < 3:
                    continue
                
                tpu_name = row[0].strip()
                acc_type = row[1].strip()
                state = row[2].strip()
                
                # Only count READY and CREATING states
                if state.upper() not in ['READY', 'CREATING']:
                    continue
                
                # Extract TPU version from accelerator type
                # Examples: v6e-64 -> v6, v5p-256 -> v5p, v5litepod-16 -> v5e
                tpu_version = None
                if 'v6e' in acc_type.lower() or (acc_type.lower().startswith('v6')):
                    tpu_version = 'v6'
                elif 'v5p' in acc_type.lower():
                    tpu_version = 'v5p'
                elif 'v5litepod' in acc_type.lower() or 'v5e' in acc_type.lower():
                    tpu_version = 'v5e'
                elif acc_type.lower().startswith('v5'):
                    tpu_version = 'v5'
                elif acc_type.lower().startswith('v4'):
                    tpu_version = 'v4'
                elif acc_type.lower().startswith('v3'):
                    tpu_version = 'v3'
                elif acc_type.lower().startswith('v2'):
                    tpu_version = 'v2'
                
                if tpu_version is None:
                    continue
                
                # Extract chip count from accelerator type (e.g., v6e-64 -> 64)
                chip_count = 0
                try:
                    # Try to extract number from accelerator type
                    numbers = re.findall(r'\d+', acc_type)
                    if numbers:
                        chip_count = int(numbers[-1])  # Take the last number (usually the chip count)
                except:
                    pass
                
                if chip_count == 0:
                    continue
                
                # Create key: v6(us-central1-b)
                key = f"{tpu_version}({zone})"
                
                if key not in usage_stats:
                    usage_stats[key] = 0
                usage_stats[key] += chip_count
                
        except subprocess.TimeoutExpired:
            print(f"{WARNING} Timeout listing TPUs in zone {zone}")
            continue
        except Exception as e:
            print(f"{WARNING} Error listing TPUs in zone {zone}: {e}")
            continue
    
    return usage_stats

def write_tpu_usage_to_sheet(usage_stats):
    """
    Write TPU usage statistics to K and L columns starting from row 6.
    K column: type (e.g., v6(us-central1-b))
    L column: chip count (e.g., 1024)
    """
    secret_path = SECRET_PATH
    sheet_id = "1MFtgLx7uzBFdiPxrIqck00ilrSslZU2w2jRwriVpKMw"
    sheet_name = "ka[experimental]"
    
    # 1. authenticate
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(secret_path, scopes=scopes)
    client = gspread.authorize(creds)
    
    # 2. open the sheet
    ws = client.open_by_key(sheet_id).worksheet(sheet_name)
    
    # 3. Prepare data - sort by type name for consistency
    sorted_stats = sorted(usage_stats.items())
    
    # 4. Clear existing data in K and L columns from row 6 onwards
    # First, find how many rows to clear (use a reasonable number, e.g., 100)
    max_rows = max(100, len(sorted_stats) + 10)
    ws.batch_clear([f"K6:L{max_rows}"])
    
    # 5. Write new data starting from row 6
    if sorted_stats:
        data_to_write = [[key, str(value)] for key, value in sorted_stats]
        print(f"data_to_write: {data_to_write}")
        ws.update(f"K6:L{5 + len(data_to_write)}", data_to_write, value_input_option='USER_ENTERED')
        print(f"{INFO} write_tpu_usage_to_sheet: Updated {len(sorted_stats)} TPU usage statistics in K and L columns")
    else:
        print(f"{WARNING} write_tpu_usage_to_sheet: No TPU usage statistics to write")
    
    return True

def read_tpu_total_counts_from_sheet():
    """
    Read TPU total counts from K and L columns starting from row 6.
    K column format: v6(asia-northeast1-b) or v5p(us-east5-a)
    L column: count (number of chips)
    Returns: dict with structure {version: {zone: total_cards}}
    Example: {'v6': {'asia-northeast1-b': 128}, 'v5': {'us-east5-a': 64}}
    """
    import re
    from .data_io import read_data
    
    secret_path = SECRET_PATH
    sheet_id = "1MFtgLx7uzBFdiPxrIqck00ilrSslZU2w2jRwriVpKMw"
    sheet_name = "ka[experimental]"
    
    # 1. authenticate
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(secret_path, scopes=scopes)
    client = gspread.authorize(creds)
    
    # 2. open the sheet
    ws = client.open_by_key(sheet_id).worksheet(sheet_name)
    
    # 3. Read K and L columns from row 6 onwards
    # Read up to row 200 to be safe
    max_row = 200
    k_col_values = ws.col_values(11)  # Column K is index 11 (1-based)
    l_col_values = ws.col_values(12)  # Column L is index 12 (1-based)
    
    counts = {}  # {version: {zone: total_cards}}
    
    # Start from row 6 (index 5 in 0-based, since col_values[0] = row 1)
    # Process up to the minimum of: max_row, length of K column, length of L column
    end_row = min(max_row, len(k_col_values), len(l_col_values))
    for i in range(5, end_row):
        # Get values (with safe indexing)
        k_value = (k_col_values[i].strip() if i < len(k_col_values) else '').strip()
        l_value = (l_col_values[i].strip() if i < len(l_col_values) else '').strip()
        
        # Skip empty rows
        if not k_value or not l_value:
            continue
        
        # Parse K column: v6(asia-northeast1-b) or v5p(us-east5-a)
        # Pattern: v<num>[p|e]?(zone)
        match = re.match(r'v(\d+)([pe]?)\s*\(([^)]+)\)', k_value)
        if not match:
            continue
        
        version_num = match.group(1)
        version_suffix = match.group(2)  # 'p' or 'e' or ''
        zone = match.group(3).strip()
        
        # Map to version: v4, v5, v6
        # v5p, v5e -> v5, v4 -> v4, v6e, v6 -> v6
        if version_num == '4':
            version = 'v4'
        elif version_num == '5':
            version = 'v5'  # v5p and v5e both map to v5
        elif version_num == '6':
            version = 'v6'  # v6e maps to v6
        else:
            continue  # Skip unknown versions
        
        # Parse L column: count
        try:
            count = int(float(l_value))  # Handle both int and float strings
        except (ValueError, TypeError):
            continue
        
        # Initialize structure
        if version not in counts:
            counts[version] = {}
        if zone not in counts[version]:
            counts[version][zone] = 0
        
        counts[version][zone] += count
    
    return counts



