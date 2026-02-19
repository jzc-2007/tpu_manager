from .helpers import *
from .constants import *
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from .users import user_from_dict
from .operate import mount_disk

import os, yaml
import gspread
from google.oauth2.service_account import Credentials



def get_monitor_config():
    data = read_data()
    for key, value in data['monitor_config'].items():
        print(f"{key}: {value}")

def set_monitor_config(args):
    data = read_and_lock_data()
    try:
        key, value = args[0], args[1]
        if key not in data['monitor_config']:
            raise ValueError(f"Monitor config {key} not found")
        if is_integer(value):
            value = int(value)
        elif is_boolean(value):
            value = to_boolean(value)
        data['monitor_config'][key] = value
        print(f"Set {key} to {value}")
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to set monitor config")
        release_lock_data()

def explain_tpu_aliases():
    data = read_data()
    for alias, name in data['tpu_aliases'].items():
        print(f"{alias}: {name}")

def add_tpu_alias(alias, name):
    data = read_and_lock_data()
    try:
        data['tpu_aliases'][alias] = name
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to add tpu alias")
        release_lock_data()

def register_tpu():
    tpu_alias, spreadsheet_name, full_name, zone, pre = None, None, None, None, False
    try:
        tpu_alias = input("Enter TPU alias(e.g. v4-32-py): ")
        spreadsheet_name = input("Enter spreadsheet name(e.g. v4-32-preemptible-yiyang): ")
        full_name = input("Enter full name(e.g. kmh-tpuvm-v4-32-preemptible-yiyang): ")
        zone = input("Enter zone(e.g. us-central1-a): ")
        pre = input("Is it preemptible? (yes/no): ").strip().lower() == 'yes'
        spot = input("Is it spot? (yes/no): ").strip().lower() == 'yes'
    except Exception as e:
        print(f"{FAIL} Failed to read input: {e}")
        return
    data = read_and_lock_data()
    try:
        if tpu_alias in data['tpu_aliases']:
            raise ValueError(f"TPU alias {tpu_alias} already exists")
        data['tpu_aliases'][tpu_alias] = full_name
        data['tpu_aliases'][spreadsheet_name] = full_name
        if zone not in data['all_tpus']:
            data['all_tpus'][zone] = []
            print(f"{WARNING} Zone {zone} not found, creating new zone entry")
        data['all_tpus'][zone].append(full_name)
        if pre:
            data['pre_info']['preemptible'].append(full_name)
        if spot:
            data['pre_info']['spot'].append(full_name)
        write_and_unlock_data(data)
        print(f"{GOOD} Successfully registered TPU {tpu_alias} with full name {full_name}")
    except Exception as e:
        print(f"{FAIL} Failed to register TPU: {e}")
        release_lock_data()

def register_tpu_and_write_spreadsheet(full_name, zone, pre=False, spot = True, tpu_alias = None, spreadsheet_name = None):
    if tpu_alias is None:
        tpu_alias = full_name
    if spreadsheet_name is None:
        spreadsheet_name = full_name
    assert full_name.startswith('kmh-tpuvm-'), f"Full name {full_name} does not start with 'kmh-tpuvm-'"
    data = read_and_lock_data()
    try:
        if tpu_alias in data['tpu_aliases']:
            raise ValueError(f"TPU alias {tpu_alias} already exists")
        data['tpu_aliases'][tpu_alias] = full_name
        data['tpu_aliases'][spreadsheet_name] = full_name
        if zone not in data['all_tpus']:
            data['all_tpus'][zone] = []
            print(f"{WARNING} Zone {zone} not found, creating new zone entry")
        data['all_tpus'][zone].append(full_name)
        if pre:
            data['pre_info']['preemptible'].append(full_name)
        if spot:
            data['pre_info']['spot'].append(full_name)
        write_and_unlock_data(data)
        print(f"{GOOD} Successfully registered TPU {tpu_alias} with full name {full_name}")
        
        # Write to spreadsheet
        try:
            secret_path = SECRET_PATH
            sheet_id = "1MFtgLx7uzBFdiPxrIqck00ilrSslZU2w2jRwriVpKMw"
            sheet_name = "ka[experimental]"
            
            # Authenticate
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_file(secret_path, scopes=scopes)
            client = gspread.authorize(creds)
            
            # Open the sheet
            ws = client.open_by_key(sheet_id).worksheet(sheet_name)
            
            # Find the last row of the TPU table.
            # IMPORTANT:
            # - We define last_row as the last row that has data in column B/C/D/E.
            # - We ignore the top 10 rows (headers), so last_row is at least 10.
            # This avoids being affected by unrelated sections (e.g. K/L usage stats).
            last_row = 10
            for sentinel_col in range(2, 6):  # COL B, C, D, E
                col_values = ws.col_values(sentinel_col)
                last_row = max(last_row, len(col_values))
            
            # Determine TPU version and type from full_name
            tpu_version = None
            tpu_type = None
            for key in NAME_TO_VER:
                if key in full_name:
                    tpu_version = NAME_TO_VER[key]
                    break
            for key in NAME_TO_TYPE:
                if key in full_name:
                    tpu_type = NAME_TO_TYPE[key]
                    break
            
            # If we can't determine type from full_name, try to extract from spreadsheet_name
            if tpu_type is None and spreadsheet_name:
                for key in NAME_TO_TYPE:
                    if key in spreadsheet_name:
                        tpu_type = NAME_TO_TYPE[key]
                        break
            
            # Prepare the row data: [empty, tpu_alias, belong, running_status, user, user_note, script_note, env, other_note]
            # Column A: empty (or can be left empty)
            # Column B: spreadsheet_name (TPU alias)
            # Column C: belong (default to empty string or 'ka' - using empty for now)
            # Column D: running_status ('free' -> will be written as '闲的')
            # Column E: user ('free' -> will be written as '闲的')
            # Column F: user_note (empty)
            # Column G: script_note ('READY')
            # Column H: env (zone)
            # Column I: other_note (empty)
            
            new_row = [
                '',  # Column A
                spreadsheet_name,  # Column B: TPU alias
                'unknown',  # Column C: belong (can be filled later)
                '闲的',  # Column D: running_status (free)
                '闲的',  # Column E: user (free)
                '.',  # Column F: user_note
                'READY',  # Column G: script_note
                zone,  # Column H: env (zone)
                '.'  # Column I: other_note
            ]
            
            # Write exactly one line under the current last line of the table (A..I).
            target_row = last_row + 1
            ws.update(
                f"A{target_row}:I{target_row}",
                [new_row],
                value_input_option='USER_ENTERED'
            )
            print(f"{GOOD} Successfully added TPU {spreadsheet_name} to spreadsheet at row {target_row}")
        except Exception as e:
            print(f"{WARNING} Failed to write to spreadsheet: {e}")
            # Don't fail the whole function if spreadsheet write fails
        
    except Exception as e:
        print(f"{FAIL} Failed to register TPU: {e}")
        release_lock_data()
    

def fang_new_tpu(new_tpu_name, old_tpu_alias):
    data = read_and_lock_data()
    try:
        # first check if the new tpu name already exists
        if new_tpu_name in data['tpu_aliases'].values():
            # look for all aliases that map to new_tpu_name
            all_aliases_for_new_name = [alias for alias, name in data['tpu_aliases'].items() if name == new_tpu_name]
            raise ValueError(f"New TPU name {new_tpu_name} already exists, with aliases: {all_aliases_for_new_name}")
        if old_tpu_alias not in data['tpu_aliases']:
            raise ValueError(f"Old TPU alias {old_tpu_alias} not found")
        old_full_name = data['tpu_aliases'][old_tpu_alias]
        zone = None
        for z in data['all_tpus']:
            if old_full_name in data['all_tpus'][z]:
                zone = z
                break
        if zone is None:
            raise ValueError(f"Old TPU full name {old_full_name} not found in any zone")
        # replace the four appear of old_full_name with new_tpu_name: 2 alias, 1 in zone, 1 in spot
        all_aliases = [alias for alias in list(data['tpu_aliases'].keys()) if data['tpu_aliases'][alias] == old_full_name]
        assert len(all_aliases) in [1,2], f"found more than 2 aliases for the old TPU. FUCK YOU! please modify manually. get aliases: {all_aliases}"
        for alias_ in all_aliases:
            data['tpu_aliases'][alias_] = new_tpu_name
        data['all_tpus'][zone].remove(old_full_name)
        data['all_tpus'][zone].append(new_tpu_name)
        if old_full_name in data['pre_info']['spot']:
            data['pre_info']['spot'].remove(old_full_name)
            data['pre_info']['spot'].append(new_tpu_name)
        else: raise NotADirectoryError(f'FUCK YOU')
        write_and_unlock_data(data)
        print(f"{GOOD} Successfully fanged new TPU {new_tpu_name} from old TPU alias {old_tpu_alias}")
    except Exception as e:
        print(f"{FAIL} Failed to fang new TPU: {e}")
        release_lock_data()

def fang_new_tpu_and_mount_disk(new_tpu_name, old_tpu_alias):
    fang_new_tpu(new_tpu_name, old_tpu_alias)
    mount_disk(new_tpu_name)

def del_registered_tpu(alias):
    data = read_and_lock_data()
    try:
        if alias not in data['tpu_aliases']:
            raise ValueError(f"TPU alias {alias} not found")
        full_name = data['tpu_aliases'][alias]
        all_aliases = [alias for alias in list(data['tpu_aliases'].keys()) if data['tpu_aliases'][alias] == full_name]
        for alias_ in all_aliases:
            del data['tpu_aliases'][alias_]
        for zone in data['all_tpus']:
            if full_name in data['all_tpus'][zone]:
                data['all_tpus'][zone].remove(full_name)
        write_and_unlock_data(data)
        print(f"{GOOD} Successfully deleted TPU alias {alias}")
    except Exception as e:
        print(f"{FAIL} Failed to delete TPU alias: {e}")
        release_lock_data()

def get_settings(user_object):
    for key, value in user_object.settings.items():
        print(f"{key}: {value}")

def set_settings(user_object, args):
    data = read_and_lock_data()
    try:
        key, value = args[0], args[1]
        if key not in user_object.settings:
            raise ValueError(f"Setting {key} not found")
        if is_integer(value):
            value = int(value)
        elif is_boolean(value):
            value = to_boolean(value)
        else:
            raise ValueError(f"Value {value} is not a valid type")
        user_object.settings[key] = value
        data['users'][user_object.name]['settings'] = user_object.settings
        print(f"Set {key} to {value}")
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to set settings")
        release_lock_data()

def add_config_alias(user_object, args):
    data = read_and_lock_data()
    try:
        alias, command = args[0], args[1]
        user_object.add_config_alias(alias, command)
        data['users'][user_object.name] = user_object.to_dict()
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to add config alias")
        release_lock_data()

def show_config_alias(user_object):
    for alias, command in user_object.config_aliases.items():
        print(f"{alias}: {command}")

def del_config_alias(user_object, args):
    data = read_and_lock_data()
    user_object = user_from_dict(data['users'][user_object.name])
    try:
        alias = args[0]
        user_object.del_config_alias(alias)
        data['users'][user_object.name] = user_object.to_dict()
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to delete config alias")
        release_lock_data()

def read_user_logs(user_object, args):
    for msg in user_object.logs:
        print(msg)

def clear_user_logs(user_object):
    data = read_and_lock_data()
    try:
        user_object.logs = []
        data['users'][user_object.name]['logs'] = user_object.logs
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to clear user logs")
        release_lock_data()

def get_wandb_notes(dir):
    # read the config file in dir/configs/remote_run_config.yaml
    # and check whether there's a key called wandb_notes
    config_path = os.path.join(dir, 'configs', 'remote_run_config.yml')
    if not os.path.exists(config_path):
        print(f"{FAIL} Config file not found at {config_path}")
        return None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'wandb_notes' in config:
        return config['wandb_notes']
    elif 'logging' in config:
        if 'wandb_notes' in config['logging']:
            return config['logging']['wandb_notes']
        else:
            print(f"{WARNING} wandb_notes not found in logging config file")
            return None
    else:
        print(f"{WARNING} wandb_notes not found in config file")
        return None