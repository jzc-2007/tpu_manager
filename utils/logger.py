from .helpers import *
from .constants import *
from .data_io import (
    read_and_lock_data,
    write_and_unlock_data,
    release_lock_data,
    read_data,
)
from .users import user_from_dict
from .operate import mount_disk

import os, yaml
import gspread
from google.oauth2.service_account import Credentials
import re
import subprocess
import sys

WRAP_MASTER_PATH = "/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/wrap_master.py"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(s):
    return ANSI_RE.sub("", s or "")


def get_tpu_zone_from_wrap_master(tpu_name, cache=True):
    """
    Resolve TPU zone from wrap_master.py output.
    If wrap_master has no output, this lookup is considered unavailable.
    """
    if not os.path.exists(WRAP_MASTER_PATH):
        raise FileNotFoundError(f"wrap_master.py not found: {WRAP_MASTER_PATH}")

    cache_flag = "true" if cache else "false"
    try:
        res = subprocess.run(
            [sys.executable, WRAP_MASTER_PATH, "--cache", cache_flag],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError("wrap_master.py timed out")

    output = res.stdout or ""
    if not output.strip():
        raise ValueError("wrap_master.py has no output, quick register cannot be used")

    clean_output = _strip_ansi(output)
    pattern = re.compile(rf"{re.escape(tpu_name)}\s*\((?P<zone>[a-z0-9-]+)\)")
    zones = []
    for line in clean_output.splitlines():
        m = pattern.search(line)
        if m:
            zones.append(m.group("zone"))

    unique_zones = list(dict.fromkeys(zones))
    if len(unique_zones) == 0:
        raise ValueError(f"TPU {tpu_name} not found in wrap_master.py output")
    if len(unique_zones) > 1:
        raise ValueError(
            f"TPU {tpu_name} matched multiple zones in wrap_master.py output: {unique_zones}"
        )

    return unique_zones[0]


def get_monitor_config():
    data = read_data()
    for key, value in data["monitor_config"].items():
        print(f"{key}: {value}")


def set_monitor_config(args):
    data = read_and_lock_data()
    try:
        key, value = args[0], args[1]
        if key not in data["monitor_config"]:
            raise ValueError(f"Monitor config {key} not found")
        if is_integer(value):
            value = int(value)
        elif is_boolean(value):
            value = to_boolean(value)
        data["monitor_config"][key] = value
        print(f"Set {key} to {value}")
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to set monitor config")
        release_lock_data()


def explain_tpu_aliases():
    data = read_data()
    for alias, name in data["tpu_aliases"].items():
        print(f"{alias}: {name}")


def add_tpu_alias(alias, name):
    data = read_and_lock_data()
    try:
        data["tpu_aliases"][alias] = name
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to add tpu alias")
        release_lock_data()


def register_tpu():
    tpu_alias, spreadsheet_name, full_name, zone, pre = None, None, None, None, False
    try:
        tpu_alias = input("Enter TPU alias(e.g. v4-32-py): ")
        spreadsheet_name = input(
            "Enter spreadsheet name(e.g. v4-32-preemptible-yiyang): "
        )
        full_name = input("Enter full name(e.g. kmh-tpuvm-v4-32-preemptible-yiyang): ")
        zone = input("Enter zone(e.g. us-central1-a): ")
        pre = input("Is it preemptible? (yes/no): ").strip().lower() == "yes"
        spot = input("Is it spot? (yes/no): ").strip().lower() == "yes"
    except Exception as e:
        print(f"{FAIL} Failed to read input: {e}")
        return
    data = read_and_lock_data()
    try:
        if tpu_alias in data["tpu_aliases"]:
            raise ValueError(f"TPU alias {tpu_alias} already exists")
        if full_name in data["tpu_aliases"].values():
            raise ValueError(f"TPU full name {full_name} already exists")
        data["tpu_aliases"][tpu_alias] = full_name
        data["tpu_aliases"][spreadsheet_name] = full_name
        if zone not in data["all_tpus"]:
            data["all_tpus"][zone] = []
            print(f"{WARNING} Zone {zone} not found, creating new zone entry")
        data["all_tpus"][zone].append(full_name)
        if pre:
            data["pre_info"]["preemptible"].append(full_name)
        if spot:
            data["pre_info"]["spot"].append(full_name)
        write_and_unlock_data(data)
        print(
            f"{GOOD} Successfully registered TPU {tpu_alias} with full name {full_name}"
        )
    except Exception as e:
        print(f"{FAIL} Failed to register TPU: {e}")
        release_lock_data()


def register_tpu_quick(full_name, cache=True):
    """
    Quick register by TPU full name.
    Zone is discovered from wrap_master.py output.
    """
    full_name = (full_name or "").strip()
    if full_name == "":
        print(f"{FAIL} quick-register: TPU name is empty")
        return
    if not full_name.startswith("kmh-tpuvm-"):
        print(
            f"{FAIL} quick-register: please pass full TPU name starting with kmh-tpuvm-"
        )
        return

    data = read_data()
    existing_aliases = [
        alias
        for alias, mapped in data.get("tpu_aliases", {}).items()
        if mapped == full_name
    ]
    if existing_aliases:
        print(
            f"{WARNING} quick-register: TPU {full_name} already registered with aliases: {existing_aliases}"
        )
        return

    try:
        zone = get_tpu_zone_from_wrap_master(full_name, cache=cache)
        print(f"{INFO} quick-register: detected zone {zone} for TPU {full_name}")
    except Exception as e:
        print(f"{FAIL} quick-register: {e}")
        return

    # Keep defaults aligned with existing non-interactive register path.
    try:
        register_tpu_and_write_spreadsheet(
            full_name=full_name,
            zone=zone,
            pre=False,
            spot=True,
            tpu_alias=full_name,
            spreadsheet_name=full_name,
        )
    except Exception as e:
        print(f"{FAIL} quick-register: register failed: {e}")


def register_tpu_and_write_spreadsheet(
    full_name, zone, pre=False, spot=True, tpu_alias=None, spreadsheet_name=None
):
    if tpu_alias is None:
        tpu_alias = full_name
    if spreadsheet_name is None:
        spreadsheet_name = full_name
    assert full_name.startswith(
        "kmh-tpuvm-"
    ), f"Full name {full_name} does not start with 'kmh-tpuvm-'"
    data = read_and_lock_data()
    try:
        if tpu_alias in data["tpu_aliases"]:
            raise ValueError(f"TPU alias {tpu_alias} already exists")
        data["tpu_aliases"][tpu_alias] = full_name
        data["tpu_aliases"][spreadsheet_name] = full_name
        if zone not in data["all_tpus"]:
            data["all_tpus"][zone] = []
            print(f"{WARNING} Zone {zone} not found, creating new zone entry")
        data["all_tpus"][zone].append(full_name)
        if pre:
            data["pre_info"]["preemptible"].append(full_name)
        if spot:
            data["pre_info"]["spot"].append(full_name)
        write_and_unlock_data(data)
        print(
            f"{GOOD} Successfully registered TPU {tpu_alias} with full name {full_name}"
        )

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
                "",  # Column A
                spreadsheet_name,  # Column B: TPU alias
                "unknown",  # Column C: belong (can be filled later)
                "闲的",  # Column D: running_status (free)
                "闲的",  # Column E: user (free)
                ".",  # Column F: user_note
                "READY",  # Column G: script_note
                zone,  # Column H: env (zone)
                ".",  # Column I: other_note
            ]

            # Write exactly one line under the current last line of the table (A..I).
            target_row = last_row + 1
            ws.update(
                f"A{target_row}:I{target_row}",
                [new_row],
                value_input_option="USER_ENTERED",
            )
            print(
                f"{GOOD} Successfully added TPU {spreadsheet_name} to spreadsheet at row {target_row}"
            )
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
        if new_tpu_name in data["tpu_aliases"].values():
            # look for all aliases that map to new_tpu_name
            all_aliases_for_new_name = [
                alias
                for alias, name in data["tpu_aliases"].items()
                if name == new_tpu_name
            ]
            raise ValueError(
                f"New TPU name {new_tpu_name} already exists, with aliases: {all_aliases_for_new_name}"
            )
        if old_tpu_alias not in data["tpu_aliases"]:
            raise ValueError(f"Old TPU alias {old_tpu_alias} not found")
        old_full_name = data["tpu_aliases"][old_tpu_alias]
        zone = None
        for z in data["all_tpus"]:
            if old_full_name in data["all_tpus"][z]:
                zone = z
                break
        if zone is None:
            raise ValueError(f"Old TPU full name {old_full_name} not found in any zone")
        # replace the four appear of old_full_name with new_tpu_name: 2 alias, 1 in zone, 1 in spot
        all_aliases = [
            alias
            for alias in list(data["tpu_aliases"].keys())
            if data["tpu_aliases"][alias] == old_full_name
        ]
        assert len(all_aliases) in [
            1,
            2,
        ], f"found more than 2 aliases for the old TPU. FUCK YOU! please modify manually. get aliases: {all_aliases}"
        for alias_ in all_aliases:
            data["tpu_aliases"][alias_] = new_tpu_name
        data["all_tpus"][zone].remove(old_full_name)
        data["all_tpus"][zone].append(new_tpu_name)
        if old_full_name in data["pre_info"]["spot"]:
            data["pre_info"]["spot"].remove(old_full_name)
            data["pre_info"]["spot"].append(new_tpu_name)
        else:
            raise NotADirectoryError(f"FUCK YOU")
        write_and_unlock_data(data)
        print(
            f"{GOOD} Successfully fanged new TPU {new_tpu_name} from old TPU alias {old_tpu_alias}"
        )
    except Exception as e:
        print(f"{FAIL} Failed to fang new TPU: {e}")
        release_lock_data()


def fang_new_tpu_and_mount_disk(user_name, new_tpu_name, old_tpu_alias):
    # check if the new tpu name is reserved by others
    reserved_user = check_reserved_user(new_tpu_name)
    if reserved_user is not None and reserved_user != user_name:
        print(f"{WARNING} TPU {new_tpu_name} is already reserved by {reserved_user}")
        return "failed"
    print(
        f"{INFO} fang_new_tpu_and_mount_disk: TPU {new_tpu_name} is not reserved by others."
    )

    fang_new_tpu(new_tpu_name, old_tpu_alias)

    ret = zhan(user_name, new_tpu_name)
    if ret != "success":
        return "failed"

    mount_disk(new_tpu_name)


def del_registered_tpu(alias):
    data = read_and_lock_data()
    try:
        if alias not in data["tpu_aliases"]:
            raise ValueError(f"TPU alias {alias} not found")
        full_name = data["tpu_aliases"][alias]
        all_aliases = [
            alias
            for alias in list(data["tpu_aliases"].keys())
            if data["tpu_aliases"][alias] == full_name
        ]
        for alias_ in all_aliases:
            del data["tpu_aliases"][alias_]
        for zone in data["all_tpus"]:
            if full_name in data["all_tpus"][zone]:
                data["all_tpus"][zone].remove(full_name)
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
        data["users"][user_object.name]["settings"] = user_object.settings
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
        data["users"][user_object.name] = user_object.to_dict()
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to add config alias")
        release_lock_data()


def show_config_alias(user_object):
    for alias, command in user_object.config_aliases.items():
        print(f"{alias}: {command}")


def del_config_alias(user_object, args):
    data = read_and_lock_data()
    user_object = user_from_dict(data["users"][user_object.name])
    try:
        alias = args[0]
        user_object.del_config_alias(alias)
        data["users"][user_object.name] = user_object.to_dict()
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
        data["users"][user_object.name]["logs"] = user_object.logs
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to clear user logs")
        release_lock_data()


def get_wandb_notes(dir):
    # read the config file in dir/configs/remote_run_config.yaml
    # and check whether there's a key called wandb_notes
    config_path = os.path.join(dir, "configs", "remote_run_config.yml")
    if not os.path.exists(config_path):
        print(f"{FAIL} Config file not found at {config_path}")
        return None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # check if config has key 'finetune: True'
    if "finetune" in config and config["finetune"] == True:
        config_path = os.path.join(dir, "configs", "finetune_config.yml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "wandb_notes" in config:
            return config["wandb_notes"]
        elif "logging" in config:
            if "wandb_notes" in config["logging"]:
                return config["logging"]["wandb_notes"]
            else:
                print(f"{WARNING} wandb_notes not found in logging config file")
                return None
        else:
            print(f"{WARNING} wandb_notes not found in finetune config file")
            return None

    if "wandb_notes" in config:
        return config["wandb_notes"]
    elif "logging" in config:
        if "wandb_notes" in config["logging"]:
            return config["logging"]["wandb_notes"]
        else:
            print(f"{WARNING} wandb_notes not found in logging config file")
            return None
    else:
        print(f"{WARNING} wandb_notes not found in config file")
        return None


def _parse_lock_filename(filename):
    """
    解析锁文件名 {user}_{vm_name}_{YYYY-MM-DD_HH-MM-SS}。
    返回 (user, vm_name, time_str) 或 None（格式不对时）。
    """
    parts = filename.split("_")
    if len(parts) < 4:
        return None
    # 时间为最后两段: YYYY-MM-DD, HH-MM-SS
    time_str = f"{parts[-2]}_{parts[-1]}"
    user = parts[0]
    vm_name = "_".join(parts[1:-2])
    return user, vm_name, time_str


def zhan(user, vm_name):
    """
    write a file under /kmh-nfs-ssd-us-mount/code/qiao/tpu_lock,
    with name {USER_VMNAME_TIME}. Time format: YYYY-MM-DD_HH-MM-SS (no space, underscores).
    also support when the vm_name is an alias.
    """
    if not vm_name.startswith("kmh-tpuvm-"):
        # this is an alias.
        data = read_data()
        vm_name = data["tpu_aliases"][vm_name]

    # first check if the vm_name is already reserved by others
    if (
        check_reserved_user(vm_name) is not None
        and check_reserved_user(vm_name) != user
    ):
        print(
            f"{FAIL} TPU {vm_name} is already reserved by {check_reserved_user(vm_name)}"
        )
        return "failed"

    time_str = get_lock_time_str()
    user_vm_name = f"{user}_{vm_name}_{time_str}"
    lock_dir = "/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock"
    with open(f"{lock_dir}/{user_vm_name}", "w") as f:
        f.write(f"{user}_{vm_name}_{time_str}")
    print(f"{GOOD} 成功创建了叫做{user_vm_name}的占卡锁")
    return "success"


def check_reserved_user(tpu):
    """
    Check the path /kmh-nfs-ssd-us-mount/code/qiao/tpu_lock, and see whether there is a file
    named {USER_VMNAME_TIME} within 30 minutes. If a file is older than 30 minutes, delete it.
    Return the reserved user name, otherwise return None.
    Time format: YYYY-MM-DD_HH-MM-SS (no space, underscores).
    """
    now = get_lock_time_str()
    lock_dir = "/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock"
    for file in os.listdir(lock_dir):
        parsed = _parse_lock_filename(file)
        if parsed is None:
            continue
        user, vm_name, time_str = parsed
        if vm_name != tpu:
            continue
        try:
            seconds_ago = lock_time_seconds_between(time_str, now)
        except (ValueError, TypeError):
            # 旧格式或非法时间，视为过期并删除
            try:
                os.remove(f"{lock_dir}/{file}")
            except OSError:
                pass
            continue
        if seconds_ago > 30 * 60:
            try:
                os.remove(f"{lock_dir}/{file}")
            except OSError:
                pass
            continue
        return user
    return None


def remove_file_lock(vm_name):
    """
    删除占卡锁文件，释放占卡。支持别名。
    """
    if not vm_name.startswith("kmh-tpuvm-"):
        # this is an alias.
        data = read_data()
        vm_name = data["tpu_aliases"][vm_name]

    lock_dir = "/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock"
    for file in os.listdir(lock_dir):
        parsed = _parse_lock_filename(file)
        if parsed is None:
            continue
        file_user, file_vm_name, _ = parsed
        if file_vm_name == vm_name:
            try:
                os.remove(f"{lock_dir}/{file}")
                print(f"{GOOD} 成功删除占卡锁 {file}，释放了 {vm_name}")
                return "success"
            except OSError as e:
                print(f"{FAIL} 删除占卡锁 {file} 失败: {e}")
                return "failed"
    print(f"{WARNING} 没有找到 {vm_name} 的锁文件")
    return "failed"
