from .helpers import DATA_PATH
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data
import os, json, time
class User():
    def __init__(self, id, name, tmux_name = None):
        self.id = id
        self.name = name
        self.tmux_name = tmux_name if tmux_name else name
        self.working_dir = {}
        self.job_data = []
        self.config_aliases = {}
        self.windows_offset = 1
        self.logs = []
        self.settings = {
            "monitor_after_run": True,
            "monitor_upd_time": 5,
            "monitor_length": 500,
            "show_length": 200,
            "time_zone": "us"
        }
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "tmux_name": self.tmux_name,
            "working_dir": self.working_dir,
            "job_data": self.job_data,
            "config_aliases": self.config_aliases,
            "settings": self.settings,
            "windows_offset": self.windows_offset,
            "logs": self.logs
        }
    def add_alias(self, alias, command):
        self.aliases[alias] = command
    def add_config_alias(self, alias, command):
        self.config_aliases[alias] = command
    def del_config_alias(self, alias):
        if alias in self.config_aliases:
            del self.config_aliases[alias]
        else:
            raise ValueError(f"Alias {alias} not found")

def user_from_dict(config_dict):
    user = User(config_dict['id'], config_dict['name'], config_dict['tmux_name'])
    user.working_dir = config_dict['working_dir']
    user.config_aliases = config_dict['config_aliases']
    user.settings = config_dict['settings']
    user.job_data = config_dict['job_data']
    user.windows_offset = config_dict['windows_offset']
    user.logs = config_dict['logs']
    return user

def create_user():
    data = read_and_lock_data()
    try:
        name = input('Enter user name:')
        if name in data['user_list']:
            raise ValueError(f"User {name} already exists")
        tmux_name = input(f'Enter tmux name, empty for default={name}:')
        if tmux_name == '':
            tmux_name = name
        # Find a mininum id not in use
        id = 0
        while id in data['id_list']:
            id += 1
        user_id = input(f'Enter user id, empty for default={id}:')
        user_id = int(user_id) if user_id != '' else id
        timezone = input('Enter timezone(us/cn), empty for default=us:')
        if timezone == '':
            timezone = 'us'
        if timezone not in ['us', 'cn']:
            raise ValueError(f"Timezone {timezone} not supported")
        user = User(user_id, name, tmux_name)
        init_dir = input('Enter initial working directory(abs. path), empty for to be set later:')
        assert os.path.exists(init_dir) or init_dir == '', f"Directory {init_dir} does not exist"
        if init_dir != '':
            user.working_dir['1'] = init_dir
        
        user.settings['time_zone'] = timezone
        data['user_list'].append(name)
        data['id_list'].append(user_id)
        data['id_user_dict'][user_id] = name
        data['user_id_dict'][name] = user_id
        data['users'][name] = user.to_dict()
        write_and_unlock_data(data)
    except:
        release_lock_data()

    # kill the tmux session if it exists
    if tmux_name in os.popen('tmux list-sessions').read():
        print(f"Killing tmux session {tmux_name}")
        os.system(f"tmux kill-session -t {tmux_name}")
        time.sleep(1)
    # create the tmux session
    print(f"Creating tmux session {tmux_name}")
    os.system(f"tmux new-session -s {tmux_name} -d")

def del_user():
    data = read_and_lock_data()
    try:
        name = input('Enter user name:')
        if name not in data['user_list']:
            raise ValueError(f"User {name} not found")
        user = data['users'][name]
        del data['users'][name]
        data['user_list'].remove(name)
        data['id_list'].remove(user['id'])
        del data['id_user_dict'][str(user['id'])]
        del data['user_id_dict'][name]
        write_and_unlock_data(data)
    except:
        print(f"Error deleting user {name}: NOT FOUND")
        release_lock_data()

def list_users():
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    for id, user in data['id_user_dict'].items():
        print(f"{id}: {user}")