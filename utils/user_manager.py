import os, json, datetime, subprocess
from .helpers import BASE_DIR, safe_input
from .terminal_ui import info, warning, error, success, log

USER_LOG_DIR = os.path.join(BASE_DIR, "user_logs.json")

def get_shell_id():
    """
    Get the shell ID by executing a shell command.
    """
    tty = os.ttyname(0).replace('/dev/', '')
    pid = str(os.getppid())
    t = subprocess.run(['ps', '-p', pid, '-o', 'lstart='], capture_output=True, text=True).stdout.strip().replace(' ', '_')
    # print([tty, t])
    return f"{tty}____{t}"

def read_log():
    """
    Read the user log file and return its content.
    """
    if not os.path.exists(USER_LOG_DIR):
        return {}
    
    with open(USER_LOG_DIR, 'r') as file:
        return json.load(file)

def write_log(data):
    """
    Write the given data to the user log file.
    """
    with open(USER_LOG_DIR, 'w') as file:
        json.dump(data, file, indent=4)

def clean_log():
    try:
        d = read_log()
        d_new = {}
        for k in d.keys():
            date = k.split('____')[-1]
            # Tue_Jul_29_20:32:54_2025
            date_obj = datetime.datetime.strptime(date, "%a_%b_%d_%H:%M:%S_%Y")
            if date_obj < datetime.datetime.now() - datetime.timedelta(days=10):
                continue
            d_new[k] = d[k]
        write_log(d_new)
        info('Log file cleaned')
    except ValueError:
        raise RuntimeError("Failed to clean log file. Contact ZHH")

def login():
    clean_log()
    d = read_log()
    shell_id = get_shell_id()
    if shell_id in d:
        # already log in
        user = d[shell_id]
        info(f'Welcome back, {user}! To log out, use the command "tpu logout".')
    else:
        user = safe_input(f'You are not logged in. Please log in. Enter your user name: ').strip()
        d[shell_id] = user
        write_log(d)
        success(f"You have logged in as {user}.")
    return user

def logout():
    d = read_log()
    shell_id = get_shell_id()
    if shell_id not in d:
        raise RuntimeError("The login session does not exist. Contact ZHH")

def need_login(func):
    """
    Decorator to ensure the user is logged in before executing the function.
    """
    def wrapper(*args, **kwargs):
        user = login()
        return func(*args, **kwargs, user=user)
    return wrapper