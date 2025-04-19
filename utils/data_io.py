from .helpers import DATA_PATH, LOCK_FILE
import json, time
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
PURPLE="\033[1;34m"
NC="\033[0m"
def read_data():
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    return data

def write_data(data):
    with open(DATA_PATH, 'w') as file:
        json.dump(data, file, indent=4)

def read_and_lock_data():
    num_ack = 0
    while True:
        num_ack += 1
        with open(LOCK_FILE, 'r') as file:
            lock = json.load(file)
        if lock['data'] == False:
            lock['data'] = True
            with open(LOCK_FILE, 'w') as file:
                json.dump(lock, file, indent=4)
            break
        else:
            time.sleep(5)
        if num_ack > 120:
            print(f"{RED}[ERROR]{NC} read_and_lock_data: Lock not released after 10 mins, this may indicate a deadlock. Please check the lock file and release it manually.")
            raise Exception("Lock not released after 10 mins, this may indicate a deadlock. Please check the lock file and release it manually.")
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    return data

def write_and_unlock_data(data):
    with open(DATA_PATH, 'w') as file:
        json.dump(data, file, indent=4)
    with open(LOCK_FILE, 'r') as file:
        lock = json.load(file)
    lock['data'] = False
    with open(LOCK_FILE, 'w') as file:
        json.dump(lock, file, indent=4)

def release_lock_data():
    with open(LOCK_FILE, 'r') as file:
        lock = json.load(file)
    lock['data'] = False
    with open(LOCK_FILE, 'w') as file:
        json.dump(lock, file, indent=4)