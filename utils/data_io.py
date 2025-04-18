from .helpers import DATA_PATH, LOCK_FILE
import json, time

def read_data():
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    return data

def write_data(data):
    with open(DATA_PATH, 'w') as file:
        json.dump(data, file, indent=4)

def read_and_lock_data():
    while True:
        with open(LOCK_FILE, 'r') as file:
            lock = json.load(file)
        if lock['data'] == False:
            lock['data'] = True
            with open(LOCK_FILE, 'w') as file:
                json.dump(lock, file, indent=4)
            break
        else:
            time.sleep(5)
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