import json, time, os
from .constants import *

def release_lock(args):
    assert len(args) == 1, "Please specify a lock type to release"
    lock_type = args[0]
    if lock_type == 'code':
        print(f"{INFO} release_lock: releasing code lock")
        with open(LOCK_PATH, 'r') as file:
            lock = json.load(file)
        lock['code']['status'] = False
        lock['code']['user'] = None
        with open(LOCK_PATH, 'w') as file:
            json.dump(lock, file, indent=4)
        print(f"{INFO} release_lock: code lock released")
    elif lock_type == 'data':
        print(f"{INFO} release_lock: releasing data lock")
        with open(LOCK_PATH, 'r') as file:
            lock = json.load(file)
        lock['data']['status'] = False
        with open(LOCK_PATH, 'w') as file:
            json.dump(lock, file, indent=4)
        print(f"{INFO} release_lock: data lock released")
    else:
        print(f"{FAIL} release_lock: unknown lock type {lock_type}")
        raise ValueError(f"Unknown lock type {lock_type}")
    
def lock(args):
    assert len(args) == 1, "Please specify a lock type to lock"
    lock_type = args[0]
    if lock_type == 'code':
        print(f"{INFO} lock: locking code")
        with open(LOCK_PATH, 'r') as file:
            lock = json.load(file)
        if lock['code']['status'] == False:
            lock['code']['status'] = True
            with open(LOCK_PATH, 'w') as file:
                json.dump(lock, file, indent=4)
        else:
            print(f"{FAIL} lock: the code is locked now.")
            raise Exception("Lock not released.")
    elif lock_type == 'data':
        print(f"{INFO} lock: locking data")
        with open(LOCK_PATH, 'r') as file:
            lock = json.load(file)
        if lock['data']['status'] == False:
            lock['data']['status'] = True
            with open(LOCK_PATH, 'w') as file:
                json.dump(lock, file, indent=4)
        else:
            print(f"{FAIL} lock: the data is locked now.")
            raise Exception("Lock not released.")
    else:
        print(f"{FAIL} lock: unknown lock type {lock_type}")
        raise ValueError(f"Unknown lock type {lock_type}")

def lock_code(username = None):
    print(f"{INFO} lock_code: locking code for user {username}")
    with open(LOCK_PATH, 'r') as file:
        lock = json.load(file)
    if lock['code']['status'] == False:
        lock['code']['status'] = True
        if username:
            lock['code']['user'] = username
        with open(LOCK_PATH, 'w') as file:
            json.dump(lock, file, indent=4)
    else:
        print(f"{FAIL} lock_code: the code is locked now.")
        raise Exception("Lock not released.")
    
def unlock_code(username = None):
    with open(LOCK_PATH, 'r') as file:
        lock = json.load(file)
    if lock['code']['status'] == False:
        print(f"{WARNING} unlock_code: the code is not locked.")
        return
    if username and lock['code']['user'] != username and lock['code']['user'] is not None:
        print(f"{FAIL} unlock_code: the code is locked by {lock['code']['user']}, you cannot unlock it.")
        print(f"If you believe this is a mistake, please contact {lock['code']['user']}.")
        return
    lock['code']['status'] = False
    lock['code']['user'] = None
    with open(LOCK_PATH, 'w') as file:
        json.dump(lock, file, indent=4)
    print(f"{INFO} unlock_code: code unlocked by {username}.")

def check_code_lock():
    with open(LOCK_PATH, 'r') as file:
        lock = json.load(file)
    return lock['code']['status']

def read_data():
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    return data

def read_queue():
    with open(QUEUE_PATH, 'r') as file:
        queue = json.load(file)
    return queue

def write_data(data):
    with open(DATA_PATH, 'w') as file:
        json.dump(data, file, indent=4)

def lock_data():
    print(f"{INFO} lock_data: locking data")
    with open(LOCK_PATH, 'r') as file:
        lock = json.load(file)
    if lock['data']['status'] == False:
        lock['data']['status'] = True
        with open(LOCK_PATH, 'w') as file:
            json.dump(lock, file, indent=4)
    else:
        print(f"{FAIL} lock_data: the data is locked now.")
        raise Exception("Lock not released.")
    

def read_and_lock_data():
    num_ack = 0
    while True:
        num_ack += 1
        with open(LOCK_PATH, 'r') as file:
            lock = json.load(file)
        if lock['data']['status'] == False:
            lock['data']['status'] = True
            with open(LOCK_PATH, 'w') as file:
                json.dump(lock, file, indent=4)
            break
        else:
            time.sleep(10)
        if num_ack > 180:
            print(f"{FAIL} read_and_lock_data: Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually.")
            raise Exception("Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually.")
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    return data

def read_and_lock_queue():
    num_ack = 0
    while True:
        num_ack += 1
        with open(QUEUE_PATH, 'r') as file:
            lock = json.load(file)
        if lock['queue']['status'] == False:
            lock['queue']['status'] = True
            with open(LOCK_PATH, 'w') as file:
                json.dump(lock, file, indent=4)
            break
        else:
            time.sleep(10)
        if num_ack > 180:
            print(f"{FAIL} read_and_lock_queue: Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually.")
            raise Exception("Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually.")
    with open(QUEUE_PATH, 'r') as file:
        queue = json.load(file)
    return queue

def write_and_unlock_data(data):
    with open(DATA_PATH, 'w') as file:
        json.dump(data, file, indent=4)
    with open(LOCK_PATH, 'r') as file:
        lock = json.load(file)
    lock['data']['status'] = False
    with open(LOCK_PATH, 'w') as file:
        json.dump(lock, file, indent=4)

def release_lock_data():
    with open(LOCK_PATH, 'r') as file:
        lock = json.load(file)
    lock['data']['status'] = False
    with open(LOCK_PATH, 'w') as file:
        json.dump(lock, file, indent=4)

def write_and_unlock_queue(queue):
    with open(QUEUE_PATH, 'w') as file:
        json.dump(queue, file, indent=4)
    with open(LOCK_PATH, 'r') as file:
        lock = json.load(file)
    lock['queue']['status'] = False
    with open(LOCK_PATH, 'w') as file:
        json.dump(lock, file, indent=4)
