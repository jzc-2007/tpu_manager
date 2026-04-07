import fcntl, json, os, tempfile, time
from .constants import *


def _mutate_lock_file(mutator):
    with open(LOCK_PATH, "r+") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            lock = json.load(file)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse lock file {LOCK_PATH}: {e}") from e
        result = mutator(lock)
        file.seek(0)
        json.dump(lock, file, indent=4)
        file.truncate()
        file.flush()
        os.fsync(file.fileno())
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    return result


def _try_acquire_lock(lock_type, username=None):
    def _mut(lock):
        if lock[lock_type]["status"] == True:
            return False
        lock[lock_type]["status"] = True
        lock[lock_type]["user"] = username
        return True

    return _mutate_lock_file(_mut)


def _clear_lock(lock_type):
    def _mut(lock):
        was_locked = lock[lock_type]["status"]
        lock[lock_type]["status"] = False
        lock[lock_type]["user"] = None
        return was_locked

    return _mutate_lock_file(_mut)


def _atomic_write_json(path, payload):
    dir_path = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=dir_path,
    )
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(payload, file, indent=4)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def release_lock(args):
    assert len(args) == 1, "Please specify a lock type to release"
    lock_type = args[0]
    assert lock_type in [
        "code",
        "data",
        "queue",
        "legacy",
        "apply",
    ], f"Unknown lock type {lock_type}"
    print(f"{INFO} release_lock: releasing {lock_type} lock")
    was_locked = _clear_lock(lock_type)
    if was_locked == False:
        print(f"{WARNING} release_lock: the {lock_type} lock is not locked.")
        return


def lock(args):
    assert len(args) == 1, "Please specify a lock type to lock"
    lock_type = args[0]
    assert lock_type in [
        "code",
        "data",
        "queue",
        "legacy",
        "apply",
    ], f"Unknown lock type {lock_type}"
    print(f"{INFO} lock: locking {lock_type}")
    if _try_acquire_lock(lock_type) == False:
        print(f"{FAIL} lock: the {lock_type} is locked now.")
        raise Exception("Lock not released.")


def lock_code(username=None):
    if _try_acquire_lock("code", username) == False:
        print(f"{FAIL} lock_code: the code is locked now.")
        raise Exception("Lock not released.")
    print(f"{INFO} lock_code: code locked by {username}.")


def unlock_code(username=None):
    result = {"status": None, "owner": None}

    def _mut(lock):
        if lock["code"]["status"] == False:
            result["status"] = "not_locked"
            return
        owner = lock["code"]["user"]
        if username and owner != username and owner is not None:
            result["status"] = "owner_mismatch"
            result["owner"] = owner
            return
        lock["code"]["status"] = False
        lock["code"]["user"] = None
        result["status"] = "released"

    _mutate_lock_file(_mut)

    if result["status"] == "not_locked":
        print(f"{WARNING} unlock_code: the code is not locked.")
        return
    if result["status"] == "owner_mismatch":
        print(
            f"{FAIL} unlock_code: the code is locked by {result['owner']}, you cannot unlock it."
        )
        print(f"If you believe this is a mistake, please contact {result['owner']}.")
        return
    print(f"{INFO} unlock_code: code unlocked by {username}.")


def check_code_lock():
    with open(LOCK_PATH, "r") as file:
        lock = json.load(file)
    return lock["code"]["status"]


def read_data():
    try:
        with open(DATA_PATH, "r") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(
            f"{FAIL} read_data: JSON parsing error in {DATA_PATH} at line {e.lineno}, column {e.colno}: {e.msg}"
        )
        print(f"Error details: {e}")
        raise
    return data


def read_queue():
    with open(QUEUE_PATH, "r") as file:
        queue = json.load(file)
    return queue


def read_legacy():
    with open(LEGACY_PATH, "r") as file:
        legacy = json.load(file)
    return legacy


def write_legacy(legacy):
    if len(legacy) > MAX_LEGACY_LENGTH:
        print(f"{WARNING} write_legacy: Legacy list is too long, truncating it.")
        legacy = legacy[: MAX_LEGACY_LENGTH // 2]
    _atomic_write_json(LEGACY_PATH, legacy)


def read_and_lock_legacy():
    num_ack = 0
    while True:
        print(num_ack)
        num_ack += 1
        if _try_acquire_lock("legacy"):
            break
        else:
            time.sleep(10)
        print(num_ack)
        if num_ack > 180:
            print(
                f"{FAIL} read_and_lock_legacy: Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually."
            )
            raise Exception(
                "Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually."
            )
    try:
        with open(LEGACY_PATH, "r") as file:
            legacy = json.load(file)
    except Exception as e:
        print(f"Error reading file: {e}")
        legacy = []
    return legacy


def release_lock_legacy():
    _clear_lock("legacy")


def write_and_unlock_legacy(legacy):
    if len(legacy) > MAX_LEGACY_LENGTH:
        print(
            f"{WARNING} write_and_unlock_legacy: Legacy list is too long, truncating it."
        )
        legacy = legacy[: MAX_LEGACY_LENGTH // 2]

    try:
        _atomic_write_json(LEGACY_PATH, legacy)
    except Exception as e:
        print(f"Error writing file: {e}")
    _clear_lock("legacy")


def write_data(data):
    _atomic_write_json(DATA_PATH, data)


def lock_data():
    lock(["data"])


def read_and_lock_data():
    num_ack = 0
    while True:
        num_ack += 1
        if _try_acquire_lock("data"):
            break
        else:
            time.sleep(10)
        if num_ack > 180:
            print(
                f"{FAIL} read_and_lock_data: Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually."
            )
            raise Exception(
                "Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually."
            )
    try:
        with open(DATA_PATH, "r") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(
            f"{FAIL} read_and_lock_data: JSON parsing error in {DATA_PATH} at line {e.lineno}, column {e.colno}: {e.msg}"
        )
        print(f"Error details: {e}")
        raise
    return data


def read_and_lock_queue():
    num_ack = 0
    while True:
        num_ack += 1
        if _try_acquire_lock("queue"):
            break
        else:
            time.sleep(10)
        if num_ack > 180:
            print(
                f"{FAIL} read_and_lock_queue: Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually."
            )
            raise Exception(
                "Lock not released after 30 mins, this may indicate a deadlock. Please check the lock file and release it manually."
            )
    with open(QUEUE_PATH, "r") as file:
        queue = json.load(file)
    return queue


def write_and_unlock_data(data):
    try:
        # Use a unique temp file to avoid collisions between concurrent writers.
        _atomic_write_json(DATA_PATH, data)
    except Exception as e:
        print(f"{FAIL} write_and_unlock_data: Failed to write data: {e}")
        raise
    _clear_lock("data")


def release_lock_data():
    _clear_lock("data")


def release_lock_queue():
    _clear_lock("queue")


def write_and_unlock_queue(queue):
    _atomic_write_json(QUEUE_PATH, queue)
    _clear_lock("queue")
