import json

from .constants import DATA_PATH, LOCK_PATH


def read_data():
    """Read the external xibo TPU job database."""
    with open(DATA_PATH, "r") as file:
        return json.load(file)


def check_code_lock():
    """Return whether the external xibo checkout is currently code-locked."""
    with open(LOCK_PATH, "r") as file:
        lock = json.load(file)
    return lock["code"]["status"]
