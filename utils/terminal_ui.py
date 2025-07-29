from .helpers import GOOD, INFO, WARNING, FAIL, LOG

def info(message):
    print(f"{INFO} {message}")

def warning(message):
    print(f"{WARNING} {message}")

def error(message):
    print(f"{FAIL} {message}")

def success(message):
    print(f"{GOOD} {message}")

def log(message):
    print(f"{LOG} {message}")