import json, os
from .helpers import *
from .constants import *

def get_passwords():
    """
    Read passwords from the authentication file autentication.json, which is located in the same directory as this script.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    auth_file_path = os.path.join(current_dir, 'authentication.json')
    if not os.path.exists(auth_file_path):
        print(f"{FAIL} Authentication file not found at {auth_file_path}")
        return []
    with open(auth_file_path, 'r') as file:
        try:
            passwords = json.load(file)
            if not isinstance(passwords, list):
                print(f"{FAIL} Authentication file is not a valid JSON array.")
                return []
            return passwords
        except json.JSONDecodeError:
            print(f"{FAIL} Authentication file is not a valid JSON.")
            return []
    
def autenticate(command):
    print(f'Begin autentication for command {command}...')
    passwords = get_passwords()
    if not passwords:
        print(f"{FAIL} No passwords found in authentication file.")
        return False
    username = input('Enter username: ')
    password = input('Enter password: ')
    for entry in passwords:
        if entry['user'] == username and entry['password'] == password:
            print(f"{GOOD} Authentication successful for user {username}.")
            return entry['priority']
    print(f"{FAIL} Authentication failed for user {username}.")
    return 0
    