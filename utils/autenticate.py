import json, os, hashlib
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
        if entry['user'] == username and entry['password'] == password_hash(password):
            print(f"{GOOD} Authentication successful for user {username}.")
            return entry['priority']
    print(f"{FAIL} Authentication failed for user {username}.")
    return 0

def add_user_password_priority(username, password, priority):
    passwords = get_passwords()
    for entry in passwords:
        if entry['user'] == username:
            print(f"{FAIL} User {username} already exists.")
            return False
    passwords.append({'user': username, 'password': password_hash(password), 'priority': priority})
    current_dir = os.path.dirname(os.path.abspath(__file__))
    auth_file_path = os.path.join(current_dir, 'authentication.json')
    with open(auth_file_path, 'w') as file:
        json.dump(passwords, file, indent=4)
    print(f"{GOOD} User {username} added successfully.")
    return True
    
def password_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()
