#!/usr/bin/env python3
"""
Create a tmux session called "try_queue" with multiple windows,
each running a gcloud command from ab.ab file.
"""

import subprocess
import os
import sys
import time
def run_tmux_command(cmd):
    """Run a tmux command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"Warning: Command failed: {cmd}")
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command '{cmd}': {e}")
        return False

def read_commands(file_path):
    """Read commands from ab.ab file."""
    commands = []
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        sys.exit(1)
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                commands.append(line)
    
    return commands

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    commands_file = os.path.join(script_dir, 'ab.ab')
    
    # Read commands from file
    commands = read_commands(commands_file)
    print(f"Found {len(commands)} commands to execute")
    
    session_name = "try_queue"
    
    # Check if session already exists
    check_cmd = f"tmux has-session -t {session_name} 2>/dev/null"
    result = subprocess.run(check_cmd, shell=True, capture_output=True)
    if result.returncode == 0:
        print(f"Session '{session_name}' already exists. Killing it first...")
        run_tmux_command(f"tmux kill-session -t {session_name}")
    
    # Create new session with first command in window 0
    if not commands:
        print("No commands found!")
        sys.exit(1)
    
    # Create new session (window 0 will be created automatically)
    create_cmd = f'tmux new-session -d -s {session_name}'
    print(f"Creating tmux session '{session_name}'...")
    
    if not run_tmux_command(create_cmd):
        print("Failed to create tmux session!")
        sys.exit(1)
    
    # Rename window 0
    run_tmux_command(f'tmux rename-window -t {session_name}:0 "window-0"')
    
    # Run first command in window 0
    # Use -l flag to send literal string, avoiding shell interpretation
    subprocess.run(['tmux', 'send-keys', '-l', '-t', f'{session_name}:0', commands[0]], check=False)
    subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}:0', 'Enter'], check=False)
    print(f"Window 0: Running command 1")
    
    # Create additional windows for remaining commands
    for i, cmd in enumerate(commands[1:], start=1):
        window_name = f"window-{i}"
        
        # Create new window
        new_window_cmd = f'tmux new-window -t {session_name} -n "{window_name}"'
        if not run_tmux_command(new_window_cmd):
            print(f"Failed to create window {i}")
            continue
        
        time.sleep(5)
        # Send command to the window using -l flag for literal string
        subprocess.run(['tmux', 'send-keys', '-l', '-t', f'{session_name}:{window_name}', cmd], check=False)
        subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}:{window_name}', 'Enter'], check=False)
        print(f"Window {i}: Running command {i+1}")
    
    print(f"\n✓ Successfully created tmux session '{session_name}' with {len(commands)} windows")
    print(f"To attach to the session, run: tmux attach -t {session_name}")
    print(f"To list windows: tmux list-windows -t {session_name}")

if __name__ == "__main__":
    main()

