def explain(cmd):
    if cmd == 'run':
        print(f"Run a job in dir<dir> on tpu with additional configs.")
        print(f"Usage: tpu run <tpu> <dir=1> <user/id=?> <additional-configs=None> <tag=None>")
    elif cmd == 'monitor':
        print(f"Monitor all the jobs of user.")
        print(f"Usage: tpu monitor <user/id=?>")
    elif cmd == 'get_attach_name' or cmd == 'gan':
        print(f"Get the attach name of the job on tpu.")
        print(f"Usage: tpu get_attach_name(gan) <tpu> <user/id=?>")
    elif cmd == 'set-dir':
        print(f"Set the current directory of user to the working directory<number>.")
        print(f"Usage: tpu set-dir <dir> <user/id=?>")
    elif cmd == 'set-cur':
        print(f"Set the current directory of user to the working directory<number>.")
        print(f"Usage: tpu set-cur <number> <user/id=?>")
    elif cmd == 'get-settings':
        print(f"Get the settings of user.")
        print(f"Usage: tpu get-settings <user/id=?>")
    elif cmd == 'set-settings':
        print(f"Set the settings of user.")
        print(f"Usage: tpu set-settings <key> <value> <user/id=?>")
    elif cmd == 'get-dir':
        print(f"Get the directory of user.")
        print(f"Usage: tpu get-dir <number> <user/id=?>")
    elif cmd == 'check':
        print(f"Check the status of the job.")
        print(f"Usage: tpu check <user/id=?>")
    elif cmd == 'check-tmux':
        print(f"Check the tmux session.")
        print(f"Usage: tpu check-tmux <user/id=?>")
    elif cmd == 'ls':
        print(f"List all the directories of user.")
        print(f"Usage: tpu ls <user/id=?>")
    elif cmd == 'kill-window' or cmd == '-kw':
        print(f"Kill the window <window_num> of user.")
        print(f"Usage: tpu kill-window <window_num> <user/id=?>")
    elif cmd == 'add-config-alias' or cmd == '-a' or cmd == '-alias':
        print(f"Add a config alias.")
        print(f"Usage: tpu add-config-alias <alias> <config> <user/id=?>")
    elif cmd == 'show-config-alias' or cmd == '-sa':
        print(f"Show all the config aliases of user.")
        print(f"Usage: tpu show-config-alias <user/id=?>")
    elif cmd == 'del-config-alias':
        print(f"Delete a config alias.")
        print(f"Usage: tpu del-config-alias <alias> <user/id=?>")
    elif cmd == 'add-tag':
        print(f"Add a tag to the job.")
        print(f"Usage: tpu add-tag <job_window_id> <tag> <user/id=?>")
    elif cmd == 'list-users' or cmd == '-lu':
        print(f"List all the users.")
        print(f"Usage: tpu list-users")
    elif cmd == 'add-user':
        print(f"Add a user.")
        print(f"Usage: tpu add-user")
    elif cmd == 'del-user':
        print(f"Delete a user.")
        print(f"Usage: tpu del-user")
    elif cmd == 'check-tpu':
        print(f"Not implemented yet")
        print(f"Usage: tpu check-tpu")
    elif cmd == 'check-tmux':
        print(f"Check the tmux session.")
        print(f"Usage: tpu check-tmux <user/id=?>")
    elif cmd == 'add-tpu-alias' or cmd == '-ta':
        print(f"Add a tpu alias.")
        print(f"Usage: tpu add-tpu-alias <alias> <tpu>")
    elif cmd == 'list-tpu-alias' or cmd == '-lta':
        print(f"List all tpu alias.")
        print(f"Usage: tpu list-tpu-alias")
    elif cmd == 'list-dir':
        print(f"List all the directories of user.")
        print(f"Usage: tpu list-dir <user/id=?>")
    elif cmd == 'set-cur':
        print(f"Set the current directory of user to the working directory<number>.")
        print(f"Usage: tpu set-cur <number> <user/id=?>")
    elif cmd == 'check-tmux':
        print(f"Check the tmux session.")
        print(f"Usage: tpu check-tmux <user/id=?>")
    elif cmd == '-kw':
        print(f"Kill the window <window_num> of user.")
        print(f"Usage: tpu -kw <window_num> <user/id=?>")
    else:
        print(f"Command {cmd} not found")

def tldr():
    Usage = """
    Usage: 
    - tpu run <tpu> <dir=1> <user/id=?> <additional-configs=None> <tag=None>
    Run a job in dir<dir> on tpu with additional configs.

    - tpu monitor <user/id=?>
    Monitor all the jobs of user.
    e.g. tpu monitor xibo

    - tpu -a <alias> <config> <user/id=?>
    Add a config alias. 
    e.g. tpu -a lr config.training.learning_rate xibo

    - tpu -sa <user/id=?>
    Show all the config aliases of user.

    - tpu -ta <alias> <config>
    Add a tpu alias.

    - tpu -lta
    List all tpu alias.

    - tpu ls <user/id=?>
    List all the directories of user.

    - tpu set-cur <number> <user/id=?>
    Set the current directory of user to the working directory<number>.

    - tpu -kw <window_num> <user/id=?>
    Kill the window <window_num> of user.
    """
    print(Usage)