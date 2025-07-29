assert __name__ == "__main__"

from utils.user_manager import need_login, login, logout
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)

inherit_parser = argparse.ArgumentParser(add_help=False)
inherit_parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")

######################### COMMANDS #########################

#### 1. User-related

## These new commands are added by ZHH ##

# login
subparsers.add_parser("login", help="Login to the system", parents=[inherit_parser]).set_defaults(func=lambda args: login())

logout_parser = subparsers.add_parser("logout", help="Logout from the system", parents=[inherit_parser])

# add user
add_user_parser = subparsers.add_parser("add-user", help="Add user", parents=[inherit_parser])

#### 2. Job-related

# run
run_parser = subparsers.add_parser("run", help="Run a job", parents=[inherit_parser])

# monitor
monitor_parser = subparsers.add_parser("monitor", help="Monitor jobs", aliases=["check"], parents=[inherit_parser])

#### 3. TPU-related

# find
find_parser = subparsers.add_parser("find", help="Find TPU by type", parents=[inherit_parser])

# describe
describe_parser = subparsers.add_parser("describe", help="Checking environment for a TPU", parents=[inherit_parser])

# solve
solve_parser = subparsers.add_parser("solve", help="Solve environment for a TPU", parents=[inherit_parser])

#### 4. Directory-related

# ls
ls_parser = subparsers.add_parser("ls", help="List all working directories for the user", parents=[inherit_parser])

#### 5. Others

# alias
alias_parser = subparsers.add_parser("alias", aliases=['-a', '-alias'], help="Set an alias, such as alias lr learning_rate", parents=[inherit_parser])
alias_parser.add_argument('key', type=str, help="The key to set an alias for, such as lr", metavar='KEY')
alias_parser.add_argument('value', type=str, help="The value to set for the alias, such as learning_rate", metavar='VALUE')

######################### LOGIC #########################

args = parser.parse_args()
args.func(args)