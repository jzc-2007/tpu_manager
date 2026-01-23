#!/usr/bin/env python3
import sys
import utils.data_io as data_io

def main():
    if len(sys.argv) != 2:
        print("usage: get_log_dir.py <id>", file=sys.stderr)
        sys.exit(1)

    target_id = sys.argv[1]

    data = data_io.read_data()
    jobs = data["users"]["sqa"]["job_data"]

    for j in jobs:
        if str(j.get("windows_id")) == target_id:
            print(j["log_dir"])
            return

    print("not found", file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
