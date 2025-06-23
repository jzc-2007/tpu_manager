DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"

while true; do
    python "$DIR/MONITOR.py"
    sleep 5
done
