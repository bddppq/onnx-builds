#!/bin/bash

set -ex

UNKNOWN=()

# defaults
PARALLEL=0

while [[ $# -gt 0 ]]
do
    arg="$1"
    case $arg in
        -p|--parallel)
            PARALLEL=1
            shift # past argument
            ;;
        *) # unknown option
            UNKNOWN+=("$1") # save it in an array for later
            shift # past argument
            ;;
    esac
done
set -- "${UNKNOWN[@]}" # leave UNKNOWN

pip install pytest
if [[ $PARALLEL == 1 ]]; then
    pip install pytest-xdist
fi

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname "$script_path")
cd "$top_dir"

if hash catchsegv 2>/dev/null; then
    PYTEST="catchsegv pytest"
else
    PYTEST="pytest"
fi

if [[ $PARALLEL == 1 ]]; then
    $PYTEST -n 3
else
    $PYTEST
fi
