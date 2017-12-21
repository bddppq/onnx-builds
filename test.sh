#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname "$script_path")
TEST_DIR="$top_dir/test"

if hash catchsegv 2>/dev/null; then
    PYTHON="catchsegv python"
else
    PYTHON="python"
fi

test_files=(
    "$TEST_DIR/test_operators.py"
    "$TEST_DIR/test_models.py"
    "$TEST_DIR/test_caffe2.py"
    "$TEST_DIR/test_verify.py"
    "$TEST_DIR/test_pytorch_helper.py"
)

if hash parallel 2>/dev/null; then
    parallel -j 2 -t --line-buffer --keep-order $PYTHON {} -v ::: "${test_files[@]}"
else
    for f in "${test_files[@]}"; do
        $PYTHON $f -v
    done
fi
