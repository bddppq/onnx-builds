#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname "$script_path")
TEST_DIR="$top_dir/test"

python "$TEST_DIR/test_operators.py" -v
python "$TEST_DIR/test_models.py" -v
python "$TEST_DIR/test_caffe2.py" -v
python "$TEST_DIR/test_verify.py" -v
python "$TEST_DIR/test_pytorch_helper.py" -v
