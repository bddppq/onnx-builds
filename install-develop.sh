#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname "$script_path")
REPOS_DIR="$top_dir/repos"

pip install ninja

pip install -r "$REPOS_DIR/pytorch/caffe2/requirements.txt"
pip install -r "$REPOS_DIR/pytorch/requirements.txt"

# Install caffe2
cd "$REPOS_DIR/pytorch" && python setup_caffe2.py develop && cd -

# Install onnx
pip install -e "$REPOS_DIR/onnx"

# Install pytorch
cd "$REPOS_DIR/pytorch" && python setup.py build develop && cd -
