#!/bin/bash

# This script installs CCache with CUDA support.
# Example usage:
#     ./ccache_setup.sh --path /installed/folder

set -e
shopt -s expand_aliases

# Setup the proxy
alias with_proxy="HTTPS_PROXY=http://fwdproxy.any:8080 HTTP_PROXY=http://fwdproxy.any:8080 FTP_PROXY=http://fwdproxy.any:8080 https_proxy=http://fwdproxy.any:8080 http_proxy=http://fwdproxy.any:8080 ftp_proxy=http://fwdproxy.any:8080 http_no_proxy='*.facebook.com|*.tfbnw.net|*.fb.com'"

# Parse options
path="$HOME/ccache"
force=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)
            shift
            path="$1"
            if [[ "$path" != /* ]]; then
              # Turn relative path to absolute path
              path="$(pwd)/$path"
            fi
            ;;
        --force)  # Force install
            force=true
            ;;
        --help)  # Force install
            echo 'usage: ./ccache_setup.py --path /installed/folder [--force]'
            exit 0
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
    shift
done

# Check whether you put nvcc in PATH
set +e
nvcc_path=$(which nvcc)
if [[ -z $nvcc_path ]]; then
  nvcc_path="/usr/local/cuda/bin/nvcc"
fi
set -e
if [ ! -f "$nvcc_path" ] && ! $force; then
  echo "nvcc is not detected in $PATH"
  exit 1
fi
echo "nvcc is detected at $nvcc_path"

if [[ "$CUDA_NVCC_EXECUTABLE" == *"ccache"* ]]; then  # Heuristic rule
  if ! $force ; then
    echo "CCache with nvcc support is already installed at $CUDA_NVCC_EXECUTABLE, please add --force"
    exit 0
  fi
fi

# Installing CCache
echo "CCache will be installed at $path"
if [ -d "$path" ]; then
    mv --backup=t "$path" "${path}.old"
fi

with_proxy git clone https://github.com/colesbury/ccache.git $path -b ccbin
cd "$path"
./autogen.sh
./configure
make install prefix="$path"

mkdir -p "$path/lib"
mkdir -p "$path/cuda"
ln -sf "$path/bin/ccache" "$path/lib/cc"
ln -sf "$path/bin/ccache" "$path/lib/c++"
ln -sf "$path/bin/ccache" "$path/lib/gcc"
ln -sf "$path/bin/ccache" "$path/lib/g++"
ln -sf "$path/bin/ccache" "$path/cuda/nvcc"
"$path/bin/ccache" -M 25Gi

export PATH="$path/lib:/usr/local/cuda/bin:$PATH"
export CUDA_NVCC_EXECUTABLE="$path/cuda/nvcc"

# Make sure the nvcc wrapped in CCache is runnable
"$path/cuda/nvcc" --version
echo "Congrats! The CCache with nvcc support is installed!"
echo -e "Please add the following lines to your bash init script:\n"
echo "################ Env Var for CCache with CUDA support ################"
echo -e "export PATH=\x22${path}/lib:\x24PATH\x22"
echo -e "export CUDA_NVCC_EXECUTABLE=\x22${path}/cuda/nvcc\x22"
echo "######################################################################"
