#!/bin/bash

# This script helps developers set up the ONNX and Caffe2 develop environment on devgpu.
# It creates an virtualenv instance, and installs all the dependencies in this environment.
# The script will creates a folder called onnx-dev folder under the $HOME directory.
# onnx, pytorch and caffe2 are installed as submodules in $HOME/onnx-dev/onnx-fb-universe/repos.
# Please source $HOME/onnx-dev/.onnx_env_init to initialize the development before starting developing.


# TODO: support python 3.

# Set script configuration
set -e
shopt -s expand_aliases

# Proxy setup
alias with_proxy="HTTPS_PROXY=http://fwdproxy.any:8080 HTTP_PROXY=http://fwdproxy.any:8080 FTP_PROXY=http://fwdproxy.any:8080 https_proxy=http://fwdproxy.any:8080 http_proxy=http://fwdproxy.any:8080 ftp_proxy=http://fwdproxy.any:8080 http_no_proxy='*.facebook.com|*.tfbnw.net|*.fb.com'"

# Set the variables
CYAN='\033[0;36m'
NC='\033[0m'
onnx_root="$HOME/onnx-dev"   # I think hardcoding the onnx root dir is fine, just like fbsource
venv="$onnx_root/onnxvenv"
onnx_init_file="$onnx_root/.onnx_env_init"
ccache_root="$onnx_root/ccache"
ccache_script="$onnx_root/ccache_install.sh"

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

# Checking to see if CuDNN is present, and install it if not exists
if [ -f /usr/local/cuda/include/cudnn.h ]; then
  echo "CuDNN header already exists!!"
else
  sudo cp -R /home/engshare/third-party2/cudnn/6.0.21/src/cuda/include/* /usr/local/cuda/include/
  sudo cp -R /home/engshare/third-party2/cudnn/6.0.21/src/cuda/lib64/* /usr/local/cuda/lib64/
fi

# TODO set the specific version for each package
# Install the dependencies for Caffe2
sudo yum install python-virtualenv freetype-devel libpng-devel glog gflags protobuf protobuf-devel protobuf-compiler -y
rpm -q protobuf  # check the version and if necessary update the value below
protoc --version  # check protoc
protoc_path=$(which protoc)
if [[ "$protoc_path" != "/bin/protoc" ]]; then
  echo "Warning: Non-default protoc is detected, the script may not work with non-default protobuf!!!"
  echo "Please try to remove the protoc at $protoc_path and rerun this script."
  exit 1
fi

# Upgrade Cmake to the right version (>3.0)
sudo yum remove cmake3 -y
sudo yum install cmake -y

# Install the dependencies for CCache
sudo yum install autoconf asciidoc -y

# Create the root folder
if [ -e "$onnx_root" ]; then
  timestamp=$(date "+%Y.%m.%d-%H.%M.%S")
  mv --backup=t "$onnx_root" "$onnx_root"."$timestamp"
fi
mkdir -p "$onnx_root"

# Set the name of virtualenv instance
with_proxy virtualenv "$venv"

# Creating a script that can be sourced in the future for the environmental variable
touch "$onnx_init_file"
{
  echo -e "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\x24LD_LIBRARY_PATH";
  echo -e "export PATH=$ccache_root/lib:/usr/local/cuda/bin:\x24PATH";
  echo "source $venv/bin/activate";
  echo 'alias with_proxy="HTTPS_PROXY=http://fwdproxy.any:8080 HTTP_PROXY=http://fwdproxy.any:8080 FTP_PROXY=http://fwdproxy.any:8080 https_proxy=http://fwdproxy.any:8080 http_proxy=http://fwdproxy.any:8080 ftp_proxy=http://fwdproxy.any:8080 http_no_proxy='"'"'*.facebook.com|*.tfbnw.net|*.fb.com'"'"'"'
} >> "$onnx_init_file"
chmod u+x "$onnx_init_file"

# Installing CCache
cd "$onnx_root"
with_proxy wget https://raw.githubusercontent.com/onnxbot/onnx-fb-universe/master/scripts/ccache_setup.sh -O "$ccache_script"
chmod u+x "$ccache_script"
"$ccache_script" --path "$ccache_root"

# Test nvcc with CCache
own_ccache=true
if [ -f "$CUDA_NVCC_EXECUTABLE" ] && [[ "$ccache_root/cuda/nvcc" == "$CUDA_NVCC_EXECUTABLE" ]]; then
  "$CUDA_NVCC_EXECUTABLE" --version
  if [ $? == 0 ]; then
    own_ccache=false
  fi
fi
if $own_ccache; then
  echo "export CUDA_NVCC_EXECUTABLE=$ccache_root/cuda/nvcc" >> "$onnx_init_file"
fi

# Loading env vars
source "$onnx_init_file"

"$CUDA_NVCC_EXECUTABLE" --version

# Create a virtualenv, activate it, upgrade pip
if [ -f "$HOME/.pip/pip.conf"]; then
  echo "Warning: $HOME/.pip/pip.conf is detected, pip install may fail!"
fi
# shellcheck disable=SC1090
with_proxy python -m pip install -U pip setuptools
with_proxy python -m pip install future numpy "protobuf>3.2" pytest-runner pyyaml typing ipython

# Cloning repos
cd "$onnx_root"
with_proxy git clone https://github.com/onnxbot/onnx-fb-universe --recursive

# Build ONNX
cd "$onnx_root/onnx-fb-universe/repos/onnx"
with_proxy python setup.py develop

# Build PyTorch
cd "$onnx_root/onnx-fb-universe/repos/pytorch"
with_proxy pip install -r "requirements.txt"
with_proxy python setup.py build develop

# Build Caffe2
set +e
cd "$onnx_root/onnx-fb-universe/repos/caffe2"
with_proxy python setup.py develop

ninja_path=$(which ninja)
if [[ ! -z $ninja_path ]]; then
  echo "Warning: ninja is installed at $ninja_path, which may cause Caffe2 building issue!!!"
  echo "Please try to remove the ninja at $ninja_path and rerun this script."
fi
set -e

# Sanity checks and useful info
python -c 'from caffe2.python import build; from pprint import pprint; pprint(build.build_options)'
python -c 'from caffe2.python import core, workspace; print("GPUs found: " + str(workspace.NumCudaDevices()))'
python -c "import onnx"

echo "Congrats, you are ready to rock!!"
echo "################ Please run the following command before development ################"
echo -e "${CYAN}source $onnx_init_file${NC}"
echo "#####################################################################################"
