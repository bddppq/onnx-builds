#!/bin/bash

set -ex
shopt -s expand_aliases

# Checking to see if CuDNN is present
if [ -f /usr/local/cuda/include/cudnn.h ]; then
  echo "CuDNN header already exists!!"
else
  sudo cp -R /home/engshare/third-party2/cudnn/6.0.21/src/cuda/include/* /usr/local/cuda/include/
  sudo cp -R /home/engshare/third-party2/cudnn/6.0.21/src/cuda/lib64/* /usr/local/cuda/lib64/
fi

# make sure we find CUDA and that nvcc is runnable
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version

# Install Caffe2 requirements from the internal yum repo
sudo yum install python-virtualenv freetype-devel libpng-devel glog gflags protobuf protobuf-devel protobuf-compiler -y

# Installing cmake
sudo yum remove cmake3 -y
sudo yum install cmake -y
sudo yum install autoconf asciidoc -y

# Proxy setup
alias with_proxy="HTTPS_PROXY=http://fwdproxy.any:8080 HTTP_PROXY=http://fwdproxy.any:8080 FTP_PROXY=http://fwdproxy.any:8080 https_proxy=http://fwdproxy.any:8080 http_proxy=http://fwdproxy.any:8080 ftp_proxy=http://fwdproxy.any:8080 http_no_proxy='\''*.facebook.com|*.tfbnw.net|*.fb.com'\'"

# Set the name of virtualenv instance
onnxroot="$HOME/onnx-dev"

if [ -d "$onnxroot" ]; then
  timestamp=`date "+%Y.%m.%d-%H.%M.%S"`
  mv --backup=t "$onnxroot" "$onnxroot"."$timestamp"
fi
mkdir -p "$onnxroot"

venv="$onnxroot/onnxvenv"

# Create a virtualenv, activate it, upgrade pip
with_proxy virtualenv "$venv"
source "$venv/bin/activate"
with_proxy pip install pip setuptools -U

# Install other Caffe2 requirements
rpm -q protobuf # check the version and if necessary update the value below
# Todo - Add Grep to find protobuf version
with_proxy pip install future numpy "protobuf>3.2" ninja pytest-runner

# Installing CCache
ccache_root="$onnxroot/ccache"
rm -rf "$ccache_root"
cd "$onnxroot"
with_proxy git clone https://github.com/colesbury/ccache.git -b ccbin
cd "$ccache_root"
./autogen.sh
./configure
make install prefix="$ccache_root"

mkdir -p "$ccache_root/lib"
mkdir -p "$ccache_root/cuda"
ln -sf "$ccache_root/bin/ccache" "$ccache_root/lib/cc"
ln -sf "$ccache_root/bin/ccache" "$ccache_root/lib/c++"
ln -sf "$ccache_root/bin/ccache" "$ccache_root/lib/gcc"
ln -sf "$ccache_root/bin/ccache" "$ccache_root/lib/g++"
ln -sf "$ccache_root/bin/ccache" "$ccache_root/cuda/nvcc"
"$ccache_root"/bin/ccache -M 25Gi

export PATH="$ccache_root/lib:/usr/local/cuda/bin:$PATH"
export CUDA_NVCC_EXECUTABLE="$ccache_root/cuda/nvcc"

# Make sure the nvcc wrapped in CCache is runnable
"$ccache_root/cuda/nvcc" --version

# Checking to see if the environment variable script is present
onnx_init_file="$onnxroot/.onnx_env_init"
if [ -f "$onnx_init_file" ]; then
    echo "Environment variable script already exists!! Moving the old version to a backup in order to generate a new one"
    mv --backup=t "$onnx_init_file" "$onnx_init_file".old
fi

# Creating a script that can be sourced in the future for the environmental variable
touch "$onnx_init_file"
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> "$onnx_init_file"
echo "export PATH=~/ccache/lib:/usr/local/cuda/bin:$PATH" >> "$onnx_init_file"
echo "export CUDA_NVCC_EXECUTABLE=~/ccache/cuda/nvcc" >> "$onnx_init_file"
chmod u+x "$onnx_init_file"

# Cloning repos
cd "$onnxroot"
with_proxy git clone https://github.com/onnxbot/onnx-fb-universe --recursive

# Installing packages in develop mode with install script
cd onnx-fb-universe
with_proxy ./install-develop.sh

# Sanity Checks
python -c 'from caffe2.python import build; from pprint import pprint; pprint(build.build_options)'
python -c 'from caffe2.python import core, workspace; print("GPUs found: " + str(workspace.NumCudaDevices()))'
python -c "import onnx"

echo "Congrats, you are ready to rock!!"
echo "BTW, don't forget to source the environment variable script by calling \"source $onnx_init_file\"" 
