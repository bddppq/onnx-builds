#!/bin/bash

set -ex
shopt -s expand_aliases

# Checking to see if CuDNN is present
if [ -f /usr/local/cuda/include/cudnn.h ]; then
  echo "Cudnn header already exists!!"

else 
  sudo cp -R /home/engshare/third-party2/cudnn/6.0.21/src/cuda/include/* /usr/local/cuda/include/
  sudo cp -R /home/engshare/third-party2/cudnn/6.0.21/src/cuda/lib64/* /usr/local/cuda/lib64/
fi

# make sure we find CUDA and that nvcc is runnable
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version

#Install Caffe2 requirements from the internal yum repo
sudo yum install python-virtualenv freetype-devel libpng-devel glog gflags protobuf protobuf-devel protobuf-compiler -y

#Installing cmake
sudo yum remove cmake3 -y
sudo yum install cmake -y
sudo yum install autoconf -y

#Proxy setup
alias with_proxy="HTTPS_PROXY=http://fwdproxy.any:8080 HTTP_PROXY=http://fwdproxy.any:8080 FTP_PROXY=http://fwdproxy.any:8080 https_proxy=http://fwdproxy.any:8080 http_proxy=http://fwdproxy.any:8080 ftp_proxy=http://fwdproxy.any:8080 http_no_proxy='\''*.facebook.com|*.tfbnw.net|*.fb.com'\'"

#Create a virtualenv, activate it, upgrade pip
cd ~
with_proxy virtualenv venv
source venv/bin/activate
with_proxy pip install pip setuptools -U

#Install other Caffe2 requirements
rpm -q protobuf # check the version and if necessary update the value below
# Todo - Add Grep to find protobuf version
with_proxy pip install future numpy protobuf==2.6.1 ninja pytest-runner

#Installing CCache
mkdir -p ~/ccache
pushd /tmp
rm -rf ccache
with_proxy git clone https://github.com/colesbury/ccache.git -b ccbin
pushd ccache
./autogen.sh
./configure
sudo yum install asciidoc -y
make install prefix=~/ccache
popd && popd

mkdir -p ~/ccache/lib
mkdir -p ~/ccache/cuda
ln -sf ~/ccache/bin/ccache ~/ccache/lib/cc
ln -sf ~/ccache/bin/ccache ~/ccache/lib/c++
ln -sf ~/ccache/bin/ccache ~/ccache/lib/gcc
ln -sf ~/ccache/bin/ccache ~/ccache/lib/g++
ln -sf ~/ccache/bin/ccache ~/ccache/cuda/nvcc
~/ccache/bin/ccache -M 25Gi

export PATH=~/ccache/lib:/usr/local/cuda/bin:$PATH
export CUDA_NVCC_EXECUTABLE=~/ccache/cuda/nvcc

# Make sure the nvcc wrapped in CCache is runnable
~/ccache/cuda/nvcc --version
#Cloning repos
with_proxy git clone https://github.com/onnxbot/onnx-fb-universe --recursive

#Installing packages with install script
cd onnx-fb-universe
with_proxy ./install-develop.sh

#Sanity Checks
# python -c 'from caffe2.python import core, workspace; print("GPUs found: " + str(workspace.NumCudaDevices()))'
# python -c "import onnx"

echo "Congrats, you are ready to rock!!"

