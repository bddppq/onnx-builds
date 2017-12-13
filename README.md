onnx-fb-builds
========

| Travis | Jenkins |
|--------|---------|
| [![Build Status](https://travis-ci.org/onnxbot/onnx-fb-universe.svg?branch=master)](https://travis-ci.org/onnxbot/onnx-fb-universe) | [![Build Status](https://ci.pytorch.org/jenkins/buildStatus/icon?job=onnx-fb-universe-master)](https://ci.pytorch.org/jenkins/job/onnx-fb-universe-master/) |

# Why Universe Repo?
* It integrates multiple repos, including onnx, onnx-caffe2, onnx-pytorch, pytorch, caffe2, to provide a convenient environment for developers.
* OnnxBot automatically updates the submodules in `repos` folder, runs the tests, and automatically detects the breaking changes from different repos.

# Installation
```shell
./install.sh    # or ./install-develop.sh to install the most of the repos in develop mode.
./test.sh
```
[Dockerfile](https://github.com/houseroad/dockerfiles/blob/master/onnx-docker/onnx-docker-gpu/Dockerfile) may be helpful for setting up the whole repo.

# Repo Structure
* `repos`: submodules
* `test`: test scripts
* `jenkins`: jenkins CI configuration
* `travis`: travis CI configuration

# Test
## Test Script
This repo contains several test scripts in `test` folder.
* `test_caffe2.py`: PyTorch ==> ONNX ==> Caffe2 end-to-end tests, in which the decimal precision is checked.
* `test_models.py`: PyTorch ==> ONNX model tests, check the output ONNX graph with expected files.
* `test_operators.py`: PyTorch ==> ONNX operator tests, check the output ONNX graph with expected files.
* `test_verify.py`: PyTorch ==> ONNX verification tests, check the exporting fails in certain conditions.

To update the `expected` files, please add flag `--accept` to the end of the command.

## Convert Test from PyTorch
Using `convert_pytorch_test.py`, we can easily turn some existing PyTorch testing cases to ONNX backend test cases. Just run:
```shell
python test/convert_pytorch_test.py
```

## Generate Test from PyTorch Operator
Script `test_operators.py` is also able to generate ONNX backend test cases based on operator tests. Just run:
```shell
python test/test_operators.py --onnx-test
```

## Check the Generated Tests
All the generated ONNX backend test data is stored in `repos/onnx/onnx/backend/test/data/model`.

But the generated test cases may not pass end-to-end tests, so we have another script `filter_generated_test.py` to check whether the cases are passed or not, and the failed data will be moved
to `test/failed/generated`.
`filter_generated_test.py` supports the following flags:
* `-v`: verbose
* `--delete`: delete failed test cases, by default, they are moved to `test/failed/generated`.
* `--no-expect`: do not generate expect files for each test, by default, expected files are generated in `test/expect` folder for debugging purpose.
