from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.autograd import Variable
from onnx import numpy_helper

import io
import onnx
import os
import shutil
import torch
import traceback

import test_pytorch_common
from common_nn import module_tests
from test_nn import new_module_tests


_generated_dir = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir, "repos", "onnx", "onnx",
    "backend", "test", "data", "generated")


# Take a test case (a dict) as input, return the test name.
def get_test_name(testcase):
    if "fullname" in testcase:
        return "test_" + testcase["fullname"]

    test_name = "test_" + testcase["constructor"].__name__
    if "desc" in testcase:
        test_name += "_" + testcase["desc"]
    return test_name


# Take a test case (a dict) as input, return the input for the module.
def gen_input(testcase):
    if "input_size" in testcase:
        return Variable(torch.randn(*testcase["input_size"]))
    elif "input_fn" in testcase:
        input = testcase["input_fn"]()
        if isinstance(input, Variable):
            return input
        return Variable(testcase["input_fn"]())


def gen_module(testcase):
    if "constructor_args" in testcase:
        args = testcase["constructor_args"]
        module = testcase["constructor"](*args)
        module.train(False)
        return module
    module = testcase["constructor"]()
    module.train(False)
    return module


def convert_tests(testcases, sets=1):
    print("Collect {} test cases from PyTorch.".format(len(testcases)))
    failed = 0
    ops = set()
    for t in testcases:
        test_name = get_test_name(t)
        module = gen_module(t)
        input = gen_input(t)
        try:
            f = io.BytesIO()
            torch.onnx._export(module, input, f)
            onnx_model = onnx.load_from_string(f.getvalue())
            output_dir = os.path.join(_generated_dir, test_name)

            for i in range(sets):
                output = module(input)
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                data_dir = os.path.join(output_dir, "test_data_set_{}".format(i))
                os.makedirs(data_dir)

                with open(os.path.join(output_dir, "model.pb"), "wb") as f:
                    f.write(onnx_model.SerializeToString())
                for index, var in enumerate([input]):
                    tensor = numpy_helper.from_array(var.data.numpy())
                    with open(os.path.join(data_dir, "input_{}.pb".format(index)), "wb") as f:
                        f.write(tensor.SerializeToString())
                for index, var in enumerate([output]):
                    tensor = numpy_helper.from_array(var.data.numpy())
                    with open(os.path.join(data_dir, "output_{}.pb".format(index)), "wb") as f:
                        f.write(tensor.SerializeToString())
                input = gen_input(t)
        except:
            traceback.print_exc()
            failed += 1

    print("Collect {} test cases from PyTorch repo, failed to export {} cases.".format(
        len(testcases), failed))
    print("Generated cases are stored in {}.".format(_generated_dir))

if __name__ == '__main__':
    testcases = module_tests + new_module_tests
    convert_tests(testcases)
