from test_pytorch_common import TestCase, run_tests, skipIfNoLapack

import torch
import torch.onnx
from torch.autograd import Variable, Function
from torch.nn import Module
import torch.nn as nn

import onnx
import onnx.checker
import onnx.helper

import google.protobuf.text_format

import itertools
import io
import unittest


def export_to_string(model, inputs, *args, **kwargs):
    f = io.BytesIO()
    torch.onnx.export(model, inputs, f, *args, **kwargs)
    return f.getvalue()


class FuncModule(Module):
    def __init__(self, f, params=tuple()):
        super(FuncModule, self).__init__()
        self.f = f
        self.params = nn.ParameterList(list(params))

    def forward(self, *args):
        return self.f(*itertools.chain(args, self.params))


class TestOperators(TestCase):
    def assertONNXExpected(self, binary_pb, subname=None):
        model_def = onnx.ModelProto.FromString(binary_pb)
        onnx.checker.check_model(model_def)
        # doc_string contains stack trace in it, strip it
        onnx.helper.strip_doc_string(model_def)
        self.assertExpected(google.protobuf.text_format.MessageToString(model_def, float_format='.15g'), subname)

    def assertONNX(self, f, args, params=tuple(), **kwargs):
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        self.assertONNXExpected(export_to_string(m, args, **kwargs))

    def assertONNXRaises(self, err, f, args, params=tuple(), **kwargs):
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        self.assertExpectedRaises(err, lambda: export_to_string(m, args, **kwargs))

    def test_basic(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)
        self.assertONNX(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), (x, y))

    def test_view(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        self.assertONNX(lambda x: x.view(1, 1), x)

    @unittest.skip("Indexing is broken by #3725")
    def test_index(self):
        x = Variable(torch.Tensor([[0]]), requires_grad=True)
        self.assertONNX(lambda x: x[0], x)

    def test_addconstant(self):
        x = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        self.assertONNX(lambda x: x + 1, x)

    def test_add_broadcast(self):
        x = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        y = Variable(torch.DoubleTensor(3), requires_grad=True)
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_add_left_broadcast(self):
        x = Variable(torch.DoubleTensor(3), requires_grad=True)
        y = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        self.assertONNXRaises(RuntimeError, lambda x, y: x + y, (x, y))

    def test_add_size1_broadcast(self):
        x = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        y = Variable(torch.DoubleTensor(2, 1), requires_grad=True)
        self.assertONNXRaises(RuntimeError, lambda x, y: x + y, (x, y))

    def test_transpose(self):
        x = Variable(torch.Tensor([[0, 1], [2, 3]]), requires_grad=True)
        self.assertONNX(lambda x: x.transpose(0, 1).transpose(1, 0), x)

    def test_chunk(self):
        x = Variable(torch.Tensor([0,1,2]), requires_grad=True)
        self.assertONNX(lambda x: x.chunk(2), x)

    def test_concat2(self):
        # volatile is of particular interest; it caused a segfault
        # with the exporter
        x = Variable(torch.randn(2, 3), volatile=True)
        y = Variable(torch.randn(2, 3), volatile=True)
        self.assertONNX(lambda inputs: torch.cat(inputs, 1), ((x, y),))

    def test_mm(self):
        m1 = Variable(torch.randn(2, 3), requires_grad=True)
        m2 = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(torch.mm, (m1, m2))

    def test_addmm(self):
        m1 = Variable(torch.randn(2, 3), requires_grad=True)
        m2 = Variable(torch.randn(3, 4), requires_grad=True)
        m3 = Variable(torch.randn(4), requires_grad=True)
        self.assertONNX(lambda x, y, z: torch.addmm(torch.addmm(z, x, y), x, y), (m1, m2, m3))

    def test_permute2(self):
        x = Variable(torch.Tensor([[[[[[0]]]]]]), requires_grad=True)
        self.assertONNX(lambda x: x.permute(0, 1, 4, 2, 5, 3), x)

    def test_pad(self):
        x = Variable(torch.Tensor([[[[0, 1, 1, 1], [2, 3, 7, 7]]]]), requires_grad=True)
        self.assertONNX(nn.ReflectionPad2d((3, 4, 1, 2)), x)

    def test_params(self):
        x = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
        y = nn.Parameter(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
        self.assertONNX(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), x, params=(y, ))

    def test_non_float_params(self):
        x = Variable(torch.LongTensor([[1, 2], [3, 4]]), requires_grad=True)
        y = nn.Parameter(torch.LongTensor([[1, 2], [3, 4]]), requires_grad=True)
        self.assertONNX(lambda x, y: x * (x + y), x, params=(y, ))

    def test_symbolic_mismatch(self):
        class MyFun(Function):
            @staticmethod
            def symbolic(g, x):
                # The inside of this function should never be invoked, because
                # we will fail due to an argument mismatch first.
                assert False

            @staticmethod
            def forward(ctx, x, y):
                return x + y

        x = Variable(torch.randn(2, 2).fill_(1.0))
        y = Variable(torch.randn(2, 2).fill_(1.0))
        # NB: Don't use expect test here, the type error wobbles depending
        # on Python version
        with self.assertRaisesRegex(TypeError, "occurred when translating MyFun"):
            export_to_string(FuncModule(MyFun().apply), (x, y))

    # TODO: Do an nn style test for these
    def test_batchnorm(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(2), x)

    def test_batchnorm_training(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(2), x, training=True)

    def test_conv(self):
        x = Variable(torch.randn(20, 16, 50, 40).fill_(1.0), requires_grad=True)
        self.assertONNX(nn.Conv2d(16, 13, 3, bias=False), x)

    def test_maxpool(self):
        x = Variable(torch.randn(20, 16, 50))
        self.assertONNX(nn.MaxPool1d(3, stride=2), x)

    @unittest.skip('Broken by onnx repo Variadic inputs commit '
                   'https://github.com/onnx/onnx/commit/fd5493013e6cd1f41a560f05965216ec106f56eb')
    def test_at_op(self):
        x = Variable(torch.randn(3, 4))

        class MyFun(Function):

            @staticmethod
            def symbolic(g, x):
                return g.at("add", x, x)

            @staticmethod
            def forward(ctx, x):
                return x + x

        class MyModule(Module):
            def forward(self, x):
                return MyFun.apply(x)

        self.assertONNX(MyModule(), x)


if __name__ == '__main__':
    run_tests()
