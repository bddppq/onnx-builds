import functools
import os
import unittest
import sys
import torch

_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, os.path.join(_root_dir, "repos", "pytorch", "test"))

from common import *

torch.set_default_tensor_type('torch.FloatTensor')


def _skipper(condition, reason):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if condition():
                raise unittest.SkipTest(reason)
            return f(*args, **kwargs)
        return wrapper
    return decorator


skipIfNoCuda = _skipper(lambda: not torch.cuda.is_available(),
                        'CUDA is not available')

skipIfTravis = _skipper(lambda: os.getenv('TRAVIS'),
                        'Skip In Travis')
