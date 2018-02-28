from torch import nn
from torch.nn.utils import rnn as rnn_utils

class RnnModelWithPackedSequence(nn.Module):
    def __init__(self, model):
        super(RnnModelWithPackedSequence, self).__init__()
        self.model = model
    def forward(self, input, *args):
        args, seq_lengths = args[:-1], args[-1]
        input = rnn_utils.pack_padded_sequence(input, seq_lengths)
        rets = self.model(input, *args)
        ret, rets = rets[0], rets[1:]
        ret, _ = rnn_utils.pad_packed_sequence(ret)
        return tuple([ret] + list(rets))
