from torch import nn

class LstmDiscardingCellState(nn.LSTM):
    def forward(self, input, *fargs, **fkwargs):
        output, (hidden, cell) = nn.LSTM.forward(self, input, *fargs, **fkwargs)
        return output, hidden
