import torch
import torch.nn as nn
import math


class Position_Encoding(nn.Module):
    def __init__(self, config):
        super(Position_Encoding, self).__init__()
        self.drop_out = nn.Dropout(config.p_drop)
        self.d_model = config.d_model

    def forward(self, x, use_gpu):
        # x (batch,seq_len,d_model)
        pos_tensor = torch.ones(x.shape[1], self.d_model)
        pos_tensor.requires_grad = False
        if use_gpu:
            pos_tensor = pos_tensor.cuda()
        for pos in range(pos_tensor.shape[0]):
            for i in range(pos_tensor.shape[1]):
                if i % 2 == 0:
                    pos_tensor[pos][i] = math.sin(pos / (10000 ** (2 * i / self.d_model)))
                else:
                    pos_tensor[pos][i] = math.cos(pos / (10000 ** (2 * i / self.d_model)))

        pos_tensor = torch.unsqueeze(pos_tensor, dim=0)
        x = x + pos_tensor
        x = self.drop_out(x)
        return x
