import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Encoder import *
from Pos_Encoding import *
import copy
from Decoder import *


class Seq2Seq(nn.Module):
    def __init__(self, config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.encoder = Encoder(config, src_embedding_num, embedding_matrix, embedding_dim_size)
        self.decoder = CRF_decoder(config, tag_num)

    def forward(self, src, trg, src_mask, use_gpu):
        # src: (batch_size,seq_len)
        # trg: (batch_size,seq_len)
        # src_mask (batch_size,seq_len)
        encoder_output = self.encoder(src, src_mask, use_gpu)
        loss = self.decoder.loss(encoder_output, trg, src_mask, use_gpu)
        return loss

