import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from syn_Encoder import *
from syn_Pos_Encoding import *
import copy
from CRF_Decoder import *

class syn_model(nn.Module):
    def __init__(self, config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size):
        super(syn_model, self).__init__()
        self.config = config
        self.encoder = syn_Encoder(config, src_embedding_num, embedding_matrix, embedding_dim_size)
        self.decoder = CRF_decoder(config.d_model, tag_num)

    def forward(self, src, trg, src_mask, use_gpu):
        # src: (batch_size,seq_len)
        # trg: (batch_size,seq_len)
        # src_mask (batch_size,seq_len)
        # 如果 return_atten 为 true 的话，返回每一层的attention，否则是一个空列表
        encoder_output = self.encoder(src, src_mask, use_gpu)
        loss,path_ = self.decoder.loss(encoder_output, trg, src_mask, use_gpu)
        return loss,path_

