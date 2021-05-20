import torch.nn as nn
from uni_Encoder import *
from CRF_Decoder import *
class Universal_Model(nn.Module):
    def __init__(self,config,src_embedding_num, tag_num, embedding_matrix, embedding_dim_size):
        super(Universal_Model, self). __init__()
        self.config = config
        self.encoder = Encoder(config, src_embedding_num, embedding_dim_size, embedding_matrix)
        self.decoder = CRF_decoder(config.d_model, tag_num)

    def forward(self, src, y, mask, use_gpu,return_attn=False):
        encoder_output, atten_list, updates_num = self.encoder(src, mask, use_gpu, return_attn)
        loss,path = self.decoder.loss(encoder_output, y, mask, use_gpu)
        return loss,path
