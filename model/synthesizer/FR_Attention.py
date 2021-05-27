import torch
import torch.nn as nn
import torch.nn.functional as F

class FR_Attention(nn.Module):
    def __init__(self,head,max_len,scale):
        super().__init__()
        self.para1 = nn.Parameter(torch.randn(head,max_len,scale))
        self.para2 = nn.Parameter(torch.randn(head,max_len,scale))
    def forward(self,src):
        # src(batch,seq_len,d_model)
        seq_len = src.shape[1]
        para1 = self.para1[:,0:seq_len,:].contiguous()
        para2 = self.para2[:,0:seq_len,:].contiguous()
        #(head,seq_len,seq_len)
        random_value = torch.einsum('nsk,nlk->nsl',para1,para2)
        # 返回值 (1,head,seq_len,seq_len)
        return random_value.unsqueeze(0)


