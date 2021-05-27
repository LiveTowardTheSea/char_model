import torch
import torch.nn as nn
import torch.nn.functional as F
class Dense_Attention(nn.Module):
    def __init__(self,d_model,head,k_dim,max_len):
        super().__init__()
        self.q_linear = nn.Linear(d_model,d_model)
        self.atten_linear = nn.Parameter(torch.randn(d_model,max_len))
        self.atten_bias = nn.Parameter(torch.randn(head,max_len))
        self.head = head
    
    def forward(self,src):
        # src(batch,seq_len,d_model)
        batch_size = src.shape[0]
        seq_len = src.shape[1]
        q = self.q_linear(src)
        query = q.reshape((batch_size,seq_len,self.head,-1))
        query = F.relu(query) #（batch,seq_len,head,k_dim)
        # 下面，我们将atten_linear进行截断
        atten_linear = self.atten_linear[:,0:seq_len]
        atten_bias = self.atten_bias[:,0:seq_len]
        #(head,k_dim,seq_len)
        atten_linear = torch.reshape(atten_linear,(self.head,-1,seq_len))
        w_result = torch.einsum('bshk,hkl->bshl',query,atten_linear) #(batch,seq_len,head,seq_len)
        result = w_result + atten_bias  #(batch,seq_len,head,seq_len)
        result = result.permute(0,2,1,3).contiguous()  #(batch,head,seq_len,seq_len)
        return result


