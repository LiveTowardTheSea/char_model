import torch
import torch.nn as nn
import torch.nn.functional as F
class tt(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(2))
    def forward(self):
        print(self.a[0],self.a[1])
        print(F.softmax(self.a,dim=-1))


b =tt()
b()
