#from __future__ import unicode_literals
#
# import json
# import codecs
# a = ['我', '是', '一', '只', '小', '猪']
# print(a)
# print(help(json.dump))
# print(help(json.load))
#
# # with codecs.open(r"test001.txt", 'w', 'utf-8') as f:
# #     json.dump(a, f,ensure_ascii=False)
# #     print('ok')
# import torch
#
# a = torch.randn(4, 4)
# b = torch.arange(0, 4)
# c = a[0, b]
# print(a)
# print(b)
# print(c)
#
# import torch
# a = torch.randn(6,6)
# b = torch.arange(0,6).reshape(2,3)
# print(a)
# print(b[:,1]) # 当前时序的所有序列
# print(a[b[:,1]]) #
# import torch
# b = torch.zeros(2)
# print(b)
# import torch
# a =torch.arange(0,25).reshape(5,5)
#
# tags_ = [[4,1,2,3], [4,3,0,1], [4,3,2,1], [4,3,2,0]]
# tags = torch.tensor(tags_, dtype=torch.long)
# print(a)
# print(tags)
# print(a[tags[:, :-1], tags[:, 1:]])
# import torch
"""不知道为什么 下面的好奇怪"""
# a = torch.randn(4,5)
# print(a)
# b=a.transpose(0,1)
# print(b)
# c =b[:-2,:-2]
# print(id(c),id(b))
# c[0][1]+=1.0
# print(c)
# print(b)
# print(a)
# import torch
# # a = torch.zeros(5, 4, 4)
# # b = torch.tensor([1.0,2,3,4])
# # a +=b
# # print(a)
# a = torch.arange(0, 48).reshape(4, 4, 3)
# print(a)
# b = torch.tensor([1,2,3,0],dtype=torch.long)
#
# for j in range(5,0,-1):
#     print(j)
#
# import torch
# a = torch.ones(3)
# b = torch.randn(3,3)
# print(a)
# print(b)
# print(a.unsqueeze(-1)+b)

# import codecs
# f = codecs.open(r'weiboNER.conll.train.txt', 'r','utf-8')
# for line in f.readlines():
#     print(line)
#     break
import chardet
import codecs
#
# with codecs.open('C:\\Users\\qiuqiu\\NER data\\Weibo NER\\weiboNER.conll.train', 'r', 'utf-8') as f:
#     data = f.read()
# type = chardet.detect(data)
# print(type)
# with codecs.open('C:\\Users\\qiuqiu\\NER data\\Weibo NER\\weiboNER.conll.train', 'r') as f:
#     lines = f.readlines()
# for line in lines:
#     print(line)
#     break
# # with codecs.open(r'weiboNER.conll.train.txt', 'r', 'GB2312') as f:
# #     data = f.readlines()
# # for line in data:
# #     print(line)
# import os
# if not os.path.exists('dataaa'):
#     os.mkdir('dataaa')
#
# a =[1,2,3,4]
# print(a[1:3])

# a= [1,2,3,4]
# b =[2.3,4.5]
# a += b
# print(a)
# import torch
# a = 0.0
# b= torch.tensor(4.6)
# a=a+b
# print(a,a.data)
import torch.nn as nn
import torch
a = nn.Linear(3,4)
input_ = torch.randn(5,3)
print(a.weight.shape)
print(a(input_))


