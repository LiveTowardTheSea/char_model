# 程序说明
+ 论文题目：基于Transformer及其变种的中文命名实体识别技术研究
+ 指导老师: 许晓伟
+ 编写者：吴秋彤
+ 编写时间：2020.06.03

## 1. 开发配置说明
- 操作系统: Ubuntu 18.04.3
- 处理器：i7-9700
- 内存：16GB
- 显卡： GeForce RTX 2080ti / 12GB
- 开发工具：Visual Code Studio

## 2. 环境说明
<pre>
Python             3.6
certifi            2020.12.5  
chardet            4.0.0  
click              8.0.0  
cycler             0.10.0  
dataclasses        0.8  
decorator          5.0.6  
filelock           3.0.12  
huggingface-hub    0.0.8  
idna               2.10  
importlib-metadata 4.0.1  
ipykernel          5.3.4    
ipython            6.1.0    
ipython-genutils   0.2.0    
jedi               0.18.0    
joblib             1.0.1    
jupyter-client     6.1.12    
jupyter-core       4.7.1    
kiwisolver         1.3.1    
matplotlib         3.3.4  
mkl-fft            1.3.0  
mkl-random         1.1.1  
mkl-service        2.3.0  
numpy              1.19.4  
packaging          20.9  
pandas             1.1.5  
parso              0.8.2  
pexpect            4.8.0  
pickleshare        0.7.5  
Pillow             8.2.0  
pip                21.0.1  
prompt-toolkit     1.0.15  
psutil             5.8.0  
ptyprocess         0.7.0  
Pygments           2.8.1  
pyparsing          2.4.7  
python-dateutil    2.8.1   
pytz               2021.1  
pyzmq              20.0.0  
regex              2021.4.4  
requests           2.25.1  
sacremoses         0.0.45  
scipy              1.5.4  
seaborn            0.11.1  
setuptools         52.0.0.post20210125  
simplegeneric      0.8.1  
six                1.15.0  
tokenizers         0.10.2  
torch              1.7.0  
tornado            6.1   
tqdm               4.60.0  
traitlets          4.3.3   
transformers       4.6.0   //用于加载Bert预训练模型
typing-extensions  3.7.4.3  
urllib3            1.26.4  
wcwidth            0.2.5  
wheel              0.36.2  
zipp               3.4.1  
</pre>

## 3. 数据集来源
- MSRA 数据集：[github链接]( https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/MSRA)
- OntoNotes5.0 数据集：[LDC链接](https://catalog.ldc.upenn.edu/LDC2013T19)，只有注册后才能免费获取

## 4. 各项目说明
### 4.1 vanilla_universal_syn文件夹
+ 三种短序列模型：[Vanilla Transformer](https://arxiv.org/abs/1706.03762)、[Universal Transformer](https://arxiv.org/abs/1807.03819)、[Synthesizer](https://arxiv.org/abs/2005.00743)

#### 4.1.1 运行说明
##### 4.1.1.1 举例
(1)训练 Vanilla  Transformer  
<code>
python main.py --model=vanilla --status=train --load_data=data/MSRA --save_model=data/MSRA/saved_model/vanilla  
</code>  
(2) 测试 Synthesizer  
<code>
python main.py --model=syn --status=test --load_data=data/MSRA --load_model=data/MSRA/saved_model/syn/epoch_29.model
</code>

##### 4.1.1.2 各参数说明
1. model：vanilla(Vanilla Transformer), universal(Universal Transformer), syn(Synthesizer)
2. status: train(训练)， test(测试)， decode(预测，更改sentence.json文件)
3. save_model: 训练时保存模型的相对路径
4. load_model: 测试或者解码时加载模型的相对路径
5. save_data: 保存数据集句子字典，标签字典，以及对应字向量的相对路径
6. load_data: 加载数据集句子字典，标签字典，以及对应字向量的相对路径
7. train: 训练集相对路径
8. dev：验证集相对路径
9. test: 测试集相对路径

此外，对于Synthesizer,若要在Self-Attention子层使用FR,在config文件修改use_fr变量为True

#### 4.1.2 目录结构说明
```
vanilla_universal_syn
│  config.py   配置不同模型超参数 
│  CRF_Decoder.py   CRF解码器
│  main.py  主函数，选择训练还是预测还是测试，选择模型
│  sentence.json    用于预测的句子
│  syn_without_fr.out   训练时输出：Synthesizer在Self-Attention子层采用dense
│  syn_with_fr.out  训练时输出:Synthesizer在Self-Attention采用Self-Attention采用dense+FR
│  transition_matrix.npy    Vanilla Transformer 训练得到的转移矩阵
│  universal_msra.out   训练时输出：Universal Transformer不共享参数
│  universal_share_msra.out 训练时输出:Universal Transformer共享参数
│  universal_updates.npy    Universal Transformer对于测试句子的步数 
│  vanilla_attention.npy    Vanilla Transformer对于测试句子的Attention Weight
│  vanilla_msra_final.out   训练时输出：Vanilla Transformer
│  visual.ipynb     可视化代码
│
├─data  数据集相关
│  │  news_char_256.vec    使用Skip-Gram模型训练得到的字向量
│  │
│  └─MSRA   MSRA数据集
│      │  char_embedding_matrix_256.npy   对于该数据集的对应字向量
│      │  msra_dev_bioes    验证集
│      │  msra_dev_bioes_2  截断长度的验证集，用于Synthesizer
│      │  msra_test_bioes   测试集
│      │  msra_test_bioes_2 截断长度的测试集，用于Synthesizer
│      │  msra_train_bioes  训练集
│      │  msra_train_bioes_2 截断长度的训练集，用于Synthesizer
│      │
│      ├─char_vocab 存储数据集字信息
│      │      itos.txt  编号->字
│      │      stoi.txt  字->编号
│      │      vocab_list.txt    字列表
│      │
│      ├─saved_model 存放训练时的模型
│      │  ├─syn  Synthesizer 
│      │  ├─syn_with_fr Synthesizer Self-Attention子层引入FR方式
│      │  ├─universal   universal不共享参数
│      │  ├─universal2  universal 共享参数
│      │  └─vanilla     Vanilla Transformer
│      └─tag_vocab  存放数据集标签信息
│              itos.txt 编号->标签
│              stoi.txt 标签->编号
│              vocab_list.txt 标签列表
│
├─model  存放模型源码
│  ├─synthesizer Synthesizer模型
│  │      Dense_Attention.py   Dense Attention方式
│  │      FR_Attention.py      Factorized Random 方式
│  │      syn_Encoder.py       Encoder
│  │      syn_model.py         总模型
│  │      syn_Pos_Encoding.py  位置编码
│  │      syn_SubLayer.py      子层：Self Attention,Layer Norm
│  │
│  ├─universal_model Universal Transformer模型
│  │      universal_Model.py   总模型
│  │      uni_Encoder.py       Encoder:实现了按位暂停
│  │      uni_Encoder_Layer.py Encoder的每一层
│  │      uni_Sublayer.py      Encoder每一个子层
│  │
│  └─vanilla_transformer       原始Transformer
│          Encoder.py          Encoder 
│          Pos_Encoding.py     位置编码
│          SubLayer.py         子层
│          vanilla_model.py    总模型
│
└─utils     用于数据集处理、模型评估
        function.py  初始化数据集vocab以及加载字向量
        iterator_dataset.py  生成数据集迭代器，桶排序
        metric.py      计算精度、召回、F值
        my_data.py     实现数据集处理的pipeline
        Vocab.py       存放数据集字、标签以及词向量的实体类
```

### 4.2 XL 文件夹
+ 模型：[Transformer-XL](https://arxiv.org/abs/1901.02860)
#### 4.2.1 运行说明
与上文相同，但要将model改为xl, load_data或者save_data的位置相对改变

#### 4.2.2 目录结构说明
- 和4.1.2大概相同，介绍一下model文件夹的不同
```
├─model
│  └─xl_model   XL模型
│          Mask_Multi_Attn.py  Self-Attention层，实现了相对位置编码
│          util.py   处理mask, 生成相对位置编码的辅助函数
│          xl_Encoder.py   XL Encoder，实现了状态重用
│          xl_EncoderLayer.py Encoder的每一层
│          xl_Model.py        总模型
│          xl_SubLayer.py     子层，Layer Norm以及Feed Foward Nework
```

### 4.3 re_sp_cm 文件夹
+ 模型：[reformer](https://arxiv.org/abs/2001.04451), [Sparse Transformer](https://arxiv.org/abs/1904.10509), [Memory Compressed Transformer](https://arxiv.org/abs/1801.10198)

#### 4.3.1 运行说明
与4.1.1.1相同，其中model参数选择为：
- reformer: Reformer
- sparse: Sparse Transformer
- compress: Memory Compressed Transformer

#### 4.3.2 目录结构说明
与4.1.2大致相同，主要不同在model文件夹，下面简要介绍一下model文件夹，这几个模型除了在Self-Attention子层都使一样的架构，不同在Self-Attention子层
```
├─model model文件夹
│      compressed_attention.py   Memory Compressed Transformer Self-Attention 子层
│      Encoder.py  三个模型通用Encoder
│      EncoderLayer.py  三个模型通用Encoder每一层
│      LSH_Attention.py  Reformer Self-Attention子层
│      Model.py  三个模型通用总模型
│      sparse_attention.py  Sparse Transformer Self-Attention子层
│      SubLayer.py  LayerNorm、Feed Forward Network 子层
```

### 4.4 bert 文件夹
- 模型：[Bert](https://arxiv.org/abs/1810.04805)
- 采用了Hugging Face预训练模型
#### 4.4.1 运行说明
- tip：需添加如4.1.2的data文件夹
- 通过 config.py文件中的decoder变量决定使用哪种 解码器（CRF、Softmax)
```
bert
│  bert_crf_lr_0.0005.out  Bert-CRF 训练时结果
│  bert_softmax_lr0.0005.out Bert-Softmax训练时结果
│  config.py  Bert配置文件
│  data_iterator.py 数据集迭代器
│  main.py  主文件，模型训练、测试、预测
│  metric.py  用于模型评估
│  mydata.py  数据处理pipeline，加载预训练Tokenizer
│  vocab.py   数据集标签字典实体类
│
└─model
        bert_model.py      Bert总模型
        CRF_Decoder.py     CRF解码器
        Encoder.py         加载Bert预训练模型
        Linear_Decoder.py  Softmax解码器
```
