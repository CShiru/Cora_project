import torch
from torch import nn
import torch.nn.functional as F

# 定义Mish激活函数与图卷积操作类
def mish(x): 
    return x * (torch.tanh(F.softplus(x)))

# 图卷积类
class GraphConvolution(nn.Module):
    def __init__(self,f_in,f_out,use_bias = True,activation=mish):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.use_bias = use_bias
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(f_in, f_out))
        self.bias = nn.Parameter(torch.FloatTensor(f_out)) if use_bias else None
        self.initialize_weights()
 
    # 对参数进行初始化
    def initialize_weights(self):
        if self.activation is None: 
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, nonlinearity='leaky_relu')
        if self.use_bias:
            nn.init.zeros_(self.bias)
 
    # 实现模型的正向处理流程
    def forward(self,input,adj): 
        support = torch.mm(input,self.weight) 
        output = torch.mm(adj,support) 
        if self.use_bias:
            output.add_(self.bias) 
        if self.activation is not None:
            output = self.activation(output) 
        return output
 
# 搭建多层图卷积网络模型
class GCN(nn.Module):
    def __init__(self, f_in, n_classes, hidden=[16], dropout_p=0.5): 
        super().__init__()
        layers = []
        # 根据参数构建多层网络
        for f_in, f_out in zip([f_in] + hidden[:-1], hidden):
            layers += [GraphConvolution(f_in, f_out)]
 
        self.layers = nn.Sequential(*layers)
        self.dropout_p = dropout_p
        self.out_layer = GraphConvolution(f_out, n_classes, activation=None)
 
    # 实现前向处理过程
    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x,adj)
        F.dropout(x,self.dropout_p,training=self.training,inplace=True)
        return self.out_layer(x,adj)
 
 
