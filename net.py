import numpy as np
import torch
import torch.nn as nn
#定义卷积-注意力网络
#定义通道注意力
class Channel_Attention(nn.Module):
    def __init__(self,in_channel,reduction_ratio,batch_size):
        super(Channel_Attention,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(in_channel,in_channel//reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channel//reduction_ratio,in_channel)
        )
        self.sigm=nn.Sigmoid()
    def forward(self,x):
        avg_out=self.avg_pool(x).view(x.size(0),x.size(1))
        # print('avg_out:'+str(avg_out.shape))
        channel_atte=self.fc(avg_out).view(x.size(0),x.size(1),1,1)
        return channel_atte
# #定义空间注意力
# class Spatial_Attention(nn.Module):
#     def __init__(self,kernel_size=7):
#         super(Spatial_Attention,self).__init__()
#         assert kernel_size in (3,7),'kernel size must be 3 or 7'
#         padding=3 if kernel_size==7 else 1

#         self.conv=nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
#         self.sigmoid=nn.Sigmoid()
#     def forward(self,x):
#         avg_out=torch.mean(x,dim=1,keepdim=True)
#         max_out,_=torch.max(x,dim=1,keepdim=True)
#         x=torch.cat([avg_out,max_out],dim=1)
#         x=self.conv(x)
#         return self.sigmoid(x)

#平滑
def flatten(x):
    return x.view(x.size(0),-1)

#LSTM网络
class LSTM_net(nn.Module):
    def __init__(self,in_channel,out_channel,hidden_size,num_layers,batch_size):
        super(LSTM_net,self).__init__()
        self.hidden_size=hidden_size
        self.LSTM=nn.LSTM(
            input_size=in_channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc=nn.Linear(hidden_size,out_channel)
    def forward(self,x):
        x=self.LSTM(x)
        x=self.fc(flatten(x))
        return x




#卷积网络
class Total_net(nn.Module):
    def __init__(self,in_planes,kernel_size,reduction_ratio,stride=1,padding=0,bias=False):
        super(Total_net,self).__init__()
        self.conv1=nn.Conv2d(
            in_channels=in_planes,
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.maxpool1=nn.MaxPool2d(kernel_size=4,stride=2)
        self.conv2=nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.maxpool2=nn.MaxPool2d(kernel_size=4,stride=2)
        self.relu=nn.ReLU()
        self.channel_attention=Channel_Attention(in_channel=64,reduction_ratio=reduction_ratio,batch_size=6)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(64,128)
        self.dropout=nn.Dropout(p=0.5)
        self.fc2=nn.Linear(128,3)
        self.sigmoid=nn.Sigmoid()
        self.LSTM=nn.LSTM(
            input_size=3,
            hidden_size=3,
            num_layers=1,
            batch_first=True
        )
    def forward(self,x):
        x=self.conv1(x)
        # print("conv1:"+str(x.shape))
        x=self.relu(x)
        x=self.maxpool1(x)
        # print("maxpool1:"+str(x.shape))
        x=self.channel_attention(x)*x
        x=self.conv2(x)
        # print("conv2:"+str(x.shape))
        x=self.maxpool2(x)
        x=self.relu(x)
        x=self.channel_attention(x)*x
        # print("maxpool2:"+str(x.shape))
        x=self.pool(x)
        # print("pool:"+str(x.shape))
        x=self.fc1(flatten(x))
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        # x=self.LSTM(x)
        # print("fc:"+str(x.shape))
        return x











