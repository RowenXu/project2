import numpy as np
import torch
import net 
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import time
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

class dataset_new(Dataset):
    def __init__(self,data,label,transform=None):
        self.data=data
        self.label=label
        self.transform=transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):     
        sample=self.data[idx]
        sample_label=self.label[idx]
        if self.transform:
            sample=self.transform(sample)
        return sample,sample_label


target='mps'
n_lead=12
bar=850
train_dataset_name='./dataset/'+str(n_lead)+'/'+str(bar)+'/train_loader.pth'
valid_dataset_name='./dataset/'+str(n_lead)+'/'+str(bar)+'/valid_loader.pth'
test_dataset_name='./dataset/'+str(n_lead)+'/'+str(bar)+'/test_loader.pth'
#----------读取数据----------
train_loader=torch.load(train_dataset_name)
valid_loader=torch.load(valid_dataset_name)
test_loader=torch.load(test_dataset_name)
#----------定义网络模型参数------
in_planes=9
kernel_size=4

model=net.Total_net(
    in_planes=in_planes,
    kernel_size=kernel_size,
    stride=1,
    padding=0,
    reduction_ratio=8,
    ).to(target)
criterion=nn.MSELoss().to(target)
LR=0.001
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
best_valid_loss=[]
best_train_loss=[]
for epoch in range(10):
    #---------训练网络模型----------
    model.train()
    print('epoch:'+str(epoch))
    running_loss=0.0
    for i,data in enumerate(train_loader,0):
        #向前传播
        inputs,labels=data
        # print(type(inputs))
        inputs=inputs.to(torch.float32).to(target)
        labels=labels.to(torch.float32).to(target)
        outputs=model(inputs)
        # print(outputs.shape)
        loss=criterion(labels,outputs)
        #向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    mean_train_loss=running_loss/len(train_loader)
    print('mean_train_loss:'+str(mean_train_loss))
    best_train_loss.append(mean_train_loss)


    #----------验证模型----------
    print('validating...')
    model.eval()
    val_losses=[]
    # outputs=[]
    # labels=[]
    with torch.no_grad():
        for data in valid_loader:
            inputs,label=data
            inputs=inputs.to(torch.float32).to(target)
            label=label.to(torch.float32).to(target)
            # labels.append(label)
            output=model(inputs)
            # outputs.append(output)
            # print('output:'+str(outputs))
            loss=criterion(output,label)
            val_losses.append(loss.item())
    mean_valid_losses=np.mean(val_losses)
    print('mean_valid_losses:'+str(mean_valid_losses))
    best_valid_loss.append(mean_valid_losses)
fig=plt.figure()
plt.plot(best_train_loss)
plt.plot(best_valid_loss)
plt.legend(['train_loss','valid_loss'])
plt.xlabel('epoch')
plt.ylabel('mseloss')
plt.show()




# #----------预测模型----------
# model.eval()
# with torch.no_grad():
#     # for data in test_loader:
#     data=test_loader
#     inputs,labels=data
#     inputs=inputs.to(torch.float32).to(target)
#     labels=labels.to(torch.float32).to(target)
#     outputs=model(inputs)
#     print('output:'+str(outputs))

# #----------保存模型----------
# torch.save(model.state_dict(),'./output/model_para_summer.pth')
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total Parameters: {total_params}")
