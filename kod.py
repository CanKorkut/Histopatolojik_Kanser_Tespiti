#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data_label = pd.read_csv("train_labels.csv")


# In[3]:


data_label.head()


# In[4]:


class_0 = data_label.loc[data_label["label"] == 0]
class_1 = data_label.loc[data_label["label"] == 1]


# In[5]:


import os
try:
    os.mkdir("./train/0")
    os.mkdir("./train/1")
except Exception as e:
    print(e)


# In[6]:


import shutil

for i in class_0['id']:
    shutil.copy("./train/" + i +".tif","./train/0/" + i +".tif")


# In[7]:


for i in class_1['id']:
    shutil.move("./train/" + i +".tif","./train/1/" + i +".tif")


# In[1]:


from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import torch


torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(12)


# In[2]:


data_transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

data_transform_valid = transforms.Compose([
        transforms.ToTensor()
    ])

cancer_dataset = datasets.ImageFolder(root='./train',
                                           transform=data_transform_train)


# In[3]:


train,valid = torch.utils.data.random_split(cancer_dataset, [int(len(cancer_dataset) * 0.8),int(len(cancer_dataset) * 0.2)])


# In[4]:


valid.dataset.transforms = data_transform_valid


# In[5]:


train_loader = DataLoader(train, batch_size=512, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid, batch_size=512, shuffle=True, num_workers=2)

dataiter = iter(valid_loader)
from matplotlib.pyplot import figure


# In[6]:


idx = 1
fig = plt.figure(figsize=(25, 4))
images = dataiter.next()

for image,label in zip(images[0],images[1]):
    
    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    plt.imshow(image.permute(1,2,0).numpy())
    if label.item() == 1:
        ax.set_title("Kanserli Hücre")
    else:
        ax.set_title("Normal Hücre")
    idx = idx+1
    if idx == 20:
        break


# In[7]:


import torch.nn as nn
import torch.nn.functional as F
vector = 0

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * 1 * 1, 2)
    def forward(self, x):
        global vector
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool(F.leaky_relu(self.bn5(self.conv5(x))))
        x = self.avg(x)
        x = x.view(-1, 512 * 1 * 1)
        vector = x
        x = self.fc(x)
        return x


# In[8]:


model = CNN()

device = "cuda"

model.to(device)


# In[9]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)


# In[10]:


def calc_acc(loader):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (100 * correct / total)


# In[11]:


total_step = len(train_loader)
train_hist = [
]
valid_hist = []
loss_hist = []

for epoch in range(8):
    total_loss = 0
    train_acc = 0
    valid_acc = 0
    correct = 0
    total = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_acc = (correct/total) * 100  
    total_loss = total_loss / i
    valid_acc = calc_acc(valid_loader)
    train_hist.append(100 * correct / total)
    valid_hist.append(valid_acc)
    loss_hist.append(total_loss)
    print ('Epoch [{}/{}], Loss: {:.4f}, Train_Acc: {:.4f}, Valid_Acc: {:.4f}'
                           .format(epoch+1, 16,total_loss,train_acc,valid_acc))
    


# In[12]:


print ('Valid_Acc: {:.4f}'.format(calc_acc(valid_loader)))


# In[13]:


train_hist = train_hist[1:]
valid_hist = valid_hist[1:]


# In[15]:


train_hist.insert(0,50)


# In[16]:


valid_hist.insert(0,50)


# In[17]:


plt.figure(2)
plt.plot(train_hist)
plt.plot(valid_hist)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Results')
plt.legend(loc='best')
plt.show()


# In[18]:


train_hist[0]=0
valid_hist[0]=0


# In[19]:


model.eval()
label_total = []
pred_total = []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        label_total.append(labels)
        pred_total.append(predicted)

print(100 * correct / total)


# In[20]:


total


# In[21]:


l = []
p = []
for i in label_total:
    for j in i:
        l.append(j.item())


# In[22]:


for i in pred_total:
    for j in i:
        p.append(j.item())


# In[23]:


from sklearn.metrics import classification_report
print(classification_report(l, p, target_names=["Kanserli","Normal"]))


# In[24]:


from sklearn import metrics

y = np.array(l)
scores = np.array(p)

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)


# In[25]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[ ]:




