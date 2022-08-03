from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt


from dataloader import dataLoader
from image_visualization import imshow
from model import train_model, visualize_model

cudnn.benchmark = True
plt.ion()   # 대화형 모드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#데이터 불러오기
image_datasets, dataloaders, dataset_sizes = dataLoader()
class_names = image_datasets['train'].classes


# 학습 데이터의 배치를 얻습니다.
inputs, classes = next(iter(dataloaders['train']))

# 배치로부터 격자 형태의 이미지를 만듭니다.
out = torchvision.utils.make_grid(inputs)

# 일부 이미지 시각화하기
imshow(out, title=[class_names[x] for x in classes])

#######################################################
#합성곱 신경망 미세 조정
#--------------------------------------------------
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len (class_names))
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


#학습 및 평가하기
model_ft = train_model(model=model_ft,
                       criterion= criterion,
                       optimizer=optimizer_ft,
                       scheduler=exp_lr_scheduler,
                       device=device,
                       dataloaders=dataloaders,
                       dataset_sizes=dataset_sizes,
                       num_epochs=25)

visualize_model(model=model_ft,
                device=device,
                dataloaders=dataloaders,
                class_names=class_names)

#--------------------------------------------------
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len (class_names))
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model=model_conv,
                        criterion= criterion,
                        optimizer=optimizer_conv,
                        scheduler=exp_lr_scheduler,
                        device=device,
                        dataloaders=dataloaders,
                        dataset_sizes=dataset_sizes,
                        num_epochs=25)

visualize_model(model=model_conv,
                device=device,
                dataloaders=dataloaders,
                class_names=class_names)


plt.ioff()
plt.show()