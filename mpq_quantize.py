import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from utils import *

import copy

def train(quant_net, train_loader, val_loader=None, device='cpu',
                      epochs=20, lr=0.0001):
    
    # Hyperparameters
    Begin_epoch = 0
    Max_epoch = epochs
    Weight_decay = 0

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
                            {'params':quant_net.parameters(),
                             'weight_decay': Weight_decay,
                             'initial_lr' : lr}], 
                             lr=lr)
    
    # Learning Rate Scheduler - Linear Decay
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (1.0 - step/Max_epoch),
        last_epoch=Begin_epoch-1
    )


    best_accuracy = 0
    best_model_state = None 
    
    for e in range(epochs):
        
        quant_net.train()
        
        running_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Prevent accumulation of gradients
            optimizer.zero_grad()
            # Make predictions
            log_ps = quant_net(images.float()) 
            loss = criterion(log_ps, labels)
            #backprop
            loss.backward()
            
            # Adaptive gradient clipping (FC layer 제외)
            parameters_list = []
            for name, p in quant_net.named_parameters():
                if 'fc' not in name and 'classifier' not in name:  # FC layer 제외
                    parameters_list.append(p)

            # Adaptive clipping 또는 일반 clipping 선택
            if len(parameters_list) > 0:
                adaptive_clip_grad(parameters_list, clip_factor=0.001)
                #torch.nn.utils.clip_grad_norm_(parameters_list, max_norm=1.0)


            optimizer.step()
            running_loss += loss.item()
        
        val_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation
        with torch.no_grad():
            quant_net.eval()
            if(val_loader != None):
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = quant_net(images.float())
                    val_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    # Get our top predictions
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                # 평균 계산 및 출력
                avg_val_loss = val_loss/len(val_loader)
                avg_accuracy = (accuracy/len(val_loader))*100

                if avg_accuracy > best_accuracy:
                     best_accuracy = avg_accuracy
                     best_model_state = quant_net.state_dict().copy()
                     #best_model_state = copy.deepcopy(quant_net.state_dict())

                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                print(f'Epoch {e+1}/{epochs}: Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_accuracy:.4f}, LR: {current_lr:.6f}')
            else:
                print(f'Epoch {e+1}/{epochs}: Train Loss: {running_loss/len(train_loader):.4f}')
        
        # Update learning rate
        lr_scheduler.step()

        
    
    if best_model_state is not None:
        quant_net.load_state_dict(best_model_state)
        print(f'\nBest model loaded with accuracy: {best_accuracy:.4f}%')
        
        


    return quant_net


Dataset_path = '/Datasets/CIFAR10/'

def get_cifar10_loaders(batch_size=256, num_workers=8, data_path="./data"):
    # 데이터 전처리
    transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 데이터셋
    Train_data = datasets.CIFAR10(root=Dataset_path, train=True, download=True, transform=transform_train)
    Test_data = datasets.CIFAR10(root=Dataset_path,  train=False, download=True, transform=transform_test)


    # DataLoader
    train_data_loader = torch.utils.data.DataLoader(
        dataset=Train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
        )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=Test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )

    return train_data_loader, test_data_loader


#operator
import brevitas.nn as qnn
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import BitWidthImplType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import StatsOp
from brevitas.inject.enum import RestrictValueType
from brevitas.inject import value

# Binary Convolution
class BinaryWeightPerChannel(Int8WeightPerTensorFloat):
     quant_type = QuantType.BINARY
     bit_width = 1
     bit_width_impl_type = BitWidthImplType.CONST
     scaling_impl_type = ScalingImplType.STATS
     scaling_stats_op = StatsOp.AVE
     scaling_per_output_channel = True
     narrow_range = False
     
     @value
     def scaling_init(module):
         return torch.mean(torch.abs(module.weight), dim=(1, 2, 3), keepdim=True)




