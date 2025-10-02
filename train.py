import torch
import torch.nn as nn
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


