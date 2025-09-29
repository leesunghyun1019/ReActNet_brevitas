import torch
import torch.nn as nn
import torch.optim as optim


# def train_quant_model(quant_net, train_loader, val_loader = None, device = 'cpu',
#                       epochs = 20, lr = 0.0001):
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(quant_net.parameters(), lr = lr)
    
#     patience = 5
#     best_val_loss = float('inf')

#     for e in range(epochs):
#         running_loss = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             # Prevent accumulation of gradients
#             optimizer.zero_grad()
#             # Make predictions
#             log_ps = quant_net(images.float())
#             loss = criterion(log_ps, labels)
#             #backprop
#             loss.backward()
#             optimizer.step()
        
#             running_loss += loss.item()

#             val_loss = 0
#             accuracy = 0

#         # Turn off gradients for validation
#         with torch.no_grad():
#             quant_net.eval()
#             if(val_loader != None):
#                 for images, labels in val_loader:
#                     images, labels = images.to(device), labels.to(device)
#                     log_ps = quant_net(images.float())
#                     val_loss += criterion(log_ps, labels)

#                     ps = torch.exp(log_ps)
#                     # Get our top predictions
#                     top_p, top_class = ps.topk(1, dim=1)
#                     equals = top_class == labels.view(*top_class.shape)
#                     accuracy += torch.mean(equals.type(torch.FloatTensor))

#         if(val_loader != None):
#             # Check for early stopping
#             avg_val_loss = val_loss/len(val_loader)
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 counter = 0
#             else:
#                 counter += 1
                
#             if counter >= patience:
#                 break

#         quant_net.train()
        
#     return quant_net


def train(quant_net, train_loader, val_loader=None, device='cpu',
                      epochs=20, lr=0.0001):
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(quant_net.parameters(), lr=lr)

    
    for e in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(quant_net.parameters(), max_norm=1.0)
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
                print(f'Epoch {e+1}/{epochs}: Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_accuracy:.4f}')
            else:
                print(f'Epoch {e+1}/{epochs}: Train Loss: {running_loss/len(train_loader):.4f}')
                
        quant_net.train()
        
    return quant_net
