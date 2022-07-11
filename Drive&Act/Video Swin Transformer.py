import torch
import timm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as tt

from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
root_dir = "D:\IRP\GitHub\Frame"

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
dataset = ImageFolder(root_dir, tt.Compose([tt.Resize(224),
                                            tt.Pad(8, padding_mode='reflect'),
                                            tt.RandomCrop(224), 
                                            tt.ToTensor(),
                                            tt.Normalize(*imagenet_stats)]))

val_pct = 0.01
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

# DataLoader (input pipeline)
batch_size = 97
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

def main():

    # Model
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True).to(device)

    # Loss and optimizer
    # F.cross_entropy computes softmax internally
    loss_fn = F.cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Set up one-cycle learning rate scheduler
    epochs = 5
    grad_clip = 0.1

    # For updating learning rate
    def update_lr(opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, epochs=epochs, steps_per_epoch=len(train_dl))

    # Train the model
    total_step = len(train_dl)
    for epoch in range(epochs):
        lrs = []
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward and optimize
            opt.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            opt.step()

            # Record & update learning rate
            lrs.append(update_lr(opt))
            sched.step()
    
        if (i+1) % 36 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, epochs, i+1, total_step, loss.item()))

    # Test the model
    model.eval()          # Turns off dropout and batchnorm layers for testing / validation.
    with torch.no_grad(): # In test phase, we don't need to compute gradients (for memory efficiency)
        correct = 0
        total = 0
        for images, labels in val_dl:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

if __name__ == "__main__":
    main()