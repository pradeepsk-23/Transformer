import torch
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

dataset = ImageFolder(root_dir, tt.Compose([tt.Resize(64), 
                                            tt.RandomCrop(64), 
                                            tt.ToTensor()]))

val_pct = 0.01
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

# DataLoader (input pipeline)
batch_size = 97
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

# Convolution block
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ResNet9
in_channels = 3
num_classes = 13
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64) # out = [64, 64, 64]
        self.conv2 = conv_block(64, 128, pool=True) # out = [128, 32, 32]
        self.res1 = nn.Sequential(conv_block(128, 128),
                                  conv_block(128, 128)) # out = [128, 32, 32]

        self.conv3 = conv_block(128, 256, pool=True) # out = [256, 16, 16]
        self.conv4 = conv_block(256, 512, pool=True) # out = [512, 8, 8]
        self.res2 = nn.Sequential(conv_block(512, 512),
                                  conv_block(512, 512)) # out = [512, 8, 8] 

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(2), # out = [512, 2, 2]
                                        nn.Flatten(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(512*2*2, num_classes))
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.res1(out2) + out2
        out4 = self.conv3(out3)
        out5 = self.conv4(out4)
        out6 = self.res2(out5) + out5
        out7 = self.classifier(out6)
        return out7

def main():

    # Model
    model = ResNet9(in_channels, num_classes).to(device)

    # Loss and optimizer
    # F.cross_entropy computes softmax internally
    loss_fn = F.cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Set up one-cycle learning rate scheduler
    epochs = 2
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

    def predict_image(img, model, classes):
        # Convert to a batch of 1
        xb = (img.unsqueeze(0)).to(device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds  = torch.max(yb, dim=1)
        # Retrieve the class label
        return classes[preds[0].item()]
    
    img, label = val_ds[32]
    plt.imshow(img.permute((1, 2, 0)))
    plt.show()
    pred = predict_image(img, model, dataset.classes)
    print('Target:', dataset.classes[label])
    print('Prediction:', pred)

if __name__ == "__main__":
    main()