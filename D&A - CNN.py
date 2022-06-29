import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from CustomDataset import DriveandAct
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
dataset = DriveandAct(
    csv_file="./Frame.csv",
    root_dir="./Frame",
    transform=transforms.ToTensor())
    
print(len(dataset))
# image, y_label = dataset[21]
# print('image.shape:', image.shape)
# plt.imshow(image.permute(1, 2, 0), cmap='gray')
# plt.show()
# print('Label:', y_label)

# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [34,12])


# DataLoader (input pipeline)
# batch_size=1
# train_dl = DataLoader(dataset, batch_size)
# test_dl = DataLoader(test_dataset, batch_size)

# for x,y in train_dl:
#     plt.imshow(x.permute(2, 3, 0, 1), cmap='gray')

# # Convolutional neural network
# num_classes = 10
# class ConvNet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
#                                     nn.BatchNorm2d(32),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
#                                     nn.BatchNorm2d(64),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
#                                     nn.BatchNorm2d(128),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
#                                     nn.BatchNorm2d(256),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Linear(256*2*2, num_classes)
        
#     def forward(self, x):
#         out1 = self.layer1(x)
#         out2 = self.layer2(out1)
#         out3 = self.layer3(out2)
#         out4 = self.layer4(out3)
#         out5 = out4.reshape(out4.size(0), -1)
#         out6 = self.fc(out5)
#         return out6            

# # Model
# model = ConvNet(num_classes).to(device)

# # Loss and optimizer
# # F.cross_entropy computes softmax internally
# loss_fn = F.cross_entropy
# opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# # Train the model
# epochs = 5
# total_step = len(train_dl)
# for epoch in range(epochs):
#     for i, (images, labels) in enumerate(train_dl):
#         # Move tensors to the configured device
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = loss_fn(outputs, labels)

#         # Backward and optimize
#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         if (i+1) % 500 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                    .format(epoch+1, epochs, i+1, total_step, loss.item()))

# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_dl:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
    
#     print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))