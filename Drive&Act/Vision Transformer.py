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

dataset = ImageFolder(root_dir, tt.Compose([tt.Resize(128), 
                                            tt.RandomCrop(128), 
                                            tt.ToTensor()]))

val_pct = 0.01
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

# DataLoader (input pipeline)
batch_size = 97
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

# Patch Embedding
class PatchEmbed(nn.Module):
    
        # Parameters:
        # image_size : Size of the image (48).
        # patch_size : Size of the patch (16).
        # in_channels : Number of input channels.
        # embed_dim : The embedding dimension.
        # Attributes:
        # num_patches : Number of patches inside of our image.
    
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Convolutional layer that does both the splitting into patches and their embedding.
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        out1 = self.projection(x)  # (batch_size, embed_dim, 3, 3)
        out2 = out1.flatten(2)  # (batch_size, embed_dim, 9(num_patches)) 
        out3 = out2.transpose(1, 2)  # (batch_size, 9, embed_dim)
        return out3

# Multi Head Attention
class Attention(nn.Module):
    
        # Parameters:
        # dim : Embedding dimension.
        # num_heads : Number of attention heads.
        # qkv_bias : If True then we include bias to the query, key and value projections.
        # attn_p : Dropout probability applied to the query, key and value tensors.
        # proj_p : Dropout probability applied to the output tensor.
    
        # Attributes:
        # scale : Normalizing constant for the dot product.
        # qkv : Linear projection for the query, key and value.
        # proj : Linear mapping that takes in the concatenated output of all attention heads and maps it into a new space.
        # attn_drop, proj_drop : Dropout layers.

    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p) 

    def forward(self, x):
        batch_size, num_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError 

        qkv = self.qkv(x)  # (batch_size, num_patches + 1, dim * 3)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)  # (batch_size, num_patches + 1, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, num_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (batch_size, num_heads, head_dim, num_patches + 1)
        dp = (q @ k_t) * self.scale # (batch_size, num_heads, num_patches + 1, num_patches + 1)
        attn = dp.softmax(dim=-1)  # (batch_size, num_heads, num_patches + 1, num_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (batch_size, num_heads, num_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (batch_size, nu_patches + 1, num_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (batch_size, nu_patches + 1, dim)

        x = self.proj(weighted_avg)  # (batch_size, num_patches + 1, dim)
        x = self.proj_drop(x)  # (batch_size, num_patches + 1, dim)

        return x

# Feed Forward Network
class MLP(nn.Module):

    # Parameters:
    # input_size : Number of input features.
    # hidden_size : Number of nodes in the hidden layer.
    # output_size : Number of output features.
    # p : Dropout probability.

    def __init__(self, input_size, hidden_size, output_size, p=0.):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), # (batch_size, num_patches + 1, hidden_size)
                                     nn.GELU(),
                                     nn.Dropout(p),
                                     nn.Linear(hidden_size, output_size), # (batch_size, num_patches + 1, output_size)
                                     nn.Dropout(p))
    
    def forward(self, x):
        return self.network(x)

# Transformer Block 
class TransformerBlock(nn.Module):
    
    # Parameters:
    # dim : Embedding dimension.
    # num_heads : Number of attention heads.
    # mlp_ratio : Determines the hidden dimension size of the `MLP` module with respect to `dim`.
    # qkv_bias : If True then we include bias to the query, key and value projections.
    # p, attn_p : Dropout probability.
    
    # Attributes:
    # norm1, norm2 : Layer normalization.
    # attn : Attention module.
    # mlp : MLP module.
 
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(input_size=embed_dim, hidden_size=hidden_size, output_size=embed_dim,)

    def forward(self, x):
        out1 = self.norm1(x)
        out2 = self.attn(out1) + x
        out3 = self.norm2(out2)
        out4 = self.mlp(out3) + out2
        return out4

# Vision Transformer
class VisionTransformer(nn.Module):

    # Parameters:
    # image_size : Size of the image (it is a square).
    # patch_size : Size of the patch (it is a square).
    # in_channels : Number of input channels.
    # num_classes : Number of classes.
    # embed_dim : Dimensionality of the token/patch embeddings.
    # depth : Number of blocks.
    # num_heads : Number of attention heads.
    # mlp_ratio : Determines the hidden dimension of the `MLP` module.
    # qkv_bias : If True then we include bias to the query, key and value projections.
    # p, attn_p : Dropout probability.

    # Attributes:
    # patch_embed : Instance of `PatchEmbed` layer.
    # class_token : Learnable parameter that will represent the first token in the sequence. It has `embed_dim` elements.
    # pos_embed : Positional embedding of the cls token + all the patches. It has `(n_patches + 1) * embed_dim` elements.
    # pos_drop : Dropout layer.
    # blocks : List of `Block` modules.
    # norm : Layer normalization.

    def __init__(self, image_size, patch_size, in_channels, num_classes,
                embed_dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, p=0, attn_p=0):
        super().__init__()

        self.patch_embed = PatchEmbed(image_size=image_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                      qkv_bias=qkv_bias, p=p, attn_p=attn_p)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        batch_size = x.shape[0]

        out1 = self.patch_embed(x)

        class_token = self.class_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        out2 = torch.cat((class_token, out1), dim=1)  # (batch_size, 1 + num_patches, embed_dim)
        out3 = out2 + self.pos_embed  # (batch_size, 1 + num_patches, embed_dim)
        out4 = self.pos_drop(out3)

        for block in self.blocks:
            out5 = block(out4)

        out6 = self.norm(out5)

        class_token_final = out6[:, 0]  # just the CLS token
        out7 = self.head(class_token_final)

        return out7

def main():

    # Model
    model = VisionTransformer(image_size=128, patch_size=16, in_channels=3, num_classes=13,
                embed_dim=768, depth=12, num_heads=12).to(device)

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