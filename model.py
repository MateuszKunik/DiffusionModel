import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    The TimeEmbedding class create a sinusoidal position embeddings, which encodes the position along the sequence into a vector.
    """
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)

        return embeddings
    

class Block(nn.Module):
    """
    The Block class defines a block of operations that performs convolution, normalization, activation functions with time embeddings.
    """
    def __init__(self, in_channels=3, out_channels=3, time_embedding_dims=128, downsample=True):
        super(Block, self).__init__()

        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = TimeEmbedding(time_embedding_dims)
        self.downsample = downsample

        if downsample:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False))
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False))
            self.transform  = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False))

        self.time_mlp = nn.Sequential(
            nn.Linear(in_features=time_embedding_dims, out_features=out_channels),
            nn.ReLU(inplace=False))

    def forward(self, x, t):
        x = self.conv1(x)

        time = self.time_mlp(self.time_embedding(t))[:, :, None, None]
        x = x.clone() + time

        return self.transform(self.conv2(x))


class UNet(nn.Module):
    """
    A simplified variant of the UNet architecture.
    """
    def __init__(self, image_channels=3, features=[64, 128, 256, 512, 1024], time_embedding_dims=128):
        super(UNet, self).__init__()
        
        self.time_embedding_dims = time_embedding_dims

        # Initializing a module list to store neural networks.
        self.downsampling = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        # Initializing a preparatory convolution layer for downsampling.
        self.prep_conv = nn.Conv2d(image_channels, features[0], 3, padding=1)

        # Creating a down part of UNet, also called "encoder".
        for in_channels, out_channels in zip(features, features[1:]):
            self.downsampling.append(Block(in_channels, out_channels, time_embedding_dims))

        # Creating a up part of UNet, also called "decoder".
        for in_channels, out_channels in zip(reversed(features), reversed(features[:-1])):
            self.upsampling.append(Block(in_channels, out_channels, time_embedding_dims, downsample=False))

        # Initializing a final convolution layer for output.
        self.final_conv = nn.Conv2d(features[0], image_channels, kernel_size=1)


    def forward(self, x, t):
        skip_connections = []

        x = self.prep_conv(x)

        # Passing input through encoder.
        for down in self.downsampling:
            x = down(x, t)
            skip_connections.append(x)

        # Passing input with skip connection through decoder.
        for up, skip_connection in zip(self.upsampling, reversed(skip_connections)):
            x = up(torch.cat((x, skip_connection), dim=1), t)

        return self.final_conv(x)