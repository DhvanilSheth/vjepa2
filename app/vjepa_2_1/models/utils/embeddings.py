import torch
import torch.nn as nn
import torch.nn.functional as F


# Part 1
# Essentially the same as PatchEmbeded3D but while also returning the tublet_embedding_size
class VideoTubletTokenizer(nn.Module):
    """
    Video given as (B, C, T, H, W) to tublet embedding
    """
    def __init__(
            self,
            in_channels=3,
            embedding_dimension=768,
            tublet_depth=2,
            tublet_height=16,
            tublet_width=16
    ):
        super().__init__()
        self.tublet_depth = tublet_depth
        self.tublet_height = tublet_height
        self.tublet_width = tublet_width
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embedding_dimension,
            kernel_size=(tublet_depth, tublet_height, tublet_width),
            stride=(tublet_depth, tublet_height, tublet_width)
        )

    def forward(self, x, **kwargs):
        # B, C, T, H, W = x.shape
        x = self.proj(x)
        tublet_embedding_size = x.shape[2:]
        x = x.flatten(2).transponse(1, 2)
        return x, tublet_embedding_size

# Part 2
# We also need to include positional embeddings.
# I want to refrain from using RoPE and want to learn the positional embeddings (this is sub-optimal but necessary for HRM style JEPA)
# A lot of places suggest that RoPE still is significantly better especially if we are going to train on a fixed image size in pre-training. 
class FactorizedPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dimensions=768,
        max_depth=32,
        max_height=16,
        max_width=16,
    ):
        super().__init__()
        # Define the spatial and temporal params to be learned
        self.spatial_position = nn.Parameter(torch.zeros(1, 1, max_height, max_width, embedding_dimensions))
        self.temporal_position = nn.Parameter(torch.zeros(1, max_depth, 1, 1, embedding_dimensions))

        # Reset the parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)

    def forward(self, x, tublet_embedding_size):
        pass
