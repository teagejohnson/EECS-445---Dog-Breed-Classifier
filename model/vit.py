"""
EECS 445 - Introduction to Machine Learning
Fall 2024 - Project 2
Target ViT
    Constructs a pytorch model for a vision transformer
    Usage: from model.vit import ViT
"""
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import torch.nn.functional as F
from train_common import patchify, get_positional_embeddings
torch.manual_seed(42)
import pdb
import random
import math
random.seed(42)


# B: Batch Size, d: hidden dimension, n: number of total patches
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_d: int, n_heads: int, mlp_ratio: int = 4):
        """
        Transformer encoder block constructor

        Args:
            hidden_d (int): Dimension of the hidden layer and attention layers.
            n_heads (int): Number of attention heads in the Multi-Head Attention mechanism.
            mlp_ratio (int, optional): Ratio to scale the hidden dimension in the MLP. Default is 4.
        """
        super(TransformerEncoder, self).__init__()

        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.multi_head_attention = MultiHeadAttention(hidden_d,n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d,mlp_ratio*hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio*hidden_d,hidden_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer Encoder block with residual connections.

        Args:
            x (Tensor): Input tensor of shape (B, n+1, d).

        Returns:
            Tensor: Output tensor of the same shape after applying multi-head attention, 
            normalization, and MLP.
        """
        # TODO: Define the foward pass of the Transformer Encoder block as illistrated in 
        #       Figure 4 of the spec using the components defined for you
        #       in the __init__ function.
        # NOTE: Don't forget about the residual connections!

        x = x + self.multi_head_attention.forward(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_features: int, num_heads: int):
        """
        Multi-Head Attention mechanism to compute attention over patches using multiple heads.

        Args:
            num_features (int): Total number of features in the input sequence (patch) embeddings.
            num_heads (int): Number of attention heads to use in the multi-head attention.
        """
        super().__init__()
        
        self.num_features = num_features
        self.num_heads = num_heads
        #Dimension of each atention head's 
        query_size = int(num_features/num_heads)

        #Note: nn.ModuleLists(list) taskes a python list of layers as its parameters
        #The object at the i'th index of the list passed to nn.ModuleList 
        #should corresopnd to the i'th attention head's K,Q, or V respective learned linear mapping
        q_modList_input = [nn.Linear(num_features,query_size) for _ in range(num_heads)]
        self.Q_mappers = nn.ModuleList(q_modList_input)

        k_modList_input = [nn.Linear(num_features,query_size) for _ in range(num_heads)]
        self.K_mappers = nn.ModuleList(k_modList_input)

        v_modList_input = [nn.Linear(num_features,query_size) for _ in range(num_heads)]
        self.V_mappers = nn.ModuleList(v_modList_input)

        self.c_proj = nn.Linear(num_features,num_features)

        self.query_size = query_size
        self.scale_factor = math.sqrt(query_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention

        Args:
            x (Tensor): Input tensor of shape (B, n+1, d).
                        Each sequence represents a sequence of patch embeddings.

        Returns:
            Tensor: Output tensor after applying multi-head attention, 
            the same shape as inputted.
        """
        result = []
        #Remember, we turned each image into a sequence of 16 dimensional "tokens" for our model
        #Loop through the batch of patch embedding sequences
        for sequence in x:
            #each element in seq_result should be a single attention head's
            # attention values
            seq_result = []
            for head in range(self.num_heads):
                #Extract the current head's respective k,q,and v learned linear mappings 
                # from self.Q_mappers, self.K_mappers, and self.V_mappers
                W_k = self.K_mappers[head]
                W_q = self.Q_mappers[head]
                W_v = self.V_mappers[head]

                #Get the given head's k,q,and v representations
                k = W_k(sequence)
                q = W_q(sequence)
                v = W_v(sequence)

                # TODO:Perform scaled dot product self attention, refer to formula
                attention = self.softmax((q @ k.T) / self.scale_factor) @ v

                #Log the current attention head's attention values
                seq_result.append(attention)

            #For the current sequence (patched image) being processed,
            #combine each attention head's attention values columnwise
            result.append(self.c_proj(torch.hstack(seq_result)))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViT(nn.Module):
    def __init__(self,
                 num_patches: int, 
                 num_blocks: int,
                 num_hidden: int,
                 num_heads: int,
                 num_classes: int = 2,
                 chw_shape: tuple = (3,64,64)):
        """
        Vision Transformer (ViT) model that processes an image by dividing it into patches,
        applying transformer encoders, and classifying the image using an MLP head.

        Args:
            num_patches (int): Number of patches to divide the image into along each dimension.
            num_blocks (int): Number of Transformer encoder blocks.
            num_hidden (int): Number of hidden dimensions in the patch embeddings.
            num_heads (int): Number of attention heads in the multi-head attention mechanism.
            num_classes (int, optional): Number of output classes for classification. Default is 2.
            chw_shape (tuple, optional): Shape of the input image in (channels, height, width). Default is (3, 64, 64).
        """

        super(ViT, self).__init__()

        # Attributes
        self.chw = chw_shape
        self.num_patches = num_patches

        #Tip: What would the size of a single patch be given the width/height 
        # of an image and the number of patches? While the final patch size should be 2D,
        # it may be easier to consider each dimesnion separately as a starting point.
        self.patch_size = (self.chw[1] / num_patches, self.chw[2] / num_patches)
        self.embedding_d = num_hidden
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # 1) Patch Tokenizer
            # input_d should hold the number of pixels in a single patch, 
            # dont forget a patch is created with pixels across all img chanels
        self.input_d = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])

        # create a linear layer to embed each patch token
        self.patch_to_token = nn.Linear(self.input_d, self.embedding_d)

        # 2) Learnable classifiation token
        # Use nn.Parameter to create a learnable classification token of shape (1,self.embedding_d)
        self.cls_token = nn.Parameter(torch.rand(1, self.embedding_d))
        
        # 3) Positional embedding
        self.pos_embed = nn.Parameter(get_positional_embeddings(self.num_patches ** 2 + 1, self.embedding_d).clone().detach())
        self.pos_embed.requires_grad = False

        # 4) Transformer encoder blocks
        # Add the number of transformer blocks specified by num_blocks
        transformer_block_list = [TransformerEncoder(num_hidden, num_heads) for _ in range(num_blocks)]
        self.transformer_blocks = nn.ModuleList(transformer_block_list)

        # 5) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_d, num_classes),
            nn.Softmax(dim=-1))
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer (ViT). B is the number of images in a batch

        Args:
            X (Tensor): Input batch of images, tensor of shape (B, channels, height, width).

        Returns:
            Tensor: Classification output of shape (batch_size, num_classes).
        """
        B, C, H, W = X.shape

        #patch images
        patches = patchify(X, self.num_patches)

        # TODO: Get linear projection of each patch. Examine the Vit class's __init__ function
        #       and think of which component is appropriate to use.
        embedded_patches = self.patch_to_token(patches)

        #add classification (sometimes called 'cls') token to the tokenized_patches
        all_tokens = torch.stack([torch.vstack((self.cls_token, embedded_patches[i])) for i in range(len(embedded_patches))])
        
        # Adding positional embedding
        pos_embed = self.pos_embed.repeat(B, 1, 1)
        all_tokens = all_tokens + pos_embed

        # TODO: run the positionaly embedded tokens
        #       through all transformer blocks stored in self.transformer_blocks
        
        for block in self.transformer_blocks:
            all_tokens = block(all_tokens)
        

        # Extract the classification token and put through mlp
        class_tokens = all_tokens[:, 0]
        output_proba = self.mlp(class_tokens)

        return output_proba