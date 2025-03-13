import torch
import timm
import numpy as np
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

""" Here, we implement the masked auto encoder framework of He. et al. (https://arxiv.org/abs/2111.06377) """

def random_indexes(size : int):
    # forward_indexes = [0, ..., size-1]
    forward_indexes = np.arange(size)

    # Shuffle indexes
    np.random.shuffle(forward_indexes)

    # Compute inverse permutation to reverse shuffling
    backward_indexes = np.argsort(forward_indexes)

    return forward_indexes, backward_indexes

def take_indexes(patches, indexes):

    # expand indexes from [batch_size, num_patches] to [batch_size, num_patches, embed_dim]
    indexes = repeat(indexes, 'b t -> b t c', c=patches.shape[-1])
    return torch.gather(patches, 1, indexes)

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        batch_size, num_patches, _ = patches.shape

        # Calculate number of remaining patches
        num_remaining_patches = int(num_patches * (1 - self.ratio))

        # Generate two lists of lists for shuffled indexes and inverse permutations
        # For each batch element, there is a list in both lists
        indexes = [random_indexes(num_patches) for _ in range(batch_size)] # [batch_size, 2, num_patches]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=0), dtype=torch.long).to(patches.device) # [batch_size, num_patches]
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=0), dtype=torch.long).to(patches.device) # [batch_size, num_patches]

        shuffled_patches = take_indexes(patches, forward_indexes)
        remaining_shuffled_patches = shuffled_patches[:, :num_remaining_patches, :]

        return remaining_shuffled_patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=64,
                 num_channels=1,
                 patch_size=4,
                 emb_dim=384,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Analogous to the ViTEncoder Module
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        self.patchify = nn.Conv2d(num_channels, emb_dim, patch_size, patch_size)
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)

        # Additional shuffle and masking logic 
        self.shuffle = PatchShuffle(mask_ratio)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        # patchify input image 
        patches = self.patchify(img)  

        # reshape from 2D conv output to [batch_size, num_patches, embed_dim]
        patches = rearrange(patches, 'b c h w -> b (h w) c') 

        # add positional embeddings w/o cls_token
        patches = patches + self.pos_embedding 

        # mask and shuffle - patches: [batch_size, num_patches*(1-mask_ratio), embed_dim]
        patches, _, backward_indexes = self.shuffle(patches)

        # concat cls_token
        cls_token_expanded = self.cls_token.expand(patches.size(0), -1, -1)

        patches = torch.cat([cls_token_expanded, patches], dim=1)

        # forward patches 
        features = self.transformer(patches)
        features = self.layer_norm(features)

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 num_channels=1,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = torch.nn.Linear(emb_dim, (patch_size**2) * num_channels)
        self.activation = torch.nn.Sigmoid()
        self.patch2img = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size, c=num_channels)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):

        num_embed_tokens = features.shape[1]

        # Add placeholders to indexes to account for mask_token 
        zeros = torch.zeros((backward_indexes.size(0), 1), dtype=torch.long).to(features.device)  
        backward_indexes = torch.cat([zeros, backward_indexes + 1], dim=1)

        # Remove encoder cls_token and append mask_token
        remaining_patches = backward_indexes.shape[1] - features.shape[1]
        mask_tokens = self.mask_token.expand(features.shape[0], remaining_patches, -1)
        features = torch.cat([features, mask_tokens], dim=1)

        # Unshuffle
        features = take_indexes(features, backward_indexes)

        # Add positional embedding
        features = features + self.pos_embedding

        # Forward features 
        features = self.transformer(features)

        # Remove cls_token
        features = features[:, 1:, :] 

        # Return to input space 
        patches = self.head(features)
        patches = self.activation(patches)

        # Generate mask
        mask = torch.zeros_like(patches)
        mask[:, :num_embed_tokens-1, :] = 1  
        backward_indexes = backward_indexes[:, 1:] - 1
        mask = take_indexes(mask, backward_indexes)
        
        mask_for_discriminator = torch.all(mask, -1, keepdim=True).float()
        mask_img = self.patch2img(mask)

        img = self.patch2img(patches)
        
        return img, mask_img, mask_for_discriminator

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=64,
                 num_channels=1,
                 patch_size=4,
                 emb_dim=192,
                 encoder_layers=12,
                 encoder_heads=3,
                 decoder_layers=4,
                 decoder_heads=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size=image_size,
                                   num_channels=num_channels,
                                   patch_size=patch_size,
                                   emb_dim=emb_dim,
                                   num_layer=encoder_layers,
                                   num_head=encoder_heads,
                                   mask_ratio=mask_ratio)
        self.decoder = MAE_Decoder(image_size=image_size,
                                   num_channels=num_channels,
                                   patch_size=patch_size,
                                   emb_dim=emb_dim,
                                   num_layer=decoder_layers,
                                   num_head=decoder_heads)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask_img, mask_for_disc = self.decoder(features, backward_indexes)
        return predicted_img, mask_img, mask_for_disc
