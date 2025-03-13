import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block


""" Here, we implement a ViT based classifier, that can be initialized with a pre-trained encoder.
This is the model that was used for classification experiments in the thesis."""



class ViTEncoder(nn.Module):
    def __init__(self, 
                 image_size=64, 
                 num_channels=1, 
                 patch_size=4, 
                 emb_dim=384, 
                 num_layer=12, 
                 num_head=3, 
                 pretrained_model = None):
        
        super().__init__()
        self.patch_size = patch_size
        if pretrained_model is not None: 
            self.num_patches = pretrained_model.encoder.num_patches
            self.cls_token = pretrained_model.encoder.cls_token
            self.pos_embedding = pretrained_model.encoder.pos_embedding
            self.patchify = pretrained_model.encoder.patchify
            self.transformer = pretrained_model.encoder.transformer
            self.layer_norm = pretrained_model.encoder.layer_norm
        else:
            self.num_patches = (image_size // self.patch_size) ** 2
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
            self.patchify = nn.Conv2d(num_channels, emb_dim, patch_size, patch_size)
            self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
            self.layer_norm = nn.LayerNorm(emb_dim)
            
            self.init_weight()


    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)


    def forward(self, img):
        # patchify input image 
        patches = self.patchify(img)  
        patches = rearrange(patches, 'b c h w -> b (h w) c') # reshape from 2D conv output to [batch_size, num_patches, embed_dim]

        # add positional embeddings w/o cls_token
        patches = patches + self.pos_embedding 

        # concat cls_token
        cls_token_expanded = self.cls_token.expand(patches.size(0), -1, -1)  
        patches = torch.cat([cls_token_expanded, patches], dim=1) 
       
        # forward patches 
        features = self.transformer(patches)
        features = self.layer_norm(features)

        # output shape is [batch_size, num_patches + 1, embed_dim]
        return features


class ViT(nn.Module):
    def __init__(self, 
                 num_classes=9, 
                 image_size=64, 
                 num_channels=1,
                 patch_size=4, 
                 emb_dim=384, 
                 encoder_layer=12, 
                 encoder_head=3,
                 mode='pairwise',
                 pretrained_model=None):
        super().__init__()
        
        self.mode = mode
        self.encoder = ViTEncoder(image_size, num_channels, patch_size, emb_dim, encoder_layer, encoder_head, pretrained_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),              # Normalize the embeddings from the encoder
            nn.Linear(emb_dim, 2048),     # Map to number of LED types
            nn.ReLU(),
            nn.Linear(2048, num_classes if num_classes !=0 else 1)     # Map to number of LED types
        )        
        

    def forward(self, img):
        features = self.encoder(img)
        cls_token = features[:, :1, :]
        classification = self.classifier(cls_token)

        # Save logits for CAM computation 
        self.logits = classification 
        
        classification = classification.squeeze(1)
        return classification
    