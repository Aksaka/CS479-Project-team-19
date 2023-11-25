from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpm_module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init


class UNet(nn.Module):
    def __init__(self, T, image_resolution, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        self.image_resolution = image_resolution
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        # self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.time_embedding = TimeEmbedding(tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )

        # # for the extract & merging
        self.embedding_dim = ch
        self.height, self.width = image_resolution
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, timestep, class_label=None):
        # Timestep embedding
        batch_size, num_frame, RGB, height, width = x.size()

        temb = self.time_embedding(timestep)
        temb = torch.repeat_interleave(temb, num_frame-2, dim=0)

        # extract features from input x
        x = self.extract_features(x)  # [batch_size, num_frame, self.embedding_dim, height, width]
        # merging features (neighboring noises average)
        x = self.merge_features(x)  # [batch_size, num_frame-2, self.embedding_dim, height, width]
        h = x.reshape(-1, self.embedding_dim, height, width)  # [batch_size*(num_frame-2), self.embedding_dim]

        # Downsampling
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
    
    def extract_features(self, x):
        # extract the x's features
        batch_size, num_frame, RGB, height, width = x.size()

        x = x.reshape(-1, RGB, height, width)
        x = self.head(x)

        return x.reshape([batch_size, num_frame, self.embedding_dim, height, width])
    
    def merge_features(self, features):
        # merging neighboring features
        # features: [batch_size, self.embedding_dim]
        feature_prev = features[:, 0:-2, :]  # [batch_size, num_frame-2, embedding_dim]
        feature_next = features[:, 2:, :]  # [batch_size, num_frame-2, embedding_dim]

        feature_merged = feature_prev + feature_next

        return feature_merged