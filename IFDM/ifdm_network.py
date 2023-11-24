import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

from pathlib import Path
from ddpm_network import UNet
from ddpm import DiffusionModel, BaseScheduler
from ddpm_data import tensor_to_pil_image, get_data_iterator, AFHQDataModule

class IFDM(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dataset_type,
                 num_diffusion_train_timesteps,
                 ch,
                 ch_mult,
                 attn,
                 num_res_blocks
                 ):
        super(IFDM, self).__init__()

        self.embedding_dim = embedding_dim

        if dataset_type == 'BattleGround':
            height = 200
            width = 200
        else:  # dataset_type == 'DrivingCar'
            height = 180
            width = 320

        # Diffusion network
        # TBD by Junhyeok Choi
        self.log_interval = 500
        self.num_diffusion_train_timesteps = num_diffusion_train_timesteps
        # self.diffusion_train_num_steps = 1000000
        self.img_size = (height, width)

        self.ddpm_network = UNet(
            T=self.num_diffusion_train_timesteps,
            image_resolution=self.img_size,
            ch=ch,
            ch_mult=ch_mult,
            attn=attn,
            num_res_blocks=num_res_blocks,
            dropout=0.1,
        )

        self.var_scheduler = BaseScheduler(
            self.num_diffusion_train_timesteps,
            beta_1=1e-4,
            beta_T=0.02,
            mode="linear",
        )

        self.ddpm = DiffusionModel(self.ddpm_network, self.var_scheduler)


    def forward(self, input, end_flag):
        # input -> extract_image_feature -> merge_feature -> diffusion
        # input: [batch_size, num_frame(30), RGB(3), height, width]
        loss, image_gen = self.IFdiffusion(input, end_flag)  # [batch_size, num_frame, height, width, RGB(3)]
        return loss, image_gen
    
    def IFdiffusion(self, images, end_flag):
        # images: [batch_size, num_frame-2, RGB, height, width]
        # merged_feature: [batch_size, num_frame, embedding_dim]
        # middle_image_next_step: [batch_size, num_frame-2, height, width, RGB]
        image_gen = images
        # if step % self.log_interval == 0:
        #     self.ddpm.eval()
        #
        #     samples = self.ddpm.p_sample_loop((batch_size_train, RGB) + self.img_size)
        #     pil_images = tensor_to_pil_image(samples)
        #     # for i, img in enumerate(pil_images):
        #     #     img.save(self.save_dir / f"step={step}-{i}.png")
        #
        #     self.ddpm.save(f"{self.save_dir}/last.ckpt")
        #     self.ddpm.train()
        # if step % 1000 == 0:
        #     self.ddpm.save(f"{self.save_dir}/{step}.ckpt")

        # images: [batch_size, num_frame, images]
        loss = self.ddpm.compute_loss(images)  # target_image. img: pred_image
        if (end_flag):
            image_gen = self.ddpm.p_sample_loop(images)

        return loss, image_gen