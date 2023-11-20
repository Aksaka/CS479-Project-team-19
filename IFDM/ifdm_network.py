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

        # Preprocessing network
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(3, 8, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(8, 16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.fc_layer1 = nn.Linear(16 * (height // 8) * (width // 8), 1028)
        # self.fc_layer2 = nn.Linear(1028, 512)
        # self.fc_layer3 = nn.Linear(512, embedding_dim)

        # Feature merging network
        # self.merge_layer = nn.Linear(2*embedding_dim, embedding_dim)

        # Diffusion network
        # TBD by Junhyeok Choi
        self.log_interval = 500
        self.num_diffusion_train_timesteps = num_diffusion_train_timesteps
        # self.diffusion_train_num_steps = 1000000
        self.img_size = (height, width)
        self.save_dir = Path("results/diffusion/{}".format(time.strftime("%Y%m%dT%H%M%S")))
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.ddpm_network = UNet(
            T=self.num_diffusion_train_timesteps,
            image_resolution=self.img_size,
            ch=ch,
            ch_mult=ch_mult,
            attn=[1],
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


    def forward(self, input):
        # input -> extract_image_feature -> merge_feature -> diffusion
        # input: [batch_size, num_frame(30), RGB(3), height, width]
        batch_size, num_frame, height, width, _ = input.size()

        # 이 파트에 처음이랑 마지막 이미지만 남기고 나머지 noise로 만드는 파트 추가해야됨
        # TBD by Junhyeok Choi
        init_image = input[:, 0, :, :, :]
        final_image = input[:, -1, :, :, :]
        new_image = input

        # image_feature = self.extract_image_feature(new_image)  # [batch_size, num_frame, embedding_dim]
        # merged_feature = self.merge_feature(image_feature)  # [batch_size, num_frame-2, embedding_dim]

        loss, middle_image_diffusion = self.IFdiffusion(input, self.num_diffusion_train_timesteps)  # [batch_size, num_frame, height, width, RGB(3)]
        new_image = torch.cat(
            (
                init_image[:, None, :, :, :],
                middle_image_diffusion,
                final_image[:, None, :, :, :]
            ), dim=1
        )

        return loss, new_image
    
    def IFdiffusion(self, images, train_num_steps):
        # images: [batch_size, num_frame-2, RGB, height, width]
        # merged_feature: [batch_size, num_frame, embedding_dim]
        # middle_image_next_step: [batch_size, num_frame-2, height, width, RGB]
        # TBD by Junhyeok Choi
        middle_image_next_step = images
        batch_size, num_frame, RGB, height, width = images.size()

        # optimizer = torch.optim.Adam(self.ddpm.network.parameters(), lr=2e-4)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=lambda t: min((t + 1) / 200, 1.0)
        # )

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

        return loss, middle_image_next_step


    # 보류
    # def extract_image_feature(self, input):
    #     # input: [batch_size, num_frame, height, width, RGB]
    #     # output: [batch_size, num_frame, 128]
    #     batch_size, num_frame, height, width, _ = input.size()
    #
    #     x = input.view(-1, 3, height, width)
    #     x = self.conv_layers(x)
    #     x = x.view(x.size(0), -1)
    #     x = F.relu(self.fc_layer1(x))
    #     x = F.relu(self.fc_layer2(x))
    #     x = self.fc_layer3(x)
    #     output = x.view(batch_size, num_frame, -1)
    #
    #     return output
    #
    # def merge_feature(self, image_feature):
    #     # image_feature: [batch_size, num_frame, embedding_dim]
    #     # output: [batch_size, num_frame-2, embedding_dim]
    #     feature_prev = image_feature[:, 0:-2, :]  # [batch_size, num_frame-2, embedding_dim]
    #     feature_next = image_feature[:, 2:, :]  # [batch_size, num_frame-2, embedding_dim]
    #
    #     feature_concat = torch.cat((feature_prev, feature_next), dim=-1)
    #     feature_merged = self.merge_layer(feature_concat)
    #
    #     return feature_merged
