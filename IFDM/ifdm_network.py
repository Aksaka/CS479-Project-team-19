import torch
import torch.nn as nn
import torch.nn.functional as F

class IFDM(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dataset_type,
                 n_iteration):
        super(IFDM, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_iteration = n_iteration

        if dataset_type == 'BattleGround':
            height = 200
            width = 200
        else:  # dataset_type == 'DrivingCar'
            height = 180
            width = 320

        # Preprocessing network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer1 = nn.Linear(16 * (height // 8) * (width // 8), 1028)
        self.fc_layer2 = nn.Linear(1028, 512)
        self.fc_layer3 = nn.Linear(512, embedding_dim)

        # Feature merging network
        self.merge_layer = nn.Linear(2*embedding_dim, embedding_dim)

        # Diffusion network
        # TBD by Junhyeok Choi


    def forward(self, input):
        # input -> extract_image_feature -> merge_feature -> diffusion
        # input: [batch_size, num_frame(30), height, width, RGB(3)]
        batch_size, num_frame, height, width, _ = input.size()

        # 이 파트에 처음이랑 마지막 이미지만 남기고 나머지 noise로 만드는 파트 추가해야됨
        # TBD by Junhyeok Choi
        init_image = input[:, 0, :, :, :]
        final_image = input[:, -1, :, :, :]
        new_image = input

        for i in range(self.n_iteration):
            image_feature = self.extract_image_feature(new_image)  # [batch_size, num_frame, embedding_dim]
            merged_feature = self.merge_feature(image_feature)  # [batch_size, num_frame-2, embedding_dim]

            middle_image_next_step = self.IFdiffusion(input[:, 1:-1, :, :, :], merged_feature)  # [batch_size, num_frame, height, width, RGB(3)]
            new_image = torch.cat(
                (
                    init_image[:, None, :, :, :],
                    middle_image_next_step,
                    final_image[:, None, :, :, :]
                ), dim=1
            )

        return new_image

    def IFdiffusion(self, middle_image, merged_feature):
        # middle_image: [batch_size, num_frame-2, height, width, RGB]
        # merged_feature: [batch_size, num_frame, embedding_dim]
        # middle_image_next_step: [batch_size, num_frame-2, height, width, RGB]
        # TBD by Junhyeok Choi
        middle_image_next_step = middle_image
        return middle_image_next_step

    def extract_image_feature(self, input):
        # input: [batch_size, num_frame, height, width, RGB]
        # output: [batch_size, num_frame, 128]
        batch_size, num_frame, height, width, _ = input.size()

        x = input.view(-1, 3, height, width)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_layer1(x))
        x = F.relu(self.fc_layer2(x))
        x = self.fc_layer3(x)
        output = x.view(batch_size, num_frame, -1)

        return output

    def merge_feature(self, image_feature):
        # image_feature: [batch_size, num_frame, embedding_dim]
        # output: [batch_size, num_frame-2, embedding_dim]
        feature_prev = image_feature[:, 0:-2, :]  # [batch_size, num_frame-2, embedding_dim]
        feature_next = image_feature[:, 2:, :]  # [batch_size, num_frame-2, embedding_dim]

        feature_concat = torch.cat((feature_prev, feature_next), dim=-1)
        feature_merged = self.merge_layer(feature_concat)

        return feature_merged