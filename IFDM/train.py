import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
import cv2

from pathlib import Path
from ddpm_data import tensor_to_pil_image


def train_epoch(model, optimizer, epoch, train_dataset, opts):
    print("Start train epoch {}".format(epoch))

    # Start train
    model.train()

    num_dataset = len(train_dataset)
    rand_indx = torch.randperm(num_dataset)
    train_dataset = train_dataset[rand_indx]

    batch_size = opts.batch_size
    num_iter = int(train_dataset.size(0)/opts.batch_size)

    for i in tqdm(range(num_iter)):
        end_flag = (i == num_iter-1)

        batch = train_dataset[i*batch_size:(i+1)*batch_size]
        train_batch(
            model,
            optimizer,
            batch,
            opts,
            epoch,
            i,
            end_flag
        )

    # avg_cost = validate(model, val_dataset, opts)
    # print("Validation cost : {}".format(avg_cost))


def train_batch(model, optimizer, dataset, opts, epoch, i, end_flag):
    dataset = dataset.to(opts.device)  # [batch_size, num_frame, height, width, 3(RGB)]
    dataset = torch.cat(
        (
            dataset[:, :, :, :, 0].unsqueeze(2),
            dataset[:, :, :, :, 1].unsqueeze(2),
            dataset[:, :, :, :, 2].unsqueeze(2)
        ), dim=2
    )

    loss, output_video_tensor = model(dataset, end_flag)  # [batch_size, num_frame, height, width, 3(RGB)]

    if (end_flag): # save the last frame when the epoch is end
        batch_size, num_frame, height, width, RGB = output_video_tensor.size()
        save_dir = opts.save_dir + '/output_{}.pt'.format(epoch)

        torch.save(output_video_tensor, save_dir)
        # save_dir = Path(opts.save_dir)
        # save_dir.mkdir(exist_ok=True, parents=True)
        # frame_array = []
        #
        # for i in range(num_frame):
        #     output_image = tensor_to_pil_image(output_video_tensor[-1, i, :, :, :])
        #     frame_array.append(output_image)
        # output = cv2.VideoWriter(opts.save_dir + '/output_{}.mp4'.format(epoch), cv2.VideoWriter_fourcc(*'DIVX'), fps=30, frameSize=(width, height))
        #
        # for i in range(num_frame):
        #     output.write(frame_array[i])
        # output.release()
        #output_image.save(save_dir / f"last_image.png")
        print('Loss: {}'.format(loss.mean()))

    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    # input i 가지고 마지막이면 저장하는 거 추가


# def validate(model, val_dataset, opts):
#     print("Validating......")
#     model.eval()
#
#     def eval_model_bat(bat):
#         with torch.no_grad():
#             pred = model(bat.to(opts.device))
#         return pred
#
#     pred = torch.cat([
#         eval_model_bat(bat)
#         for bat
#         in tqdm(DataLoader(val_dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
#     ], 0)
#
#     return get_loss(val_dataset, pred).cpu()
