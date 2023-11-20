from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time

from pathlib import Path
from ddpm_data import tensor_to_pil_image


def train_epoch(model, optimizer, epoch, train_dataset, opts):
    print("Start train epoch {}".format(epoch))

    # Start train
    model.train()
    training_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=1)

    batch_size = opts.batch_size
    num_iter = int(train_dataset.size(0)/opts.batch_size)

    for i in tqdm(range(num_iter)):
        batch = train_dataset[i*batch_size:(i+1)*batch_size]
        train_batch(
            model,
            optimizer,
            batch,
            opts,
            i,
            num_iter
        )

    # avg_cost = validate(model, val_dataset, opts)
    # print("Validation cost : {}".format(avg_cost))


def train_batch(model, optimizer, dataset, opts, i, max_i):
    dataset = dataset.to(opts.device)  # [batch_size, num_frame, height, width, 3(RGB)]
    dataset = torch.cat(
        (
            dataset[:, :, :, :, 0].unsqueeze(2),
            dataset[:, :, :, :, 1].unsqueeze(2),
            dataset[:, :, :, :, 2].unsqueeze(2)
        ), dim=2
    )
    optimizer.zero_grad()
    loss, output_video = model(dataset)  # [batch_size, num_frame, height, width, 3(RGB)]

    if ( i == (max_i - 1) ):
        save_dir = Path(opts.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        output_image = tensor_to_pil_image(output_video[-1, -1, :, :, :])
        output_image.save(save_dir / f"last_image.png") 


    loss.backward()
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
