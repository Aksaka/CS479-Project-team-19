from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


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
            opts
        )

    # avg_cost = validate(model, val_dataset, opts)
    # print("Validation cost : {}".format(avg_cost))


def train_batch(model, optimizer, dataset, opts):
    dataset = dataset.to(opts.device)  # [batch_size, num_frame, height, width, 3(RGB)]
    dataset = torch.cat(
        (
            dataset[:, :, :, :, 0].unsqueeze(2),
            dataset[:, :, :, :, 1].unsqueeze(2),
            dataset[:, :, :, :, 2].unsqueeze(2)
        ), dim=2
    )
    output_video = model(dataset)  # [batch_size, num_frame, height, width, 3(RGB)]


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
