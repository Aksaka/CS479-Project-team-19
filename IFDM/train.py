from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

def train_epoch(model, optimizer, epoch, train_dataset, val_dataset, opts):
    print("Start train epoch {}".format(epoch))

    # Start train
    model.train()
    training_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=1)
    for batch_id, batch in enumerate(tqdm(training_dataloader)):
        train_batch(
            model,
            optimizer,
            batch,
            opts
        )

    avg_cost = validate(model, val_dataset, opts)
    print("Validation cost : {}".format(avg_cost))


def train_batch(model, optimizer, dataset, opts):
    dataset = dataset.to(opts.device)  # [batch_size, num_frame, height, width, 3(RGB)]
    output_video = model(dataset)  # [batch_size, num_frame, height, width, 3(RGB)]
    loss = get_loss(dataset, output_video)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_loss(ground_truth, pred):
    loss_func = nn.MSELoss()
    loss = loss_func(ground_truth, pred)
    return loss


def validate(model, val_dataset, opts):
    print("Validating......")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            pred = model(bat.to(opts.device))
        return pred

    pred = torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(val_dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)

    return get_loss(val_dataset, pred).cpu()
