import os
import json
import pprint as pp
import torch
import torch.optim as optim
import numpy as np
from ifdm_network import IFDM
from options import get_options
from train import train_epoch

def run(opts):
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Initialize model
    model = IFDM(
        opts.embedding_dim,
        opts.dataset_type,
        opts.n_iteration
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=opts.lr)

    # Load Dataset
    if opts.dataset_type == "BattleGround":
        dataset_path1 = "BattleGroundDance/BattleGroundDanceDataset.npy"
        dataset_path2 = "BattleGroundEtc/BattleGroundEtcDataset.npy"
        dataset1 = torch.from_numpy(np.load(dataset_path1))
        dataset2 = torch.from_numpy(np.load(dataset_path2))

        dataset = torch.cat((dataset1, dataset2), dim=0)
    else:  # opts.dataset_type == "DrivingCar"
        dataset_path = "DrivingCar/DrivingCarDataset.npy"
        dataset = torch.from_numpy((np.load(dataset_path)))

    num_dataset = len(dataset)
    rand_indx = torch.randperm(num_dataset)
    dataset_shuffled = dataset[rand_indx]

    val_dataset = dataset[0:int(num_dataset*opts.val_ratio)]
    train_dataset = dataset[int(num_dataset*opts.val_ratio):]

    for epoch in range(opts.n_epochs):
        # Load Dataset
        train_dataset = None
        # [batch_size, num_frame, image]
        # Train Model
        output_video = train_epoch(
            model,
            optimizer,
            epoch,
            train_dataset,
            val_dataset,
            opts
        )

        if epoch == opts.n_epochs-1:
            pass
            # save()

if __name__ == "__main__":
    run(get_options())