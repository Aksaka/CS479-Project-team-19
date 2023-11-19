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

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Initialize model
    model = IFDM(
        opts.embedding_dim,
        opts.dataset_type
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(opts.device)

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
        dataset_path1 = "dataset/DrivingCarDataset.npy"
        dataset1 = torch.from_numpy((np.load(dataset_path1)))
        dataset_path2 = "dataset/DrivingCarSunDataset.npy"
        dataset2 = torch.from_numpy((np.load(dataset_path1)))

        dataset = np.concatenate((dataset1, dataset2), axis=0)

    num_dataset = len(dataset)
    rand_indx = torch.randperm(num_dataset)
    train_dataset = torch.from_numpy(dataset[rand_indx])

    for epoch in range(opts.n_epoch):
        # Train Model
        train_epoch(
            model,
            optimizer,
            epoch,
            train_dataset,
            opts
        )

if __name__ == "__main__":
    run(get_options())