import os
import time
import argparse
import torch

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description='Diffusion model to generate video through initial and final image'
    )

    parser.add_argument('--batch_size', default=15, help="The number of instances per batch during training")
    parser.add_argument('--num_sample_per_dataset', default=15, help="The number of dataset per sample")
    parser.add_argument('--eval_batch_size', default=256, help="The number of instances per batch during training")
    parser.add_argument('--n_epoch', default=100, help="The number of epochs to train")
    parser.add_argument('--n_iteration', default=1000, help="The number of epochs to train")
    parser.add_argument('--val_ratio', default=0.1, help="The ratio of the validation dataset")
    parser.add_argument('--seed', type=int, default=1, help="Random seed to use")
    parser.add_argument('--lr', default=1e-4, help="Learning rate")

    parser.add_argument('--dataset_type', default='DrivingCar', help="Dataset to be used in the training, 'DrivingCar' 'BattleGround'")
    parser.add_argument('--embedding_dim', default=128, help="Dimension of feature embedding")

    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda

    return opts