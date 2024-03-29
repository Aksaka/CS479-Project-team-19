import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import tensor_to_pil_image
from dotmap import DotMap
from ddpm import DiffusionModule, BaseScheduler
from network import UNet
from pytorch_lightning import seed_everything
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    now = get_current_time()
    save_dir = Path(f"results/diffusion-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    image_resolution = config.image_resolution

    var_scheduler = BaseScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode="linear",
    )
    # if isinstance(var_scheduler, DDIMScheduler):
    #     var_scheduler.set_timesteps(20)  # 20 steps are enough in the case of DDIM.

    network = UNet(
        T=config.num_diffusion_train_timesteps,
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
    )

    ddpm = DiffusionModule(network, var_scheduler)

    if args.use_cuda and torch.cuda.device_count() > 1:
        ddpm = torch.nn.DataParallel(ddpm)

    ddpm.to(device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    dataset = torch.load('CarDataset64/DrivingCarDataset_5_500.pt')
    dataset = torch.cat(
        (
            dataset[:, :, :, :, 0].unsqueeze(2),
            dataset[:, :, :, :, 1].unsqueeze(2),
            dataset[:, :, :, :, 2].unsqueeze(2)
        ), dim=2
    ).float()
    dataset = 2 * (dataset / 255 - 0.5)
    dataset = dataset.clone().detach()
    # normalize between [-1, +1]
    idx = torch.arange(config.batch_size)

    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            video = dataset[idx].to(device)

            if step % config.log_interval == 0 and step != 0:
                ddpm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                # samples = ddpm.p_sample_loop((4,3,64,64))
                video_sample = dataset[0].unsqueeze(0).to(device)
                samples = ddpm.p_sample_loop(video_sample)
                pil_images = tensor_to_pil_image(samples.squeeze())
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                ddpm.save(f"{save_dir}/last.ckpt")
                ddpm.train()

            if step % 1000 == 0:
                ddpm.save(f"{save_dir}/{step}.ckpt")

            loss = ddpm.compute_loss(video)
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)
            idx = (idx+2) % config.batch_size

    print(f"last.ckpt is saved at {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action='store_true', help='Disable CUDA')
    parser.add_argument("--use_cuda", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,  ######
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=1000,
        help="max number of images per category for AFHQ dataset",
    )
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=108)

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)
