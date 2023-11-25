import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class BaseScheduler(nn.Module):
    """
    Variance scheduler of DDPM.
    """

    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        mode: str = "linear",
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)


class DiffusionModel(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: BaseScheduler):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    @property
    def device(self):
        return self.network.parameters().device

    def q_sample(self, x0, t, noise=None):
        """
        sample x_t from q(x_t | x_0) of DDPM.

        Input:
            x0 (`torch.Tensor`): clean data to be mapped to timestep t in the forward process of DDPM.
            t (`torch.Tensor`): timestep
            noise (`torch.Tensor`, optional): random Gaussian noise. if None, randomly sample Gaussian noise in the function.
        Output:
            xt (`torch.Tensor`): noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alphas_prod_t = extract(self.var_scheduler.alphas_cumprod, t, x0)
        xt = alphas_prod_t.sqrt() * x0 + (1 - alphas_prod_t).sqrt() * noise

        return xt
    
    @torch.no_grad()
    def p_sample(self, xt, t):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            xt (`torch.Tensor`): samples at arbitrary timestep t.
            t (`torch.Tensor`): current timestep in a reverse process.
        Ouptut:
            x_t_prev (`torch.Tensor`): one step denoised sample. (= x_{t-1})

        """
        batch_size, num_frame, RGB, height, width = xt.size()
        init_image = xt[:, 0, :, :, :].unsqueeze(1)
        noise_image = xt[:, 1:-1, :, :, :]
        final_image = xt[:, -1, :, :, :].unsqueeze(1)

        # init_image = xt[0, 0, :, :, :][None, :, :, :]
        # noise_image = xt[0, 1:-1, :, :, :][:, :, :, :]
        # final_image = xt[0, -1, :, :, :][None, :, :, :]

        if isinstance(t, int):
            t = torch.tensor([t]).to(xt.device).repeat([batch_size])
        
        eps_factor = (1 - extract(self.var_scheduler.alphas, t, noise_image)) / (
            1 - extract(self.var_scheduler.alphas_cumprod, t, noise_image)
        ).sqrt()
        eps_theta = self.network(xt, t)  # [num_frame-2, 3(RGB), height, width]
        eps_theta = eps_theta.reshape([batch_size, -1, RGB, height, width])

        sigma = extract(self.var_scheduler.betas, t, noise_image).sqrt()
        #print(sigma.device)
        alpha = extract(self.var_scheduler.alphas, t, noise_image)

        if t[0].item() == 0:
            z = torch.zeros_like(noise_image)
        else:
            z = torch.randn_like(noise_image)

        # x_t_prev = (xt - eps_factor * eps_theta) / (alpha.sqrt()) + sigma * z
        noise_image_prev = (noise_image - eps_factor * eps_theta.squeeze(0)) / (alpha.sqrt()) + sigma * z

        x_t_prev = torch.cat(
            (
                init_image,
                noise_image_prev,
                final_image
            ), 1
        )  # [num_frame, 3, height, width]

        return x_t_prev
    
    @torch.no_grad()
    def p_sample_loop(self, images):
        """
        The loop of the reverse process of DDPM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """

        batch_size, num_frame, RGB, height, width = images.size()

        init_image = images[:, 0, :, :, :].unsqueeze(1)
        final_image = images[:, -1, :, :, :].unsqueeze(1)

        x0_pred = torch.zeros([batch_size, num_frame-2, RGB, height, width]).to(images.device)

        x0_pred = torch.cat(
            (
                init_image,
                x0_pred,
                final_image
            ), dim=1
        )

        for t in range(self.var_scheduler.num_train_timesteps - 1, -1, -1):
            x0_pred = self.p_sample(x0_pred, t)

        return x0_pred

    def compute_loss(self, x0):
        # x0: [batch_size, num_frame, images]
        # DO NOT change the code outside this part.
        # compute noise matching loss.]

        batch_size, num_frame, RGB, height, width = x0.size()
        t = (
            torch.randint(0, self.var_scheduler.num_train_timesteps, size=(batch_size,))
            .to(x0.device)
            .long()
        )

        noise = torch.randn_like(x0)  # [batch_size, num_frame, 3, height, width]
        xt_pred = self.q_sample(x0, t, noise)   # noisy image [batch_size, num_frame, 3, height, width]
        xt_pred = torch.cat(
            (
                x0[:, 0, :, :, :].unsqueeze(1),
                xt_pred[:, 1:-1, :, :, :],
                x0[:, -1, :, :, :].unsqueeze(1),
            ), dim=1
        )
        eps_theta = self.network(xt_pred, t)
        noise = noise[:, 1:-1, :, :, :].reshape(-1, RGB, height, width)

        loss = F.mse_loss(noise, eps_theta)
        ######################
        return loss