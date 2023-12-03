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


class DiffusionModule(nn.Module):
    """
    A high-level wrapper of DDPM and DDIM.
    If you want to sample data based on the DDIM's reverse process, use `ddim_p_sample()` and `ddim_p_sample_loop()`.
    """

    def __init__(self, network: nn.Module, var_scheduler: BaseScheduler):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    @property
    def device(self):
        return self.network.parameters().device

    @property
    def image_resolution(self):
        # For image diffusion model.
        return getattr(self.network, "image_resolution", None)

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

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Compute xt.
        alphas_prod_t = extract(self.var_scheduler.alphas_cumprod, t, x0)
        xt = alphas_prod_t.sqrt() * x0 + (1 - alphas_prod_t).sqrt() * noise
        #######################

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
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute x_t_prev.
        batch_size, num_frame, RGB, height, width = xt.size()
        init_image = xt[:, 0, :, :, :].unsqueeze(1)
        noise_image = xt[:, 1:-1, :, :, :]
        final_image = xt[:, -1, :, :, :].unsqueeze(1)

        if isinstance(t, int):
            t = torch.tensor([t]).to(xt.device)

        eps_factor = (1 - extract(self.var_scheduler.alphas, t, noise_image)) / (
            1 - extract(self.var_scheduler.alphas_cumprod, t, noise_image)
        ).sqrt()

        eps_theta = self.network(xt, t)  # [batch_size, num_frame-2, RGB, height, width]
        ###
        if t == 0:
            z = torch.zeros_like(noise_image)
        else:
            z = torch.randn_like(noise_image)

        alpha_t = extract(self.var_scheduler.alphas, t, noise_image)
        sigma = extract(self.var_scheduler.betas, t, noise_image).sqrt()
        noise_image_prev = (noise_image - eps_factor * eps_theta) / alpha_t.sqrt() + sigma * z

        x_t_prev = torch.cat(
            (
                init_image,
                noise_image_prev,
                final_image
            ), 1
        )
        ## Algorithm 2 in DDPM paper
        #######################
        return x_t_prev

    @torch.no_grad()
    def p_sample_loop(self, video):
        """
        The loop of the reverse process of DDPM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # sample x0 based on Algorithm 2 of DDPM paper.
        # x0_pred = torch.zeros(shape).to(self.device)
        batch_size, num_frame, RGB, height, width = video.size()

        init_image = video[0, 0, :, :, :][None, None, :, :, :]
        final_image = video[0, -1, :, :, :][None, None, :, :, :]

        x0_pred = torch.randn([1, num_frame-2, RGB, height, width]).to(video.device)
        x0_pred = torch.cat(
            (
                init_image,
                x0_pred,
                final_image
            ), dim=1
        )

        num_timesteps = self.var_scheduler.num_train_timesteps

        for i in range(num_timesteps):
            t = num_timesteps - i - 1
            x0_pred = self.p_sample(x0_pred, t)
        ######################
        return x0_pred

    # @torch.no_grad()
    # def ddim_p_sample(self, xt, t, t_prev, eta=0.0):
    #     """
    #     One step denoising function of DDIM: $x_t{\tau_i}$ -> $x_{\tau{i-1}}$.
    #
    #     Input:
    #         xt (`torch.Tensor`): noisy data at timestep $\tau_i$.
    #         t (`torch.Tensor`): current timestep (=\tau_i)
    #         t_prev (`torch.Tensor`): next timestep in a reverse process (=\tau_{i-1})
    #         eta (float): correspond to η in DDIM which controls the stochasticity of a reverse process.
    #     Output:
    #        x_t_prev (`torch.Tensor`): one step denoised sample. (= $x_{\tau_{i-1}}$)
    #     """
    #     ######## TODO ########
    #     # DO NOT change the code outside this part.
    #     # compute x_t_prev based on ddim reverse process.
    #     # alpha_prod_t = extract(self.var_scheduler.alphas_cumprod, t, xt)
    #     # if t_prev >= 0:
    #     #     alpha_prod_t_prev = extract(self.var_scheduler.alphas_cumprod, t_prev, xt)
    #     #     alpha_t_prev = extract(self.var_scheduler.alphas, t_prev, xt)
    #     # else:
    #     #     alpha_prod_t_prev = torch.ones_like(alpha_prod_t)
    #     #     alpha_t_prev = torch.ones_like(alpha_prod_t)
    #
    #     # x_t_prev = xt
    #     ###
    #     alpha_t = extract(self.var_scheduler.alphas, t, xt)
    #     alpha_t_prev = extract(self.var_scheduler.alphas, t_prev, xt)
    #     noise = torch.randn_like(xt)
    #     # alpha_t_prev = extract(self.var_scheduler.alphas, t_prev, xt)
    #
    #     sigma_t = eta * ((1 - alpha_t/alpha_t_prev) * (1 - alpha_t_prev) / (1 - alpha_t)).sqrt()
    #     # sigma_t = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)).sqrt()
    #     eps_theta = self.network(xt, t)
    #
    #     x_t_prev = alpha_t_prev.sqrt() * (xt - (1 - alpha_t).sqrt() * eps_theta) / alpha_t.sqrt() + \
    #                (1 - alpha_t_prev - sigma_t ** 2).sqrt() * eps_theta + \
    #                sigma_t * noise
    #     ######################
    #     return x_t_prev
    #
    # @torch.no_grad()
    # def ddim_p_sample_loop(self, shape, num_inference_timesteps=50, eta=0.0):
    #     """
    #     The loop of the reverse process of DDIM.
    #
    #     Input:
    #         shape (`Tuple`): The shape of output. e.g., (num particles, 2)
    #         num_inference_timesteps (`int`): the number of timesteps in the reverse process.
    #         eta (`float`): correspond to η in DDIM which controls the stochasticity of a reverse process.
    #     Output:
    #         x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
    #     """
    #     ######## TODO ########
    #     # DO NOT change the code outside this part.
    #     # sample x0 based on Algorithm 2 of DDPM paper.
    #     step_ratio = self.var_scheduler.num_train_timesteps // num_inference_timesteps
    #     timesteps = (
    #         (np.arange(0, num_inference_timesteps) * step_ratio)
    #         .round()[::-1]
    #         .copy()
    #         .astype(np.int64)
    #     )
    #     timesteps = torch.from_numpy(timesteps)
    #     prev_timesteps = timesteps - step_ratio
    #
    #     x0_pred = torch.randn(shape).to(self.device)
    #     ###
    #     for t, t_prev in zip(timesteps, prev_timesteps):
    #         if t.item() == 0:
    #             continue
    #         t = t.to(self.device)
    #         t_prev = t_prev.to(self.device)
    #         x0_pred = self.ddim_p_sample(x0_pred, t, t_prev, eta)
    #     ######################
    #
    #     return x0_pred

    def compute_loss(self, video):
        """
        The simplified noise matching loss corresponding Equation 14 in DDPM paper.

        Input:
            x0 (`torch.Tensor`): clean data
        Output:
            loss: the computed loss to be backpropagated.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        batch_size, num_frame, RGB, height, width = video.size()
        t = (
            torch.randint(0, self.var_scheduler.num_train_timesteps, size=(batch_size,))
            .to(video.device)
            .long()
        )
        x0 = video[:, 1, :, :, :]
        noise = torch.randn_like(x0)
        xt_pred = self.q_sample(x0, t, noise)
        # xt_pred = torch.cat(
        #     (
        #         x0[:, 0, :, :, :].unsqueeze(1),
        #         xt_pred[:, 1:-1, :, :, :],
        #         x0[:, -1, :, :, :].unsqueeze(1),
        #     ), dim=1
        # )  # [batch_size, num_frame, RGB, height, width]
        eps_theta = self.network(video, t)  # [batch_size, num_frame-2, RGB, height, width]
        eps_theta = eps_theta.reshape([batch_size, -1, RGB, height, width])
        loss = F.mse_loss(noise, eps_theta)
        ######################
        return loss

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
