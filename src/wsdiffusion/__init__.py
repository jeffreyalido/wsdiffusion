import math

import torch

from src import utils, model_arch, likelihood_models


__all__ = ["utils", "model_arch", "likelihood_models"]

class Beta:
    """noise schedule"""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return (self.beta_min + (self.beta_max - self.beta_min) * t).view(-1, 1, 1, 1)

    def integrate(self, t: torch.Tensor) -> torch.Tensor:
        return (t * ((self.beta_max - self.beta_min) * t + 2 * self.beta_min)) / 2

    def calc_R1(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.integrate(t)).view(-1, 1, 1, 1)

    def calc_R2(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.integrate(t)).view(-1, 1, 1, 1)

    def calc_SNR(self, t: torch.Tensor) -> torch.Tensor:
        return self.calc_R1(t) / (torch.sqrt(1 - self.calc_R2(t)))

    def calc_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.calc_R1(t)


def sample_noise(
    noise_args: dict,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Sample Gaussian Random Fields (GRFs) with different kernel types.

    Args:
        noise_args = {
            "sizes": (B, C, H, W),
            "gaussian_noise_type": "isotropic" or "anisotropic",
            "std_lims": (min_std, max_std),
            "gauss_std_dist": "uniform" or "log_uniform",
            "n": fraction of gray-scale images in case of anisotropic noise,
            "isotropic": bool, whether to use isotropic kernels
        }

    Returns:
        torch.Tensor: Generated GRF tensor.
    """
    B, C, H, W = noise_args.get("sizes")
    if noise_args.get("gaussian_noise_type") == "rgb_anisotropic":
        z = torch.randn(B, 3, H, W, device=device)
        k = utils.generate_gaussian_kernels(
            B=z.shape[0],
            std_lims=(noise_args.get("std_lims")[0], noise_args.get("std_lims")[1]),
            size=(z.shape[-2], z.shape[-1]),
            gauss_std_dist=noise_args.get("gauss_std_dist"),
            device=device,
            isotropic=noise_args.get("isotropic"),
        )
        sampled_noise = utils.conv2d(obj=z, kernel=k)
    elif noise_args.get("gaussian_noise_type") == "mix_anisotropic":
        B_gray = int(B * noise_args.get("n"))
        B_rgb = B - B_gray

        z_gray = torch.randn(B_gray, 1, H, W, device=device)
        z_rgb = torch.randn(B_rgb, 3, H, W, device=device)

        z_gray = torch.repeat_interleave(z_gray, 3, dim=1)
        z = torch.cat([z_gray, z_rgb], dim=0)

        k = utils.generate_gaussian_kernels(
            B=z.shape[0],
            std_lims=(noise_args.get("std_lims")[0], noise_args.get("std_lims")[1]),
            size=(z.shape[-2], z.shape[-1]),
            gauss_std_dist=noise_args.get("gauss_std_dist"),
            device=device,
            isotropic=noise_args.get("isotropic"),
        )
        c_alpha = torch.rand(1, device=device)
        noise = torch.randn_like(z) * c_alpha

        sampled_noise = (utils.conv2d(obj=z, kernel=k) + noise) / math.sqrt(1 + c_alpha)
    elif noise_args.get("gaussian_noise_type") == "gray_anisotropic":
        z = torch.randn(B, 1, H, W, device=device)
        z = torch.repeat_interleave(z, 3, dim=1)
        k = utils.generate_gaussian_kernels(
            B=z.shape[0],
            std_lims=(noise_args.get("std_lims")[0], noise_args.get("std_lims")[1]),
            size=(z.shape[-2], z.shape[-1]),
            gauss_std_dist=noise_args.get("gauss_std_dist"),
            device=device,
            isotropic=noise_args.get("isotropic"),
        )
        sampled_noise = utils.conv2d(obj=z, kernel=k)
    elif noise_args.get("gaussian_noise_type") == "isotropic":
        sampled_noise = torch.randn(B, C, H, W, device=device)
    else:
        raise ValueError(
            f"Unknown gaussian_noise_type: {noise_args.get('gaussian_noise_type')}"
        )

    return sampled_noise


def forward_sampling_VP(
    x0: torch.Tensor,
    sampled_noise: torch.Tensor,
    t: torch.Tensor,
    beta,
) -> torch.Tensor:
    """transition kernel for VP SDE SBR

    Args:
        x0 (torch.Tensor): (B, 3, H, W) tensor of uncorrupted image (B: batch size, H: height, W: width)
        grf (torch.Tensor): (B, 3, H, W) tensor of Gaussian random field (B: batch size, H: height, W: width)
        t (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    R2 = beta.calc_R2(t)  # equal to alpha_t**2
    diffusion_term = sampled_noise * torch.sqrt(1 - R2)

    drift_coef = beta.calc_alpha_t(t)
    return x0 * drift_coef + diffusion_term


def VP_score(
    z: torch.Tensor, k: torch.Tensor, t: torch.Tensor, beta: Beta
) -> torch.Tensor:
    """calculate the score for the VP SDE SBR

    Args:
        z (torch.Tensor): same exact noise sample that was used for the forward sampling
        k (torch.Tensor): same exact kernel that was used for the forward sampling
        t (torch.Tensor): same exact time step that was used for the forward sampling

    Returns:
        torch.Tensor: score for the VP SDE SBR
    """
    R = beta.calc_R2(t)
    scalar = 1 / (1e-10 + torch.sqrt(1 - R))
    return utils.deconvolve(x=z, k=k, eps=3.2e4) * scalar


def VP_GGscore(noise: torch.Tensor, beta: Beta, t: torch.Tensor) -> torch.Tensor:
    """calculate G*G*score for the VP SDE

    Args:
        noise (torch.Tensor): same exact noise sample that was used for the forward sampling (Kz)
        t (torch.Tensor): same exact time step that was used for the forward sampling

    Returns:
        torch.Tensor: whitened score for the VP SDE
    """
    scalar = 1 * beta(t) / torch.sqrt(1 - beta.calc_alpha_t(t) ** 2)
    return scalar * noise


def add_noise(
    x: torch.tensor,
    t: float,
    noise_args: dict,
    beta: Beta = Beta(),
    device: str = "cuda",
):
    if not noise_args:
        noise_args = {
            "sizes": x.size(),
            "gaussian_noise_type": "isotropic",
            "std_lims": (0.1, 0.1),
            "gauss_std_dist": "uniform",
            "isotropic": True,
        }
    noise = sample_noise(
        noise_args=noise_args,
        device=device,
    )

    t = torch.tensor([t]).to(device)  # Start time

    return forward_sampling_VP(x, noise, t, beta)


def solve_inv_ODE(
    model: torch.nn.Module,
    y: torch.tensor,
    A: callable,
    lambd: float,
    likelihood: str = "jalal",
    beta: Beta = Beta(),
    device: str = "cuda",
    noise_args: dict = None,
):
    """_summary_

    Args:
        model (torch.nn.Module): trained ws-diffusion model prior
        y (torch.tensor): noisy measurement
        A (callable): likelihood forward operator
        lambd (float): regularization parameter for the inverse problem
        likelihood (str, optional): likelihood model to use. Defaults to "jalal".
        beta (Beta, optional): Beta schedule for the diffusion process. Defaults to Beta().
        device (str, optional): device to run the computation on. Defaults to "cuda".
        noise_args (dict, optional): additional arguments for noise sampling. Defaults to None.

    Returns:
        _type_: x_hat, the reconstructed image
    """
    T = 1000
    if not noise_args:
        noise_args = {
            "sizes": y.size(),
            "gaussian_noise_type": "isotropic",
            "std_lims": (1.0, 1.0),
            "gauss_std_dist": "uniform",
            "isotropic": True,
        }
    x = sample_noise(
        noise_args=noise_args,
        device=device,
    )
    x.requires_grad = True
    t_array = torch.linspace(1001 - T, 1000, T).to(device)  # Start time
    t_array = (t_array.int()) / 1000
    for _, t in enumerate(reversed(t_array)):
        with torch.no_grad():
            ggscore = model(x, torch.tensor([t], device=device))

        if likelihood == "jalal":
            log_likelihood = torch.norm(y - A(x), 2)
            x_prime = (
                (2 - torch.sqrt(1 - beta(t) / 1000)) * x
                + ggscore
                / 1000
                / 2 
            )
            meas_match = torch.autograd.grad(outputs=log_likelihood, inputs=x)[0]
            ggscore_mag = torch.norm(ggscore, 2)
            meas_match_mag = torch.norm(meas_match, 2)
            meas_match /= meas_match_mag
            scalar = lambd * ggscore_mag
            meas_match *= scalar
            x = x_prime - meas_match / 1000 / 2
        elif likelihood == "dps":
            x_hat = ggscore * (1 - beta.calc_R2(t)) / beta(t) + x
            log_likelihood = torch.norm(y - A(x_hat), 2)
            x_prime = (2 - torch.sqrt(1 - beta(t) / 1000)) * x + ggscore / 1000 / 2
            meas_match = torch.autograd.grad(outputs=log_likelihood, inputs=x)[0]
            ggscore_mag = torch.norm(ggscore, 2)
            meas_match /= torch.norm(meas_match, 2)
            scalar = lambd * ggscore_mag
            meas_match *= scalar
            x = x_prime - meas_match / 1000 / 2

    # tweedie's formula
    with torch.no_grad():
        x = (
            model(x, torch.tensor([t], device=device)) * (1 - beta.calc_R2(t)) / beta(t)
            + x
        )

    return x
