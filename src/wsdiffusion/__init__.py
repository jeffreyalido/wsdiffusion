import math

import torch

from src import utils, model_arch


__all__ = ["utils", "model_arch"]


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


# def sample_noise(
#     sizes: tuple | list,
#     gaussian_noise: str,
#     gauss_std: tuple = (11, 15),
#     gauss_std_dist: str = "uniform",
#     n: float = 0.5,
#     device: str = "cuda",
#     c_alpha: float = 0.5,
#     isotropic: bool = True,
# ) -> torch.Tensor:
def sample_noise(
    noise_args: dict,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Sample Gaussian Random Fields (GRFs) with different kernel types.

    Args:
        sizes (): Dimensions of the input tensor (B, C, H, W).
        kname (str): Type of kernel to use ('gaussian', 'scale_free', 'combined', 'none').
        gauss_std (tuple, optional): Standard deviation limits for Gaussian kernel. Defaults to (11, 15).
        length_scale_bounds (tuple, optional): Bounds for length scale in scale-free kernel. Defaults to (0.3, 1).
        length_scale_dist (str, optional): Distribution type for length scale. Defaults to "uniform".
        n (float, optional): Split ratio for 'combined' mode. Defaults to 0.5 for half gray, half rgb.
        device (str, optional): Device to perform computation on. Defaults to "cuda".

    Returns:
        torch.Tensor: Generated GRF tensor.
    """
    B, C, H, W = noise_args.get("sizes")
    if noise_args.get("gaussian_noise") == "isotropic":
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
    elif noise_args.get("gaussian_noise") == "anisotropic":
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
    gauss_kname: str,
    gauss_std: float = 2,
    isotropic: bool = True,
    beta: Beta = Beta(),
):
    device = x.device
    noise = sample_noise(
        sizes=x.size(),
        kname=gauss_kname,
        gauss_std=(gauss_std, gauss_std),
        device=device,
        isotropic=isotropic,
    )
    t = torch.tensor([t]).to(device)  # Start time

    return forward_sampling_VP(x, noise, t, beta)
