from collections import OrderedDict

import numpy as np
import torch
from matplotlib.pyplot import imshow
from PIL import Image
from torchvision import transforms

import wsdiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_tensor(t):
    return (t * 2) - 1


def linear_normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


def gaussian_kernel(size: int, std: tuple, isotropic: bool = True) -> torch.Tensor:
    """Generates a 2D Gaussian kernel using PyTorch.

    Args:
        size (int): The size (height and width) of the output 2D kernel.
        std (tuple): (std_x, std_y).
        isotropic (bool): Whether to generate an isotropic (circularly symmetric) kernel.
                          If False, anisotropy_scale is applied to std_y.
        anisotropy_scale (float): Scaling factor for anisotropy if isotropic is False.

    Returns:
        torch.Tensor: A 2D tensor representing the Gaussian kernel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    std_x, std_y = std

    # Generate coordinate grid
    x = torch.arange(size, dtype=torch.float32, device=device)
    y = torch.arange(size, dtype=torch.float32, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")

    center = (size) / 2.0

    # Handle cases where std is very small
    if std_x < 0.5 and std_y < 0.5:
        kernel = torch.zeros((size, size), dtype=torch.float32, device=device)
        kernel[int(center), int(center)] = 1.0
        return kernel
    elif std_x < 0.5:
        kernel = torch.exp(-((y_grid - center) ** 2) / (2 * std_y**2))
        kernel[int(center), :] = kernel[int(center), :]  # keep only row spread
    elif std_y < 0.5:
        kernel = torch.exp(-((x_grid - center) ** 2) / (2 * std_x**2))
        kernel[:, int(center)] = kernel[:, int(center)]  # keep only column spread
    else:
        kernel = torch.exp(
            -(
                ((x_grid - center) ** 2) / (2 * std_x**2)
                + ((y_grid - center) ** 2) / (2 * std_y**2)
            )
        )

    kernel /= kernel.sum()
    return kernel


# Function to generate a batch of 2D Gaussian kernels with varying standard deviations
def generate_gaussian_kernels(
    B: int,
    std_lims: tuple | list,
    size: tuple,
    device: str,
    gauss_std_dist: str = "uniform",
    isotropic: bool = True,
) -> torch.Tensor:
    """
    Generates a batch of 2D Gaussian kernels with varying standard deviations.

    Args:
        B (int): The number of kernels to generate.
        std_lims (tuple): A tuple (min_std, max_std) specifying the range of standard deviations.
        size (tuple): A tuple (H, W) specifying the height and width of each kernel.
        gauss_std_dist (str): Distribution type to sample standard deviations, 'uniform' or 'log-uniform'.

    Returns:
        torch.Tensor: A 4D tensor of shape (B, 1, H, W) containing the Gaussian kernels.
    """

    H, W = size
    kernels = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

    match gauss_std_dist:
        case "log-uniform":
            # Log-uniformly sample standard deviations
            log_min_std = np.log(std_lims[0])
            log_max_std = np.log(std_lims[1])
            stds_x = torch.exp(
                torch.rand(B, device=device) * (log_max_std - log_min_std) + log_min_std
            )
            stds_y = torch.exp(
                torch.rand(B, device=device) * (log_max_std - log_min_std) + log_min_std
            )
        case "uniform":
            # Uniformly sample standard deviations
            stds_x = (
                torch.rand(B, device=device) * (std_lims[1] - std_lims[0]) + std_lims[0]
            )
            stds_y = (
                torch.rand(B, device=device) * (std_lims[1] - std_lims[0]) + std_lims[0]
            )
        case _:
            raise ValueError(f"Unknown gauss_std_dist value: {gauss_std_dist}")

    for i, (std_x, std_y) in enumerate(zip(stds_x, stds_y)):
        if isotropic:
            kernel = gaussian_kernel(max(H, W), (std_x.item(), std_x.item()), isotropic)
        else:
            kernel = gaussian_kernel(max(H, W), (std_x.item(), std_y.item()), isotropic)
        # Resize kernel if necessary (if H != W or kernel size is different)
        if kernel.shape[0] != H or kernel.shape[1] != W:
            kernel = kernel[:H, :W]
        kernels[i, 0, :, :] = kernel

    return kernels


def crop(arr: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
    """Crops the center of the tensor

    Args:
        arr (torch.Tensor): Input tensor
        new_height (int): New height of the crop
        new_width (int): New width of the crop

    Returns:
        torch.Tensor: Cropped tensor
    """
    # Assuming arr is a 2D tensor; adjust if your tensors are in a different format (e.g., with channels)
    height, width = arr.shape[-2], arr.shape[-1]

    start_row = (height - new_height) // 2
    start_col = (width - new_width) // 2

    cropped_arr = arr[
        ..., start_row : start_row + new_height, start_col : start_col + new_width
    ]

    return cropped_arr


def pad0(arr: torch.Tensor) -> torch.Tensor:
    """
    Pads the last two dimensions (spatial dimensions) of a 4D tensor with zeros.

    Args:
        arr (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Padded tensor.
    """
    # Calculate padding sizes for height and width
    pad_height = arr.size(-2) // 2
    pad_width = arr.size(-1) // 2

    # Create a padding tuple for the last two dimensions

    # Apply padding to the last two dimensions
    padded_arr = torch.nn.functional.pad(
        arr,
        pad=(pad_width, pad_width, pad_height, pad_height),
        mode="constant",
        value=0,
    )

    return padded_arr


def fft2d(x: torch.Tensor, norm: str = "backward") -> torch.Tensor:
    """2D Fast Fourier Transform

    Args:
        x (Tensor): _description_

    Returns:
        Tensor: complex-valued 2D FFT
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), norm=norm))


def ifft2d(x: torch.Tensor, norm: str = "backward") -> torch.Tensor:
    """2D Inverse Fast Fourier Transform

    Args:
        x (np.ndarray): thing to fft
        norm (str): normalization

    Returns:
        np.ndarray: complex-valued 2D iFFT
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x), norm=norm))


def radial_average_psd(batch_tensor):
    """
    Compute radially averaged, max-normalized power spectral density (PSD)
    for a batch of RGB images using binary ring masks.

    Args:
        batch_tensor (torch.Tensor): Tensor of shape (B, 3, H, W)

    Returns:
        torch.Tensor: Normalized radial PSDs of shape (B, R)
    """
    B, C, H, W = batch_tensor.shape
    assert C == 3, "Input must have 3 channels"
    device = batch_tensor.device

    # Convert to grayscale
    gray_batch = batch_tensor.mean(dim=1)  # (B, H, W)

    # Compute 2D FFT and power spectrum
    fft = torch.fft.fftshift(torch.fft.fft2(gray_batch), dim=(-2, -1))
    power = torch.abs(fft) ** 2  # (B, H, W)

    # Create radius grid
    y = torch.arange(H, device=device) - H // 2
    x = torch.arange(W, device=device) - W // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    r = r.round().long()  # Integer radius values

    r_max = r.max().item() + 1  # Number of bins
    psd = torch.zeros((B, r_max), device=device)
    counts = torch.zeros((r_max,), device=device)

    for radius in range(r_max):
        mask = r == radius
        count = mask.sum()
        if count == 0:
            continue
        ring_power = power[:, mask]  # (B, N_pixels_in_ring)
        psd[:, radius] = ring_power.mean(dim=1)
        counts[radius] = count

    # Normalize PSD so each row has max value of 1
    psd = psd / psd.max()

    return psd  # (B, R)


def psd(x: torch.Tensor) -> torch.Tensor:
    """Power spectral density

    Args:
        x (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """

    psd = abs(fft2d(x, norm="ortho"))
    return psd


def conv2d(obj: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        obj (torch.Tensor): _description_
        kernel (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    assert obj.shape[-2:] == kernel.shape[-2:]

    conv_result = crop(
        ifft2d(fft2d(pad0(obj)) * fft2d(pad0(kernel))).real,
        obj.shape[-2],
        obj.shape[-1],
    )
    return conv_result


def conv2d_b(
    obj: torch.Tensor, fft_k: torch.Tensor, std: float = 0.1, std_std: float = 0.03
) -> torch.Tensor:
    """for when we have fft(k) directly

    Args:
        obj (torch.Tensor): _description_
        kernel (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """

    # replicate fft_k along the 2nd axis 3 times
    fft_k = torch.repeat_interleave(fft_k, 3, dim=1)

    std = torch.randn(obj.shape[0], 1, 1, 1).to(obj.device) * std_std + std
    mu = torch.randn(obj.shape[0], 1, 1, 1).to(obj.device) * 0.05

    mu = 0
    # conv_result = ifft2d(fft2d(obj) * torch.sqrt(fft_k)).real
    conv_result = torch.abs(torch.fft.ifft2(obj * torch.sqrt(fft_k)))
    std_sample = torch.std(conv_result, dim=(1, 2, 3), keepdim=True)
    conv_result = (
        (conv_result - torch.mean(conv_result, dim=(1, 2, 3), keepdim=True))
        / std_sample
        * std
    ) + mu

    return conv_result


def deconvolve(
    x: torch.Tensor, k: torch.Tensor, eps: float = torch.finfo(torch.float64).eps
) -> torch.Tensor:
    """Deconvolves an image using the provided kernel in the frequency domain. \\Sigma^-1 operation

    Args:
        x (torch.Tensor): image to be deconvolved
        k (torch.Tensor): kernel used for deconvolution
        eps (float, optional): small value to avoid division by zero. Defaults to torch.finfo(torch.float64).eps.

    Returns:
        torch.Tensor: deconvolved image
    """
    fft_k = fft2d(pad0(k))
    # Compute the deconvolution
    return crop(ifft2d(fft2d(pad0(x)) / (fft_k + eps)), x.shape[-2], x.shape[-1]).real


def show_tensor_image(image):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0).contiguous()),  # CHW to HWC
            # transforms.Lambda(lambda t: t * 255.0),
            # transforms.Lambda(lambda t: linear_normalize(t)),
            # transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    imshow(reverse_transforms(image))


def get_tensor_image(image, lin_normalize=False, batch=False):
    # make a custom transform for linear normalization
    if lin_normalize:
        normalize1 = transforms.Lambda(lambda t: t)
        normalize2 = transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min()))
        normalize = transforms.Compose([normalize1, normalize2])
    else:
        normalize = transforms.Lambda(lambda t: (t + 1) / 2)
    reverse_transforms = transforms.Compose(
        [
            normalize,
            transforms.Lambda(lambda t: t.permute(1, 2, 0).contiguous()),  # CHW to HWC
            # transforms.Lambda(lambda t: t * 255.0),
            # transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.Lambda(lambda t: t.numpy().astype(np.float32)),
        ]
    )
    # Take first image of batch if in batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


def radial_decay_mask(size, decay_rate=1.0):
    """
    Creates a 2D mask that decays exponentially radially outwards, starting from 1 at the center.

    Args:
    size (int): The size of the square mask (size x size).
    decay_rate (float): The rate at which the mask decays. Higher values mean faster decay.

    Returns:
    torch.Tensor: The 2D exponential decay mask.
    """
    # Create a grid of distances from the center
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    center_y, center_x = size // 2, size // 2
    r = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Normalize radial distances to be between 0 and 1
    r_norm = r / r.max()

    # Apply exponential decay
    mask = torch.exp(-decay_rate * r_norm)

    return mask


def psnr(recon, gt):
    mse = torch.mean((recon - gt) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def save_image(image, path):
    image = get_tensor_image(image.detach().cpu())

    # tifffile.imwrite(path, image)
    # image /= image.max()
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    # imsave(path, image)
    image = Image.fromarray(image)
    image.save(path)


def load_image_from_path(path):
    """
    Load a JPG image from the specified path and convert it to a tensor.

    Args:
        path (str): Path to the JPG image file.

    Returns:
        torch.Tensor: Image tensor of shape (C, H, W).
    """
    image = Image.open(path).convert("RGB")
    
    transform1 = transforms.ToTensor()
    transform2 = transforms.Lambda(lambda t: normalize_tensor(t))
    transform = transforms.Compose([transform1, transform2])

    return transform(image).unsqueeze(0)  # Add batch dimension


def load_model(model_path, n_blocks):
    model = wsdiffusion.model_arch.UNet(n_blocks=n_blocks)
    cp = torch.load(model_path)  # uncorrelated
    model_state_dict = cp["model_state_dict"]

    # rename the keys to remove the "_orig_mod." prefix

    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[10:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return torch.compile(model)