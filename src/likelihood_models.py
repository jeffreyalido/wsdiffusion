import torch
import torch.nn.functional as F

import wsdiffusion



def horizontal_blur(image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Applies a horizontal blur to the input image using a 1D convolution.
    Supports both even and odd kernel sizes.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W).
        kernel_size (int): Size of the blur kernel (must be >= 1).

    Returns:
        torch.Tensor: Blurred image tensor of the same shape as input.
    """
    if kernel_size < 1:
        raise ValueError("Kernel size must be >= 1.")

    # Create a horizontal blur kernel (1D average filter)
    kernel = torch.ones(1, 1, 1, kernel_size, device=image.device) / kernel_size

    # Ensure input image has a batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

    # Calculate asymmetric padding for even-sized kernels
    pad_left = kernel_size // 2
    pad_right = kernel_size - pad_left - 1

    # Apply asymmetric padding (left and right along width)
    image_padded = F.pad(image, (pad_left, pad_right, 0, 0), mode="replicate")

    # Apply convolution with groups for channel-wise filtering
    blurred_image = F.conv2d(
        image_padded,
        kernel.expand(image.shape[1], 1, 1, kernel_size),
        groups=image.shape[1],
    )

    return blurred_image.squeeze(0) if image.dim() == 4 and image.shape[0] == 1 else blurred_image


def lens_blur(images: torch.Tensor, std: float) -> torch.Tensor:
    """
    Convolve a 32x32 kernel with a batch of RGB images.

    Args:
        images (torch.Tensor): Input tensor of shape [B, 3, 32, 32]
        kernel (torch.Tensor): 2D kernel of shape [32, 32]

    Returns:
        torch.Tensor: Output tensor of shape [B, 1, 1, 1]
    """
    # assert (
    #     images.ndim == 4 and images.shape[1] == 3 and images.shape[2:] == (32, 32)
    # ), "Expected input shape [B, 3, 32, 32]"

    kernel = wsdiffusion.utils.gaussian_kernel(images.shape[-1], std=(std, std))

    output = wsdiffusion.utils.conv2d(images, kernel)

    return output