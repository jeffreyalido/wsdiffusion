import matplotlib.pyplot as plt
import wsdiffusion
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the model
model_path = ... # use yours or download WSDM weights from hf https://huggingface.co/jeffreyalido
model = wsdiffusion.utils.load_model(model_path)

# simulate measurement

y = wsdiffusion.utils.load_image_from_path("validation_data/sample_0_y2.png").to(device)
plt.imshow(wsdiffusion.utils.get_tensor_image(y.cpu()))
plt.axis('off')
plt.show()


noise_args = {
    "sizes": y.size(),
    "gaussian_noise_type": "gray_anisotropic",
    "std_lims": (3.0, 3.0),
    "gauss_std_dist": "uniform",
    "isotropic": False,
}


# need to sweep lambda due to the sensitivity of the method
xs = []
for lambda_ in [0.1, 0.5, 1.0, 2.0]:
    print(f"Solving for lambda={lambda_}")
    x = wsdiffusion.solve_inv_ODE(
        model=wsdiffusion.load_model_from_checkpoint(
            "/home/jalido/wsdiffusion/checkpoints/gray_anisotropic_model.ckpt",
            device,
        ),
        y=y.to(device),
        A=lambda x: wsdiffusion.likelihood_models.lens_blur(x, std=1.0),
        lambd=lambda_,
        likelihood="jalal",
        beta=wsdiffusion.Beta(),
        device=device,
        noise_args=noise_args,
    )
    xs.append(x.cpu())
    plt.imshow(wsdiffusion.utils.get_tensor_image(x.cpu()))
    plt.axis('off')
    plt.show()