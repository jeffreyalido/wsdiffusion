import torch
from torchvision import datasets, transforms

import wsdiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data transforms
data_transforms = [
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    wsdiffusion.utils.normalize_tensor,
]

data_transform = transforms.Compose(data_transforms)

# Load datasets
train_dataset = datasets.CIFAR10(
    root="./data",
    download=True,
    transform=data_transform,
    train=True,
)
val_dataset = datasets.CIFAR10(
    root="./data",
    download=True,
    transform=data_transform,
    train=False,
)

# Create dataloaders
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

# Initialize model and training components
num_epochs = 1000
lr = 3e-5

model = wsdiffusion.model_arch.UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
beta = wsdiffusion.Beta()


def wsloss_criterion(
    predicted_ws: torch.Tensor, gt_ws: torch.Tensor, beta, t
) -> torch.Tensor:
    # weight each difference by beta(t)
    weighted_diff = beta(t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (
        predicted_ws - gt_ws
    )
    return torch.mean(weighted_diff**2)


noise_args = {
    "std_lims": (0.1, 0.1),
    "gauss_std_dist": "uniform",
    "gaussian_noise_type": "isotropic",
    "isotropic": True,
}

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0

    for batch_idx, (x0, _) in enumerate(train_dataloader):
        x0 = x0.to(device)
        optimizer.zero_grad()

        # Sample noise and timesteps
        noise_args.update({"sizes": x0.size()})
        Kz = wsdiffusion.sample_noise(noise_args, device=device)
        t = torch.randint(1, 1000, (x0.size(0),)).to(device).float() / 1000

        # Forward diffusion
        xt = wsdiffusion.forward_sampling_VP(x0, sampled_noise=Kz, t=t, beta=beta)
        ws_gt = wsdiffusion.VP_GGscore(noise=Kz, beta=beta, t=t)

        # Model prediction
        ws_pred = model(xt, t)
        loss = wsloss_criterion(ws_pred, ws_gt, beta, t)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (x0, _) in enumerate(val_dataloader):
            x0 = x0.to(device)

            # Sample noise and timesteps
            noise_args.update({"sizes": x0.size()})
            Kz = wsdiffusion.sample_noise(noise_args, device=device)
            t = torch.randint(1, 1000, (x0.size(0),)).to(device).float() / 1000

            # Forward diffusion
            xt = wsdiffusion.forward_sampling_VP(x0, sampled_noise=Kz, t=t, beta=beta)
            ws_gt = wsdiffusion.VP_GGscore(noise=Kz, beta=beta, t=t)

            # Model prediction
            ws_pred = model(xt, t)
            loss = wsloss_criterion(ws_pred, ws_gt, beta, t)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)

    # Print epoch statistics
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
    )

    # Optional: Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            f"checkpoint_epoch_{epoch + 1}.pth",
        )

print("Training complete!")
