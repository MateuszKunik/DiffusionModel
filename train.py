import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils import *


class DDPM():
    """
    The DDPM class is an implementation of the Denoising Diffusion Probabilistic Model.
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, image_size=256, image_channels=3, device="cuda"):
        super().__init__()
        
        # The hyperparameters initialization.
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.image_channels = image_channels
        self.device = device

        # The beta tensor contains noise amounts that are applied at every timestep of the diffusion process.
        self.beta = self.noise_schedule().to(device)
        # The alpha tensor contains amounts of image information that are preserved after every timestep of the process.
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = torch.from_numpy(np.append(1.0, self.alpha_hat[:-1].cpu())).to(device=self.device).float()


    def noise_schedule(self):
        """
        This function prepares the noise schedule for the forward diffusion process.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    

    def sample_timesteps(self, n):
        """
        This function returns n integers (time-steps).
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def forward_process(self, x, t):
        """
        This function samples a noised image from a normal distribution,
        where mean = sqrt(alpha_hat) * x and variance = 1 - alpha_hat.
        The function also returns the noise itself.
        """
        # Creating Gaussian noise of x shape.
        noise = torch.randn_like(x)

        # Computing the mean and the variance of distribution.
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]

        # Computing a noised image (image noising).
        noised_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        
        return noised_image, noise
    

    def sample(self, model, n):
        """
        This function denoises n initial images (samples) from a standard normal distribution to reconstruct an original images.
        """
        # Switching the model to evaluation mode.
        model.eval()
        with torch.no_grad():
            # Creating an initial images from N(0, I).
            x = torch.randn((n, self.image_channels, self.image_size, self.image_size)).to(self.device)
            
            # The main process of denoising an images.
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                
                beta_t = self.beta[t][:, None, None, None]
                alpha_t = self.alpha[t][:, None, None, None]
                alpha_hat_t = self.alpha_hat[t][:, None, None, None]
                alpha_hat_prev_t = self.alpha_hat_prev[t][:, None, None, None]

                # Computing predicted original sample from predicted noise
                x_0 = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
                x_0 = x_0.clamp(-1, 1)

                # Computing the mean and variance of the diffusion posterior.
                mean_coef1 = beta_t * torch.sqrt(alpha_hat_prev_t) / (1 - alpha_hat_t)
                mean_coef2 = (1 - alpha_hat_prev_t) * torch.sqrt(alpha_t) / (1 - alpha_hat_t)               
                variance = beta_t * (1 - alpha_hat_prev_t) / (1 - alpha_hat_t)

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = mean_coef1 * x_0 + mean_coef2 * x + variance * noise

        # Switching the model back to training mode.
        model.train()

        # Transforming the reconstructed images.
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train_fn(model, diffusion, loader, optimizer, loss_fn, num_epochs, device, scheduler=None, sampling=True):
    """
    
    """

    time = datetime.now().strftime("%Y%m%d%H%M")
    train_loss = 0.0
    loss_history = []

    # Put the model into training mode
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch} / {num_epochs}\n-------")

        for batch_idx, (images, _) in enumerate(tqdm(loader)):
            # Put data on target device
            images = images.to(device)

            # Forward pass
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.forward_process(images, t)
            predicted_noise = model(x_t, t)
            
            # Calculate loss per batch
            loss = loss_fn(noise, predicted_noise)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        # Calculate and save average loss
        train_loss /= len(loader)
        loss_history.append(train_loss)
        print(f"Train loss: {train_loss:.4f}\n")

        if sampling and ((epoch % (num_epochs // 10) == 0) or (epoch + 1 == num_epochs)):
            sampled_images = diffusion.sample(model, images.shape[0])
            plot_images(sampled_images)

    # Saving model and optimizer state.
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()}

    save_checkpoint(checkpoint, filename=f"checkpoint_{time}.pth.tar")

    return loss_history