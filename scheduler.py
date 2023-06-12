import torch


class LinearScheduler():
    """
    The LinearScheduler class performs a linear schedule for adding noise to images in a diffusion models.
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def get_alpha(self):
        return self.alpha
    
    def get_alpha_hat(self):
        return self.alpha_hat
    
    def get_beta(self):
        return self.beta
    

class CosineScheduler():
    """
    The CosineScheduler class implements a cosine schedule for noise generation in a diffusion models.
    It is particularly useful for applications like image denoising or inpainting, where gradually reducing the noise level is desired.
    """
    def __init__(self, noise_steps=1000, s=0.01, device="cuda"):

        self.noise_steps = noise_steps
        self.s = s
        
        self.timesteps = torch.arange(0, self.noise_steps, dtype=torch.float32).to(device)
        self.cosine = torch.pow(
            torch.cos(
                (self.timesteps / self.noise_steps + self.s) / (1 + self.s) * torch.pi / 2.0), 2)
        self.alpha_hat = self.cosine / self.cosine[0]
        self.beta = torch.clip(
            1 - self.alpha_hat / torch.roll(self.alpha_hat, shifts=1, dims=0), min=0, max=0.999)
        self.alpha = 1. - self.beta

    def get_alpha(self):
        return self.alpha

    def get_alpha_hat(self):
        return self.alpha_hat

    def get_beta(self):
        return self.beta