import os.path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.dataset import MNISTdata


class VAE(nn.Module):
    """
    VAE for generating MNIST images.
    """
    def __init__(self, bottleneck_dim: int = 2, beta: float = 1.):
        super().__init__()
        self.image_size = 28
        self.beta = beta
        self.bottleneck_dim = bottleneck_dim
        self.encoder = self._get_encoder_layers()
        self.decoder = self._get_decoder_layers()
        self.mu_layer = nn.Linear(256, self.bottleneck_dim)
        self.sigma_layer = nn.Linear(256, self.bottleneck_dim)

    def _get_encoder_layers(self):
        layers = [nn.Linear(self.image_size**2, 512),
                  nn.ReLU(),
                  nn.Linear(512, 256),
                  nn.ReLU()]
        layers = nn.Sequential(*layers)
        return layers

    def _get_decoder_layers(self):
        layers = [nn.Linear(self.bottleneck_dim, 256),
                  nn.ReLU(),
                  nn.Linear(256, 512),
                  nn.ReLU(),
                  nn.Linear(512, self.image_size**2)]
        layers = nn.Sequential(*layers)
        return layers

    @staticmethod
    def sampling(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        b = self.encoder(x.view(-1, self.image_size**2))
        mu, log_var = self.mu_layer(b), self.sigma_layer(b)
        z = self.sampling(mu, log_var)
        return self.decoder(z).view(x.shape), mu, log_var


class Trainer:
    def __init__(self, batch_size=64, lr=1e-3):
        self.vae = VAE(beta=0.5)
        self.loss_function_sum = nn.MSELoss(reduction='sum')
        self.dataloader = MNISTdata().train_dataloader(batch_size=batch_size)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

    def training_loop(self, epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            for idx, (data, _) in enumerate(self.dataloader):
                self.vae.zero_grad()
                vae_loss = self._step(data)
                vae_loss.backward()
                epoch_loss += vae_loss.item()
                self.optimizer.step()
            print(f'loss: {epoch_loss}')

    def _get_loss(self, recon_x, x, mu, log_var):
        mse = self.loss_function_sum(x, recon_x)
        # print(f'mse: {mse.item()}')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # print(f'kld: {kld.item()}')
        return mse + self.vae.beta * kld

    def _step(self, input_data):
        recon, mu, logvar = self.vae(input_data)
        loss = self._get_loss(recon, input_data, mu, logvar)
        return loss

    def save_model(self):
        folder = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.vae.state_dict(), os.path.join(folder, 'vae.pt'))

    def plot_images(self):
        images, _ = next(iter(self.dataloader))

        # Plot real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(make_grid(images[:64], padding=5, normalize=True), (1, 2, 0)))

        # Plot reconstructed images
        reconstruction, _, _ = self.vae(images)
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Reconstructed Images")
        plt.imshow(np.transpose(make_grid(reconstruction.detach()[:64], padding=5, normalize=True), (1, 2, 0)))
        plt.show()


if __name__ == '__main__':
    tr = Trainer()
    tr.training_loop(50)
    tr.save_model()
    tr.plot_images()
