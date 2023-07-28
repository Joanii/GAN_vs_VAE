import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from typing import Tuple
from src.VAE import VAE
from src.GAN import Generator


def load_models() -> Tuple[VAE, Generator]:
    """
    Load the two models for sample generation: VAE and Generator part of the GAN.
    :return: vae, generator
    """
    vae_model = VAE()
    vae_model.load_state_dict(torch.load(os.path.join('models', 'vae.pt')))
    gen_model = Generator()
    gen_model.load_state_dict(torch.load(os.path.join('models', 'generator.pt')))
    return vae_model, gen_model


def generate_samples(n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate samples with the VAE decoder and Generator by feeding them with noise.
    :param n_samples: number of samples that are generated
    :return: images generated with Generator (GAN), images generated with VAE (decoder)
    """
    gen_img = gen(torch.randn(n_samples, 128))
    vae_img = vae.decoder(torch.randn(n_samples, vae.bottleneck_dim)).view(n_samples, 1, 28, 28)
    return gen_img, vae_img


def plotting(gen_images: torch.Tensor, vae_images: torch.Tensor):
    """
    Plot the generated images for comparison.
    :param gen_images: GAN generated images
    :param vae_images: VAE generated images
    :return:
    """
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("VAE generated images")
    plt.imshow(np.transpose(make_grid(vae_images.detach(), padding=5, normalize=True), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("GAN generated images")
    plt.imshow(np.transpose(make_grid(gen_images.detach(), padding=5, normalize=True), (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    n_samples = 64
    vae, gen = load_models()
    gen_img, vae_img = generate_samples(n_samples)
    plotting(gen_img, vae_img)
