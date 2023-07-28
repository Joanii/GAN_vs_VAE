import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from src.VAE import VAE
from src.GAN import Generator


def load_models():
    vae_model = VAE()
    vae_model.load_state_dict(torch.load(os.path.join('models', 'vae.pt')))
    gen_model = Generator()
    gen_model.load_state_dict(torch.load(os.path.join('models', 'generator.pt')))
    return vae_model, gen_model


def generate_samples(n_samples):
    gen_img = gen(torch.randn(n_samples, 128))
    vae_img = vae.decoder(torch.randn(n_samples, vae.bottleneck_dim)).view(n_samples, 1, 28, 28)
    return gen_img, vae_img


def plotting(gen_images, vae_images):
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
