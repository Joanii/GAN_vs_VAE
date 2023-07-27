import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

from src.dataset import MNISTdata


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.layers(x)
        return output.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 784
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.layers(x)


class Training:
    def __init__(self, lr=2e-4, batch_size=64):
        self.batch_size = batch_size
        self.dataloader = MNISTdata().train_dataloader(batch_size=self.batch_size)
        self.discriminator = Discriminator()
        self.optimizer_dis = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.generator = Generator()
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.loss_function = nn.BCELoss()

    def training_loop(self, epochs):
        for epoch in range(epochs):
            for batch_idx, (data, _) in enumerate(self.dataloader):
                # Discriminator step real data
                self.discriminator.zero_grad()
                dis_output = self.discriminator(data)
                real_labels = torch.ones((data.shape[0], 1))
                dis_loss = self.loss_function(dis_output, real_labels)
                dis_loss.backward()

                # Discriminator step fake data
                noise = torch.randn(data.shape[0], 128)
                gen_images = self.generator(noise)
                gen_labels = torch.zeros((data.shape[0], 1))
                output = self.discriminator(gen_images.detach())
                dis_gen_loss = self.loss_function(output, gen_labels)
                dis_gen_loss.backward()
                total_loss = dis_loss + dis_gen_loss
                self.optimizer_dis.step()

                # Generator step
                self.generator.zero_grad()
                discriminator_output = self.discriminator(gen_images)
                gen_loss = self.loss_function(discriminator_output, real_labels)
                gen_loss.backward()
                self.optimizer_gen.step()
                print(f'discriminator loss: {total_loss}, generator loss: {gen_loss}')

    def plot_images(self):
        real_images, _ = next(iter(self.dataloader))

        # Plot real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(make_grid(real_images[:64], padding=5, normalize=True), (1, 2, 0)))

        # Plot fake images
        fake_img = self.generator(torch.randn(self.batch_size, 128))
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(make_grid(fake_img.detach()[:64], padding=5, normalize=True), (1, 2, 0)))
        plt.show()


if __name__ == '__main__':
    tr = Training()
    tr.training_loop(10)
    tr.plot_images()
