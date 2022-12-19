import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data.dataset import ColorizationDataset
from src.model.descriminator.patchGAN import PatchGAN
from src.model.generator.unet import Unet
from torchvision import transforms
from src.model.fix import generator as G, discriminator as D

from src.model.loss import Loss
from src.model.pix2pix import Pix2Pix
from utils import freeze_weights

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
])
grayscale = transforms.Grayscale(1)

if __name__ == '__main__':
    # img = torch.randn(1, 1, 64, 64)
    # unet = Unet(5, 1, 512, 3, 64)
    # unet(img)
    DEVICE = "cuda"
    # model = Pix2Pix(5, 4, 1, 512, 2, 64).to(DEVICE)
    d = ColorizationDataset("./data/imagenet-mini/train", train_transform)
    d_ = DataLoader(d, batch_size=4, shuffle=True)

    # generator = Unet(5, 1, 512, 2, 64).to(DEVICE)
    # discriminator = PatchGAN(5, 3, 64).to(DEVICE)
    generator = G(64).to(DEVICE)
    discriminator = D(64).to(DEVICE)
    loss_fn = Loss()
    l1_scale_coeff = 1

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),
                                               lr=0.0002, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(),
                                           lr=0.0002, betas=(0.5, 0.999))
    for L, ab, stats in d_:
        L = L.to(DEVICE, non_blocking=True)
        ab = ab.to(DEVICE, non_blocking=True)
        generated_img = generator(L)

        freeze_weights(discriminator, freeze=False)
        optimizer_discriminator.zero_grad()
        real_img = torch.cat([L, ab], dim=1)
        gen_img = torch.cat([L, generated_img.detach()], dim=1)
        disc_real_out = discriminator(L, ab)
        disc_gen_out = discriminator(L, generated_img.detach())
        disc_real_loss = loss_fn.D_loss(disc_real_out, True)
        disc_gen_loss = loss_fn.D_loss(disc_gen_out, False)
        disc_loss = disc_real_loss + disc_gen_loss
        disc_loss.backward()
        optimizer_discriminator.step()

        freeze_weights(discriminator, freeze=True)
        optimizer_generator.zero_grad()
        gen_img = torch.cat([L, generated_img], dim=1)
        disc_gen_out = discriminator(L, generated_img)
        G_ce_loss, G_l1_loss = loss_fn.G_loss(disc_gen_out, generated_img, ab)
        generator_loss = G_ce_loss + l1_scale_coeff * G_l1_loss
        generator_loss.backward()
        optimizer_generator.step()

