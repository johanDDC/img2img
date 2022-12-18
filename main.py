import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data.dataset import ColorizationDataset
from src.model.generator.unet import Unet
from torchvision import transforms

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
    model = Pix2Pix(5, 4, 1, 512, 2, 64).to(DEVICE)
    d = ColorizationDataset("./data/imagenet-mini/train", train_transform)
    d_ = DataLoader(d, batch_size=4, shuffle=True)
    optimizer_discriminator = torch.optim.Adam(model.descriminator.parameters(),
                                               lr=0.0002, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(model.generator.parameters(),
                                           lr=0.0002, betas=(0.5, 0.999))
    for L, ab, stats in d_:
        L = L.to(DEVICE, non_blocking=True)
        ab = ab.to(DEVICE, non_blocking=True)
        generated_img = model.generator(L)

        optimizer_discriminator.zero_grad()
        freeze_weights(model.descriminator, freeze=False)
        desc_loss = model([L, generated_img], true_output=ab, G_step=False)
        desc_loss[0].backward()
        optimizer_discriminator.step()

        optimizer_generator.zero_grad()
        freeze_weights(model.descriminator, freeze=True)
        generator_loss = model([L, generated_img], true_output=ab, G_step=True)
        generator_loss[0].backward()
        optimizer_generator.step()

