from utils import Dict

PIX2PIX_CONFIG = {
    "data": {
        "train_dir": "./data/imagenet-mini/train",
        "val_dir": "./data/imagenet-mini/val",
        "img_resolution": 256,
    },
    "model": {
        "generator": {
            "num_layers": 5,
            "in_channels": 1,
            "out_channels": 2,
            "inner_channels": 512,
            "start_num_filters": 64, # num channels after first conv
        },
        "discriminator": {
            "num_layers": 5
        }
    },
    "optimizer":{
        "lr": 2e-4,
        "betas": (0.5, 0.999)
    },
    "log": {
        "steps": 1000,
        "log_per_epoch": True
    },
    "train":{
        "num_epoches": 100,
        "batch_size": 4,
    }
}

PIX2PIX_CONFIG = Dict(PIX2PIX_CONFIG)