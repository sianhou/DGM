import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from ddpm.Diffusion import UNet, GaussianDiffusionSampler


def main(model_config=None):
    modelConfig = {
        "state": "train",  # or eval
        "epoch": 625,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs",
        "sampledImgName": "SampledNoGuidenceImgs",
        "nrow": 8
    }

    device = torch.device(modelConfig["device"])

    # data
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model
    model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                 attn=modelConfig["attn"],
                 num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
    ckpt = torch.load(os.path.join(
        modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
    model.load_state_dict(ckpt)

    # sampler
    with torch.no_grad():
        model.eval()

        fid_metric = FrechetInceptionDistance(normalize=True).to(
            device=device, non_blocking=True
        )

        for data_iter_step, (samples, labels) in enumerate(dataloader):
            if data_iter_step < modelConfig["epoch"]:
                samples = samples.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                fid_metric.update(samples, real=True)

                # sampler
                sampler = GaussianDiffusionSampler(
                    model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
                # Sampled from standard normal distribution
                noisyImage = torch.randn(
                    size=[modelConfig["batch_size"], 3, 32, 32], device=device)
                saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
                save_image(saveNoisy, os.path.join(
                    modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"] + f"-{data_iter_step}.png"),
                           nrow=modelConfig["nrow"])
                sampledImgs = sampler(noisyImage)
                sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                save_image(sampledImgs, os.path.join(
                    modelConfig["sampled_dir"], modelConfig["sampledImgName"] + f"-{data_iter_step}.png"),
                           nrow=modelConfig["nrow"])

                fid_metric.update(sampledImgs, real=False)

                print(f"running FID ({data_iter_step}): {fid_metric.compute()}")

        running_fid = fid_metric.compute().detach().cpu()
        print("FID: ", running_fid)


if __name__ == "__main__":
    main()
