import argparse
import logging
import os
from copy import deepcopy
from glob import glob
from time import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffusion import create_diffusion
from models import DiT


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        if p.requires_grad:
            ema_p.mul_(decay).add_(p, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler(f"{logging_dir}/log.txt"))
    return logger


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = str(args.input_size) + "-" + str(args.patch_size) + "-" + str(args.hidden_size)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)

    logger.info(vars(args))
    logger.info(f"Experiment device is {device}")
    logger.info(f"Experiment directory created at {experiment_dir}")

    # define model
    model = DiT(input_size=args.input_size,
                patch_size=args.patch_size,
                in_channels=args.in_channels,
                hidden_size=args.hidden_size,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                class_dropout_prob=args.class_dropout_prob,
                num_classes=args.num_classes,
                learn_sigma=args.learn_sigma
                ).to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # diffusion
    diffusion = create_diffusion(timestep_respacing="",
                                 learn_sigma=args.learn_sigma)  # default: 1000 steps, linear noise schedule

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    transform = transforms.Compose([
        transforms.Resize(args.input_size),  # 调整到指定大小
        transforms.ToTensor(),  # [0,255] → [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # [0,1] → [-1,1]
                             std=[0.5, 0.5, 0.5])
    ])

    # dataset
    dataset = ImageFolder(args.data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,  # 全量 batch_size
        shuffle=True,  # 单机直接打乱
        num_workers=1,  # 单线程
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # restart
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # 随机 timestep
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()

            # EMA step 更新
            update_ema(ema, model, decay=0.9999)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

        # Save DiT checkpoint:
        if epoch % args.ckpt_every == 0 and train_steps > 0:
            checkpoint = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args
            }
            checkpoint_path = f"{checkpoint_dir}/{epoch:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--input-size", type=int, choices=[32, 64, 128, 256, 512], default=256)
    parser.add_argument("--in-channels", type=int, required=True)
    parser.add_argument("--patch-size", type=int, choices=[2, 4, 8, 32], default=4)
    parser.add_argument("--hidden-size", type=int, choices=[384, 768, 1024, 1152], default=1152)
    parser.add_argument("--depth", type=int, choices=[2, 4, 8, 12, 16, 20, 24, 28], default=8)
    parser.add_argument("--num-heads", type=int, choices=[4, 8, 12, 16], default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--learn-sigma", action="store_false", help="Disable learning sigma")
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
