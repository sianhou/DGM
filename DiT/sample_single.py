import argparse

import torch
from torchvision.utils import save_image

from diffusion import create_diffusion
from models import DiT


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
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
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Labels to condition the model with (feel free to change):
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, args.in_channels, args.input_size, args.input_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # Save and display images:
    save_image(samples, "sample.png", nrow=args.num_classes // 2, normalize=True, value_range=(-1, 1))


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
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default=None)

    args = parser.parse_args()
    main(args)
