import argparse
import os

from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm


def save_cifar_as_folders(
        root,
        out_dir,
        dataset="cifar10",
        split="train"
):
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.ToTensor()

    if dataset == "cifar10":
        ds = CIFAR10(
            root=root,
            train=(split == "train"),
            download=True
        )
        num_classes = 10
    elif dataset == "cifar100":
        ds = CIFAR100(
            root=root,
            train=(split == "train"),
            download=True
        )
        num_classes = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    # 创建类别文件夹
    for i in range(num_classes):
        os.makedirs(os.path.join(out_dir, str(i)), exist_ok=True)

    print(f"Saving {dataset} {split} set to {out_dir}")

    for idx, (img, label) in enumerate(tqdm(ds)):
        class_dir = os.path.join(out_dir, str(label))
        img_path = os.path.join(class_dir, f"{idx:06d}.png")

        if isinstance(img, Image.Image):
            img.save(img_path)
        else:
            Image.fromarray(img).save(img_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--root", type=str, default="./temp")
    parser.add_argument("--out", type=str, default="./cifar10_train")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])

    args = parser.parse_args()

    save_cifar_as_folders(
        root=args.root,
        out_dir=os.path.join(args.out, args.split),
        dataset=args.dataset,
        split=args.split
    )
