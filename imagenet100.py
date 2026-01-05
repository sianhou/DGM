import os
import tarfile

from tqdm import tqdm

TRAIN_TAR = "D:/ILSVRC2012_img_train.tar"
CLASS_LIST = "imagenet100_classes.txt"
OUT_DIR = "data/imagenet100/train"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# 读取 100 类
# -------------------------
with open(CLASS_LIST) as f:
    classes = [line.strip() for line in f if line.strip()]

assert len(classes) == 100, "❌ imagenet100_classes.txt 必须正好 100 行"

print(f"✅ 读取 {len(classes)} 个 ImageNet-100 类别")

# -------------------------
# 打开总 tar
# -------------------------
with tarfile.open(TRAIN_TAR, "r") as train_tar:
    members = {m.name: m for m in train_tar.getmembers()}

    for cls in tqdm(classes, desc="Extract ImageNet-100"):
        cls_tar_name = f"{cls}.tar"
        assert cls_tar_name in members, f"❌ 找不到 {cls_tar_name}"

        cls_out_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(cls_out_dir, exist_ok=True)

        # 读取类别 tar（内嵌 tar）
        cls_tar_file = train_tar.extractfile(members[cls_tar_name])
        with tarfile.open(fileobj=cls_tar_file) as cls_tar:
            cls_tar.extractall(path=cls_out_dir)

print("\n🎉 ImageNet-100 训练集构建完成")
print(f"📁 输出目录: {OUT_DIR}")
