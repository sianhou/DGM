import math
import platform
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ==========================================
# 1. 配置 (Ada Lovelace Config)
# ==========================================
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 32
    patch_size = 2
    in_channels = 3
    num_classes = 10

    # Modern ViT 架构参数
    hidden_size = 384
    depth = 8
    num_heads = 6
    mlp_ratio = 4.0
    bottleneck_dim = 64

    in_context_len = 8
    in_context_start = 2

    # --- 针对 4060 Ti 的显存优化 ---
    # 4060 Ti 8G 建议: 64
    # 4060 Ti 16G 建议: 128
    batch_size = 64

    # 梯度
    gradient_accumulation_steps = 4

    lr = 5e-4
    epochs = 200

    num_workers = 0 if platform.system() == 'Windows' else 4

    noise_scale = 1.0
    save_dir = "./jit_results_4060ti"
    resume_path = None


# ==========================================
# 2. 基础组件
# # ==========================================
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len, num_cls_token=0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        freqs = 1. / (10000 ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len)
        freqs = torch.einsum('i,j->ij', t, freqs)
        freqs = torch.cat((freqs[:, None, :].repeat(1, pt_seq_len, 1),
                           freqs[None, :, :].repeat(pt_seq_len, 1, 1)), dim=-1)
        freqs = rearrange(freqs, 'h w d -> (h w) d')
        freqs = repeat(freqs, 'n d -> n (d r)', r=2)
        if num_cls_token > 0:
            pad = torch.zeros(num_cls_token, freqs.shape[1])
            freqs = torch.cat([pad, freqs], dim=0)
        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        seq_len = x.shape[-2]
        f_cos = self.freqs_cos[:seq_len]
        f_sin = self.freqs_sin[:seq_len]
        return x * f_cos + x_rotated * f_sin


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, rope_func):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope_func)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, bottleneck_dim):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2).contiguous()


# ==========================================
# 3. JiT 主模型
# ==========================================
class JiT(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=2,
                 in_channels=3,
                 hidden_size=384,
                 depth=8,
                 num_heads=6,
                 mlp_ratio=4.0,
                 class_dropout_prob: float = 0,
                 num_classes: Optional[int] = None,
                 bottleneck_dim=64,
                 in_context_len=8,
                 in_context_start=2
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.t_embedder = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(),
                                        nn.Linear(hidden_size, hidden_size))
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        if num_classes is not None:
            self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(img_size, patch_size, in_channels,
                                               hidden_size, bottleneck_dim)
        hw_seq_len = img_size // patch_size
        head_dim = hidden_size // num_heads
        self.feat_rope = VisionRotaryEmbeddingFast(dim=head_dim // 2, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(dim=head_dim // 2, pt_seq_len=hw_seq_len,
                                                             num_cls_token=in_context_len)
        self.blocks = nn.ModuleList(
            [JiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm_final = RMSNorm(hidden_size)
        self.linear_final = nn.Linear(hidden_size, patch_size ** 2 * self.out_channels)
        self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def get_time_embedding(self, t):
        half_dim = 256 // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim).to(
            t.device)
        args = t[:, None].float() * freqs[None]
        return self.t_embedder(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))

    def unpatchify(self, x):
        p = self.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        return torch.einsum('nhwpqc->nchpwq', x.reshape(shape=(x.shape[0], h, w, p, p, c))).reshape(
            shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, extra):
        x = self.x_embedder(x)
        c = self.get_time_embedding(t)
        if self.num_classes and "label" not in extra:
            # Hack to deal with ddp find_unused_parameters not working with activation checkpointing...
            # self.num_classes corresponds to the pad index of the embedding table
            extra["label"] = torch.full(
                (x.size(0),), self.num_classes, dtype=torch.long, device=x.device
            )

        if self.num_classes is not None and "label" in extra:
            y = extra["label"]
            assert (
                    y.shape == x.shape[:1]
            ), f"Labels have shape {y.shape}, which does not match the batch dimension of the input {x.shape}"

            y = self.y_embedder(y)  # (N, D)
            c = t + y

        for i, block in enumerate(self.blocks):
            if i == self.in_context_start:
                ctx = self.y_embedder(y).unsqueeze(1).repeat(1, self.in_context_len, 1)
                x = torch.cat([ctx, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)
        if self.in_context_len > 0: x = x[:, self.in_context_len:]
        shift, scale = self.adaLN_final(c).chunk(2, dim=1)
        return self.unpatchify(self.linear_final(modulate(self.norm_final(x), shift, scale)))


if __name__ == '__main__':
    batch_size = 8
    model = JiT()
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size])
    y = model(x, t, None)

    print(y.shape)
