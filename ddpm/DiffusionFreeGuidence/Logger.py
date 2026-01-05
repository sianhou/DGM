from torch.utils.tensorboard import SummaryWriter
import torch
import time

class DGMLogger(object):
    def __init__(self, log_dir, log_interval=100):
        self.writer = SummaryWriter(log_dir)
        self.log_interval = log_interval
        self.start_time = None

    def start_iter(self):
        self.start_time = time.time()

    def log_losses(self, step, loss_dict):
        for k, v in loss_dict.items():
            self.writer.add_scalar(f"loss/{k}", v.item(), step)

    def log_perf(self, step, batch_size):
        iter_time = time.time() - self.start_time
        self.writer.add_scalar("perf/iter_time", iter_time, step)
        self.writer.add_scalar(
            "perf/samples_per_sec", batch_size / iter_time, step
        )

        if torch.cuda.is_available():
            self.writer.add_scalar(
                "gpu/mem_alloc",
                torch.cuda.memory_allocated() / 1024 ** 2,
                step
            )
            self.writer.add_scalar(
                "gpu/mem_reserved",
                torch.cuda.memory_reserved() / 1024 ** 2,
                step
            )

    def log_diffusion_stats(self, step, t, noise, pred_noise, x0, x0_pred):
        self.writer.add_scalar(
            "diffusion/mean_t", t.float().mean().item(), step
        )

        noise_norm = noise.norm(dim=(1, 2, 3))
        pred_noise_norm = pred_noise.norm(dim=(1, 2, 3))

        self.writer.add_scalar(
            "diffusion/noise_norm", noise_norm.mean().item(), step
        )
        self.writer.add_scalar(
            "diffusion/pred_noise_norm", pred_noise_norm.mean().item(), step
        )

        self.writer.add_scalar(
            "diffusion/x0_recon_error",
            (x0 - x0_pred).pow(2).mean().item(),
            step
        )

        # Histogram（关键！）
        self.writer.add_histogram("hist/timestep", t, step)
        self.writer.add_histogram("hist/noise_norm", noise_norm, step)
        self.writer.add_histogram("hist/pred_noise_norm", pred_noise_norm, step)

    def log_images(self, step, x0=None, xt=None, x0_pred=None, generated=None):
        if x0 is not None:
            self.writer.add_images("samples/x0", x0, step)
        if xt is not None:
            self.writer.add_images("samples/xt", xt, step)
        if x0_pred is not None:
            self.writer.add_images("samples/x0_pred", x0_pred, step)
        if generated is not None:
            self.writer.add_images("samples/generated", generated, step)

    def close(self):
        self.writer.close()