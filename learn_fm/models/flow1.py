import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Flow(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_blocks=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input is concatenated with time t -> input_size + 1
        self.head = MLPBlock(input_size + 1, hidden_size)

        self.blocks = nn.ModuleList(
            [MLPBlock(hidden_size, hidden_size) for _ in range(num_blocks)]
        )

        self.final = nn.Linear(hidden_size, input_size)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

        # Optional but common for flows / diffusion
        nn.init.zeros_(self.final.weight)

    def forward(self, x_t, t):
        y = self.head(torch.cat((x_t, t), -1))
        for model in self.blocks:
            y = model(y)
        return self.final(y)

    def step(self, x_t, t_start, t_end):
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        return x_t + + (t_end - t_start) * self(x_t=x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2,
                                                t=t_start + (t_end - t_start) / 2)


if __name__ == "__main__":
    batch_size = 32
    print(f"Test flow")
    flow = Flow(input_size=2, hidden_size=64, num_blocks=2)

    x = torch.randn(batch_size, 2)
    t = torch.rand((batch_size, 1))
    y = flow(x, t)
    print(f"y.shape: {y.shape}")
