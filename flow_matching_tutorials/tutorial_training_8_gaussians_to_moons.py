import os
import time

import torch
from torchdyn.core import NeuralODE

from model.cfm import VariancePreservingConditionalFlowMatcher
from model.mlp import MLP
from utils import sample_moons, sample_8gaussians, plot_trajectories, torch_wrapper

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)

sigma = 0.1
dim = 2
batch_size = 256
steps = 20000
seed = 0

# # Conditional Flow Matching with sigma = 0.0
# torch.manual_seed(seed)
# model = MLP(dim=dim, time_varying=True)
# optimizer = torch.optim.Adam(model.parameters())
# FM = ConditionalFlowMatcher(sigma=0.0)
#
# start = time.time()
# for k in range(20000):
#     optimizer.zero_grad()
#
#     x0 = sample_8gaussians(batch_size)
#     x1 = sample_moons(batch_size)
#
#     t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
#
#     vt = model(torch.cat([xt, t[:, None]], dim=-1))
#     loss = torch.mean((vt - ut) ** 2)
#
#     loss.backward()
#     optimizer.step()
#
#     if (k + 1) % 5000 == 0:
#         end = time.time()
#         print(f"{k + 1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
#         start = end
#         node = NeuralODE(
#             torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
#         )
#         with torch.no_grad():
#             traj = node.trajectory(
#                 sample_8gaussians(1024),
#                 t_span=torch.linspace(0, 1, 100),
#             )
#             plot_trajectories(traj.cpu().numpy(), f"{savedir}/cfm_v0_step_{k + 1}.png")
# torch.save(model, f"{savedir}/cfm_v0.pt")
#
# # Conditional Flow Matching with sigma = 0.1
# torch.manual_seed(seed)
# model = MLP(dim=dim, time_varying=True)
# optimizer = torch.optim.Adam(model.parameters())
# FM = ConditionalFlowMatcher(sigma=sigma)
#
# start = time.time()
# for k in range(20000):
#     optimizer.zero_grad()
#
#     x0 = sample_8gaussians(batch_size)
#     x1 = sample_moons(batch_size)
#
#     t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
#
#     vt = model(torch.cat([xt, t[:, None]], dim=-1))
#     loss = torch.mean((vt - ut) ** 2)
#
#     loss.backward()
#     optimizer.step()
#
#     if (k + 1) % 5000 == 0:
#         end = time.time()
#         print(f"{k + 1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
#         start = end
#         node = NeuralODE(
#             torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
#         )
#         with torch.no_grad():
#             traj = node.trajectory(
#                 sample_8gaussians(1024),
#                 t_span=torch.linspace(0, 1, 100),
#             )
#             plot_trajectories(traj.cpu().numpy(), f"{savedir}/cfm_v1_step_{k + 1}.png")
# torch.save(model, f"{savedir}/cfm_v1.pt")

# # OT Conditional Flow Matching with sigma = 0.1
# torch.manual_seed(seed)
# model = MLP(dim=dim, time_varying=True)
# optimizer = torch.optim.Adam(model.parameters())
# FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
#
# start = time.time()
# for k in range(20000):
#     optimizer.zero_grad()
#
#     x0 = sample_8gaussians(batch_size)
#     x1 = sample_moons(batch_size)
#
#     t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
#
#     vt = model(torch.cat([xt, t[:, None]], dim=-1))
#     loss = torch.mean((vt - ut) ** 2)
#
#     loss.backward()
#     optimizer.step()
#
#     if (k + 1) % 5000 == 0:
#         end = time.time()
#         print(f"{k + 1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
#         start = end
#         node = NeuralODE(
#             torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
#         )
#         with torch.no_grad():
#             traj = node.trajectory(
#                 sample_8gaussians(1024),
#                 t_span=torch.linspace(0, 1, 100),
#             )
#             plot_trajectories(traj.cpu().numpy(), f"{savedir}/otcfm_step_{k + 1}.png")
# torch.save(model, f"{savedir}/otcfm.pt")

# # Schrödinger Bridge Conditional Flow Matching with sigma = 0.1
# torch.manual_seed(seed)
# model = MLP(dim=dim, time_varying=True)
# optimizer = torch.optim.Adam(model.parameters())
# FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
#
# start = time.time()
# for k in range(20000):
#     optimizer.zero_grad()
#
#     x0 = sample_8gaussians(batch_size)
#     x1 = sample_moons(batch_size)
#
#     t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
#
#     vt = model(torch.cat([xt, t[:, None]], dim=-1))
#     loss = torch.mean((vt - ut) ** 2)
#
#     loss.backward()
#     optimizer.step()
#
#     if (k + 1) % 5000 == 0:
#         end = time.time()
#         print(f"{k + 1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
#         start = end
#         node = NeuralODE(
#             torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
#         )
#         with torch.no_grad():
#             traj = node.trajectory(
#                 sample_8gaussians(1024),
#                 t_span=torch.linspace(0, 1, 100),
#             )
#             plot_trajectories(traj.cpu().numpy(), f"{savedir}/sbcfm_step_{k + 1}.png")
# torch.save(model, f"{savedir}/sbcfm.pt")

# Variance Preserving Bridge Conditional Flow Matching with sigma = 0.1
torch.manual_seed(seed)
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)

start = time.time()
for k in range(20000):
    optimizer.zero_grad()

    x0 = sample_8gaussians(batch_size)
    x1 = sample_moons(batch_size)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 5000 == 0:
        end = time.time()
        print(f"{k + 1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        with torch.no_grad():
            traj = node.trajectory(
                sample_8gaussians(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            plot_trajectories(traj.cpu().numpy(), f"{savedir}/vpcfm_step_{k + 1}.png")
torch.save(model, f"{savedir}/vpcfm.pt")
