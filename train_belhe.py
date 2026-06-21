import math
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import model
import samplers
import utils_belhe as utils

torch.manual_seed(30)

# Modified from Vector Graphics configuration
config = {
    'NUM_ITERS': 300000,
    'BATCH_SIZE': 2**19,
#    'LR': 0.01,
#    'FNAME': 'data/vg/shapes/img.msh_curved.npz',
    'SAMPLING': 'grid',
    'SAMPLING_GRID_SIZE': 1024
}

SAVE_INT = 10000
FEATURE_DIM = 5
ACCEL_GRID_DIMS = (2000, 2000)
LR = 5e-4
BETAS = (0.9, 0.99)
NUM_QUERIES_SQRT = config['SAMPLING_GRID_SIZE']
NUM_ITERS = config['NUM_ITERS']
BATCH_SIZE = config['BATCH_SIZE']

results_dir = os.path.join('results', "belhe")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# SharpNet sampler
class Belhe(nn.Module):
    def __init__(self, **kwargs):
        super(Belhe, self).__init__()
        if kwargs.keys().__len__() > 0:
            print(f"Belhe received extra arguments: {kwargs.keys()}")

        self._vertices = nn.Parameter(torch.tensor([
            # For the line
            [0.27721, 0.42631],
            [0.40487, 0.79068],
            # For the polygon
            [0.45444, 0.42631],
            [0.83864, 0.39781],
            [0.91548, 0.81547],
            [0.40487, 0.57752],
        ], dtype=torch.float32), requires_grad=False)
        self._edges = nn.Parameter(torch.tensor([
            [0, 1],
            [2, 3], [3, 4], [4, 5], [5, 2],
        ], dtype=torch.int32), requires_grad=False)

    @staticmethod
    def _C0_2d(pt: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        def _dot(a: torch.Tensor, b: torch.Tensor)  ->torch.Tensor:
            return torch.sum(a*b, dim=-1, keepdim=True)
        def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a0 = a[:,:,[0]]
            a1 = a[:,:,[1]]
            b0 = b[:,:,[0]]
            b1 = b[:,:,[1]]
            return a0 * b1 - a1 * b0

        v = face-pt.unsqueeze(dim=-2) # P x F x 2 x 2
        va = v[:,:,0,:] # P x F x 2
        vb = v[:,:,1,:]
        vd = face[:,:,1,:] - face[:,:,0,:]
        ld = vd.norm(dim=-1, keepdim=True) # P x F x 1
        l = torch.abs(_cross2d(va, vb)) / ld
        L_ = torch.square(l)
        l_ = torch.sqrt(L_)
        def f(t):
            return t * (torch.log(torch.square(t) + L_) - 2) / 2 + l_ * torch.atan2(t, l_)
        res = f(_dot(vb,vd) / ld) - f(_dot(va,vd) / ld)
        return res.squeeze(dim=-1) / torch.pi # P x F

    def forward(self, x: torch.Tensor, keepdim: bool=False) -> torch.Tensor:
        assert x.shape[-1] == 2, "Input must have 2 elements in the last dimension"
        shape = x.shape[:-1]
        x = x.reshape(-1, 2)
        rpt = x.unsqueeze(dim=1) # P x {1} x 2

        vertices = self._vertices.detach()
        edges = self._edges.detach()
        # A lite version of BEMquery that solves the laplacian equation.
        # The data is small enough for dense non-mollifier implementation.

        redge = torch.index_select(vertices, 0, edges.flatten()).reshape(5, 2, 2).unsqueeze(dim=0)  # (1, num_edges, 2, 2)
        out = self._C0_2d(rpt, redge).sum(dim=-1, keepdim=True)   # P, 1

        out = torch.reshape(out, (*shape, -1))
        if not keepdim:
            out = out.squeeze(-1)
        return out

class BelheSampler(samplers.BaseSampler):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = Belhe().to("cuda")

    def __call__(self, Q):
        Q = Q[:, [1, 0]]    # Swap x and y to match the coordinate convention of our code
        return self.dataset.forward(Q, keepdim=True)

sampler = BelheSampler()

vmin = -1.00
vmax = -0.10

mesh = np.load('data/wos/overview/img.msh_curved.npz')
mesh2 = {
    "vertices": mesh["vertices"],
    "continuous_triangles": np.concatenate([mesh["continuous_triangles"], mesh["linear_triangles"]], axis=0),
    "linear_triangles": np.empty((0,), dtype=np.int32),
    "curved_triangles": np.empty((0,), dtype=np.int32),
}

model = model.DANN(mesh=mesh2, FEATURE_DIM=FEATURE_DIM, ACCEL_GRID_DIMS=ACCEL_GRID_DIMS, OUT_DIM=1)
optim = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)

# ckpt = torch.load('results/belhe/checkpoint_300000.pth')
# model.load_state_dict(ckpt['model'])
model = model.to("cuda")
# iter_count = 300002

iter_count = 0
for idx in range(NUM_ITERS):
    if config['SAMPLING'] == 'triangle':
        Q = samplers.get_stratified_in_triangles(model=model)
    elif config['SAMPLING'] == 'grid':
        Q = samplers.get_stratified_random(NUM_QUERIES_SQRT)
    else:
        assert False
    randperm = torch.randperm(Q.shape[0])
    Q = Q[randperm]

    gt = sampler(Q)

    for batch_idx in range((Q.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE):
        Qb = Q[batch_idx*BATCH_SIZE: (batch_idx+1)*BATCH_SIZE]
        gtb = gt[batch_idx*BATCH_SIZE: (batch_idx+1)*BATCH_SIZE]

        predb = model(Q=Qb)
        if "wos" == "rendering":
            loss = ((predb-gtb).square()/(predb.detach().square() + 0.01)).mean() # rendering
        else:
            # loss = (predb-gtb).square().mean() # vg, wos
            loss = (predb-gtb).abs().mean() # for SharpNet

        loss.backward()
        optim.step()
        optim.zero_grad()

        print(f"Iter {iter_count}, Loss: {loss.item()}")
    
        if iter_count % SAVE_INT == 0:
            torch.save({'model': model.state_dict()}, os.path.join(results_dir, f"checkpoint_{iter_count}.pth"))
            locs = [(0.5, 0.5, 1)]
            fig = utils.visualise_output(
                query_field=lambda Q: model(Q=Q),
                query_gt = lambda Q: sampler(Q),
                xmin = 0.0,
                xmax = 1.0,
                ymin = 0.0,
                ymax = 1.0,
                vmin = vmin,
                vmax = vmax,
                levels = [-0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20],
                resolution = 2048,
            )
            fig.savefig(os.path.join(results_dir, f"output_{iter_count}.png"), transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        iter_count += 1
    if iter_count > NUM_ITERS:
        break

# Final output
with torch.no_grad():
    xs = torch.linspace(0.0, 1.0, 2049)
    ys = torch.linspace(0.0, 1.0, 2049)

    N = 64
    X = xs.split(N)
    Y = ys.split(N)
    dim = 1

    u = np.empty([2049, 2049, dim])
    for yi, ys in enumerate(Y):
        for xi, xs in enumerate(X):
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            xx = xx.reshape(-1, 1)
            yy = yy.reshape(-1, 1)
            pts = torch.cat([yy, xx], dim=1).to("cuda")     # Concatenate in (y, x) order to mimic the convention of DANN
            val = model(Q=pts)
            val = val.reshape(len(ys), len(xs), dim).detach().cpu().numpy()
            u[yi*N:yi*N+len(ys), xi*N:xi*N+len(xs), :] = val

    u = u.squeeze(-1)

    xs = np.linspace(0.0, 1.0, 2049)
    ys = np.linspace(0.0, 1.0, 2049)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    utils.write_height_mesh(os.path.join(results_dir, f"output_{iter_count}_mesh.ply"), xx, yy, u)

    # Draw raw image
    figure = plt.figure()
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.imshow(u, origin="lower", cmap="winter", zorder=0, vmin=vmin, vmax=vmax, extent=(0,1,0,1))
    figure.savefig(os.path.join(results_dir, f"output_{iter_count}_raw.png"), transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(figure)