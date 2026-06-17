"""
AcousticPINN — Physics-Informed Neural Network
Governing equation: ∂²p/∂t² = c²(∂²p/∂x² + ∂²p/∂y²)
"""

import torch
import torch.nn as nn
import numpy as np


class FourierEmbedding(nn.Module):
    """Random Fourier Feature embedding to help capture high-freq solutions."""
    def __init__(self, in_dim=3, embed_dim=64, scale=1.0):
        super().__init__()
        B = torch.randn(in_dim, embed_dim) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B  # (N, embed_dim)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*embed_dim)


class AcousticPINN(nn.Module):
    """
    Inputs  : x, y, t  (normalized to [-1,1] and [0,1])
    Output  : acoustic pressure p(x,y,t)
    """
    def __init__(self, hidden=128, layers=6, fourier_embed=True, embed_dim=32):
        super().__init__()
        self.use_fourier = fourier_embed
        if fourier_embed:
            self.embed = FourierEmbedding(in_dim=3, embed_dim=embed_dim, scale=2.0)
            in_dim = embed_dim * 2
        else:
            in_dim = 3

        net = []
        dims = [in_dim] + [hidden] * layers + [1]
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        inp = torch.stack([x, y, t], dim=-1)          # (N, 3)
        if self.use_fourier:
            inp = self.embed(inp)                      # (N, 2*embed_dim)
        return self.net(inp).squeeze(-1)               # (N,)


class WaveEquationLoss(nn.Module):
    """
    Computes all PINN loss components for the 2D acoustic wave equation.

    Loss = w_pde * L_pde + w_bc * L_bc + w_ic * L_ic + w_src * L_src
    """
    def __init__(self, c=1.0, w_pde=1.0, w_bc=10.0, w_ic=5.0, w_src=20.0):
        super().__init__()
        self.c = c
        self.w_pde = w_pde
        self.w_bc  = w_bc
        self.w_ic  = w_ic
        self.w_src = w_src

    def forward(self, model, colloc, bc_pts, ic_pts, src_pts, src_vals):
        """
        colloc  : (N,3) tensor [x,y,t] — interior collocation points
        bc_pts  : (M,3) tensor [x,y,t] — boundary points (p=0, absorbing)
        ic_pts  : (K,3) tensor [x,y,0] — initial condition points
        src_pts : (S,3) tensor [x,y,t] — source points near drone
        src_vals: (S,)  tensor          — harmonic target at source
        """
        losses = {}

        # ── PDE loss ─────────────────────────────────────────────────────────
        x = colloc[:,0].requires_grad_(True)
        y = colloc[:,1].requires_grad_(True)
        t = colloc[:,2].requires_grad_(True)

        p = model(x, y, t)

        p_t  = torch.autograd.grad(p.sum(), t, create_graph=True)[0]
        p_tt = torch.autograd.grad(p_t.sum(), t, create_graph=True)[0]
        p_x  = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_xx = torch.autograd.grad(p_x.sum(), x, create_graph=True)[0]
        p_y  = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        p_yy = torch.autograd.grad(p_y.sum(), y, create_graph=True)[0]

        residual = p_tt - self.c**2 * (p_xx + p_yy)
        losses['pde'] = (residual**2).mean()

        # ── Boundary condition loss (absorbing / p=0 at walls) ───────────────
        xb = bc_pts[:,0]; yb = bc_pts[:,1]; tb = bc_pts[:,2]
        p_bc = model(xb, yb, tb)
        losses['bc'] = (p_bc**2).mean()

        # ── Initial condition loss (quiescent field at t=0) ──────────────────
        xi = ic_pts[:,0]; yi = ic_pts[:,1]; ti = ic_pts[:,2]
        p_ic = model(xi, yi, ti)
        losses['ic'] = (p_ic**2).mean()

        # ── Source condition loss ────────────────────────────────────────────
        xs = src_pts[:,0]; ys = src_pts[:,1]; ts = src_pts[:,2]
        p_src = model(xs, ys, ts)
        losses['src'] = ((p_src - src_vals)**2).mean()

        total = (self.w_pde * losses['pde'] +
                 self.w_bc  * losses['bc']  +
                 self.w_ic  * losses['ic']  +
                 self.w_src * losses['src'])
        losses['total'] = total
        return losses, residual.detach()
