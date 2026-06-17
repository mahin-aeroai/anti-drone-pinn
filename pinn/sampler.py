"""
Collocation point samplers for the acoustic PINN.
Supports uniform, Latin Hypercube, and adaptive (residual-guided) sampling.
"""

import torch
import numpy as np


def uniform_collocation(n, x_range=(-1,1), y_range=(-1,1), t_range=(0,1)):
    x = torch.FloatTensor(n).uniform_(*x_range)
    y = torch.FloatTensor(n).uniform_(*y_range)
    t = torch.FloatTensor(n).uniform_(*t_range)
    return torch.stack([x, y, t], dim=1)


def lhs_collocation(n, x_range=(-1,1), y_range=(-1,1), t_range=(0,1)):
    """Latin Hypercube Sampling — better space coverage than pure random."""
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=3)
    sample  = sampler.random(n=n)
    lower   = [x_range[0], y_range[0], t_range[0]]
    upper   = [x_range[1], y_range[1], t_range[1]]
    scaled  = qmc.scale(sample, lower, upper)
    return torch.FloatTensor(scaled)


def boundary_points(n, x_range=(-1,1), y_range=(-1,1), t_range=(0,1)):
    """Points on the 4 spatial walls at random times."""
    n4  = n // 4
    t   = torch.FloatTensor(n).uniform_(*t_range)
    # left, right, bottom, top
    xl  = torch.full((n4,), x_range[0])
    xr  = torch.full((n4,), x_range[1])
    yb  = torch.FloatTensor(n4).uniform_(*y_range)
    yt  = torch.FloatTensor(n4).uniform_(*y_range)
    xb2 = torch.FloatTensor(n4).uniform_(*x_range)
    xt2 = torch.FloatTensor(n4).uniform_(*x_range)
    yb2 = torch.full((n4,), y_range[0])
    yt2 = torch.full((n4,), y_range[1])

    xs = torch.cat([xl, xr, xb2, xt2])
    ys = torch.cat([yb, yt, yb2, yt2])
    return torch.stack([xs, ys, t[:len(xs)]], dim=1)


def initial_condition_points(n, x_range=(-1,1), y_range=(-1,1)):
    """Points at t=0."""
    x = torch.FloatTensor(n).uniform_(*x_range)
    y = torch.FloatTensor(n).uniform_(*y_range)
    t = torch.zeros(n)
    return torch.stack([x, y, t], dim=1)


def source_points(n, src_x, src_y, radius=0.08,
                  t_range=(0,1), freq=1.0, amp=1.0):
    """
    Points clustered near the drone source + harmonic target values.
    Returns (pts, target_pressure)
    """
    angles = torch.FloatTensor(n).uniform_(0, 2*np.pi)
    r      = torch.FloatTensor(n).uniform_(0, radius)
    x      = src_x + r * torch.cos(angles)
    y      = src_y + r * torch.sin(angles)
    t      = torch.FloatTensor(n).uniform_(*t_range)
    pts    = torch.stack([x, y, t], dim=1)
    target = amp * torch.sin(2 * np.pi * freq * t)
    return pts, target


def wall_clustered_collocation(n, src_x, src_y, x_range=(-1,1),
                                y_range=(-1,1), t_range=(0,1), cluster_frac=0.3):
    """
    Mix uniform + source-clustered points.
    Helps resolve high-gradient regions near the drone.
    """
    n_cluster = int(n * cluster_frac)
    n_uniform  = n - n_cluster

    uni = uniform_collocation(n_uniform, x_range, y_range, t_range)

    # Gaussian cluster around source
    x_c = torch.randn(n_cluster) * 0.15 + src_x
    y_c = torch.randn(n_cluster) * 0.15 + src_y
    t_c = torch.FloatTensor(n_cluster).uniform_(*t_range)
    x_c = x_c.clamp(*x_range); y_c = y_c.clamp(*y_range)
    cluster = torch.stack([x_c, y_c, t_c], dim=1)

    return torch.cat([uni, cluster], dim=0)
