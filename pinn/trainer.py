"""
PINN Trainer — Phase 1
Adam warm-up → L-BFGS polish (same two-optimizer strategy from the Graetz paper).
"""

import torch
import numpy as np
from pinn.model   import AcousticPINN, WaveEquationLoss
from pinn.sampler import (wall_clustered_collocation, boundary_points,
                           initial_condition_points, source_points)


class PINNTrainer:
    def __init__(self, cfg):
        """
        cfg keys:
          c, src_x, src_y, freq, amp, hidden, layers,
          n_colloc, n_bc, n_ic, n_src,
          adam_epochs, adam_lr, lbfgs_epochs,
          w_pde, w_bc, w_ic, w_src, device
        """
        self.cfg = cfg
        self.device = torch.device(cfg.get('device', 'cpu'))

        self.model = AcousticPINN(
            hidden=cfg.get('hidden', 128),
            layers=cfg.get('layers', 6)
        ).to(self.device)

        self.loss_fn = WaveEquationLoss(
            c=cfg['c'],
            w_pde=cfg.get('w_pde', 1.0),
            w_bc =cfg.get('w_bc',  10.0),
            w_ic =cfg.get('w_ic',  5.0),
            w_src=cfg.get('w_src', 20.0),
        )
        self.history = []

    def _sample_batch(self):
        cfg = self.cfg
        colloc   = wall_clustered_collocation(
                        cfg.get('n_colloc', 2000),
                        cfg['src_x'], cfg['src_y']).to(self.device)
        bc       = boundary_points(cfg.get('n_bc', 400)).to(self.device)
        ic       = initial_condition_points(cfg.get('n_ic', 400)).to(self.device)
        src, tgt = source_points(
                        cfg.get('n_src', 200),
                        cfg['src_x'], cfg['src_y'],
                        freq=cfg['freq'], amp=cfg['amp'])
        return colloc, bc, ic, src.to(self.device), tgt.to(self.device)

    def train(self, callback=None):
        cfg = self.cfg

        # ── Adam phase ────────────────────────────────────────────────────────
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.get('adam_lr', 1e-3)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.get('adam_epochs', 500)
        )

        for ep in range(cfg.get('adam_epochs', 500)):
            optimizer.zero_grad()
            colloc, bc, ic, src, tgt = self._sample_batch()
            losses, residual = self.loss_fn(self.model, colloc, bc, ic, src, tgt)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            rec = {k: v.item() for k, v in losses.items()}
            rec['epoch'] = ep + 1
            rec['phase'] = 'adam'
            self.history.append(rec)

            if callback:
                callback(ep, cfg.get('adam_epochs', 500), rec, 'adam')

        # ── L-BFGS polish phase ───────────────────────────────────────────────
        lbfgs_epochs = cfg.get('lbfgs_epochs', 100)
        if lbfgs_epochs > 0:
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                lr=0.1, max_iter=20, history_size=50,
                line_search_fn='strong_wolfe'
            )
            colloc, bc, ic, src, tgt = self._sample_batch()  # fixed batch for L-BFGS
            ep_lbfgs = [0]

            def closure():
                lbfgs.zero_grad()
                losses, _ = self.loss_fn(self.model, colloc, bc, ic, src, tgt)
                losses['total'].backward()
                return losses['total']

            for ep in range(lbfgs_epochs):
                lbfgs.step(closure)
                with torch.no_grad():
                    losses, _ = self.loss_fn(self.model, colloc, bc, ic, src, tgt)
                rec = {k: v.item() for k, v in losses.items()}
                rec['epoch'] = cfg.get('adam_epochs', 500) + ep + 1
                rec['phase'] = 'lbfgs'
                self.history.append(rec)
                if callback:
                    callback(ep, lbfgs_epochs, rec, 'lbfgs')

        return self.history

    def save(self, path):
        torch.save({'model': self.model.state_dict(), 'cfg': self.cfg}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])

    @torch.no_grad()
    def evaluate_field(self, t_val, grid_n=100):
        xs = np.linspace(-1, 1, grid_n)
        ys = np.linspace(-1, 1, grid_n)
        XX, YY = np.meshgrid(xs, ys)
        x_flat = torch.FloatTensor(XX.ravel()).to(self.device)
        y_flat = torch.FloatTensor(YY.ravel()).to(self.device)
        t_flat = torch.full_like(x_flat, t_val)
        p_flat = self.model(x_flat, y_flat, t_flat).cpu().numpy()
        return xs, ys, p_flat.reshape(grid_n, grid_n)

    def evaluate_residual_map(self, t_val, grid_n=50):
        xs = np.linspace(-1, 1, grid_n)
        ys = np.linspace(-1, 1, grid_n)
        XX, YY = np.meshgrid(xs, ys)
        x_flat = torch.FloatTensor(XX.ravel()).to(self.device)
        y_flat = torch.FloatTensor(YY.ravel()).to(self.device)
        t_flat = torch.full_like(x_flat, t_val)

        colloc = torch.stack([x_flat, y_flat, t_flat], dim=1)
        dummy_bc = boundary_points(4).to(self.device)
        dummy_ic = initial_condition_points(4).to(self.device)
        src, tgt = source_points(4, self.cfg['src_x'], self.cfg['src_y'])

        _, residual = self.loss_fn(
            self.model, colloc, dummy_bc, dummy_ic,
            src.to(self.device), tgt.to(self.device)
        )
        return xs, ys, residual.cpu().numpy().reshape(grid_n, grid_n)
