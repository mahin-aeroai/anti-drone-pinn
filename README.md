# 🛸 Physics-Informed Aeroacoustic Anti-Drone Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-1%20of%208-0ea5e9?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Aerospace%20×%20ML-6366f1?style=flat-square)

**A research-grade Physics-Informed Neural Network system for modeling and detecting drone-generated acoustic pressure fields — without labeled data.**

*The physics itself is the teacher.*

[Overview](#overview) · [Physics](#governing-physics) · [Architecture](#pinn-architecture) · [Modules](#simulation-modules) · [Install](#installation) · [Roadmap](#roadmap)

</div>

---

## Overview

Standard neural networks need thousands of labeled examples. This system needs none.

Instead of learning from data, the PINN is trained to satisfy the **acoustic wave equation** through automatic differentiation. The governing PDE becomes the loss function — the network is penalized for violating the laws of physics, and rewarded for obeying them.

Applied to anti-drone surveillance: given a drone's rotor frequency and position, the system models how its acoustic pressure field propagates through air, identifies its Blade Passing Frequency (BPF) harmonics via FFT, and learns the full spatiotemporal pressure field from physics alone.

```
Standard NN:  data → labels → loss → learn
PINN:         equation → residual → loss → learn
```

---

## Governing Physics

The 2D acoustic wave equation drives all learning:

```
∂²p/∂t² = c² · (∂²p/∂x² + ∂²p/∂y²)
```

| Symbol | Meaning |
|--------|---------|
| `p(x,y,t)` | Acoustic pressure field |
| `c` | Speed of sound (temperature-corrected) |
| `∇²p` | Spatial Laplacian — wave spreading |
| `∂²p/∂t²` | Temporal acceleration of pressure |

Speed of sound is computed from ambient temperature:

```
c = 331.3 · √(1 + T/273.15)   [m/s]
```

Rotor Blade Passing Frequency:

```
BPF = (RPM / 60) × N_blades
```

---

## PINN Architecture

```
Input (x, y, t)
      │
      ▼
┌─────────────────────────┐
│  Random Fourier Embed   │  ← captures high-frequency solutions
│  sin(Bx) ⊕ cos(Bx)     │
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│   6× Linear + Tanh      │  ← 96 hidden units per layer
│   Xavier initialization │
└─────────────────────────┘
      │
      ▼
   p(x, y, t)              ← scalar acoustic pressure
```

### Loss Function

```
L_total = L_pde + 10·L_bc + 5·L_ic + 20·L_src

L_pde  = || ∂²p/∂t² − c²∇²p ||²        ← wave equation residual
L_bc   = || p(boundary, t) ||²           ← absorbing wall condition
L_ic   = || p(x, y, 0) ||²              ← quiescent initial field
L_src  = || p(src) − A·sin(ωt) ||²      ← harmonic drone source
```

### Two-Phase Optimizer

```
Phase 1 — Adam (fast exploration)
  └─ Cosine annealing LR schedule
  └─ Gradient clipping (norm=1.0)
  └─ Source-clustered collocation sampling

Phase 2 — L-BFGS (second-order polish)
  └─ Strong Wolfe line search
  └─ Breaks through Adam's loss plateau
```

This two-phase strategy is directly inspired by the Graetz problem convergence analysis — Adam explores the loss landscape, L-BFGS polishes through the plateau.

---

## Collocation Sampling Strategy

Rather than uniform random sampling, points are clustered near the drone source to resolve high-gradient regions:

```
Total points = 70% uniform + 30% Gaussian cluster near drone
```

Boundary points cover all 4 spatial walls. Initial condition points are sampled at `t=0` across the full domain.

---

## Signal Processing

| Method | Output |
|--------|--------|
| FFT | Frequency spectrum, BPF peaks, harmonic series |
| STFT | Spectrogram, time-frequency evolution |
| Bessel solution | Analytical ground truth: `p = A·J₀(kr)·sin(ωt)` |
| BPF detection | Peak prominence analysis, drone type estimation |
| Rotor harmonics | Multi-harmonic synthesis: BPF × 1,2,...,N |

---

## Simulation Modules

| Module | Status | Description |
|--------|--------|-------------|
| `pinn/model.py` | ✅ Phase 1 | Fourier-embedded PINN + Wave equation loss |
| `pinn/sampler.py` | ✅ Phase 1 | Uniform, LHS, source-clustered, boundary, IC samplers |
| `pinn/trainer.py` | ✅ Phase 1 | Adam → L-BFGS trainer, save/load, field evaluation |
| `signal_processing/synthetic.py` | ✅ Phase 1 | Analytical Bessel field, rotor harmonics, sensor array |
| `signal_processing/processor.py` | ✅ Phase 1 | FFT, STFT, BPF peak detection, SNR estimation |
| `streamlit_app/app.py` | ✅ Phase 1 | Full interactive dashboard |

---

## Project Structure

```
anti_drone_PINN/
├── pinn/
│   ├── model.py                 # AcousticPINN + WaveEquationLoss
│   ├── sampler.py               # Collocation point strategies
│   ├── trainer.py               # PINNTrainer: Adam + L-BFGS
│   └── __init__.py
├── signal_processing/
│   ├── synthetic.py             # Bessel solution, rotor harmonics
│   └── processor.py             # FFT, STFT, BPF detection
├── streamlit_app/
│   └── app.py                   # Streamlit dashboard
├── data/                        # WAV files, microphone recordings
├── models/                      # Model definitions (future phases)
├── trained_models/              # Saved checkpoints
├── notebooks/                   # Experimental notebooks
├── visualization/               # Standalone plot utilities
├── utils/                       # Shared helpers
└── requirements.txt
```

---

## Installation

```bash
# Clone
git clone https://github.com/mahin-aeroai/anti-drone-pinn.git
cd anti-drone-pinn

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
cd streamlit_app
streamlit run app.py
```

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
plotly>=5.14.0
librosa>=0.10.0
scikit-learn>=1.3.0
streamlit>=1.28.0
```

---

## Dashboard

The Streamlit dashboard provides:

- **Analytical pressure field** — Bessel solution ground truth
- **PINN learned field** — what the network inferred from physics
- **Physics residual map** — where the wave equation is violated
- **FFT spectrum** — BPF harmonics with peak detection
- **STFT spectrogram** — time-frequency evolution
- **Training loss curves** — Adam and L-BFGS phases with log scale

Sidebar controls: drone position, RPM, drone type, temperature, training epochs, learning rate, time snapshot.

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Stationary drone · Wave equation PINN · FFT · Spectrogram | ✅ Complete |
| 2 | Multi-drone swarm · Interference patterns · Constructive/destructive waves | 🔜 Next |
| 3 | Moving drone · Velocity · Trajectory · Doppler shift | 🔜 Planned |
| 4 | Environmental effects · Wind · Humidity · Temperature gradients | 🔜 Planned |
| 5 | Rotor harmonic modeling · Quadcopter / Hexacopter / Fixed-wing | 🔜 Planned |
| 6 | CNN classifier · LSTM sequence model · Autoencoder anomaly detector | 🔜 Planned |
| 7 | Sensor array · Acoustic beamforming · Localization & triangulation | 🔜 Planned |
| 8 | Real-time inference · Edge deployment · TinyML on Raspberry Pi | 🔜 Planned |

---

## Background

This project sits at the intersection of **aerospace engineering** and **machine learning** — specifically the insight that physical laws, when encoded as differentiable loss functions, can replace labeled datasets entirely.

The two-phase Adam → L-BFGS training strategy is motivated by the Graetz problem study on PINNs for convection-diffusion, where second-order optimization was shown to break through the loss plateau that first-order methods cannot escape.

Rotor noise theory follows Gutin (1948) and Lighthill's aeroacoustic analogy (1952). The Bessel function analytical solution serves as ground truth for validating the PINN field at moderate frequencies.

---

## References

- Raissi, M., Perdikaris, P., Lagaris, I.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics.
- Lighthill, M.J. (1952). *On sound generated aerodynamically.* Proceedings of the Royal Society.
- Gutin, L. (1948). *On the sound field of a rotating propeller.* NACA Technical Memorandum.
- Tancik, M. et al. (2020). *Fourier features let networks learn high frequency functions in low dimensional domains.* NeurIPS.

---

## Author

**Mahin Nandipa**
B.Tech Aerospace Engineering, VIT Bhopal
PG Certificate AI/ML, IIIT Hyderabad (TalentSprint × Accenture)

[GitHub](https://github.com/mahin-aeroai) · [LinkedIn](https://linkedin.com/in/mahin-nandipa-58189b392)

---

<div align="center">
<sub>Physics-Informed Neural Networks · Aeroacoustics · Anti-Drone Surveillance · PyTorch · Streamlit</sub>
</div>
