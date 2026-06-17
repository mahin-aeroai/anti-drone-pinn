# Physics-Informed Aeroacoustic Anti-Drone Detection System

**Research-grade PINN project · Aerospace × ML**

## Governing Physics

```
∂²p/∂t² = c²(∂²p/∂x² + ∂²p/∂y²)
```

The PINN learns the acoustic pressure field from the wave equation alone — no labeled data required.

## Architecture

- **Fourier Feature Network** (Random Fourier Embeddings + Tanh MLP)
- **Auto-differentiation losses**: PDE + BC + IC + Source
- **Two-phase optimizer**: Adam (explore) → L-BFGS (polish)

## Loss Function

```
L_total = L_pde + 10·L_bc + 5·L_ic + 20·L_src
```

## Project Structure

```
anti_drone_PINN/
├── pinn/
│   ├── model.py          # AcousticPINN + WaveEquationLoss
│   ├── sampler.py        # Collocation point strategies
│   └── trainer.py        # PINNTrainer (Adam + L-BFGS)
├── signal_processing/
│   ├── synthetic.py      # Analytical Bessel solution + rotor harmonics
│   └── processor.py      # FFT, STFT, BPF detection
├── streamlit_app/
│   └── app.py            # Phase 1 dashboard
├── data/                 # WAV files, microphone recordings
├── trained_models/       # Saved checkpoints
└── requirements.txt
```

## Installation & Run

```bash
pip install -r requirements.txt
cd streamlit_app
streamlit run app.py
```

## Phases

| Phase | Feature |
|-------|---------|
| ✅ 1 | Stationary drone · Wave equation PINN · FFT · Spectrogram |
| 🔜 2 | Multi-drone swarm · Interference patterns |
| 🔜 3 | Moving drone · Doppler shift |
| 🔜 4 | Environmental effects (wind, humidity, temperature) |
| 🔜 5 | Rotor harmonic modeling per drone class |
| 🔜 6 | CNN/LSTM classifier · Autoencoder anomaly detection |
| 🔜 7 | Sensor array · Acoustic beamforming · Localization |
| 🔜 8 | Real-time inference · Edge deployment |

## Key References

- Raissi et al. (2019) — Physics-Informed Neural Networks
- Lighthill (1952) — Aeroacoustic analogy
- Gutin (1948) — Rotor noise theory
- Lévêque approximation — boundary layer acoustics
