"""
Anti-Drone Acoustic PINN — Phase 1 Streamlit Dashboard
Physics: ∂²p/∂t² = c²(∂²p/∂x² + ∂²p/∂y²)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from pinn.model   import AcousticPINN, WaveEquationLoss
from pinn.sampler import (wall_clustered_collocation, boundary_points,
                           initial_condition_points, source_points)
from signal_processing.synthetic  import SyntheticAcousticField
from signal_processing.processor  import (compute_fft, compute_spectrogram,
                                           detect_bpf_peaks, estimate_drone_type, snr_db)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AeroAcoustic PINN", page_icon="🛸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #07090f; color: #c5d5e8; }
section[data-testid="stSidebar"] { background: #0b0f1a; border-right: 1px solid #1a2f4a; }
.stButton>button { background:#0ea5e9; color:#000; font-weight:700; border:none;
                   border-radius:4px; width:100%; padding:0.45rem 0; }
.stButton>button:hover { background:#38bdf8; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top: 1rem; }

.hdr { border-bottom:1px solid #1a2f4a; padding-bottom:.8rem; margin-bottom:1.2rem; }
.hdr-title { font-size:1.8rem; font-weight:700; color:#e0efff; letter-spacing:-.03em; margin:0; }
.hdr-sub { font-size:.75rem; color:#3a6a8a; font-family:'JetBrains Mono',monospace; margin-top:.2rem; }

.kpi { background:#0c1220; border:1px solid #1a2f4a; border-radius:5px;
        padding:.6rem 1rem; margin-bottom:.5rem; }
.kpi-lbl { font-size:.65rem; color:#3a6a8a; font-family:'JetBrains Mono',monospace;
            text-transform:uppercase; letter-spacing:.08em; }
.kpi-val { font-size:1.3rem; font-weight:700; color:#38bdf8;
            font-family:'JetBrains Mono',monospace; }

.tag { display:inline-block; background:#0c1220; border:1px solid #1a2f4a;
        border-radius:3px; padding:.1rem .45rem; font-size:.65rem;
        font-family:'JetBrains Mono',monospace; color:#38bdf8; margin:.1rem; }

.sec { font-size:.65rem; color:#3a6a8a; font-family:'JetBrains Mono',monospace;
        text-transform:uppercase; letter-spacing:.1em; border-left:2px solid #0ea5e9;
        padding-left:.4rem; margin-bottom:.4rem; margin-top:.8rem; }

.eq  { background:#0c1220; border:1px solid #1a2f4a; border-radius:5px;
        padding:.75rem 1rem; font-family:'JetBrains Mono',monospace;
        font-size:.8rem; color:#7dd3fc; margin-bottom:.8rem; }

.alert-green { background:#052e16; border:1px solid #166534; border-radius:5px;
                padding:.5rem .8rem; font-size:.75rem; color:#86efac; margin-top:.5rem; }
.alert-red   { background:#450a0a; border:1px solid #991b1b; border-radius:5px;
                padding:.5rem .8rem; font-size:.75rem; color:#fca5a5; margin-top:.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────── HEADER ──────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-title">🛸 AeroAcoustic PINN — Anti-Drone Detection</div>
  <div class="hdr-sub">
    ∂²p/∂t² = c²∇²p &nbsp;·&nbsp; Phase 1: Stationary Source &nbsp;·&nbsp;
    Wave Equation · PyTorch Autodiff · Fourier Feature Network
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────── SIDEBAR ─────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec">Drone Source</div>', unsafe_allow_html=True)
    src_x = st.slider("X position (norm.)", -0.8, 0.8, 0.0, 0.05)
    src_y = st.slider("Y position (norm.)", -0.8, 0.8, 0.0, 0.05)

    st.markdown('<div class="sec">Rotor Acoustics</div>', unsafe_allow_html=True)
    drone_type = st.selectbox("Drone type", ["Quadcopter (4-blade)", "Hexacopter (6-blade)", "Fixed-wing UAV"])
    rpm        = st.slider("RPM", 2000, 12000, 5000, 250)
    n_blades   = {"Quadcopter (4-blade)": 4, "Hexacopter (6-blade)": 6, "Fixed-wing UAV": 2}[drone_type]
    bpf        = (rpm / 60) * n_blades
    freq_norm  = min(bpf / 400.0, 2.0)   # normalised for PINN
    amp        = st.slider("Amplitude", 0.1, 2.0, 1.0, 0.1)

    st.markdown('<div class="sec">Atmosphere</div>', unsafe_allow_html=True)
    temp_c  = st.slider("Temperature (°C)", -20, 50, 20, 1)
    c_ms    = 331.3 * np.sqrt(1 + temp_c / 273.15)   # temperature-corrected
    c_norm  = c_ms / 343.0

    st.markdown('<div class="sec">PINN Training</div>', unsafe_allow_html=True)
    adam_ep  = st.slider("Adam epochs", 100, 1000, 400, 50)
    lbfgs_ep = st.slider("L-BFGS epochs", 0, 200, 60, 10)
    n_col    = st.slider("Collocation pts", 500, 4000, 1500, 250)
    lr       = st.select_slider("Learning rate", [5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)

    st.markdown('<div class="sec">Visualise</div>', unsafe_allow_html=True)
    t_snap   = st.slider("Time snapshot t", 0.0, 1.0, 0.5, 0.02)
    grid_n   = st.select_slider("Grid resolution", [60, 80, 100, 120], value=80)

    train_btn = st.button("▶ Train PINN")
    reset_btn = st.button("↺ Reset")

# Derived display values
wavelength = c_ms / bpf
st.sidebar.markdown(f"""
<div class="kpi"><div class="kpi-lbl">BPF</div><div class="kpi-val">{bpf:.0f} Hz</div></div>
<div class="kpi"><div class="kpi-lbl">Speed of Sound</div><div class="kpi-val">{c_ms:.1f} m/s</div></div>
<div class="kpi"><div class="kpi-lbl">Wavelength</div><div class="kpi-val">{wavelength:.2f} m</div></div>
""", unsafe_allow_html=True)

# ─────────────────────────────── EQUATION PANEL ──────────────────────────────
col_eq, col_kpi = st.columns([3, 1])
with col_eq:
    st.markdown('<div class="sec">Governing Physics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="eq">
    ∂²p/∂t² = c² · (∂²p/∂x² + ∂²p/∂y²)<br>
    <span style="color:#4a7fa5">
    c = {c_ms:.1f} m/s (T={temp_c}°C) &nbsp;·&nbsp;
    BPF = {bpf:.0f} Hz &nbsp;·&nbsp;
    source @ ({src_x:.2f}, {src_y:.2f}) &nbsp;·&nbsp;
    {drone_type}
    </span><br>
    <span style="color:#2a4f6a">
    Loss = L_pde + 10·L_bc + 5·L_ic + 20·L_src
    </span>
    </div>
    """, unsafe_allow_html=True)

with col_kpi:
    st.markdown('<div class="sec">Phase tags</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <span class="tag">Phase 1</span>
    <span class="tag">Stationary src</span>
    <span class="tag">Adam+L-BFGS</span>
    <span class="tag">Fourier embed</span>
    <span class="tag">{drone_type.split()[0]}</span>
    <span class="tag">BPF {bpf:.0f}Hz</span>
    """, unsafe_allow_html=True)

# ─────────────────────────────── MODEL STATE ─────────────────────────────────
if reset_btn:
    for k in ['trained', 'history', 'model_state']:
        st.session_state.pop(k, None)
    st.rerun()

@st.cache_resource
def get_model():
    return AcousticPINN(hidden=96, layers=6, fourier_embed=True, embed_dim=32)

model = get_model()

if 'model_state' in st.session_state:
    model.load_state_dict(st.session_state['model_state'])

loss_fn = WaveEquationLoss(c=c_norm, w_pde=1.0, w_bc=10.0, w_ic=5.0, w_src=20.0)

# ─────────────────────────────── TRAINING ────────────────────────────────────
if train_btn:
    st.markdown('<div class="sec">Training</div>', unsafe_allow_html=True)
    prog   = st.progress(0)
    status = st.empty()
    history = []

    # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, adam_ep)

    for ep in range(adam_ep):
        optimizer.zero_grad()
        colloc = wall_clustered_collocation(n_col, src_x, src_y)
        bc     = boundary_points(max(n_col//5, 200))
        ic     = initial_condition_points(max(n_col//5, 200))
        src_pts, src_tgt = source_points(max(n_col//8, 150), src_x, src_y,
                                          freq=freq_norm, amp=amp)
        losses, _ = loss_fn(model, colloc, bc, ic, src_pts, src_tgt)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); sched.step()

        rec = {k: v.item() for k, v in losses.items()}
        rec.update({'epoch': ep+1, 'phase': 'adam'})
        history.append(rec)

        if ep % max(1, adam_ep//25) == 0:
            prog.progress((ep+1) / (adam_ep + lbfgs_ep + 1))
            status.markdown(
                f'<div class="sec">Adam {ep+1}/{adam_ep} · Loss {rec["total"]:.4f} · PDE {rec["pde"]:.4f}</div>',
                unsafe_allow_html=True)

    # L-BFGS
    if lbfgs_ep > 0:
        lbfgs  = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20,
                                     history_size=50, line_search_fn='strong_wolfe')
        colloc_f = wall_clustered_collocation(n_col, src_x, src_y)
        bc_f     = boundary_points(max(n_col//5, 200))
        ic_f     = initial_condition_points(max(n_col//5, 200))
        sp_f, st_f = source_points(max(n_col//8, 150), src_x, src_y,
                                    freq=freq_norm, amp=amp)

        def closure():
            lbfgs.zero_grad()
            l, _ = loss_fn(model, colloc_f, bc_f, ic_f, sp_f, st_f)
            l['total'].backward()
            return l['total']

        for ep in range(lbfgs_ep):
            lbfgs.step(closure)
            with torch.no_grad():
                ls, _ = loss_fn(model, colloc_f, bc_f, ic_f, sp_f, st_f)
            rec = {k: v.item() for k, v in ls.items()}
            rec.update({'epoch': adam_ep + ep + 1, 'phase': 'lbfgs'})
            history.append(rec)
            prog.progress((adam_ep + ep + 1) / (adam_ep + lbfgs_ep + 1))
            if ep % max(1, lbfgs_ep//5) == 0:
                status.markdown(
                    f'<div class="sec">L-BFGS {ep+1}/{lbfgs_ep} · Loss {rec["total"]:.5f}</div>',
                    unsafe_allow_html=True)

    prog.progress(1.0)
    st.session_state['trained'] = True
    st.session_state['history'] = history
    st.session_state['model_state'] = model.state_dict()
    status.markdown(
        f'<div class="alert-green">✓ Training complete · Final loss: {history[-1]["total"]:.5f}</div>',
        unsafe_allow_html=True)

# ─────────────────────────────── VISUALISATIONS ──────────────────────────────
trained = st.session_state.get('trained', False)
history = st.session_state.get('history', [])

# Always show analytical reference + signal analysis
st.markdown('<div class="sec">Analytical Reference Field (Bessel Solution)</div>', unsafe_allow_html=True)

synth = SyntheticAcousticField(c=c_ms, freq=bpf, amp=amp, src_x=src_x*5, src_y=src_y*5)
xs_a, ys_a, PP_a = synth.field_snapshot(t=t_snap * (1/bpf), grid_n=120)

sr = 16000
t_arr = np.linspace(0, 1, sr)
sensor_sig = synth.time_series_at_sensor(2.0, 0.0, t_arr)
rotor_sig  = synth.rotor_harmonics(n_blades=n_blades, rpm=rpm, n_harmonics=6,
                                    t_arr=t_arr, sx=2.0, sy=0.0)
freqs_fft, mag_fft = compute_fft(rotor_sig, sr)
t_spec, f_spec, Sxx = compute_spectrogram(rotor_sig, sr)
peaks = detect_bpf_peaks(freqs_fft, mag_fft)

# ── Row 1: Analytical field + FFT + Spectrogram ──────────────────────────────
fig1 = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Analytical Pressure Field", "FFT — Rotor Harmonics", "STFT Spectrogram"],
    horizontal_spacing=0.08
)

fig1.add_trace(go.Heatmap(z=PP_a, x=xs_a, y=ys_a, colorscale='RdBu', zmid=0,
    colorbar=dict(x=0.30, len=0.9, thickness=10, tickfont=dict(color='#6a9abf', size=8)),
    name='Analytical p'), row=1, col=1)
fig1.add_trace(go.Scatter(x=[src_x*5], y=[src_y*5], mode='markers',
    marker=dict(symbol='hexagram', size=14, color='#facc15',
                line=dict(color='#fff', width=1)), name='Drone'), row=1, col=1)

# BPF peak lines
for pf, pm in peaks[:6]:
    fig1.add_vline(x=pf, line=dict(color='#f472b6', width=1, dash='dot'), row=1, col=2)

fig1.add_trace(go.Scatter(x=freqs_fft[freqs_fft<5000],
    y=mag_fft[freqs_fft<5000],
    line=dict(color='#38bdf8', width=1.2), name='FFT',
    fill='tozeroy', fillcolor='rgba(56,189,248,0.08)'), row=1, col=2)

fig1.add_trace(go.Heatmap(z=Sxx, x=t_spec, y=f_spec[f_spec<5000],
    colorscale='Inferno',
    colorbar=dict(x=0.99, len=0.9, thickness=10, tickfont=dict(color='#6a9abf', size=8)),
    name='Spectrogram'), row=1, col=3)

_layout = dict(height=380, paper_bgcolor='#07090f', plot_bgcolor='#0c1220',
    font=dict(family='JetBrains Mono', color='#6a9abf', size=9),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor='#0c1220', bordercolor='#1a2f4a', borderwidth=1, font=dict(size=8)))
fig1.update_layout(**_layout)
for ax in ['xaxis','yaxis','xaxis2','yaxis2','xaxis3','yaxis3']:
    fig1.update_layout(**{ax: dict(gridcolor='#1a2f4a', zerolinecolor='#1a2f4a')})
fig1.update_xaxes(title_text='Frequency (Hz)', row=1, col=2)
fig1.update_yaxes(title_text='dB', row=1, col=2)
fig1.update_xaxes(title_text='Time (s)', row=1, col=3)
fig1.update_yaxes(title_text='Frequency (Hz)', row=1, col=3)
st.plotly_chart(fig1, use_container_width=True)

# BPF detection summary
if peaks:
    fund = peaks[0][0]
    est  = estimate_drone_type(fund)
    snr  = snr_db(rotor_sig)
    st.markdown(f"""
    <div class="alert-green">
    🔊 Detected peaks: {', '.join([f'{f:.0f}Hz' for f,_ in peaks[:5]])} &nbsp;·&nbsp;
    Fundamental: <strong>{fund:.0f} Hz</strong> &nbsp;·&nbsp;
    Estimated: <strong>{est}</strong> &nbsp;·&nbsp;
    SNR: <strong>{snr:.1f} dB</strong>
    </div>""", unsafe_allow_html=True)

# ── Row 2: PINN results (if trained) ─────────────────────────────────────────
if trained and history:
    st.markdown('<div class="sec">PINN Learned Field vs Residual vs Training Loss</div>',
                unsafe_allow_html=True)

    # Evaluate PINN field
    xs_p = np.linspace(-1, 1, grid_n)
    ys_p = np.linspace(-1, 1, grid_n)
    XX, YY = np.meshgrid(xs_p, ys_p)
    with torch.no_grad():
        xf = torch.FloatTensor(XX.ravel())
        yf = torch.FloatTensor(YY.ravel())
        tf = torch.full_like(xf, t_snap)
        PP_pinn = model(xf, yf, tf).numpy().reshape(grid_n, grid_n)

    # Residual map
    rg = 40
    xs_r = np.linspace(-1, 1, rg)
    ys_r = np.linspace(-1, 1, rg)
    XX_r, YY_r = np.meshgrid(xs_r, ys_r)
    xr = torch.FloatTensor(XX_r.ravel()).requires_grad_(True)
    yr = torch.FloatTensor(YY_r.ravel()).requires_grad_(True)
    tr = torch.full_like(xr, t_snap).requires_grad_(True)
    pr = model(xr, yr, tr)
    pr_tt = torch.autograd.grad(
        torch.autograd.grad(pr.sum(), tr, create_graph=True)[0].sum(), tr, create_graph=False
    )[0]
    pr_xx = torch.autograd.grad(
        torch.autograd.grad(pr.sum(), xr, create_graph=True)[0].sum(), xr, create_graph=False
    )[0]
    pr_yy = torch.autograd.grad(
        torch.autograd.grad(pr.sum(), yr, create_graph=True)[0].sum(), yr, create_graph=False
    )[0]
    residual_map = (pr_tt - c_norm**2 * (pr_xx + pr_yy)).detach().numpy()
    RR = np.abs(residual_map).reshape(rg, rg)

    fig2 = make_subplots(rows=1, cols=3,
        subplot_titles=["PINN Pressure Field  p(x,y,t)", "|Wave Eq. Residual|", "Training Loss"],
        horizontal_spacing=0.08)

    fig2.add_trace(go.Heatmap(z=PP_pinn, x=xs_p, y=ys_p, colorscale='RdBu', zmid=0,
        colorbar=dict(x=0.30, len=0.9, thickness=10, tickfont=dict(color='#6a9abf', size=8)),
        name='PINN p'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=[src_x], y=[src_y], mode='markers',
        marker=dict(symbol='hexagram', size=14, color='#facc15',
                    line=dict(color='#fff', width=1)), name='Drone'), row=1, col=1)

    fig2.add_trace(go.Heatmap(z=RR, x=xs_r, y=ys_r, colorscale='Inferno',
        colorbar=dict(x=0.65, len=0.9, thickness=10, tickfont=dict(color='#6a9abf', size=8)),
        name='|Residual|'), row=1, col=2)

    ep_all = [h['epoch'] for h in history]
    fig2.add_trace(go.Scatter(x=ep_all, y=[h['total'] for h in history],
        name='Total', line=dict(color='#38bdf8', width=2)), row=1, col=3)
    fig2.add_trace(go.Scatter(x=ep_all, y=[h['pde'] for h in history],
        name='PDE', line=dict(color='#f472b6', width=1.5, dash='dot')), row=1, col=3)
    fig2.add_trace(go.Scatter(x=ep_all, y=[h['src'] for h in history],
        name='Source', line=dict(color='#34d399', width=1.5, dash='dash')), row=1, col=3)

    # Phase boundary line
    adam_done = max((h['epoch'] for h in history if h.get('phase') == 'adam'), default=0)
    if adam_done and lbfgs_ep > 0:
        fig2.add_vline(x=adam_done, line=dict(color='#facc15', width=1, dash='dash'),
                       annotation_text="L-BFGS", row=1, col=3)

    fig2.update_layout(**_layout)
    fig2.update_yaxes(type='log', row=1, col=3)
    fig2.update_xaxes(title_text='Epoch', row=1, col=3)
    fig2.update_yaxes(title_text='Loss (log)', row=1, col=3)
    for ax in ['xaxis','yaxis','xaxis2','yaxis2','xaxis3','yaxis3']:
        fig2.update_layout(**{ax: dict(gridcolor='#1a2f4a', zerolinecolor='#1a2f4a')})

    st.plotly_chart(fig2, use_container_width=True)

    # KPI row
    last = history[-1]
    st.markdown(f"""
    <div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:.8rem">
      <div class="kpi" style="flex:1"><div class="kpi-lbl">Total Loss</div>
        <div class="kpi-val">{last['total']:.5f}</div></div>
      <div class="kpi" style="flex:1"><div class="kpi-lbl">PDE Residual</div>
        <div class="kpi-val">{last['pde']:.5f}</div></div>
      <div class="kpi" style="flex:1"><div class="kpi-lbl">BC Loss</div>
        <div class="kpi-val">{last.get('bc',0):.5f}</div></div>
      <div class="kpi" style="flex:1"><div class="kpi-lbl">Source Loss</div>
        <div class="kpi-val">{last['src']:.5f}</div></div>
      <div class="kpi" style="flex:1"><div class="kpi-lbl">Epochs</div>
        <div class="kpi-val">{last['epoch']}</div></div>
      <div class="kpi" style="flex:1"><div class="kpi-lbl">BPF</div>
        <div class="kpi-val">{bpf:.0f} Hz</div></div>
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#0c1220;border:1px dashed #1a2f4a;border-radius:8px;
                padding:2.5rem;text-align:center;color:#3a6a8a;margin-top:.5rem">
      <div style="font-size:2rem;margin-bottom:.4rem">🛸</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.8rem">
        Set parameters → click <strong style="color:#38bdf8">▶ Train PINN</strong>
      </div>
      <div style="font-size:.7rem;margin-top:.3rem">
        Analytical field and FFT are always live above.
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────── FOOTER ──────────────────────────────────────
st.markdown("""
<div style="margin-top:1.5rem;padding-top:.8rem;border-top:1px solid #1a2f4a;
            font-family:'JetBrains Mono',monospace;font-size:.65rem;
            color:#1e3a5a;text-align:center">
  Physics-Informed Aeroacoustic Anti-Drone Detection · Phase 1 · Wave Equation · PyTorch · Streamlit
  &nbsp;·&nbsp; Next: Phase 2 — Multiple drone swarm + interference patterns
</div>""", unsafe_allow_html=True)
