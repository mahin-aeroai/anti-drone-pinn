"""
Signal processing utilities for acoustic drone detection.
Phase 1: FFT, STFT spectrogram, BPF peak detection.
"""

import numpy as np
from scipy.signal import spectrogram, find_peaks
from scipy.fft import fft, fftfreq


def compute_fft(signal, sr=16000):
    """Returns (frequencies, magnitude_dB)."""
    N    = len(signal)
    yf   = np.abs(fft(signal))[:N//2]
    xf   = fftfreq(N, 1/sr)[:N//2]
    db   = 20 * np.log10(yf + 1e-12)
    return xf, db


def compute_spectrogram(signal, sr=16000, nperseg=512, noverlap=384):
    """Returns (times, frequencies, power_dB)."""
    f, t, Sxx = spectrogram(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db    = 10 * np.log10(Sxx + 1e-12)
    return t, f, Sxx_db


def detect_bpf_peaks(freqs, magnitude_db, min_freq=50, max_freq=5000,
                      prominence=6, n_peaks=6):
    """
    Find dominant frequency peaks (BPF + harmonics).
    Returns list of (frequency, magnitude_dB) tuples.
    """
    mask   = (freqs >= min_freq) & (freqs <= max_freq)
    f_cut  = freqs[mask]
    m_cut  = magnitude_db[mask]
    peaks, props = find_peaks(m_cut, prominence=prominence)
    if len(peaks) == 0:
        return []
    # Sort by prominence
    order  = np.argsort(props['prominences'])[::-1][:n_peaks]
    peaks  = peaks[order]
    return [(f_cut[p], m_cut[p]) for p in sorted(peaks)]


def estimate_drone_type(bpf_hz):
    """
    Rough drone type estimation from fundamental BPF.
    BPF = RPM/60 * n_blades
    """
    if bpf_hz < 80:
        return "Fixed-wing UAV (low RPM)"
    elif bpf_hz < 150:
        return "Hexacopter (6-blade)"
    elif bpf_hz < 300:
        return "Quadcopter (4-blade)"
    else:
        return "High-speed racing drone"


def snr_db(signal, noise_floor_frac=0.1):
    """Estimate SNR from a signal."""
    sig_power  = np.mean(signal**2)
    n          = int(len(signal) * noise_floor_frac)
    noise_est  = np.std(signal[:n])**2 + 1e-12
    return 10 * np.log10(sig_power / noise_est)
