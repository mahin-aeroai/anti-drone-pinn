"""
Synthetic acoustic field generator using the analytical solution
for a point harmonic source in 2D free space.

p(x,y,t) = A * J0(k*r) * sin(ωt)

where J0 is the Bessel function of the first kind, k = ω/c, r = distance from source.
Used for:
 1. Ground truth comparison
 2. FFT / spectrogram generation
 3. Sensor array simulation
"""

import numpy as np
from scipy.special import j0


class SyntheticAcousticField:
    def __init__(self, c=343.0, freq=150.0, amp=1.0,
                 src_x=0.0, src_y=0.0,
                 domain=(-5, 5, -5, 5)):
        """
        c     : speed of sound (m/s)
        freq  : source frequency (Hz)
        amp   : pressure amplitude (Pa)
        src_x : source x position (m)
        src_y : source y position (m)
        domain: (x_min, x_max, y_min, y_max) in meters
        """
        self.c    = c
        self.freq = freq
        self.amp  = amp
        self.src  = np.array([src_x, src_y])
        self.omega = 2 * np.pi * freq
        self.k     = self.omega / c
        self.domain = domain

    def pressure(self, x, y, t):
        r = np.sqrt((x - self.src[0])**2 + (y - self.src[1])**2) + 1e-8
        return self.amp * j0(self.k * r) * np.sin(self.omega * t)

    def field_snapshot(self, t, grid_n=200):
        xs = np.linspace(self.domain[0], self.domain[1], grid_n)
        ys = np.linspace(self.domain[2], self.domain[3], grid_n)
        XX, YY = np.meshgrid(xs, ys)
        PP = self.pressure(XX, YY, t)
        return xs, ys, PP

    def time_series_at_sensor(self, sx, sy, t_arr):
        """Simulate microphone recording at position (sx, sy)."""
        return self.pressure(sx, sy, t_arr)

    def rotor_harmonics(self, n_blades=4, rpm=5000, n_harmonics=5,
                        t_arr=None, sx=1.0, sy=0.0):
        """
        Multi-harmonic source: BPF and harmonics.
        BPF = (RPM/60) * n_blades
        """
        if t_arr is None:
            t_arr = np.linspace(0, 1, 16000)
        bpf = (rpm / 60.0) * n_blades
        signal = np.zeros_like(t_arr)
        for h in range(1, n_harmonics + 1):
            f_h = bpf * h
            k_h = 2 * np.pi * f_h / self.c
            r   = np.sqrt((sx - self.src[0])**2 + (sy - self.src[1])**2) + 1e-8
            amp_h = self.amp / h  # amplitude falls off with harmonic order
            signal += amp_h * j0(k_h * r) * np.sin(2 * np.pi * f_h * t_arr)
        return signal

    def sensor_array(self, positions, t):
        """
        Compute pressure at a list of sensor positions at time t.
        positions: list of (x,y) tuples
        """
        return np.array([self.pressure(px, py, t) for px, py in positions])
