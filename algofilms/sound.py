"""A code-only soundtrack that follows the process: a detuned, noisy cluster while the error is high,
resolving into a clean chord as the error falls, with soft ticks for events (a blob split, a thread drawn).

numpy only. Writes 16-bit stereo WAV. It is a bed, not a song: swap in a real track for anything serious."""
from __future__ import annotations

import wave

import numpy as np

SR = 44100
CHORDS = {
    "maj9": [0, 12, 19, 23, 26, 28, 31],     # C2 C3 G3 B3 D4 E4 G4
    "min9": [0, 12, 19, 22, 26, 27, 31],
    "sus":  [0, 12, 19, 24, 26, 31, 33],
}


def _lowpass(x: np.ndarray, a: float) -> np.ndarray:
    """one-pole low-pass via IIR expressed with cumulative sums would be complex; use FFT brick-ish shelf instead."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (f / a) ** 2)
    return np.fft.irfft(X, len(x))


def _reverb(x: np.ndarray, seconds: float = 2.6, mix: float = 0.32, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.standard_normal((n, 2)) * np.exp(-np.linspace(0, 7, n))[:, None]
    ir[:, 0] = _lowpass(ir[:, 0], 5000); ir[:, 1] = _lowpass(ir[:, 1], 5000)
    ir /= np.abs(ir).sum(0, keepdims=True) ** 0.5 * 8
    L = len(x) + n
    nfft = 1 << (L - 1).bit_length()
    wet = np.stack([np.fft.irfft(np.fft.rfft(x[:, c], nfft) * np.fft.rfft(ir[:, c], nfft), nfft)[:len(x)] for c in range(2)], 1)
    return x * (1 - mix) + wet * mix


def score(duration: float, error, events=(), root_hz: float = 65.41, chord: str = "maj9", seed: int = 7,
          swell_at: float | None = None) -> np.ndarray:
    """error: callable t -> [0,1] (1 = chaos, 0 = solved) or an array sampled evenly over the duration.
    events: times in seconds for soft ticks. Returns float32 (n, 2) in [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    if callable(error):
        e = np.array([error(x) for x in np.linspace(0, duration, 400)])
    else:
        e = np.asarray(error, np.float64)
    e = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(e)), np.clip(e, 0, 1))
    out = np.zeros((n, 2))
    semis = CHORDS[chord]
    for i, s in enumerate(semis):
        f0 = root_hz * 2 ** (s / 12)
        det = rng.uniform(0.6, 1.6) * rng.choice([-1, 1])
        drift = 0.15 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 6))
        f = f0 * 2 ** ((det * e ** 1.3 + drift * e) / 12)
        ph = 2 * np.pi * np.cumsum(f) / SR
        v = np.sin(ph) + 0.25 * np.sin(2 * ph) * (1 - e) + 0.12 * np.sin(3 * ph)
        amp = (0.55 if i == 0 else 0.32) * (0.8 + 0.2 * np.sin(2 * np.pi * rng.uniform(0.03, 0.09) * t + i))
        amp *= np.clip((t - i * 0.35) / 2.0, 0, 1)          # voices enter one by one
        pan = 0.5 + 0.35 * np.sin(i * 2.1)
        out[:, 0] += v * amp * (1 - pan); out[:, 1] += v * amp * pan
    hiss = _lowpass(rng.standard_normal(n), 2500) * 0.9 * e ** 2
    out += hiss[:, None] * np.array([0.9, 1.0])
    for k, te in enumerate(events):
        i0 = int(te * SR)
        if i0 >= n:
            continue
        L = min(int(0.12 * SR), n - i0)
        tt = np.arange(L) / SR
        f = root_hz * 8 * 2 ** (rng.choice([0, 2, 4, 7, 9, 12]) / 12)
        blip = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 38) * 0.09
        p = rng.uniform(0.2, 0.8)
        out[i0:i0 + L, 0] += blip * (1 - p); out[i0:i0 + L, 1] += blip * p
    if swell_at is not None:
        g = np.clip((t - swell_at) / 2.5, 0, 1)
        sw = sum(np.sin(2 * np.pi * root_hz * 2 ** ((s + 12) / 12) * t) for s in semis[2:]) * 0.12 * g ** 2
        out += sw[:, None]
    out[:, 0] = _lowpass(out[:, 0], 1800 + 5000 * 0); out[:, 1] = _lowpass(out[:, 1], 1800)
    out = _reverb(out)
    fade = np.minimum(1, np.minimum(t / 1.2, (duration - t) / 2.0))
    out *= np.clip(fade, 0, 1)[:, None]
    out *= 0.89 / max(1e-6, np.abs(out).max())
    return out.astype(np.float32)


def write_wav(path: str, x: np.ndarray):
    x16 = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(x16.tobytes())
