"""Interactive tuning utility to create EQ presets."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE
from .presets import save_preset

DEFAULT_DURATION = 2.0  # seconds to record each sample


def _record_sample(duration: float, samplerate: int) -> np.ndarray:
    """Record mono audio for *duration* seconds and return as 1-D float32 array."""
    print(f"Recording {duration} s …")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("   ✔ done\n")
    return audio[:, 0]


def _dominant_freq(x: np.ndarray, fs: int) -> float | None:
    """Return dominant frequency of *x* (Hz) or None if level is too low."""
    # Hanning window then FFT
    xw = x * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(xw))
    freqs = np.fft.rfftfreq(len(xw), 1 / fs)
    idx = np.argmax(mag)
    if mag[idx] < 1e-4:
        return None
    return float(freqs[idx])


def run_tuner(preset_path: Path, samplerate: int = SAMPLE_RATE) -> None:
    print("=== Voice-Equalizer Tuner ===")
    print("You will be prompted to record isolated sounds you want to amplify.")
    print("Leave the name empty when you are finished.\n")

    bands: List[dict] = []
    while True:
        name = input("Sound name (empty to finish): ").strip()
        if not name:
            break
        input("Press ENTER then immediately make the sound …")
        sample = _record_sample(DEFAULT_DURATION, samplerate)
        freq = _dominant_freq(sample, samplerate)
        if freq is None:
            print("   ⚠️  Could not detect a dominant frequency – skipping.\n")
            continue
        print(f"   ➜ Detected {freq:.1f} Hz for '{name}'\n")
        # --- fine-tune ---------------------------------------------------
        try:
            gain_str = input("   Gain in dB (default 6): ").strip()
            gain_db = float(gain_str) if gain_str else 6.0
        except ValueError:
            print("   Invalid gain – using 6 dB.")
            gain_db = 6.0

        try:
            q_str = input("   Bandwidth Q (default 4): ").strip()
            Q = float(q_str) if q_str else 4.0
        except ValueError:
            print("   Invalid Q – using 4.0.")
            Q = 4.0

        bands.append({"name": name, "freq": freq, "gain_db": gain_db, "Q": Q})

    if not bands:
        print("No bands recorded, aborting.")
        sys.exit(1)

    save_preset(preset_path, bands)
    print(f"Preset saved to {preset_path} containing {len(bands)} band(s).") 