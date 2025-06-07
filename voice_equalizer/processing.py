from __future__ import annotations

import queue
import sys
from typing import List
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi

from .config import SAMPLE_RATE

if TYPE_CHECKING:
    from .viewer import WaveSpectrogramViewer

__all__ = ["design_eq", "AudioProcessor"]

# Helper --------------------------------------------------------------------

def _peaking_sos(f0: float, gain_db: float, Q: float, fs: int) -> np.ndarray:
    """Return SOS for a peaking EQ filter.

    Formula from Robert Bristow-Johnson's Audio EQ Cookbook.
    """
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    return np.asarray([[b[0], b[1], b[2], a[0], a[1], a[2]]])

# -----------------------
# Filter design utilities
# -----------------------

def design_eq(fs: int, extra_bands: List[dict] | None = None) -> List[np.ndarray]:
    """Return a list of second-order sections (SOS) representing our EQ chain."""
    sos_chain: List[np.ndarray] = []

    # 1. High-pass @80 Hz (order 2)
    hp_sos = butter(N=2, Wn=80, btype="high", fs=fs, output="sos")
    sos_chain.append(hp_sos)

    # 2. 60 Hz mains hum notch (Q≈30)
    f0 = 60
    Q = 30.0
    b, a = iirnotch(w0=f0, Q=Q, fs=fs)
    sos_chain.append(np.asarray([[b[0], b[1], b[2], a[0], a[1], a[2]]]))

    # 3. Bass-boost: low-pass branch later mixed in (+6 dB)
    bass_lp_sos = butter(N=2, Wn=120, btype="low", fs=fs, output="sos")
    sos_chain.append(bass_lp_sos)  # index = 2

    # 4. Presence bump @3 kHz (+3 dB, Q≈1)
    f0 = 3000.0
    Q = 1.0
    gain_db = 3.0
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    presence_sos = np.asarray([[b[0], b[1], b[2], a[0], a[1], a[2]]])
    sos_chain.append(presence_sos)

    # 5. Low-pass @9 kHz (order 4)
    lp_sos = butter(N=4, Wn=9000, btype="low", fs=fs, output="sos")
    sos_chain.append(lp_sos)

    # ---- user-tuned extra bands ----
    if extra_bands:
        for band in extra_bands:
            f0 = float(band["freq"])
            gain = float(band.get("gain_db", 6.0))
            Q = float(band.get("Q", 4.0))
            sos_chain.append(_peaking_sos(f0, gain, Q, fs))

    return sos_chain


class AudioProcessor:
    """Handle real-time streaming, filtering, and routing."""

    def __init__(
        self,
        input_device: str,
        output_device: str,
        *,
        samplerate: int = SAMPLE_RATE,
        frame_size: int = 1024,
        preset_bands: List[dict] | None = None,
        monitor: "WaveSpectrogramViewer" | None = None,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.samplerate = samplerate
        self.frame_size = frame_size
        self.monitor = monitor

        self.sos_chain = design_eq(samplerate, extra_bands=preset_bands)
        self.zi: List[np.ndarray] = [sosfilt_zi(sos) for sos in self.sos_chain]
        self.bass_idx = 2  # bass LP branch index
        self.q: queue.Queue[None] = queue.Queue()

    # ---- processing ----
    def _process(self, indata: np.ndarray) -> np.ndarray:
        x = indata.copy()
        for i, sos in enumerate(self.sos_chain):
            if i == self.bass_idx:
                low, self.zi[i] = sosfilt(sos, x, zi=self.zi[i])
                x, self.zi[i] = sosfilt(sos, x, zi=self.zi[i])
                x += low * 2.0  # +6 dB boost
            else:
                x, self.zi[i] = sosfilt(sos, x, zi=self.zi[i])

        # Noise-gate below −45 dBFS
        rms = np.sqrt(np.mean(x ** 2))
        if rms < 10 ** (-45 / 20):
            x *= 0.0
        return x

    # ---- stream ----
    def run(self):
        try:
            with sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.samplerate,
                blocksize=self.frame_size,
                dtype="float32",
                channels=1,
                callback=self._callback,
            ):
                print("[voice-equalizer] running – press Ctrl-C to stop")
                while True:
                    self.q.get()
        except KeyboardInterrupt:
            print("\n[voice-equalizer] stopped")
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)

    def _callback(self, indata, outdata, frames, time, status):  # noqa: D401, N803
        if status:
            print(status, file=sys.stderr)
        processed = self._process(indata[:, 0])
        if self.monitor is not None:
            # Send data to viewer (copy to avoid sharing memviews across threads)
            try:
                self.monitor.add_data(indata[:, 0].copy(), processed.copy())
            except Exception:
                pass
        outdata[:] = processed.reshape(-1, 1)
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass 