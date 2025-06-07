from __future__ import annotations

import threading
from collections import deque
from typing import Deque

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


class WaveSpectrogramViewer:
    """Real-time 2×2 waveform & spectrogram display (thread-safe)."""

    def __init__(self, samplerate: int, buffer_seconds: int = 3):
        self.sr = samplerate
        self.nbuf = buffer_seconds * samplerate
        self.pre_buf: Deque[np.ndarray] = deque(maxlen=self.nbuf)
        self.post_buf: Deque[np.ndarray] = deque(maxlen=self.nbuf)
        self._lock = threading.Lock()

        self._setup_figure()

        # Animation timer updates every 50 ms in the *main* thread
        self._anim = animation.FuncAnimation(
            self.fig, self._animate, interval=50, blit=False
        )

    # ------------------------------------------------------------------
    def _setup_figure(self):
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 6))
        (self.l_pre,) = self.ax[0, 0].plot([], [], lw=0.8)
        (self.l_post,) = self.ax[0, 1].plot([], [], lw=0.8)
        self.ax[0, 0].set_title("Waveform – pre")
        self.ax[0, 1].set_title("Waveform – post")
        self.ax[1, 0].set_title("Spectrogram – pre")
        self.ax[1, 1].set_title("Spectrogram – post")

        # initialise spectrogram images
        empty_spec = np.zeros((129, 1))
        self.spec_pre = self.ax[1, 0].imshow(
            empty_spec,
            origin="lower",
            aspect="auto",
            extent=[-3, 0, 0, self.sr / 2],
            cmap="magma",
            vmin=-100,
            vmax=0,
        )
        self.spec_post = self.ax[1, 1].imshow(
            empty_spec,
            origin="lower",
            aspect="auto",
            extent=[-3, 0, 0, self.sr / 2],
            cmap="magma",
            vmin=-100,
            vmax=0,
        )

        for axes in self.ax.flat:
            axes.set_xlabel("")
            axes.set_ylabel("")

        self.fig.tight_layout()

    # ------------------------------------------------------------------
    def add_data(self, pre: np.ndarray, post: np.ndarray):
        """Thread-safe push of latest audio frames (called from audio thread)."""
        with self._lock:
            self.pre_buf.append(pre.copy())
            self.post_buf.append(post.copy())

    # ------------------------------------------------------------------
    def _animate(self, _frame):
        """Timer-driven update executed in the GUI (main) thread."""
        with self._lock:
            pre = np.concatenate(list(self.pre_buf)) if self.pre_buf else np.array([])
            post = np.concatenate(list(self.post_buf)) if self.post_buf else np.array([])

        # --- waveforms ---
        if pre.size:
            t_pre = np.linspace(-len(pre) / self.sr, 0, len(pre))
            self.l_pre.set_data(t_pre, pre)
        if post.size:
            t_post = np.linspace(-len(post) / self.sr, 0, len(post))
            self.l_post.set_data(t_post, post)

        for axes in (self.ax[0, 0], self.ax[0, 1]):
            axes.set_xlim(-self.nbuf / self.sr, 0)
            axes.set_ylim(-1, 1)

        # --- spectrograms ---
        nfft = 256
        def _spec(signal: np.ndarray):
            if signal.size < nfft:
                return None
            S = np.abs(np.fft.rfft(signal[-self.nbuf :], n=nfft))
            return 20 * np.log10(S + 1e-6).reshape(-1, 1)

        sp_pre = _spec(pre)
        if sp_pre is not None:
            self.spec_pre.set_data(sp_pre)
        sp_post = _spec(post)
        if sp_post is not None:
            self.spec_post.set_data(sp_post)

        return ()  # not using blit contents

    # ------------------------------------------------------------------
    def show(self):
        plt.show()

    def close(self):
        plt.close(self.fig) 