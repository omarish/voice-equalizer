"""Voice-Equalizer package entry-point.

Run with:
    python -m voice_equalizer tune  my_preset.json
    python -m voice_equalizer stream --preset my_preset.json [other args]
    # or simply `python -m voice_equalizer` for default streaming preset.
"""

import argparse
import sounddevice as sd

from pathlib import Path

from .processing import AudioProcessor
from .presets import load_preset
from .tuner import run_tuner
from .config import SAMPLE_RATE

def build_parser() -> argparse.ArgumentParser:
    """Return argument parser with *tune* and *stream* subcommands."""
    parser = argparse.ArgumentParser(prog="voice-equalizer", description="Real-time microphone equalizer")
    sub = parser.add_subparsers(dest="command", required=False)

    # ---- common defaults ----
    default_input, default_output = sd.default.device

    # tune command ---------------------------------------------------------
    p_tune = sub.add_parser("tune", help="Interactively create a preset JSON file")
    p_tune.add_argument("preset", type=Path, help="Output JSON path for the new preset")
    p_tune.add_argument("--samplerate", "-r", type=int, default=SAMPLE_RATE, help="Sample rate during recording")

    # stream command -------------------------------------------------------
    p_stream = sub.add_parser("stream", help="Stream with optional preset")
    p_stream.add_argument("--preset", type=Path, help="Path to preset JSON")
    p_stream.add_argument("--input", "-i", default=default_input, help="Input device name or index")
    p_stream.add_argument("--output", "-o", default=default_output, help="Output device name or index")
    p_stream.add_argument("--samplerate", "-r", type=int, default=SAMPLE_RATE, help="Sample rate (Hz)")
    p_stream.add_argument("--frames", "-f", type=int, default=1024, help="Block size in frames")
    p_stream.add_argument("--gui", action="store_true", help="Show live waveform/spectrogram window")

    return parser


def main() -> None:  # noqa: D401
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "tune":
        run_tuner(args.preset, samplerate=args.samplerate)
        return

    # Default to streaming if no subcommand or command == stream
    bands = load_preset(args.preset) if getattr(args, "preset", None) else None

    viewer = None
    if getattr(args, "gui", False):
        from .viewer import WaveSpectrogramViewer

        viewer = WaveSpectrogramViewer(args.samplerate)

    proc = AudioProcessor(
        input_device=args.input,
        output_device=args.output,
        samplerate=args.samplerate,
        frame_size=args.frames,
        preset_bands=bands,
        monitor=viewer,
    )

    try:
        if viewer is not None:
            import threading

            t = threading.Thread(target=proc.run, daemon=True)
            t.start()
            viewer.show()  # blocks until window closed
        else:
            proc.run()
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":  # pragma: no cover
    main()