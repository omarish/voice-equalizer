from importlib import import_module

# Delayed import to avoid circular dependencies
processing = import_module("voice_equalizer.processing")

design_eq = processing.design_eq  # type: ignore[attr-defined]
AudioProcessor = processing.AudioProcessor  # type: ignore[attr-defined]

__all__ = ["design_eq", "AudioProcessor"] 