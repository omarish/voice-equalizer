.PHONY: venv install setup run devices clean

# Create a dedicated uv virtual environment in .venv
venv:
	uv venv .venv

# Install project (editable) and its dependencies into the venv
install: venv
	uv pip -p .venv/bin/python install -e .

# Convenience alias – full setup in one go
setup: install

# Run the equalizer with optional env-override variables
#   make run INPUT="Built-in Microphone" OUTPUT="BlackHole 2ch"
INPUT ?=
OUTPUT ?=
SAMPLERATE ?= 44100
FRAMES ?= 1024
PRESET ?=
GUI ?=0

run:
	.venv/bin/python -m voice_equalizer stream \
		$(if $(INPUT),--input "$(INPUT)") \
		$(if $(OUTPUT),--output "$(OUTPUT)") \
		$(if $(PRESET),--preset "$(PRESET)") \
		$(if $(filter 1,$(GUI)),--gui) \
		--samplerate $(SAMPLERATE) --frames $(FRAMES)

# List available audio devices (uses sounddevice module)
devices: venv
	.venv/bin/python -m sounddevice

# Remove the virtual environment
clean:
	rm -rf .venv

# Interactive tuning
tune:
	.venv/bin/python -m voice_equalizer tune $(PRESET)

.PHONY: lint
lint:
	ruff check . --fix