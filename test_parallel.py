#!/usr/bin/env python3
"""Test parallel speed auto-detection by forcing no metadata"""

import numpy as np
import wave
import click
from muwave.audio.fsk import FSKConfig, FSKModulator, FSKDemodulator
from muwave.core.config import Config
from muwave.ui.interface import UIManager
import time
import concurrent.futures
import os

# Setup
ui = UIManager()
cfg = Config()

# Generate test signal (2-channel, medium speed)
config = FSKConfig(
    sample_rate=44100,
    base_frequency=1800,
    frequency_step=120,
    num_frequencies=16,
    symbol_duration_ms=60,  # medium speed
    start_frequency=800,
    end_frequency=900,
    signal_duration_ms=200,
    silence_ms=50,
    volume=0.8,
    num_channels=2,
)

modulator = FSKModulator(config)
text = "Quick parallel test"
audio = modulator.encode_text(text, signature_length=8, include_metadata=False)

# Save
audio_int16 = np.int16(audio * 32767)
with wave.open('test_parallel_no_meta.wav', 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(44100)
    wf.writeframes(audio_int16.tobytes())

ui.print_success("Generated test file without metadata")

# Now decode with parallel testing
with wave.open('test_parallel_no_meta.wav', 'rb') as wf:
    audio_samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    audio_samples = audio_samples.astype(np.float32) / 32768.0

# Detect boundaries
temp_config = FSKConfig(
    sample_rate=44100,
    base_frequency=1800,
    frequency_step=120,
    num_frequencies=16,
    symbol_duration_ms=35,
    start_frequency=800,
    end_frequency=900,
    signal_duration_ms=200,
    silence_ms=50,
    num_channels=1,
)
temp_demod = FSKDemodulator(temp_config)

detected, start_pos = temp_demod.detect_start_signal(audio_samples)
ui.print_success(f"Start detected at {start_pos}")

data_samples = audio_samples[start_pos:]
end_detected, end_pos = temp_demod.detect_end_signal(data_samples)
message_samples = data_samples[:end_pos] if end_detected else data_samples

# Parallel speed testing
ui.print_info("Testing parallel speed auto-detection...")
test_speeds = [
    ("ultra-fast", 20),
    ("fast", 35),
    ("medium", 60),
    ("slow", 120),
]

num_channels = 2
max_workers = os.cpu_count() or 4
ui.print_info(f"Using {max_workers} threads...")

def _test_speed(args):
    speed_name, symbol_dur = args
    local_cfg = FSKConfig(
        sample_rate=44100,
        base_frequency=1800,
        frequency_step=120,
        num_frequencies=16,
        symbol_duration_ms=symbol_dur,
        start_frequency=800,
        end_frequency=900,
        signal_duration_ms=200,
        silence_ms=50,
        volume=0.8,
        num_channels=num_channels,
    )
    demod = FSKDemodulator(local_cfg)
    text, signature, confidence = demod.decode_text(
        message_samples,
        signature_length=8,
        repetitions=1,
        read_metadata=False,
    )
    return (speed_name, symbol_dur, text, signature, confidence)

start_time = time.time()
best_confidence = 0.0
best_result = None

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_map = {executor.submit(_test_speed, ts): ts for ts in test_speeds}
    for future in concurrent.futures.as_completed(future_map):
        speed_name, symbol_dur, text, signature, confidence = future.result()
        ui.console.print(f"  {speed_name} ({symbol_dur}ms): ", end="")
        if text is not None:
            ui.console.print(f"[green]✓[/] {confidence:.2%}")
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = (speed_name, symbol_dur, text, signature, confidence)
        else:
            ui.console.print(f"[red]✗[/] {confidence:.2%}")

elapsed = time.time() - start_time

if best_result:
    speed_name, symbol_dur, text, signature, confidence = best_result
    ui.print_success(f"Best: {speed_name} ({symbol_dur}ms) - {confidence:.2%} confidence")
    ui.console.print(f"\n[bold]Decoded text:[/] {text}")
    ui.console.print(f"[bold]Time:[/] {elapsed:.3f}s")
else:
    ui.print_error("No successful decode")
