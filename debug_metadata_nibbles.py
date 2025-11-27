#!/usr/bin/env python3
"""Debug script to analyze metadata nibble decoding."""

import numpy as np
from scipy.io import wavfile
from muwave.audio.fsk import FSKDemodulator, FSKConfig

# Load the audio file
sample_rate, audio_data = wavfile.read('test_format.wav')

# Convert to mono if stereo
if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1)

# Normalize to float32 in range [-1, 1]
if audio_data.dtype == np.int16:
    audio_data = audio_data.astype(np.float32) / 32768.0
elif audio_data.dtype == np.int32:
    audio_data = audio_data.astype(np.float32) / 2147483648.0

print(f"Total samples: {len(audio_data)}")

# Create demodulator
config = FSKConfig(
    sample_rate=44100,
    signal_duration_ms=500,
    silence_ms=50,
)
demod = FSKDemodulator(config)

# Detect start signal
detected, start_pos = demod.detect_start_signal(audio_data)
print(f"\nStart signal detected: {detected}")
print(f"Start position: {start_pos}")

# Get message samples (after start signal)
message_samples = audio_data[start_pos:]
print(f"Message samples: {len(message_samples)}")

# Manually decode metadata with debug output
metadata_symbol_samples = int(sample_rate * 35.0 / 1000)
metadata_byte_samples = metadata_symbol_samples * 2

# Create temporary single-channel frequency list
temp_frequencies = np.array([config.base_frequency + i * config.frequency_step
                             for i in range(config.num_frequencies)])

print(f"\nMetadata decoding setup:")
print(f"  Symbol samples: {metadata_symbol_samples}")
print(f"  Byte samples: {metadata_byte_samples}")
print(f"  Frequencies: {temp_frequencies}")

# Skip initial silence after start signal
silence_ms = config.silence_ms
silence_samples = int(sample_rate * silence_ms / 1000)
pos = silence_samples
skip = max(1, metadata_symbol_samples // 15)

print(f"\nPosition tracking:")
print(f"  Initial silence skip: {silence_samples}")
print(f"  Starting position: {pos}")
print(f"  Skip within symbols: {skip}")

# Decode byte 1 (channel count)
print(f"\nDecoding byte 1 (channel count):")
print(f"  High nibble range: {pos + skip} to {pos + metadata_symbol_samples - skip}")
high_samples = message_samples[pos + skip:pos + metadata_symbol_samples - skip]
high_samples = high_samples - np.mean(high_samples)
high_idx, high_conf = demod._detect_frequency(high_samples, temp_frequencies)
print(f"  High nibble: {high_idx} (confidence: {high_conf:.2f})")

print(f"  Low nibble range: {pos + metadata_symbol_samples + skip} to {pos + metadata_byte_samples - skip}")
low_samples = message_samples[pos + metadata_symbol_samples + skip:pos + metadata_byte_samples - skip]
low_samples = low_samples - np.mean(low_samples)
low_idx, low_conf = demod._detect_frequency(low_samples, temp_frequencies)
print(f"  Low nibble: {low_idx} (confidence: {low_conf:.2f})")

byte1 = (high_idx << 4) | low_idx
print(f"  Byte 1 value: {byte1} (expected: channel count, probably 2)")
pos += metadata_byte_samples

# Decode byte 2 (symbol duration)
print(f"\nDecoding byte 2 (symbol duration):")
print(f"  High nibble range: {pos + skip} to {pos + metadata_symbol_samples - skip}")
high_samples = message_samples[pos + skip:pos + metadata_symbol_samples - skip]
high_samples = high_samples - np.mean(high_samples)
high_idx, high_conf = demod._detect_frequency(high_samples, temp_frequencies)
print(f"  High nibble: {high_idx} (confidence: {high_conf:.2f})")

print(f"  Low nibble range: {pos + metadata_symbol_samples + skip} to {pos + metadata_byte_samples - skip}")
low_samples = message_samples[pos + metadata_symbol_samples + skip:pos + metadata_byte_samples - skip]
low_samples = low_samples - np.mean(low_samples)
low_idx, low_conf = demod._detect_frequency(low_samples, temp_frequencies)
print(f"  Low nibble: {low_idx} (confidence: {low_conf:.2f})")

byte2 = (high_idx << 4) | low_idx
print(f"  Byte 2 value: {byte2} (expected: 35 for 35ms)")

print(f"\nFinal decoded metadata:")
print(f"  Channels: {byte1}")
print(f"  Duration: {byte2}ms")
print(f"  Expected: 2 channels, 35ms")
