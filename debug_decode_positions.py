#!/usr/bin/env python3
"""Debug script to analyze decode position tracking."""

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

# Get message samples
message_samples = audio_data[start_pos:]
print(f"Message samples: {len(message_samples)}")

# Decode metadata
detected_channels, detected_duration, metadata_end_pos = demod.decode_metadata(message_samples, 0)
print(f"\nMetadata decoded:")
print(f"  Channels: {detected_channels}")
print(f"  Duration: {detected_duration}ms")
print(f"  Metadata end position: {metadata_end_pos}")

# Calculate expected positions
silence_samples = int(sample_rate * config.silence_ms / 1000)
metadata_symbol_samples = int(sample_rate * 35.0 / 1000)
metadata_byte_samples = metadata_symbol_samples * 2
metadata_bytes = 2

print(f"\nExpected values:")
print(f"  Silence samples: {silence_samples}")
print(f"  Metadata symbol samples: {metadata_symbol_samples}")
print(f"  Metadata byte samples: {metadata_byte_samples}")
print(f"  Total metadata region: {silence_samples + metadata_bytes * metadata_byte_samples + silence_samples//2}")

# Now decode with the detected settings
config.num_channels = detected_channels
config.symbol_duration_ms = float(detected_duration)
demod._frequencies = demod._generate_frequencies()

# Try decoding signature
symbol_samples = int(sample_rate * detected_duration / 1000)
byte_samples = symbol_samples if detected_channels >= 2 else symbol_samples * 2
signature_length = 8

print(f"\nData decoding:")
print(f"  Symbol samples: {symbol_samples}")
print(f"  Byte samples: {byte_samples}")
print(f"  Starting position for signature: {metadata_end_pos}")
print(f"  Signature will span: {metadata_end_pos} to {metadata_end_pos + signature_length * byte_samples}")

# Skip past signature and length
signature_end = metadata_end_pos + signature_length * byte_samples
silence_samples_half = int(sample_rate * config.silence_ms / 2000)
length_end = signature_end + silence_samples_half + 2 * byte_samples

print(f"  After signature: {signature_end}")
print(f"  After half-silence: {signature_end + silence_samples_half}")
print(f"  After length bytes: {length_end}")
print(f"  Data decoding starts at: {length_end}")
