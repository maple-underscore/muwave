#!/usr/bin/env python3
"""Debug script to see what actual bytes are decoded."""

import numpy as np
from scipy.io import wavfile
from muwave.audio.fsk import FSKDemodulator
from muwave.core.config import Config

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

# Create demodulator with proper config from config.yaml
cfg = Config()
config = cfg.create_fsk_config(num_channels=1)
demod = FSKDemodulator(config)

# Detect start/end
detected, start_pos = demod.detect_start_signal(audio_data)
print(f"Start signal: {detected} at {start_pos}")

message_samples = audio_data[start_pos:]
end_detected, end_pos = demod.detect_end_signal(message_samples)
print(f"End signal: {end_detected} at {end_pos}")

message_samples = message_samples[:end_pos] if end_detected else message_samples

# Decode
data, signature, confidence = demod.decode_data(
    message_samples,
    signature_length=8,
    repetitions=2,
    read_metadata=True,
)

if data:
    print(f"\nDecoded {len(data)} bytes with {confidence:.2%} confidence")
    print(f"First 50 bytes: {data[:50]}")
    print(f"As hex: {data[:50].hex()}")
    print(f"\nFirst 50 bytes as text:")
    try:
        print(repr(data[:50].decode('utf-8', errors='replace')))
    except:
        print("(decode error)")
    
    # Decode format
    from muwave.utils.formats import FormatEncoder
    text, format_meta = FormatEncoder.decode(data)
    print(f"\nFormat: {format_meta.format_type if format_meta else 'None'}")
    print(f"Decoded text length: {len(text)}")
    print(f"First 100 chars of text:")
    print(repr(text[:100]))
else:
    print("No data decoded")
