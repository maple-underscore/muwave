#!/usr/bin/env python3
"""Test script to diagnose audio issues."""

import numpy as np
import sounddevice as sd

print("Testing audio devices...")

# Query devices
print("\nAvailable audio devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['name']}")
    print(f"   Input channels: {device['max_input_channels']}")
    print(f"   Output channels: {device['max_output_channels']}")
    print(f"   Default sample rate: {device['default_samplerate']}")

# Get default devices
print(f"\nDefault input device: {sd.default.device[0]}")
print(f"Default output device: {sd.default.device[1]}")

# Test playback
print("\nGenerating test tone (440Hz for 1 second)...")
frequency = 440  # Hz
duration = 1  # seconds
sample_rate = 44100

t = np.linspace(0, duration, int(sample_rate * duration), False)
samples = 0.3 * np.sin(2 * np.pi * frequency * t)

print("Playing test tone through default output device...")
try:
    sd.play(samples, sample_rate)
    sd.wait()
    print("✓ Test tone played successfully!")
except Exception as e:
    print(f"✗ Failed to play test tone: {e}")

# Test the muwave transmitter
print("\nTesting muwave Transmitter...")
try:
    from muwave.protocol.transmitter import Transmitter
    from muwave.core.party import Party
    from muwave.audio.device import AudioDevice
    
    party = Party(name="test")
    audio_device = AudioDevice(sample_rate=44100)
    transmitter = Transmitter(
        party=party,
        config={},
        audio_device=audio_device,
    )
    
    print("Transmitting message: 'hello'")
    transmitter.transmit("hello")
    print("✓ Muwave transmission completed!")
    
except Exception as e:
    print(f"✗ Muwave transmission failed: {e}")
    import traceback
    traceback.print_exc()
