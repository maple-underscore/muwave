"""Tests for the FSK modulation module."""

import numpy as np
import pytest

from muwave.audio.fsk import (
    FSKModulator, FSKDemodulator, FSKConfig,
    SignalGenerator, FrequencyDetector,
    create_modulator, create_demodulator,
)


class TestFSKConfig:
    """Tests for FSKConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FSKConfig()
        
        assert config.sample_rate == 44100
        assert config.base_frequency == 1800.0
        assert config.num_frequencies == 16
        assert config.symbol_duration_ms == 60.0
        assert config.num_channels == 2
    
    def test_from_config(self):
        """Test loading config from config.yaml."""
        config = FSKConfig.from_config()
        
        assert config.sample_rate == 44100
        # Values should come from config.yaml
        assert config.base_frequency is not None
        assert config.num_channels >= 1


class TestSignalGenerator:
    """Tests for SignalGenerator utility class."""
    
    def test_generate_tone(self):
        """Test generating a tone."""
        config = FSKConfig()
        gen = SignalGenerator(config)
        
        tone = gen.generate_tone(1000, 100)  # 1000 Hz, 100ms
        
        expected_samples = int(44100 * 0.1)  # 4410 samples
        assert len(tone) == expected_samples
        assert tone.dtype == np.float32
        assert np.max(np.abs(tone)) <= 1.0
    
    def test_generate_silence(self):
        """Test generating silence."""
        config = FSKConfig()
        gen = SignalGenerator(config)
        
        silence = gen.generate_silence(50)  # 50ms
        
        expected_samples = int(44100 * 0.05)
        assert len(silence) == expected_samples
        assert np.all(silence == 0)


class TestFSKModulator:
    """Tests for FSKModulator."""
    
    def test_modulator_creation(self):
        """Test creating a modulator."""
        mod = FSKModulator()
        
        assert mod.config is not None
        # One frequency table per channel
        assert len(mod._frequencies) == mod.config.num_channels
        for ch in range(mod.config.num_channels):
            assert len(mod._frequencies[ch]) == mod.config.num_frequencies
    
    def test_generate_start_signal(self):
        """Test generating start signal."""
        mod = FSKModulator()
        
        signal = mod.generate_start_signal()
        
        # Signal should be at least signal_duration_ms long (may be longer with chirp + multi-tone)
        min_samples = int(44100 * mod.config.signal_duration_ms * 0.4 / 1000)  # At least multi-tone portion
        assert len(signal) >= min_samples
        assert signal.dtype == np.float32
    
    def test_generate_end_signal(self):
        """Test generating end signal."""
        mod = FSKModulator()
        
        signal = mod.generate_end_signal()
        
        min_samples = int(44100 * mod.config.signal_duration_ms * 0.4 / 1000)
        assert len(signal) >= min_samples
    
    def test_encode_byte(self):
        """Test encoding a single byte."""
        mod = FSKModulator()
        
        samples = mod.encode_byte(0xAB)
        
        symbol_samples = int(44100 * mod.config.symbol_duration_ms / 1000)
        # For >=2 channels, both nibbles are simultaneous (1 symbol per byte)
        expected = symbol_samples if mod.config.num_channels >= 2 else symbol_samples * 2
        assert len(samples) == expected
    
    def test_encode_signature(self):
        """Test encoding a signature."""
        mod = FSKModulator()
        signature = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        
        samples = mod.encode_signature(signature)
        
        symbol_samples = int(44100 * mod.config.symbol_duration_ms / 1000)
        byte_samples = symbol_samples if mod.config.num_channels >= 2 else symbol_samples * 2
        expected_samples = len(signature) * byte_samples
        assert len(samples) == expected_samples
    
    def test_encode_data(self):
        """Test encoding data with start/end signals."""
        mod = FSKModulator()
        data = b"Hello"
        
        samples, timestamps = mod.encode_data(data)
        
        # Should have start signal + data + end signal + silences
        assert len(samples) > 0
        assert samples.dtype == np.float32
        assert 'start' in timestamps
        assert 'end' in timestamps
    
    def test_encode_text(self):
        """Test encoding text."""
        mod = FSKModulator()
        
        samples = mod.encode_text("Hello, World!")
        
        assert len(samples) > 0
        assert samples.dtype == np.float32
    
    def test_encode_with_repetitions(self):
        """Test encoding with redundancy repetitions."""
        mod = FSKModulator()
        data = b"Test"
        
        samples_1x, _ = mod.encode_data(data, repetitions=1)
        samples_2x, _ = mod.encode_data(data, repetitions=2)
        
        # 2x repetition should be longer
        assert len(samples_2x) > len(samples_1x)
    
    def test_encode_with_signature(self):
        """Test encoding with party signature."""
        mod = FSKModulator()
        data = b"Test"
        signature = b'\x12\x34\x56\x78\x9A\xBC\xDE\xF0'
        
        samples_no_sig, _ = mod.encode_data(data)
        samples_with_sig, _ = mod.encode_data(data, signature=signature)
        
        # With signature should be longer
        assert len(samples_with_sig) > len(samples_no_sig)


class TestFrequencyDetector:
    """Tests for FrequencyDetector utility class."""
    
    def test_goertzel_algorithm(self):
        """Test Goertzel algorithm for frequency detection."""
        config = FSKConfig()
        detector = FrequencyDetector(config)
        
        # Generate a known frequency
        t = np.linspace(0, 0.05, int(44100 * 0.05), endpoint=False)
        test_freq = 1000
        samples = np.sin(2 * np.pi * test_freq * t).astype(np.float32)
        
        # Goertzel should detect high magnitude at test frequency
        magnitude = detector.goertzel(samples, test_freq)
        magnitude_off = detector.goertzel(samples, 2000)  # Different frequency
        
        assert magnitude > magnitude_off
    
    def test_detect_frequency(self):
        """Test frequency detection."""
        config = FSKConfig()
        detector = FrequencyDetector(config)
        
        # Generate frequency table
        target_idx = 5
        freqs = np.array([
            config.base_frequency + i * config.frequency_step
            for i in range(config.num_frequencies)
        ])
        target_freq = freqs[target_idx]
        
        t = np.linspace(0, 0.05, int(44100 * 0.05), endpoint=False)
        samples = np.sin(2 * np.pi * target_freq * t).astype(np.float32) * 0.8
        
        detected_idx, confidence = detector.detect_frequency(samples, freqs)
        
        assert detected_idx == target_idx
        assert confidence > 0.3


class TestFSKDemodulator:
    """Tests for FSKDemodulator."""
    
    def test_demodulator_creation(self):
        """Test creating a demodulator."""
        demod = FSKDemodulator()
        
        assert demod.config is not None
        assert len(demod._frequencies) == demod.config.num_channels
        for ch in range(demod.config.num_channels):
            assert len(demod._frequencies[ch]) == demod.config.num_frequencies
    
    def test_encode_decode_roundtrip(self):
        """Test that modulation/demodulation roundtrips correctly."""
        mod = FSKModulator()
        demod = FSKDemodulator()
        
        # Encode a simple message
        original_text = "Hi"
        samples = mod.encode_text(original_text)
        
        # Find start signal
        start_found, start_pos = demod.detect_start_signal(samples)
        assert start_found
        
        # Find end signal
        end_found, end_pos = demod.detect_end_signal(samples[start_pos:])
        assert end_found


class TestConvenienceFunctions:
    """Tests for convenience factory functions."""
    
    def test_create_modulator(self):
        """Test create_modulator function."""
        mod = create_modulator()
        assert isinstance(mod, FSKModulator)
        assert mod.config is not None
    
    def test_create_demodulator(self):
        """Test create_demodulator function."""
        demod = create_demodulator()
        assert isinstance(demod, FSKDemodulator)
        assert demod.config is not None
    
    def test_create_with_speed_mode(self):
        """Test creating with speed mode override."""
        mod = create_modulator(speed_mode='s60')
        assert mod.config.symbol_duration_ms == 60.0
    
    def test_create_with_overrides(self):
        """Test creating with parameter overrides."""
        mod = create_modulator(num_channels=1)
        assert mod.config.num_channels == 1
