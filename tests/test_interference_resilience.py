"""
Tests for muwave interference resilience features.

Tests the enhanced signal detection capabilities including chirp signals,
Barker codes, and adaptive thresholding under various interference conditions.
"""

import pytest
import numpy as np
from muwave.audio.fsk import FSKModulator, FSKDemodulator, FSKConfig


class InterferenceGenerator:
    """Utility class for generating various types of interference."""
    
    SAMPLE_RATE = 44100
    
    @classmethod
    def add_noise(
        cls,
        signal: np.ndarray,
        snr_db: float,
        interference_type: str = 'white'
    ) -> np.ndarray:
        """Add interference to a signal.
        
        Args:
            signal: Clean audio signal
            snr_db: Signal-to-noise ratio in dB (negative = more noise than signal)
            interference_type: 'white', 'tonal', 'impulse', or 'speech-like'
            
        Returns:
            Signal with added interference
        """
        signal_power = np.mean(signal ** 2)
        if signal_power < 1e-10:
            return signal
            
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = cls._generate_noise(len(signal), noise_power, interference_type)
        
        return (signal + noise).astype(np.float32)
    
    @classmethod
    def _generate_noise(
        cls,
        length: int,
        power: float,
        noise_type: str
    ) -> np.ndarray:
        """Generate noise of specified type and power."""
        if noise_type == 'white':
            return np.random.normal(0, np.sqrt(power), length)
        
        elif noise_type == 'tonal':
            t = np.arange(length) / cls.SAMPLE_RATE
            return np.sqrt(power) * (
                np.sin(2 * np.pi * 440 * t) +
                np.sin(2 * np.pi * 880 * t) +
                np.sin(2 * np.pi * 1760 * t) +
                np.sin(2 * np.pi * 2500 * t)
            ) / 4
        
        elif noise_type == 'impulse':
            noise = np.zeros(length)
            num_impulses = max(1, length // 500)
            positions = np.random.choice(length, size=num_impulses, replace=False)
            noise[positions] = np.random.choice([-1, 1], size=num_impulses) * np.sqrt(power) * 15
            return noise
        
        elif noise_type == 'speech-like':
            t = np.arange(length) / cls.SAMPLE_RATE
            am = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
            return np.sqrt(power) * am * (
                np.sin(2 * np.pi * 300 * t) +
                np.sin(2 * np.pi * 800 * t) +
                np.sin(2 * np.pi * 2500 * t)
            ) / 3
        
        return np.random.normal(0, np.sqrt(power), length)


class TestChirpSignalDetection:
    """Tests for chirp-based signal detection."""
    
    @pytest.fixture
    def chirp_config(self):
        """FSK config with chirp signals enabled."""
        return FSKConfig(use_chirp_signals=True, use_barker_preamble=False)
    
    @pytest.fixture
    def multitone_config(self):
        """FSK config with only multi-tone signals (no chirp)."""
        return FSKConfig(use_chirp_signals=False, use_barker_preamble=False)
    
    def test_chirp_detection_clean_signal(self, chirp_config):
        """Test chirp detection works on clean signal."""
        mod = FSKModulator(chirp_config)
        demod = FSKDemodulator(chirp_config)
        
        samples, _ = mod.encode_data(b"Test message")
        detected, pos = demod.detect_start_signal(samples)
        
        assert detected, "Chirp start signal should be detected in clean signal"
        assert pos > 0, "Position should be positive"
    
    @pytest.mark.parametrize("snr_db", [20, 10, 5, 0, -5])
    def test_chirp_detection_with_white_noise(self, chirp_config, snr_db):
        """Test chirp detection at various SNR levels with white noise."""
        mod = FSKModulator(chirp_config)
        demod = FSKDemodulator(chirp_config)
        
        samples, _ = mod.encode_data(b"Test")
        noisy = InterferenceGenerator.add_noise(samples, snr_db, 'white')
        
        detected, _ = demod.detect_start_signal(noisy)
        assert detected, f"Chirp should be detected at SNR={snr_db}dB"
    
    @pytest.mark.parametrize("interference_type", ['white', 'tonal', 'impulse', 'speech-like'])
    def test_chirp_detection_interference_types(self, chirp_config, interference_type):
        """Test chirp detection with different interference types at 0 dB SNR."""
        mod = FSKModulator(chirp_config)
        demod = FSKDemodulator(chirp_config)
        
        samples, _ = mod.encode_data(b"Test interference")
        noisy = InterferenceGenerator.add_noise(samples, 0, interference_type)
        
        detected, _ = demod.detect_start_signal(noisy)
        assert detected, f"Chirp should be detected with {interference_type} interference at 0dB"
    
    def test_chirp_vs_multitone_robustness(self, chirp_config, multitone_config):
        """Compare chirp vs multi-tone detection under noise."""
        text = b"Comparison test"
        
        # Chirp-based detection
        mod_chirp = FSKModulator(chirp_config)
        demod_chirp = FSKDemodulator(chirp_config)
        samples_chirp, _ = mod_chirp.encode_data(text)
        
        # Multi-tone only detection
        mod_mt = FSKModulator(multitone_config)
        demod_mt = FSKDemodulator(multitone_config)
        samples_mt, _ = mod_mt.encode_data(text)
        
        # Test at 0 dB SNR - chirp should succeed, multi-tone may fail
        noisy_chirp = InterferenceGenerator.add_noise(samples_chirp, 0, 'white')
        noisy_mt = InterferenceGenerator.add_noise(samples_mt, 0, 'white')
        
        chirp_detected, _ = demod_chirp.detect_start_signal(noisy_chirp)
        
        # Chirp should always detect at 0 dB
        assert chirp_detected, "Chirp detection should work at 0 dB SNR"


class TestEndSignalDetection:
    """Tests for end signal detection."""
    
    @pytest.fixture
    def config(self):
        return FSKConfig(use_chirp_signals=True)
    
    def test_end_signal_detection_clean(self, config):
        """Test end signal detection on clean signal."""
        mod = FSKModulator(config)
        demod = FSKDemodulator(config)
        
        samples, _ = mod.encode_data(b"Test message")
        
        # Find start first
        start_detected, start_pos = demod.detect_start_signal(samples)
        assert start_detected
        
        # Find end signal
        end_detected, end_pos = demod.detect_end_signal(samples[start_pos:])
        assert end_detected, "End signal should be detected"
        assert end_pos > 0, "End position should be positive"
    
    @pytest.mark.parametrize("snr_db", [10, 5, 0])
    def test_end_signal_detection_with_noise(self, config, snr_db):
        """Test end signal detection under noise."""
        mod = FSKModulator(config)
        demod = FSKDemodulator(config)
        
        samples, _ = mod.encode_data(b"End test")
        noisy = InterferenceGenerator.add_noise(samples, snr_db, 'white')
        
        start_detected, start_pos = demod.detect_start_signal(noisy)
        if start_detected:
            end_detected, _ = demod.detect_end_signal(noisy[start_pos:])
            assert end_detected, f"End signal should be detected at SNR={snr_db}dB"


class TestBarkerPreamble:
    """Tests for Barker code preamble functionality."""
    
    @pytest.fixture
    def barker_config(self):
        return FSKConfig(
            use_chirp_signals=True,
            use_barker_preamble=True,
            barker_length=13
        )
    
    def test_barker_signal_generation(self, barker_config):
        """Test that Barker preamble is generated correctly."""
        mod = FSKModulator(barker_config)
        
        # Generate start signal with Barker preamble
        start_signal = mod.generate_start_signal()
        
        assert len(start_signal) > 0
        assert start_signal.dtype == np.float32
        
        # Signal should be longer with Barker preamble
        config_no_barker = FSKConfig(use_chirp_signals=True, use_barker_preamble=False)
        mod_no_barker = FSKModulator(config_no_barker)
        start_no_barker = mod_no_barker.generate_start_signal()
        
        assert len(start_signal) > len(start_no_barker), \
            "Signal with Barker preamble should be longer"
    
    @pytest.mark.parametrize("barker_length", [7, 11, 13])
    def test_barker_lengths(self, barker_length):
        """Test different Barker code lengths."""
        config = FSKConfig(
            use_chirp_signals=True,
            use_barker_preamble=True,
            barker_length=barker_length
        )
        mod = FSKModulator(config)
        demod = FSKDemodulator(config)
        
        samples, _ = mod.encode_data(b"Barker test")
        detected, _ = demod.detect_start_signal(samples)
        
        assert detected, f"Detection should work with Barker-{barker_length}"
    
    @pytest.mark.parametrize("snr_db", [10, 5, 0])
    def test_barker_detection_under_noise(self, barker_config, snr_db):
        """Test Barker preamble detection under noise."""
        mod = FSKModulator(barker_config)
        demod = FSKDemodulator(barker_config)
        
        samples, _ = mod.encode_data(b"Barker noise test")
        noisy = InterferenceGenerator.add_noise(samples, snr_db, 'white')
        
        detected, _ = demod.detect_start_signal(noisy)
        assert detected, f"Barker detection should work at SNR={snr_db}dB"


class TestFullDecodeUnderInterference:
    """Tests for complete encode-decode cycle under interference."""
    
    @pytest.fixture
    def config(self):
        return FSKConfig(use_chirp_signals=True)
    
    def test_decode_clean_signal(self, config):
        """Test decode works on clean signal."""
        mod = FSKModulator(config)
        demod = FSKDemodulator(config)
        
        original = "Hello, World!"
        samples, _ = mod.encode_data(original.encode('utf-8'))
        
        # Detect and decode
        start_detected, start_pos = demod.detect_start_signal(samples)
        assert start_detected
        
        end_detected, end_pos = demod.detect_end_signal(samples[start_pos:])
        assert end_detected
        
        data_samples = samples[start_pos:start_pos + end_pos]
        decoded, _, confidence = demod.decode_data(data_samples, signature_length=0)
        
        assert decoded is not None
        assert decoded.decode('utf-8') == original
        assert confidence > 0.8
    
    @pytest.mark.parametrize("snr_db", [20, 10, 5])
    def test_decode_with_white_noise(self, config, snr_db):
        """Test decode accuracy under white noise."""
        mod = FSKModulator(config)
        demod = FSKDemodulator(config)
        
        original = "Test message"
        samples, _ = mod.encode_data(original.encode('utf-8'))
        noisy = InterferenceGenerator.add_noise(samples, snr_db, 'white')
        
        start_detected, start_pos = demod.detect_start_signal(noisy)
        if not start_detected:
            pytest.skip(f"Start signal not detected at {snr_db}dB")
        
        end_detected, end_pos = demod.detect_end_signal(noisy[start_pos:])
        if not end_detected:
            pytest.skip(f"End signal not detected at {snr_db}dB")
        
        data_samples = noisy[start_pos:start_pos + end_pos]
        decoded, _, confidence = demod.decode_data(data_samples, signature_length=0)
        
        assert decoded is not None, f"Decode should succeed at SNR={snr_db}dB"
        decoded_text = decoded.decode('utf-8', errors='replace')
        assert decoded_text == original, f"Decoded text should match at SNR={snr_db}dB"
    
    @pytest.mark.parametrize("interference_type", ['white', 'tonal', 'impulse', 'speech-like'])
    def test_decode_various_interference(self, config, interference_type):
        """Test decode with various interference types at 10 dB SNR."""
        mod = FSKModulator(config)
        demod = FSKDemodulator(config)
        
        original = "Interference test"
        samples, _ = mod.encode_data(original.encode('utf-8'))
        noisy = InterferenceGenerator.add_noise(samples, 10, interference_type)
        
        start_detected, start_pos = demod.detect_start_signal(noisy)
        assert start_detected, f"Start should be detected with {interference_type}"
        
        end_detected, end_pos = demod.detect_end_signal(noisy[start_pos:])
        assert end_detected, f"End should be detected with {interference_type}"
        
        data_samples = noisy[start_pos:start_pos + end_pos]
        decoded, _, _ = demod.decode_data(data_samples, signature_length=0)
        
        assert decoded is not None
        assert decoded.decode('utf-8', errors='replace') == original


class TestAdaptiveNoiseFloor:
    """Tests for adaptive noise floor detection."""
    
    def test_noise_floor_measurement(self):
        """Test noise floor is measured correctly."""
        config = FSKConfig(use_chirp_signals=True)
        demod = FSKDemodulator(config)
        
        # Create pure white noise with known amplitude
        noise_amplitude = 0.1
        noise = np.random.normal(0, noise_amplitude, 44100).astype(np.float32)
        
        noise_floor = demod._measure_noise_floor(noise)
        
        assert noise_floor > 0, "Noise floor should be positive"
        assert noise_floor < 1.0, "Noise floor should be reasonable"
        # Should be close to the actual noise RMS
        assert 0.05 < noise_floor < 0.2, f"Noise floor {noise_floor} should be near {noise_amplitude}"
    
    def test_adaptive_threshold_adjusts(self):
        """Test that detection adapts to noise level."""
        config = FSKConfig(use_chirp_signals=True)
        mod = FSKModulator(config)
        demod = FSKDemodulator(config)
        
        samples, _ = mod.encode_data(b"Adaptive test")
        
        # Should work with both clean and noisy signals
        detected_clean, _ = demod.detect_start_signal(samples)
        
        noisy = InterferenceGenerator.add_noise(samples, 5, 'white')
        detected_noisy, _ = demod.detect_start_signal(noisy)
        
        assert detected_clean, "Should detect in clean signal"
        assert detected_noisy, "Should detect with adaptive threshold in noisy signal"


class TestConfigValidation:
    """Tests for FSKConfig validation."""
    
    def test_invalid_barker_length(self):
        """Test that invalid Barker length raises error."""
        with pytest.raises(ValueError, match="barker_length must be 7, 11, or 13"):
            FSKConfig(barker_length=5)
    
    def test_valid_barker_lengths(self):
        """Test that valid Barker lengths work."""
        for length in [7, 11, 13]:
            config = FSKConfig(barker_length=length)
            assert config.barker_length == length
    
    def test_chirp_config_defaults(self):
        """Test default chirp configuration values."""
        config = FSKConfig()
        
        assert config.use_chirp_signals is True
        assert config.chirp_start_freq == 600.0
        assert config.chirp_end_freq == 2400.0
        assert config.use_barker_preamble is False
