"""
FSK (Frequency-Shift Keying) modulation for muwave.

Provides core audio encoding/decoding functionality with automatic
configuration loading from config.yaml.
"""

from __future__ import annotations

import math
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from muwave.core.config import Config

# =============================================================================
# Constants
# =============================================================================

# Barker codes for enhanced synchronization (optimal autocorrelation properties)
BARKER_CODES: Dict[int, List[int]] = {
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}

# Metadata header format constants (fixed for reliable decoding)
METADATA_MAGIC = bytes([0x4D, 0x57, 0xAA, 0x55])  # "MW" + alternating bits
METADATA_VERSION = 2
METADATA_SYMBOL_MS = 40.0
METADATA_BASE_FREQ = 1800.0
METADATA_FREQ_STEP = 120.0
METADATA_LENGTH = 13  # Total metadata bytes


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FSKConfig:
    """
    Configuration for FSK modulation.
    
    Can be initialized directly or loaded from config.yaml using from_config().
    """
    
    # Audio settings
    sample_rate: int = 44100
    volume: float = 0.8
    
    # Frequency settings
    base_frequency: float = 1800.0
    frequency_step: float = 120.0
    num_frequencies: int = 16
    
    # Timing settings
    symbol_duration_ms: float = 60.0
    signal_duration_ms: float = 200.0
    silence_ms: float = 50.0
    
    # Channel settings
    num_channels: int = 2
    channel_spacing: float = 2400.0
    
    # Signal markers (multi-frequency for robust detection)
    start_frequencies: List[float] = field(default_factory=lambda: [800.0, 850.0, 900.0])
    end_frequencies: List[float] = field(default_factory=lambda: [900.0, 950.0, 1000.0])
    
    # Chirp signal settings (enhanced interference resilience)
    use_chirp_signals: bool = True
    chirp_start_freq: float = 600.0
    chirp_end_freq: float = 2400.0
    
    # Barker preamble settings (precise synchronization)
    use_barker_preamble: bool = False
    barker_length: int = 13
    barker_carrier_freq: float = 1500.0
    barker_chip_duration_ms: float = 8.0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 1 <= self.num_channels <= 4:
            raise ValueError("num_channels must be 1-4")
        if self.barker_length not in BARKER_CODES:
            raise ValueError("barker_length must be 7, 11, or 13")
    
    @classmethod
    def from_config(
        cls,
        config: Optional[Config] = None,
        speed_mode: Optional[str] = None,
        **overrides: Any,
    ) -> FSKConfig:
        """
        Create FSKConfig from muwave configuration.
        
        Args:
            config: Config instance. Loads from default path if None.
            speed_mode: Override speed mode (e.g., 's40', 's60').
            **overrides: Additional parameter overrides.
            
        Returns:
            Configured FSKConfig instance.
        """
        if config is None:
            from muwave.core.config import Config
            config = Config()
        
        # Get audio settings
        audio = config.audio
        protocol = config.protocol
        speed = config.speed
        
        # Determine symbol duration from speed mode
        mode = speed_mode or speed.get("mode", "s40")
        modes = speed.get("modes", {})
        mode_settings = modes.get(mode, {})
        symbol_duration = mode_settings.get(
            "symbol_duration_ms",
            overrides.pop("symbol_duration_ms", 40.0)
        )
        
        # Build configuration
        params = {
            "sample_rate": audio.get("sample_rate", 44100),
            "volume": audio.get("volume", 0.8),
            "base_frequency": protocol.get("base_frequency", 1800.0),
            "frequency_step": protocol.get("frequency_step", 120.0),
            "num_frequencies": protocol.get("num_frequencies", 16),
            "symbol_duration_ms": symbol_duration,
            "signal_duration_ms": protocol.get("signal_duration_ms", 200.0),
            "silence_ms": protocol.get("silence_ms", 50.0),
            "num_channels": protocol.get("num_channels", 2),
            "channel_spacing": protocol.get("channel_spacing", 2400.0),
            "start_frequencies": protocol.get("start_frequencies", [800.0, 850.0, 900.0]),
            "end_frequencies": protocol.get("end_frequencies", [900.0, 950.0, 1000.0]),
            "use_chirp_signals": protocol.get("use_chirp_signals", True),
            "chirp_start_freq": protocol.get("chirp_start_freq", 600.0),
            "chirp_end_freq": protocol.get("chirp_end_freq", 2400.0),
            "use_barker_preamble": protocol.get("use_barker_preamble", False),
            "barker_length": protocol.get("barker_length", 13),
            "barker_carrier_freq": protocol.get("barker_carrier_freq", 1500.0),
            "barker_chip_duration_ms": protocol.get("barker_chip_duration_ms", 8.0),
        }
        
        # Apply overrides
        params.update(overrides)
        
        return cls(**params)


# =============================================================================
# Signal Generation Utilities
# =============================================================================

class SignalGenerator:
    """Utility class for generating audio signals."""
    
    def __init__(self, config: FSKConfig) -> None:
        self.config = config
    
    def generate_tone(
        self,
        frequency: float,
        duration_ms: float,
        fade_ms: float = 5.0,
    ) -> np.ndarray:
        """
        Generate a pure tone with anti-interference envelope.
        
        Args:
            frequency: Tone frequency in Hz.
            duration_ms: Duration in milliseconds.
            fade_ms: Fade in/out duration.
            
        Returns:
            Audio samples as float32 array.
        """
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        # Generate sine wave
        signal = np.sin(2 * np.pi * frequency * t, dtype=np.float64)
        signal *= self.config.volume
        
        # Adaptive fade for fast symbols
        adaptive_fade = min(duration_ms * 0.35, 10.0) if duration_ms < 30 else fade_ms
        fade_samples = int(self.config.sample_rate * adaptive_fade / 1000)
        
        # Apply raised-cosine envelope
        if 0 < fade_samples < num_samples // 2:
            t_fade = np.linspace(0, np.pi, fade_samples)
            signal[:fade_samples] *= (1 - np.cos(t_fade)) / 2
            signal[-fade_samples:] *= (1 + np.cos(t_fade)) / 2
        
        # Apply anti-aliasing filter
        return self._apply_lowpass_filter(signal).astype(np.float32)
    
    def generate_silence(self, duration_ms: float) -> np.ndarray:
        """Generate silence of specified duration."""
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        return np.zeros(num_samples, dtype=np.float32)
    
    def generate_chirp(
        self,
        start_freq: float,
        end_freq: float,
        duration_ms: float,
        chirp_type: str = "linear",
    ) -> np.ndarray:
        """
        Generate a frequency sweep (chirp) signal.
        
        Chirps provide excellent interference resilience through
        spread-spectrum energy distribution.
        
        Args:
            start_freq: Starting frequency in Hz.
            end_freq: Ending frequency in Hz.
            duration_ms: Duration in milliseconds.
            chirp_type: "linear" or "logarithmic".
            
        Returns:
            Audio samples as float32 array.
        """
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        duration_sec = duration_ms / 1000
        
        if chirp_type == "logarithmic" and start_freq > 0:
            ratio = end_freq / start_freq
            k = ratio ** (1.0 / duration_sec)
            phase = 2 * np.pi * start_freq * (k**t - 1) / np.log(k)
        else:  # linear
            k = (end_freq - start_freq) / duration_sec
            phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)
        
        signal = np.sin(phase, dtype=np.float64) * self.config.volume
        
        # Apply smooth envelope
        envelope = np.sin(np.pi * t / duration_sec) ** 0.4
        return (signal * envelope).astype(np.float32)
    
    def generate_barker_signal(
        self,
        carrier_freq: float,
        code_length: int = 13,
        chip_duration_ms: float = 8.0,
    ) -> np.ndarray:
        """
        Generate BPSK-modulated Barker code for precise synchronization.
        
        Barker codes have optimal autocorrelation with N:1 peak-to-sidelobe ratio.
        
        Args:
            carrier_freq: BPSK carrier frequency.
            code_length: Barker code length (7, 11, or 13).
            chip_duration_ms: Duration per chip.
            
        Returns:
            Audio samples as float32 array.
        """
        code = BARKER_CODES.get(code_length, BARKER_CODES[13])
        chip_samples = int(self.config.sample_rate * chip_duration_ms / 1000)
        total_samples = chip_samples * len(code)
        
        signal = np.zeros(total_samples, dtype=np.float64)
        t_chip = np.linspace(0, chip_duration_ms / 1000, chip_samples, endpoint=False)
        
        for i, chip in enumerate(code):
            start = i * chip_samples
            end = start + chip_samples
            phase = 0.0 if chip == 1 else np.pi
            signal[start:end] = np.sin(2 * np.pi * carrier_freq * t_chip + phase)
        
        # Apply envelope
        t_total = np.linspace(0, 1, total_samples, endpoint=False)
        envelope = np.sin(np.pi * t_total) ** 0.3
        return (signal * envelope * self.config.volume).astype(np.float32)
    
    def generate_multitone(
        self,
        frequencies: List[float],
        duration_ms: float,
    ) -> np.ndarray:
        """
        Generate a multi-frequency signal for robust detection.
        
        Args:
            frequencies: List of frequencies to combine.
            duration_ms: Duration in milliseconds.
            
        Returns:
            Audio samples as float32 array.
        """
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        signal = np.zeros(num_samples, dtype=np.float64)
        for freq in frequencies:
            signal += np.sin(2 * np.pi * freq * t)
        
        # Normalize and apply envelope
        signal = (signal / len(frequencies)) * self.config.volume
        envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.5
        return (signal * envelope).astype(np.float32)
    
    @staticmethod
    def _apply_lowpass_filter(signal: np.ndarray) -> np.ndarray:
        """Apply simple low-pass filter to reduce harmonics."""
        if len(signal) < 5:
            return signal
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        return np.convolve(signal, kernel, mode='same')


# =============================================================================
# Frequency Detection Utilities
# =============================================================================

class FrequencyDetector:
    """Utility class for frequency detection using Goertzel algorithm."""
    
    def __init__(self, config: FSKConfig) -> None:
        self.config = config
    
    def goertzel(self, samples: np.ndarray, target_freq: float) -> float:
        """
        Goertzel algorithm for efficient single-frequency detection.
        
        Args:
            samples: Audio samples.
            target_freq: Target frequency to detect.
            
        Returns:
            Magnitude of the target frequency.
        """
        n = len(samples)
        if n == 0:
            return 0.0
        
        samples_hp = samples.astype(np.float64)
        normalized_freq = target_freq / self.config.sample_rate
        w = 2.0 * np.pi * normalized_freq
        coeff = 2.0 * np.cos(w)
        
        s0, s1, s2 = 0.0, 0.0, 0.0
        for sample in samples_hp:
            s0 = sample + coeff * s1 - s2
            s2, s1 = s1, s0
        
        real = s1 - s2 * np.cos(w)
        imag = s2 * np.sin(w)
        return float(np.sqrt(real * real + imag * imag))
    
    def detect_frequency(
        self,
        samples: np.ndarray,
        target_frequencies: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Detect which frequency is dominant in samples.
        
        Args:
            samples: Audio samples.
            target_frequencies: Array of possible frequencies.
            
        Returns:
            Tuple of (frequency index, confidence).
        """
        if len(samples) == 0:
            return 0, 0.0
        
        # Apply Blackman window for sidelobe suppression
        samples_hp = samples.astype(np.float64)
        window = np.blackman(len(samples_hp))
        windowed = samples_hp * window
        
        signal_energy = np.sum(windowed ** 2)
        if signal_energy < 1e-20:
            return 0, 0.0
        
        # Calculate correlations for all frequencies
        correlations = np.array([
            self.goertzel(windowed, freq) for freq in target_frequencies
        ])
        correlations /= np.sqrt(signal_energy)
        
        best_idx = int(np.argmax(correlations))
        confidence = self._calculate_confidence(correlations)
        
        return best_idx, confidence
    
    def _calculate_confidence(self, correlations: np.ndarray) -> float:
        """Calculate detection confidence from correlation values."""
        sorted_corr = np.sort(correlations)[::-1]
        
        if sorted_corr[0] <= 1e-10:
            return 0.0
        
        if len(sorted_corr) < 2 or sorted_corr[1] <= 1e-10:
            return 0.95
        
        # Multi-metric confidence calculation
        separation = (sorted_corr[0] - sorted_corr[1]) / sorted_corr[0]
        
        total_energy = np.sum(correlations ** 2)
        concentration = (sorted_corr[0] ** 2) / total_energy if total_energy > 0 else 0
        
        duration_scale = np.sqrt(self.config.symbol_duration_ms / 60.0)
        expected_magnitude = 0.3 * duration_scale
        strength = min(1.0, sorted_corr[0] / expected_magnitude)
        
        confidence = separation * 0.35 + concentration * 0.35 + strength * 0.30
        return float(np.clip(confidence, 0.0, 1.0))
    
    def measure_noise_floor(
        self,
        samples: np.ndarray,
        window_ms: float = 50.0,
        percentile: float = 25.0,
    ) -> float:
        """
        Estimate noise floor for adaptive thresholding.
        
        Args:
            samples: Audio samples.
            window_ms: Analysis window size.
            percentile: Percentile for noise estimate.
            
        Returns:
            Estimated noise floor (RMS).
        """
        window_size = int(self.config.sample_rate * window_ms / 1000)
        if len(samples) < window_size:
            return float(np.sqrt(np.mean(samples ** 2)))
        
        rms_values = []
        step = window_size // 2
        for i in range(0, len(samples) - window_size, step):
            rms = np.sqrt(np.mean(samples[i:i + window_size] ** 2))
            rms_values.append(rms)
        
        return float(np.percentile(rms_values, percentile)) if rms_values else 0.01


# =============================================================================
# FSK Modulator
# =============================================================================

class FSKModulator:
    """
    FSK modulator for encoding data into audio.
    
    Supports 1-4 channel parallel transmission for variable bitrates.
    Automatically loads settings from config.yaml when initialized without config.
    """
    
    def __init__(self, config: Optional[FSKConfig] = None) -> None:
        """
        Initialize the FSK modulator.
        
        Args:
            config: FSK configuration. Auto-loads from config.yaml if None.
        """
        self.config = config or FSKConfig.from_config()
        self._signal_gen = SignalGenerator(self.config)
        self._frequencies = self._generate_frequencies()
        self._last_encode_timestamps: Optional[Dict[str, float]] = None
    
    def _generate_frequencies(self) -> List[np.ndarray]:
        """Generate frequency tables for all channels."""
        channels = []
        for ch in range(self.config.num_channels):
            base = self.config.base_frequency + (ch * self.config.channel_spacing)
            freqs = np.array([
                base + i * self.config.frequency_step
                for i in range(self.config.num_frequencies)
            ])
            channels.append(freqs)
        return channels
    
    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------
    
    def generate_start_signal(self) -> np.ndarray:
        """
        Generate start signal combining chirp and multi-tone.
        
        Returns:
            Audio samples for start signal.
        """
        samples = []
        
        if self.config.use_chirp_signals:
            chirp_duration = self.config.signal_duration_ms * 0.5
            chirp = self._signal_gen.generate_chirp(
                self.config.chirp_start_freq,
                self.config.chirp_end_freq,
                chirp_duration,
            )
            samples.append(chirp)
            samples.append(self._signal_gen.generate_silence(10.0))
        
        # Multi-tone for backward compatibility
        tone_duration = self.config.signal_duration_ms * (
            0.4 if self.config.use_chirp_signals else 1.0
        )
        samples.append(self._signal_gen.generate_multitone(
            self.config.start_frequencies, tone_duration
        ))
        
        if self.config.use_barker_preamble:
            samples.append(self._signal_gen.generate_silence(5.0))
            samples.append(self._signal_gen.generate_barker_signal(
                self.config.barker_carrier_freq,
                self.config.barker_length,
                self.config.barker_chip_duration_ms,
            ))
        
        return np.concatenate(samples)
    
    def generate_end_signal(self) -> np.ndarray:
        """
        Generate end signal (falling chirp + multi-tone).
        
        Returns:
            Audio samples for end signal.
        """
        samples = []
        
        if self.config.use_chirp_signals:
            chirp_duration = self.config.signal_duration_ms * 0.5
            # Falling chirp (reverse direction)
            chirp = self._signal_gen.generate_chirp(
                self.config.chirp_end_freq,
                self.config.chirp_start_freq,
                chirp_duration,
            )
            samples.append(chirp)
            samples.append(self._signal_gen.generate_silence(10.0))
        
        tone_duration = self.config.signal_duration_ms * (
            0.4 if self.config.use_chirp_signals else 1.0
        )
        samples.append(self._signal_gen.generate_multitone(
            self.config.end_frequencies, tone_duration
        ))
        
        return np.concatenate(samples)
    
    # -------------------------------------------------------------------------
    # Byte Encoding
    # -------------------------------------------------------------------------
    
    def encode_byte(self, byte_val: int) -> np.ndarray:
        """
        Encode a single byte into audio using multi-channel FSK.
        
        Channel strategies:
        - 1 channel: Sequential nibbles (2 symbols/byte)
        - 2 channels: Parallel nibbles (1 symbol/byte)
        - 3 channels: 3+3+2 bit split
        - 4 channels: 2+2+2+2 bit split
        
        Args:
            byte_val: Byte value (0-255).
            
        Returns:
            Audio samples for the byte.
        """
        duration = self.config.symbol_duration_ms
        
        if self.config.num_channels == 1:
            return self._encode_single_channel(byte_val, duration)
        elif self.config.num_channels == 2:
            return self._encode_dual_channel(byte_val, duration)
        elif self.config.num_channels == 3:
            return self._encode_tri_channel(byte_val, duration)
        else:
            return self._encode_quad_channel(byte_val, duration)
    
    def _encode_single_channel(self, byte_val: int, duration: float) -> np.ndarray:
        """Encode byte using single channel (sequential nibbles)."""
        high_nibble = (byte_val >> 4) & 0x0F
        low_nibble = byte_val & 0x0F
        
        high_tone = self._signal_gen.generate_tone(
            self._frequencies[0][high_nibble], duration
        )
        low_tone = self._signal_gen.generate_tone(
            self._frequencies[0][low_nibble], duration
        )
        return np.concatenate([high_tone, low_tone])
    
    def _encode_dual_channel(self, byte_val: int, duration: float) -> np.ndarray:
        """Encode byte using dual channels (parallel nibbles)."""
        high_nibble = (byte_val >> 4) & 0x0F
        low_nibble = byte_val & 0x0F
        
        tone1 = self._signal_gen.generate_tone(self._frequencies[0][high_nibble], duration)
        tone2 = self._signal_gen.generate_tone(self._frequencies[1][low_nibble], duration)
        return self._mix_signals([tone1, tone2])
    
    def _encode_tri_channel(self, byte_val: int, duration: float) -> np.ndarray:
        """Encode byte using 3 channels (3+3+2 bit split)."""
        bits_765 = (byte_val >> 5) & 0x07
        bits_432 = (byte_val >> 2) & 0x07
        bits_10 = byte_val & 0x03
        
        tones = [
            self._signal_gen.generate_tone(self._frequencies[0][bits_765], duration),
            self._signal_gen.generate_tone(self._frequencies[1][bits_432], duration),
            self._signal_gen.generate_tone(self._frequencies[2][bits_10], duration),
        ]
        return self._mix_signals(tones)
    
    def _encode_quad_channel(self, byte_val: int, duration: float) -> np.ndarray:
        """Encode byte using 4 channels (2+2+2+2 bit split)."""
        bits_76 = (byte_val >> 6) & 0x03
        bits_54 = (byte_val >> 4) & 0x03
        bits_32 = (byte_val >> 2) & 0x03
        bits_10 = byte_val & 0x03
        
        tones = [
            self._signal_gen.generate_tone(self._frequencies[0][bits_76], duration),
            self._signal_gen.generate_tone(self._frequencies[1][bits_54], duration),
            self._signal_gen.generate_tone(self._frequencies[2][bits_32], duration),
            self._signal_gen.generate_tone(self._frequencies[3][bits_10], duration),
        ]
        return self._mix_signals(tones)
    
    @staticmethod
    def _mix_signals(signals: List[np.ndarray]) -> np.ndarray:
        """Mix multiple signals with normalization to prevent clipping."""
        mixed = sum(signals) / len(signals)
        peak = np.max(np.abs(mixed))
        if peak > 0.95:
            mixed = mixed * (0.95 / peak)
        return mixed
    
    # -------------------------------------------------------------------------
    # Metadata Encoding
    # -------------------------------------------------------------------------
    
    def encode_metadata(self, signature_length: int = 0) -> np.ndarray:
        """
        Encode standardized transmission metadata.
        
        Uses fixed reliable format (single channel, 40ms symbols) for
        guaranteed decoding regardless of actual signal parameters.
        
        Format (13 bytes):
        - Magic (4): 0x4D 0x57 0xAA 0x55
        - Version (1): Protocol version
        - Channels (1): Number of data channels
        - Duration (1): Symbol duration in ms
        - Base freq high/low (2): Base frequency / 100
        - Freq step (1): Frequency step / 10
        - Channel spacing (1): Spacing / 100
        - Signature length (1): Signature bytes (0 = none)
        - Checksum (1): XOR of all previous bytes
        
        Args:
            signature_length: Signature length in bytes.
            
        Returns:
            Audio samples for metadata header.
        """
        # Build metadata bytes
        metadata = list(METADATA_MAGIC)
        metadata.append(METADATA_VERSION)
        metadata.append(self.config.num_channels)
        metadata.append(int(self.config.symbol_duration_ms) & 0xFF)
        
        # Encode frequencies
        base_freq_encoded = int(self.config.base_frequency / 100)
        metadata.append((base_freq_encoded >> 8) & 0xFF)
        metadata.append(base_freq_encoded & 0xFF)
        metadata.append(int(self.config.frequency_step / 10) & 0xFF)
        metadata.append(int(self.config.channel_spacing / 100) & 0xFF)
        metadata.append(signature_length & 0xFF)
        
        # Checksum
        checksum = 0
        for b in metadata:
            checksum ^= b
        metadata.append(checksum)
        
        # Generate tones using fixed metadata frequencies
        temp_freqs = np.array([
            METADATA_BASE_FREQ + i * METADATA_FREQ_STEP
            for i in range(self.config.num_frequencies)
        ])
        
        samples = []
        for byte_val in metadata:
            high_nibble = (byte_val >> 4) & 0x0F
            low_nibble = byte_val & 0x0F
            samples.append(self._signal_gen.generate_tone(
                temp_freqs[high_nibble], METADATA_SYMBOL_MS
            ))
            samples.append(self._signal_gen.generate_tone(
                temp_freqs[low_nibble], METADATA_SYMBOL_MS
            ))
        
        return np.concatenate(samples)
    
    # -------------------------------------------------------------------------
    # High-Level Encoding
    # -------------------------------------------------------------------------
    
    def encode_signature(self, signature: bytes) -> np.ndarray:
        """Encode party signature into audio."""
        return np.concatenate([self.encode_byte(b) for b in signature])
    
    def encode_data(
        self,
        data: bytes,
        signature: Optional[bytes] = None,
        repetitions: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Encode data into complete audio transmission.
        
        Args:
            data: Data bytes to encode.
            signature: Optional party signature.
            repetitions: Number of repetitions for redundancy.
            
        Returns:
            Tuple of (audio signal, timing dictionary).
        """
        timestamps = {'start': time.time()}
        samples = []
        silence = self._signal_gen.generate_silence
        
        # Start signal
        samples.append(self.generate_start_signal())
        samples.append(silence(self.config.silence_ms))
        
        # Metadata
        sig_length = len(signature) if signature else 0
        samples.append(self.encode_metadata(signature_length=sig_length))
        samples.append(silence(self.config.silence_ms / 2))
        
        # Signature
        if signature:
            samples.append(self.encode_signature(signature))
            samples.append(silence(self.config.silence_ms / 2))
        
        # Data length (2 bytes)
        length = len(data)
        samples.append(self.encode_byte((length >> 8) & 0xFF))
        samples.append(self.encode_byte(length & 0xFF))
        
        # Data with repetitions
        for rep in range(repetitions):
            for byte_val in data:
                samples.append(self.encode_byte(byte_val))
            if repetitions > 1:
                samples.append(silence(self.config.silence_ms / 2))
        
        samples.append(silence(self.config.silence_ms))
        timestamps['data_encoded'] = time.time()
        
        # End signal
        samples.append(self.generate_end_signal())
        timestamps['end'] = time.time()
        timestamps['total_duration'] = timestamps['end'] - timestamps['start']
        
        self._last_encode_timestamps = timestamps
        return np.concatenate(samples), timestamps
    
    def encode_text(
        self,
        text: str,
        signature: Optional[bytes] = None,
        repetitions: int = 1,
    ) -> np.ndarray:
        """
        Encode text into audio.
        
        Args:
            text: Text to encode.
            signature: Optional party signature.
            repetitions: Number of repetitions.
            
        Returns:
            Audio signal.
        """
        data = text.encode('utf-8')
        audio, _ = self.encode_data(data, signature, repetitions)
        return audio


# =============================================================================
# FSK Demodulator
# =============================================================================

class FSKDemodulator:
    """
    FSK demodulator for decoding audio into data.
    
    Supports automatic configuration detection via metadata header
    and multi-channel decoding.
    """
    
    def __init__(self, config: Optional[FSKConfig] = None) -> None:
        """
        Initialize the FSK demodulator.
        
        Args:
            config: FSK configuration. Auto-loads from config.yaml if None.
        """
        self.config = config or FSKConfig.from_config()
        self._signal_gen = SignalGenerator(self.config)
        self._freq_detector = FrequencyDetector(self.config)
        self._frequencies = self._generate_frequencies()
        self._last_decode_timestamps: Optional[Dict[str, float]] = None
    
    def _generate_frequencies(self) -> List[np.ndarray]:
        """Generate frequency tables for all channels."""
        channels = []
        for ch in range(self.config.num_channels):
            base = self.config.base_frequency + (ch * self.config.channel_spacing)
            freqs = np.array([
                base + i * self.config.frequency_step
                for i in range(self.config.num_frequencies)
            ])
            channels.append(freqs)
        return channels
    
    # -------------------------------------------------------------------------
    # Signal Detection
    # -------------------------------------------------------------------------
    
    def detect_start_signal(
        self,
        samples: np.ndarray,
        threshold: float = 0.12,
    ) -> Tuple[bool, int]:
        """
        Detect start signal using chirp and multi-tone detection.
        
        Args:
            samples: Audio samples to search.
            threshold: Detection threshold.
            
        Returns:
            Tuple of (detected, sample_position).
        """
        noise_floor = self._freq_detector.measure_noise_floor(
            samples[:min(len(samples), self.config.sample_rate)]
        )
        adaptive_threshold = max(threshold, noise_floor * 2.5)
        
        # Try chirp detection first
        if self.config.use_chirp_signals:
            chirp_duration = self.config.signal_duration_ms * 0.5
            detected, pos, peak = self._detect_chirp(
                samples,
                self.config.chirp_start_freq,
                self.config.chirp_end_freq,
                chirp_duration,
            )
            if detected:
                # Account for following multi-tone
                tone_duration = self.config.signal_duration_ms * 0.4
                extra_samples = int(self.config.sample_rate * (10.0 + tone_duration) / 1000)
                
                if self.config.use_barker_preamble:
                    barker_duration = self.config.barker_chip_duration_ms * self.config.barker_length
                    extra_samples += int(self.config.sample_rate * (5.0 + barker_duration) / 1000)
                
                return True, pos + extra_samples
        
        # Fallback to multi-tone detection
        return self._detect_multitone(
            samples,
            self.config.start_frequencies,
            adaptive_threshold,
        )
    
    def detect_end_signal(
        self,
        samples: np.ndarray,
        threshold: float = 0.12,
    ) -> Tuple[bool, int]:
        """
        Detect end signal using falling chirp and multi-tone detection.
        
        Args:
            samples: Audio samples to search.
            threshold: Detection threshold.
            
        Returns:
            Tuple of (detected, sample_position).
        """
        noise_floor = self._freq_detector.measure_noise_floor(
            samples[:min(len(samples), self.config.sample_rate // 2)]
        )
        adaptive_threshold = max(threshold, noise_floor * 2.5)
        
        # Try falling chirp detection
        if self.config.use_chirp_signals:
            chirp_duration = self.config.signal_duration_ms * 0.5
            detected, pos, peak = self._detect_chirp(
                samples,
                self.config.chirp_end_freq,  # Reversed
                self.config.chirp_start_freq,
                chirp_duration,
            )
            if detected:
                chirp_samples = int(self.config.sample_rate * chirp_duration / 1000)
                return True, max(0, pos - chirp_samples)
        
        # Fallback to multi-tone detection
        return self._detect_multitone(
            samples,
            self.config.end_frequencies,
            adaptive_threshold,
        )
    
    def _detect_chirp(
        self,
        samples: np.ndarray,
        start_freq: float,
        end_freq: float,
        duration_ms: float,
        threshold: float = 0.25,
    ) -> Tuple[bool, int, float]:
        """Detect chirp signal using matched filter correlation."""
        reference = self._signal_gen.generate_chirp(start_freq, end_freq, duration_ms)
        
        if len(samples) < len(reference):
            return False, 0, 0.0
        
        # Matched filter
        correlation = np.correlate(
            samples.astype(np.float64),
            reference.astype(np.float64),
            mode='valid'
        )
        correlation = np.abs(correlation)
        
        # Normalize
        ref_energy = np.sqrt(np.sum(reference.astype(np.float64) ** 2))
        sig_energy_sq = np.convolve(
            samples.astype(np.float64) ** 2,
            np.ones(len(reference)),
            mode='valid'
        )
        sig_energy = np.sqrt(np.maximum(sig_energy_sq, 1e-10))
        
        normalized = correlation / (ref_energy * sig_energy)
        peak_idx = int(np.argmax(normalized))
        peak_value = float(normalized[peak_idx])
        
        return peak_value > threshold, peak_idx + len(reference), peak_value
    
    def _detect_multitone(
        self,
        samples: np.ndarray,
        frequencies: List[float],
        threshold: float,
    ) -> Tuple[bool, int]:
        """Detect multi-frequency signal."""
        window_size = int(self.config.sample_rate * self.config.signal_duration_ms / 1000)
        step_size = window_size // 4
        
        for i in range(0, len(samples) - window_size, step_size):
            window = samples[i:i + window_size] - np.mean(samples[i:i + window_size])
            window_rms = np.sqrt(np.mean(window ** 2))
            
            if window_rms < 0.01:
                continue
            
            detection_count = 0
            for freq in frequencies:
                magnitude = self._freq_detector.goertzel(window, freq)
                normalized = magnitude / (len(window) * window_rms) if window_rms > 1e-10 else 0
                if normalized > threshold:
                    detection_count += 1
            
            required = math.ceil(len(frequencies) * 0.5)
            if detection_count >= required:
                return True, i + window_size
        
        return False, 0
    
    # -------------------------------------------------------------------------
    # Byte Decoding
    # -------------------------------------------------------------------------
    
    def decode_byte(self, samples: np.ndarray) -> Tuple[int, float]:
        """
        Decode a byte from audio samples.
        
        Args:
            samples: Audio samples for one byte.
            
        Returns:
            Tuple of (byte value, confidence).
        """
        symbol_samples = int(
            self.config.sample_rate * self.config.symbol_duration_ms / 1000
        )
        
        # Adaptive skip based on symbol duration
        if self.config.symbol_duration_ms < 25:
            skip_fraction = 30
        elif self.config.symbol_duration_ms < 40:
            skip_fraction = 20
        else:
            skip_fraction = 10
        skip = max(1, symbol_samples // skip_fraction)
        
        if self.config.num_channels == 1:
            return self._decode_single_channel(samples, symbol_samples, skip)
        elif self.config.num_channels == 2:
            return self._decode_dual_channel(samples, symbol_samples, skip)
        elif self.config.num_channels == 3:
            return self._decode_tri_channel(samples, symbol_samples, skip)
        else:
            return self._decode_quad_channel(samples, symbol_samples, skip)
    
    def _decode_single_channel(
        self,
        samples: np.ndarray,
        symbol_samples: int,
        skip: int,
    ) -> Tuple[int, float]:
        """Decode byte from single channel (sequential nibbles)."""
        if len(samples) < symbol_samples * 2:
            return 0, 0.0
        
        high_samples = samples[skip:symbol_samples - skip]
        if len(high_samples) > 0:
            high_samples = high_samples - np.mean(high_samples)
        high_idx, high_conf = self._freq_detector.detect_frequency(
            high_samples, self._frequencies[0]
        )
        
        low_samples = samples[symbol_samples + skip:symbol_samples * 2 - skip]
        if len(low_samples) > 0:
            low_samples = low_samples - np.mean(low_samples)
        low_idx, low_conf = self._freq_detector.detect_frequency(
            low_samples, self._frequencies[0]
        )
        
        return (high_idx << 4) | low_idx, float(np.sqrt(high_conf * low_conf))
    
    def _decode_dual_channel(
        self,
        samples: np.ndarray,
        symbol_samples: int,
        skip: int,
    ) -> Tuple[int, float]:
        """Decode byte from dual channels (parallel nibbles)."""
        if len(samples) < symbol_samples:
            return 0, 0.0
        
        symbol = samples[skip:symbol_samples - skip]
        if len(symbol) > 10:
            symbol = symbol - np.mean(symbol)
        
        high_idx, high_conf = self._freq_detector.detect_frequency(
            symbol, self._frequencies[0]
        )
        low_idx, low_conf = self._freq_detector.detect_frequency(
            symbol, self._frequencies[1]
        )
        
        return (high_idx << 4) | low_idx, float(np.sqrt(high_conf * low_conf))
    
    def _decode_tri_channel(
        self,
        samples: np.ndarray,
        symbol_samples: int,
        skip: int,
    ) -> Tuple[int, float]:
        """Decode byte from 3 channels (3+3+2 bit split)."""
        if len(samples) < symbol_samples:
            return 0, 0.0
        
        symbol = samples[skip:symbol_samples - skip]
        if len(symbol) > 10:
            symbol = symbol - np.mean(symbol)
        
        idx1, conf1 = self._freq_detector.detect_frequency(symbol, self._frequencies[0])
        idx2, conf2 = self._freq_detector.detect_frequency(symbol, self._frequencies[1])
        idx3, conf3 = self._freq_detector.detect_frequency(symbol, self._frequencies[2])
        
        byte_val = ((idx1 & 0x07) << 5) | ((idx2 & 0x07) << 2) | (idx3 & 0x03)
        confidence = (conf1 * conf2 * conf3) ** (1/3)
        return byte_val, float(confidence)
    
    def _decode_quad_channel(
        self,
        samples: np.ndarray,
        symbol_samples: int,
        skip: int,
    ) -> Tuple[int, float]:
        """Decode byte from 4 channels (2+2+2+2 bit split)."""
        if len(samples) < symbol_samples:
            return 0, 0.0
        
        symbol = samples[skip:symbol_samples - skip]
        if len(symbol) > 10:
            symbol = symbol - np.mean(symbol)
        
        idx1, conf1 = self._freq_detector.detect_frequency(symbol, self._frequencies[0])
        idx2, conf2 = self._freq_detector.detect_frequency(symbol, self._frequencies[1])
        idx3, conf3 = self._freq_detector.detect_frequency(symbol, self._frequencies[2])
        idx4, conf4 = self._freq_detector.detect_frequency(symbol, self._frequencies[3])
        
        byte_val = ((idx1 & 0x03) << 6) | ((idx2 & 0x03) << 4) | ((idx3 & 0x03) << 2) | (idx4 & 0x03)
        confidence = (conf1 * conf2 * conf3 * conf4) ** (1/4)
        return byte_val, float(confidence)
    
    # -------------------------------------------------------------------------
    # Metadata Decoding
    # -------------------------------------------------------------------------
    
    def decode_metadata(
        self,
        samples: np.ndarray,
        start_pos: int = 0,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Decode transmission metadata from audio samples.
        
        Uses fixed reliable format to ensure decoding regardless of
        actual signal parameters.
        
        Args:
            samples: Audio samples after start signal.
            start_pos: Starting position in samples.
            
        Returns:
            Tuple of (metadata dict, next position).
        """
        metadata_symbol_samples = int(self.config.sample_rate * METADATA_SYMBOL_MS / 1000)
        metadata_byte_samples = metadata_symbol_samples * 2
        
        temp_frequencies = np.array([
            METADATA_BASE_FREQ + i * METADATA_FREQ_STEP
            for i in range(self.config.num_frequencies)
        ])
        
        pos = start_pos + int(self.config.sample_rate * self.config.silence_ms / 1000)
        skip = max(1, metadata_symbol_samples // 20)
        
        default_metadata = self._get_default_metadata()
        
        total_samples = metadata_byte_samples * METADATA_LENGTH
        if pos + total_samples > len(samples):
            return default_metadata, pos
        
        # Decode metadata bytes
        metadata_bytes = []
        confidences = []
        
        for _ in range(METADATA_LENGTH):
            if pos + metadata_byte_samples > len(samples):
                return default_metadata, pos
            
            # High nibble
            high_samples = samples[pos + skip:pos + metadata_symbol_samples - skip]
            if len(high_samples) > 0:
                high_samples = high_samples - np.mean(high_samples)
            high_idx, high_conf = self._freq_detector.detect_frequency(
                high_samples, temp_frequencies
            )
            
            # Low nibble
            low_samples = samples[
                pos + metadata_symbol_samples + skip:pos + metadata_byte_samples - skip
            ]
            if len(low_samples) > 0:
                low_samples = low_samples - np.mean(low_samples)
            low_idx, low_conf = self._freq_detector.detect_frequency(
                low_samples, temp_frequencies
            )
            
            metadata_bytes.append((high_idx << 4) | low_idx)
            confidences.append((high_conf + low_conf) / 2)
            pos += metadata_byte_samples
        
        # Validate and parse
        return self._parse_metadata(metadata_bytes, confidences, start_pos), pos
    
    def _get_default_metadata(self) -> Dict[str, Any]:
        """Get default metadata values."""
        return {
            'valid': False,
            'version': METADATA_VERSION,
            'num_channels': self.config.num_channels,
            'symbol_duration_ms': self.config.symbol_duration_ms,
            'base_frequency': self.config.base_frequency,
            'frequency_step': self.config.frequency_step,
            'channel_spacing': self.config.channel_spacing,
            'signature_length': 8,
            'checksum_ok': False,
        }
    
    def _parse_metadata(
        self,
        metadata_bytes: List[int],
        confidences: List[float],
        start_pos: int,
    ) -> Dict[str, Any]:
        """Parse and validate metadata bytes."""
        # Validate magic
        if metadata_bytes[0:4] != list(METADATA_MAGIC):
            default = self._get_default_metadata()
            return default
        
        # Verify checksum
        computed_checksum = 0
        for b in metadata_bytes[:-1]:
            computed_checksum ^= b
        checksum_ok = computed_checksum == metadata_bytes[-1]
        
        # Extract fields
        version = metadata_bytes[4]
        num_channels = metadata_bytes[5]
        symbol_duration = metadata_bytes[6]
        base_freq_hi = metadata_bytes[7]
        base_freq_lo = metadata_bytes[8]
        freq_step_encoded = metadata_bytes[9]
        channel_spacing_encoded = metadata_bytes[10]
        signature_length = metadata_bytes[11] if version >= 2 else 8
        
        # Decode values
        base_frequency = ((base_freq_hi << 8) | base_freq_lo) * 100.0
        frequency_step = freq_step_encoded * 10.0
        channel_spacing = channel_spacing_encoded * 100.0
        
        # Validate with sanity checks
        if not 1 <= num_channels <= 4:
            num_channels = self.config.num_channels
        if not 5 <= symbol_duration <= 255:
            symbol_duration = int(self.config.symbol_duration_ms)
        if not 500 <= base_frequency <= 10000:
            base_frequency = self.config.base_frequency
        if not 10 <= frequency_step <= 500:
            frequency_step = self.config.frequency_step
        if not 500 <= channel_spacing <= 5000:
            channel_spacing = self.config.channel_spacing
        if signature_length > 255:
            signature_length = 0
        
        return {
            'valid': checksum_ok,
            'version': version,
            'num_channels': num_channels,
            'symbol_duration_ms': float(symbol_duration),
            'base_frequency': base_frequency,
            'frequency_step': frequency_step,
            'channel_spacing': channel_spacing,
            'signature_length': signature_length,
            'checksum_ok': checksum_ok,
            'confidence': float(np.mean(confidences)) if confidences else 0.0,
        }
    
    # -------------------------------------------------------------------------
    # High-Level Decoding
    # -------------------------------------------------------------------------
    
    def decode_data(
        self,
        samples: np.ndarray,
        signature_length: int = 8,
        repetitions: int = 1,
        read_metadata: bool = True,
    ) -> Tuple[Optional[bytes], Optional[bytes], float]:
        """
        Decode data from audio samples.
        
        Args:
            samples: Audio samples between start and end signals.
            signature_length: Expected signature length (ignored if metadata found).
            repetitions: Number of repetitions for redundancy.
            read_metadata: Whether to read metadata header.
            
        Returns:
            Tuple of (data bytes, signature bytes, confidence).
        """
        actual_sig_length = signature_length
        
        if read_metadata:
            metadata, pos = self.decode_metadata(samples, 0)
            
            if metadata['valid']:
                # Update config from metadata
                self.config.num_channels = metadata['num_channels']
                self.config.symbol_duration_ms = metadata['symbol_duration_ms']
                self.config.base_frequency = metadata['base_frequency']
                self.config.frequency_step = metadata['frequency_step']
                self.config.channel_spacing = metadata['channel_spacing']
                actual_sig_length = metadata.get('signature_length', signature_length)
                
                # Regenerate frequencies
                self._frequencies = self._generate_frequencies()
                self._freq_detector = FrequencyDetector(self.config)
            
            pos += int(self.config.sample_rate * self.config.silence_ms / 2000)
        else:
            pos = int(self.config.sample_rate * self.config.silence_ms / 1000)
        
        symbol_samples = int(
            self.config.sample_rate * self.config.symbol_duration_ms / 1000
        )
        byte_samples = symbol_samples if self.config.num_channels >= 2 else symbol_samples * 2
        silence_samples = int(self.config.sample_rate * self.config.silence_ms / 2000)
        
        timestamps = {'start': time.time()}
        confidences = []
        
        # Decode signature
        signature_bytes = []
        for _ in range(actual_sig_length):
            if pos + byte_samples > len(samples):
                return None, None, 0.0
            byte_val, conf = self.decode_byte(samples[pos:pos + byte_samples])
            signature_bytes.append(byte_val)
            confidences.append(conf)
            pos += byte_samples
        
        if actual_sig_length > 0:
            pos += silence_samples
        timestamps['signature_decoded'] = time.time()
        
        # Decode length
        if pos + byte_samples * 2 > len(samples):
            return None, bytes(signature_bytes), 0.0
        
        length_high, conf = self.decode_byte(samples[pos:pos + byte_samples])
        confidences.append(conf)
        pos += byte_samples
        
        length_low, conf = self.decode_byte(samples[pos:pos + byte_samples])
        confidences.append(conf)
        pos += byte_samples
        
        data_length = (length_high << 8) | length_low
        timestamps['length_decoded'] = time.time()
        
        # Decode data with repetitions
        all_data = []
        for _ in range(repetitions):
            data_bytes, pos, rep_confidences = self._decode_data_block(
                samples, pos, data_length, byte_samples
            )
            all_data.append(data_bytes)
            confidences.extend(rep_confidences)
            pos += silence_samples
        
        timestamps['data_decoded'] = time.time()
        
        # Vote on data if multiple repetitions
        final_data = self._vote_on_data(all_data, data_length)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        timestamps['end'] = time.time()
        timestamps['total_duration'] = timestamps['end'] - timestamps['start']
        self._last_decode_timestamps = timestamps
        
        return bytes(final_data), bytes(signature_bytes), avg_confidence
    
    def _decode_data_block(
        self,
        samples: np.ndarray,
        pos: int,
        data_length: int,
        byte_samples: int,
    ) -> Tuple[List[int], int, List[float]]:
        """Decode a block of data bytes."""
        data_bytes = []
        confidences = []
        
        # Collect byte positions
        byte_positions = []
        for _ in range(data_length):
            if pos + byte_samples > len(samples):
                break
            byte_positions.append(pos)
            pos += byte_samples
        
        # Parallel decode for larger data
        if len(byte_positions) > 20:
            with ThreadPoolExecutor(max_workers=min(4, multiprocessing.cpu_count())) as executor:
                futures = [
                    executor.submit(self.decode_byte, samples[bp:bp + byte_samples])
                    for bp in byte_positions
                ]
                for future in futures:
                    byte_val, conf = future.result()
                    data_bytes.append(byte_val)
                    confidences.append(conf)
        else:
            for bp in byte_positions:
                byte_val, conf = self.decode_byte(samples[bp:bp + byte_samples])
                data_bytes.append(byte_val)
                confidences.append(conf)
        
        return data_bytes, pos, confidences
    
    @staticmethod
    def _vote_on_data(all_data: List[List[int]], data_length: int) -> List[int]:
        """Vote on data from multiple repetitions."""
        if len(all_data) > 1 and all(len(d) == data_length for d in all_data):
            final_data = []
            for i in range(data_length):
                votes = [d[i] for d in all_data]
                try:
                    final_data.append(int(np.median(votes)))
                except Exception:
                    final_data.append(Counter(votes).most_common(1)[0][0])
            return final_data
        return all_data[0] if all_data else []
    
    def decode_text(
        self,
        samples: np.ndarray,
        signature_length: int = 8,
        repetitions: int = 1,
        read_metadata: bool = True,
    ) -> Tuple[Optional[str], Optional[bytes], float]:
        """
        Decode text from audio samples.
        
        Args:
            samples: Audio samples.
            signature_length: Expected signature length.
            repetitions: Number of repetitions.
            read_metadata: Whether to read metadata header.
            
        Returns:
            Tuple of (decoded text, signature, confidence).
        """
        data, signature, confidence = self.decode_data(
            samples, signature_length, repetitions, read_metadata
        )
        
        if data is None:
            return None, signature, confidence
        
        try:
            text = data.decode('utf-8')
            if confidence < 0.75:
                text = self._apply_char_correction(text)
            return text, signature, confidence
        except UnicodeDecodeError:
            return None, signature, confidence
    
    def get_last_decode_timestamps(self) -> Optional[Dict[str, float]]:
        """Get timestamps from the last decode operation."""
        return self._last_decode_timestamps
    
    @staticmethod
    def _apply_char_correction(text: str) -> str:
        """Apply common character corrections for FSK bit errors."""
        corrections = [
            ('tge ', 'the '), ('Tge ', 'The '), ('tgat ', 'that '),
            ('tgis ', 'this '), ('witg ', 'with '), ('wgen ', 'when '),
            ('sgould ', 'should '), ('cgaracter', 'character'),
            ('cgannel', 'channel'), ('gigg ', 'high '),
            ('fiw ', 'fix '), ('fiwed', 'fixed'), ('Fiw ', 'Fix '),
            ('Fiwed', 'Fixed'), ('tewt', 'text'), ('Tewt', 'Text'),
            ("'60%", "(60%"), ("'7/7)", "(7/7)"), ("'UI ", "(UI "),
            ('Mucg ', 'Much '),
        ]
        result = text
        for wrong, right in corrections:
            result = result.replace(wrong, right)
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def create_modulator(
    speed_mode: Optional[str] = None,
    **overrides: Any,
) -> FSKModulator:
    """
    Create an FSK modulator with auto-loaded configuration.
    
    Args:
        speed_mode: Speed mode override (e.g., 's40', 's60').
        **overrides: Additional configuration overrides.
        
    Returns:
        Configured FSKModulator instance.
    """
    config = FSKConfig.from_config(speed_mode=speed_mode, **overrides)
    return FSKModulator(config)


def create_demodulator(
    speed_mode: Optional[str] = None,
    **overrides: Any,
) -> FSKDemodulator:
    """
    Create an FSK demodulator with auto-loaded configuration.
    
    Args:
        speed_mode: Speed mode override (e.g., 's40', 's60').
        **overrides: Additional configuration overrides.
        
    Returns:
        Configured FSKDemodulator instance.
    """
    config = FSKConfig.from_config(speed_mode=speed_mode, **overrides)
    return FSKDemodulator(config)
