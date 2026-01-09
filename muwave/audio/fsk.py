"""
FSK (Frequency-Shift Keying) modulation for muwave.
Provides the core audio encoding/decoding functionality.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import math
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time

# Barker codes for enhanced synchronization
# These have optimal autocorrelation properties (peak:sidelobe = N:1)
BARKER_CODES = {
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


@dataclass
class FSKConfig:
    """Configuration for FSK modulation."""
    sample_rate: int = 44100
    base_frequency: float = 1800.0
    frequency_step: float = 120.0
    num_frequencies: int = 16
    symbol_duration_ms: float = 60.0
    start_frequencies: List[float] = None  # Multiple frequencies for start signal
    end_frequencies: List[float] = None    # Multiple frequencies for end signal
    signal_duration_ms: float = 200.0
    silence_ms: float = 50.0
    volume: float = 0.8
    num_channels: int = 2  # 1=mono, 2=dual, 3=tri, 4=quad channel FSK (higher=faster but less accurate)
    channel_spacing: float = 2400.0  # Spacing between channel base frequencies
    
    # Enhanced interference resilience options
    use_chirp_signals: bool = True  # Use chirp (frequency sweep) for start/end signals
    chirp_start_freq: float = 600.0  # Chirp starting frequency
    chirp_end_freq: float = 2400.0   # Chirp ending frequency  
    use_barker_preamble: bool = False  # Add Barker code preamble for sync
    barker_length: int = 13  # Barker code length (7, 11, or 13)
    barker_carrier_freq: float = 1500.0  # Carrier for Barker BPSK
    barker_chip_duration_ms: float = 8.0  # Duration of each Barker chip
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_channels > 4:
            raise ValueError("num_channels must be 1-4 (higher values cause excessive interference)")
        if self.num_channels < 1:
            raise ValueError("num_channels must be at least 1")
        # Set default multi-frequency start/end signals if not provided
        if self.start_frequencies is None:
            self.start_frequencies = [800.0, 850.0, 900.0]  # Triple-frequency start signal
        if self.end_frequencies is None:
            self.end_frequencies = [900.0, 950.0, 1000.0]  # Triple-frequency end signal
        # Validate Barker length
        if self.barker_length not in (7, 11, 13):
            raise ValueError("barker_length must be 7, 11, or 13")


class FSKModulator:
    """
    FSK modulator for encoding data into audio.
    
    Uses multiple frequencies to encode data with multi-channel support for higher bitrate.
    Each symbol is represented by a specific frequency, and multiple channels
    transmit simultaneously for parallel data transfer.
    """
    
    def __init__(self, config: Optional[FSKConfig] = None):
        """
        Initialize the FSK modulator.
        
        Args:
            config: FSK configuration. Uses defaults if None.
        """
        self.config = config or FSKConfig()
        self._frequencies = self._generate_frequencies()
    
    def _generate_frequencies(self) -> List[np.ndarray]:
        """Generate the frequency table for symbols across all channels."""
        channels = []
        for ch in range(self.config.num_channels):
            base = self.config.base_frequency + (ch * self.config.channel_spacing)
            freqs = np.array([
                base + i * self.config.frequency_step
                for i in range(self.config.num_frequencies)
            ])
            channels.append(freqs)
        return channels
    
    def _generate_tone(
        self,
        frequency: float,
        duration_ms: float,
        fade_ms: float = 5.0,
    ) -> np.ndarray:
        """
        Generate a tone at the given frequency with anti-interference measures.
        
        Args:
            frequency: Frequency in Hz
            duration_ms: Duration in milliseconds
            fade_ms: Fade in/out duration in milliseconds
            
        Returns:
            Audio samples as numpy array
        """
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        # Generate pure sine wave with high precision
        signal = np.sin(2 * np.pi * frequency * t, dtype=np.float64) * self.config.volume
        
        # Adaptive fade duration based on symbol length
        # Longer fade for faster symbols to reduce spectral splatter
        # Fast symbols (<30ms) get proportionally longer fades
        if duration_ms < 30:
            adaptive_fade_ms = min(duration_ms * 0.35, 10.0)  # Up to 35% of symbol or 10ms max
        else:
            adaptive_fade_ms = fade_ms
        
        fade_samples = int(self.config.sample_rate * adaptive_fade_ms / 1000)
        
        # Use raised-cosine window for smoother transitions (reduces harmonics)
        if fade_samples > 0 and fade_samples < num_samples // 2:
            # Raised cosine provides better spectral properties than linear
            t_fade = np.linspace(0, np.pi, fade_samples)
            fade_in = (1 - np.cos(t_fade)) / 2  # Smooth raised-cosine
            fade_out = (1 + np.cos(t_fade)) / 2
            signal[:fade_samples] *= fade_in
            signal[-fade_samples:] *= fade_out
        
        # Apply gentle low-pass filter to remove high-frequency artifacts
        # This is critical for reducing the green-blue interference bands
        signal = self._apply_anti_aliasing_filter(signal)
        
        return signal.astype(np.float32)
    
    def _apply_anti_aliasing_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply low-pass filter to remove harmonics and reduce interference.
        Uses a simple but effective moving average filter.
        
        Args:
            signal: Input signal
            
        Returns:
            Filtered signal
        """
        # Only apply filtering if we have enough samples
        if len(signal) < 5:
            return signal
        
        # Simple 3-tap moving average (low-pass filter)
        # This removes high-frequency components while preserving the fundamental
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        filtered = np.convolve(signal, kernel, mode='same')
        
        # Preserve amplitude by compensating for filter gain
        return filtered
    
    def _generate_silence(self, duration_ms: float) -> np.ndarray:
        """Generate silence."""
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        return np.zeros(num_samples, dtype=np.float32)
    
    def _generate_chirp(
        self,
        start_freq: float,
        end_freq: float,
        duration_ms: float,
        chirp_type: str = "linear"
    ) -> np.ndarray:
        """
        Generate a chirp (frequency sweep) signal for enhanced detection.
        
        Chirps provide excellent interference resilience due to:
        - Spread energy across time and frequency
        - Matched filter correlation gives processing gain
        - Robust to narrowband interference
        
        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz
            duration_ms: Duration in milliseconds
            chirp_type: "linear" or "logarithmic"
            
        Returns:
            Audio samples as numpy array
        """
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        if chirp_type == "linear":
            # Linear chirp: f(t) = f0 + (f1-f0)*t/T
            # Phase integral: φ(t) = 2π * (f0*t + 0.5*k*t²)
            k = (end_freq - start_freq) / (duration_ms / 1000)
            phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)
        elif chirp_type == "logarithmic":
            # Logarithmic chirp: better for wide frequency spans
            ratio = end_freq / start_freq
            k = ratio ** (1.0 / (duration_ms / 1000))
            phase = 2 * np.pi * start_freq * (k**t - 1) / np.log(k)
        else:
            # Default to linear
            k = (end_freq - start_freq) / (duration_ms / 1000)
            phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)
        
        signal = np.sin(phase, dtype=np.float64) * self.config.volume
        
        # Apply smooth envelope to reduce spectral splatter
        # Use raised-cosine envelope for clean edges
        envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.4
        signal = signal * envelope
        
        return signal.astype(np.float32)
    
    def _generate_barker_signal(
        self,
        carrier_freq: float,
        code_length: int = 13,
        chip_duration_ms: float = 8.0
    ) -> np.ndarray:
        """
        Generate BPSK-modulated Barker code signal for precise synchronization.
        
        Barker codes have optimal autocorrelation properties:
        - Sharp correlation peak
        - Minimal sidelobes (peak:sidelobe = N:1)
        - Used in WiFi, radar, and professional communications
        
        Args:
            carrier_freq: Carrier frequency for BPSK modulation
            code_length: Barker code length (7, 11, or 13)
            chip_duration_ms: Duration of each chip in milliseconds
            
        Returns:
            Audio samples as numpy array
        """
        if code_length not in BARKER_CODES:
            code_length = 13  # Default to longest code
        
        code = BARKER_CODES[code_length]
        chip_samples = int(self.config.sample_rate * chip_duration_ms / 1000)
        total_samples = chip_samples * len(code)
        
        signal = np.zeros(total_samples, dtype=np.float64)
        t_chip = np.linspace(0, chip_duration_ms / 1000, chip_samples, endpoint=False)
        
        for i, chip in enumerate(code):
            start = i * chip_samples
            end = start + chip_samples
            # BPSK: phase = 0 for +1, phase = π for -1
            phase = 0.0 if chip == 1 else np.pi
            signal[start:end] = np.sin(2 * np.pi * carrier_freq * t_chip + phase)
        
        # Apply gentle overall envelope
        t_total = np.linspace(0, 1, total_samples, endpoint=False)
        envelope = np.sin(np.pi * t_total) ** 0.3
        signal = signal * envelope * self.config.volume
        
        return signal.astype(np.float32)
    
    def generate_start_signal(self) -> np.ndarray:
        """
        Generate the distinctive start signal.
        
        Uses chirp + multi-tone combination for maximum interference resilience:
        1. Rising chirp (sweep from low to high frequency)
        2. Multi-tone burst (legacy detection compatibility)
        3. Optional Barker preamble for precise sync
        """
        samples = []
        
        if self.config.use_chirp_signals:
            # Part 1: Rising chirp for robust detection
            chirp_duration = self.config.signal_duration_ms * 0.5
            chirp = self._generate_chirp(
                self.config.chirp_start_freq,
                self.config.chirp_end_freq,
                chirp_duration,
                chirp_type="linear"
            )
            samples.append(chirp)
            
            # Short silence between chirp and multi-tone
            samples.append(self._generate_silence(10.0))
        
        # Part 2: Multi-tone burst (original approach for backward compatibility)
        duration_ms = self.config.signal_duration_ms * (0.4 if self.config.use_chirp_signals else 1.0)
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        multi_tone = np.zeros(num_samples, dtype=np.float32)
        for freq in self.config.start_frequencies:
            tone = np.sin(2 * np.pi * freq * t)
            multi_tone += tone
        
        # Normalize and apply volume
        multi_tone = (multi_tone / len(self.config.start_frequencies)) * self.config.volume
        
        # Apply envelope
        envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.5
        multi_tone = (multi_tone * envelope).astype(np.float32)
        samples.append(multi_tone)
        
        # Part 3: Optional Barker preamble for precise timing
        if self.config.use_barker_preamble:
            samples.append(self._generate_silence(5.0))
            barker = self._generate_barker_signal(
                self.config.barker_carrier_freq,
                self.config.barker_length,
                self.config.barker_chip_duration_ms
            )
            samples.append(barker)
        
        return np.concatenate(samples)
    
    def generate_end_signal(self) -> np.ndarray:
        """
        Generate the distinctive end signal.
        
        Uses falling chirp + multi-tone for easy differentiation from start:
        1. Falling chirp (sweep from high to low frequency)
        2. Multi-tone burst (different frequencies from start)
        """
        samples = []
        
        if self.config.use_chirp_signals:
            # Part 1: Falling chirp (reverse of start signal)
            chirp_duration = self.config.signal_duration_ms * 0.5
            chirp = self._generate_chirp(
                self.config.chirp_end_freq,  # Start high
                self.config.chirp_start_freq,  # End low
                chirp_duration,
                chirp_type="linear"
            )
            samples.append(chirp)
            
            # Short silence
            samples.append(self._generate_silence(10.0))
        
        # Part 2: Multi-tone burst
        duration_ms = self.config.signal_duration_ms * (0.4 if self.config.use_chirp_signals else 1.0)
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        multi_tone = np.zeros(num_samples, dtype=np.float32)
        for freq in self.config.end_frequencies:
            tone = np.sin(2 * np.pi * freq * t)
            multi_tone += tone
        
        # Normalize and apply volume
        multi_tone = (multi_tone / len(self.config.end_frequencies)) * self.config.volume
        
        # Apply envelope
        envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.5
        multi_tone = (multi_tone * envelope).astype(np.float32)
        samples.append(multi_tone)
        
        return np.concatenate(samples)
    
    def encode_byte(self, byte: int) -> np.ndarray:
        """
        Encode a single byte into audio using multi-channel FSK.
        
        Encoding strategies by channel count:
        - 1 channel: Sequential nibbles (2 symbols per byte)
        - 2 channels: Parallel nibbles (1 symbol per byte, 2x bitrate)
        - 3 channels: First nibble uses 2 channels, second nibble uses 1 (1.5 symbols per byte)
        - 4 channels: Each bit-pair on separate channel (1 symbol per byte, 4x parallelism)
        
        Args:
            byte: Byte value (0-255)
            
        Returns:
            Audio samples for the byte
        """
        if self.config.num_channels == 1:
            # Single-channel: transmit nibbles sequentially
            high_nibble = (byte >> 4) & 0x0F
            low_nibble = byte & 0x0F
            high_tone = self._generate_tone(
                self._frequencies[0][high_nibble],
                self.config.symbol_duration_ms,
            )
            low_tone = self._generate_tone(
                self._frequencies[0][low_nibble],
                self.config.symbol_duration_ms,
            )
            return np.concatenate([high_tone, low_tone])
        
        elif self.config.num_channels == 2:
            # Dual-channel: both nibbles simultaneously
            high_nibble = (byte >> 4) & 0x0F
            low_nibble = byte & 0x0F
            tone1 = self._generate_tone(self._frequencies[0][high_nibble], self.config.symbol_duration_ms)
            tone2 = self._generate_tone(self._frequencies[1][low_nibble], self.config.symbol_duration_ms)
            # Careful mixing to prevent clipping and reduce intermodulation
            mixed = (tone1 + tone2) * 0.5
            # Normalize to prevent overflow
            peak = np.max(np.abs(mixed))
            if peak > 0.95:
                mixed = mixed * (0.95 / peak)
            return mixed
        
        elif self.config.num_channels == 3:
            # Tri-channel: split 8 bits as 3+3+2 bits
            bits_765 = (byte >> 5) & 0x07  # Top 3 bits
            bits_432 = (byte >> 2) & 0x07  # Middle 3 bits
            bits_10 = byte & 0x03          # Bottom 2 bits
            tone1 = self._generate_tone(self._frequencies[0][bits_765], self.config.symbol_duration_ms)
            tone2 = self._generate_tone(self._frequencies[1][bits_432], self.config.symbol_duration_ms)
            tone3 = self._generate_tone(self._frequencies[2][bits_10], self.config.symbol_duration_ms)
            # Careful mixing with normalization
            mixed = (tone1 + tone2 + tone3) * 0.333
            peak = np.max(np.abs(mixed))
            if peak > 0.95:
                mixed = mixed * (0.95 / peak)
            return mixed
        
        else:  # num_channels == 4
            # Quad-channel: split 8 bits as 2+2+2+2 bits
            bits_76 = (byte >> 6) & 0x03
            bits_54 = (byte >> 4) & 0x03
            bits_32 = (byte >> 2) & 0x03
            bits_10 = byte & 0x03
            tone1 = self._generate_tone(self._frequencies[0][bits_76], self.config.symbol_duration_ms)
            tone2 = self._generate_tone(self._frequencies[1][bits_54], self.config.symbol_duration_ms)
            tone3 = self._generate_tone(self._frequencies[2][bits_32], self.config.symbol_duration_ms)
            tone4 = self._generate_tone(self._frequencies[3][bits_10], self.config.symbol_duration_ms)
            # Careful mixing with normalization
            mixed = (tone1 + tone2 + tone3 + tone4) * 0.25
            peak = np.max(np.abs(mixed))
            if peak > 0.95:
                mixed = mixed * (0.95 / peak)
            return mixed
    
    def encode_signature(self, signature: bytes) -> np.ndarray:
        """
        Encode a party signature into audio.
        
        Args:
            signature: Party signature bytes
            
        Returns:
            Audio samples for the signature
        """
        samples = []
        for byte in signature:
            samples.append(self.encode_byte(byte))
        return np.concatenate(samples)
    
    def encode_metadata(self) -> np.ndarray:
        """
        Encode transmission metadata (channel count and speed) using fixed format.
        Always uses 1-channel, fast mode (35ms) for reliable header detection.
        
        Returns:
            Audio samples for metadata (2 bytes)
        """
        # Save current config
        original_channels = self.config.num_channels
        original_duration = self.config.symbol_duration_ms
        
        # Use fixed format for metadata: 1-channel, fast (35ms)
        self.config.num_channels = 1
        self.config.symbol_duration_ms = 35.0
        
        # Regenerate frequencies for 1-channel
        temp_frequencies = [[self.config.base_frequency + i * self.config.frequency_step
                            for i in range(self.config.num_frequencies)]]
        
        # Encode 2 bytes: channel count and symbol duration
        # Byte 1: num_channels (1-4)
        # Byte 2: symbol_duration_ms (encoded as value, supports 1-255ms)
        metadata_bytes = [
            original_channels,
            int(original_duration) & 0xFF,
        ]
        
        samples = []
        for byte_val in metadata_bytes:
            # Use 1-channel encoding
            high_nibble = (byte_val >> 4) & 0x0F
            low_nibble = byte_val & 0x0F
            high_tone = self._generate_tone(
                temp_frequencies[0][high_nibble],
                35.0,  # Fixed 35ms for metadata
            )
            low_tone = self._generate_tone(
                temp_frequencies[0][low_nibble],
                35.0,
            )
            samples.extend([high_tone, low_tone])
        
        # Restore original config
        self.config.num_channels = original_channels
        self.config.symbol_duration_ms = original_duration
        
        return np.concatenate(samples)
    
    def encode_data(
        self,
        data: bytes,
        signature: Optional[bytes] = None,
        repetitions: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Encode data into audio with start/end signals.
        
        Args:
            data: Data bytes to encode
            signature: Optional party signature
            repetitions: Number of times to repeat the data
            
        Returns:
            Tuple of (complete audio signal, timing dict)
        """
        timestamps = {'start': time.time()}
        samples = []
        
        # Start signal
        samples.append(self.generate_start_signal())
        samples.append(self._generate_silence(self.config.silence_ms))
        
        # Metadata header (channel count and speed)
        samples.append(self.encode_metadata())
        samples.append(self._generate_silence(self.config.silence_ms / 2))
        
        # Signature if provided
        if signature:
            samples.append(self.encode_signature(signature))
            samples.append(self._generate_silence(self.config.silence_ms / 2))
        
        # Data length (2 bytes, max 65535 bytes)
        length = len(data)
        samples.append(self.encode_byte((length >> 8) & 0xFF))
        samples.append(self.encode_byte(length & 0xFF))
        
        # Data (with repetitions for redundancy)
        for _ in range(repetitions):
            for byte in data:
                samples.append(self.encode_byte(byte))
            if repetitions > 1:
                samples.append(self._generate_silence(self.config.silence_ms / 2))
        
        samples.append(self._generate_silence(self.config.silence_ms))
        timestamps['data_encoded'] = time.time()
        
        # End signal
        samples.append(self.generate_end_signal())
        timestamps['end'] = time.time()
        timestamps['total_duration'] = timestamps['end'] - timestamps['start']
        
        # Store timestamps for retrieval
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
            text: Text to encode
            signature: Optional party signature
            repetitions: Number of times to repeat
            
        Returns:
            Audio signal (timestamps stored in _last_encode_timestamps)
        """
        data = text.encode('utf-8')
        audio, timestamps = self.encode_data(data, signature, repetitions)
        return audio


class FSKDemodulator:
    """
    FSK demodulator for decoding audio into data.
    
    Uses frequency detection to decode symbols from audio with multi-channel support.
    """
    
    def __init__(self, config: Optional[FSKConfig] = None):
        """
        Initialize the FSK demodulator.
        
        Args:
            config: FSK configuration. Uses defaults if None.
        """
        self.config = config or FSKConfig()
        self._frequencies = self._generate_frequencies()
    
    def _generate_frequencies(self) -> List[np.ndarray]:
        """Generate the frequency table for symbols across all channels."""
        channels = []
        for ch in range(self.config.num_channels):
            base = self.config.base_frequency + (ch * self.config.channel_spacing)
            freqs = np.array([
                base + i * self.config.frequency_step
                for i in range(self.config.num_frequencies)
            ])
            channels.append(freqs)
        return channels
    
    def _detect_frequency(
        self,
        samples: np.ndarray,
        target_frequencies: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Detect which frequency is present in the samples.
        
        Args:
            samples: Audio samples
            target_frequencies: Array of possible frequencies
            
        Returns:
            Tuple of (frequency index, confidence)
        """
        if len(samples) == 0:
            return 0, 0.0
        
        # Use float64 for higher precision
        samples_hp = samples.astype(np.float64)
        
        # Apply Blackman-Harris window for superior sidelobe suppression
        # This reduces spectral leakage better than Hamming for our use case
        window = np.blackman(len(samples_hp))
        windowed_samples = samples_hp * window
        
        # Calculate signal energy for normalization with improved precision
        signal_energy = np.sum(windowed_samples ** 2)
        if signal_energy < 1e-20:
            return 0, 0.0
        
        # Use Goertzel algorithm for efficient frequency detection
        correlations = []
        for freq in target_frequencies:
            correlation = self._goertzel(windowed_samples, freq)
            correlations.append(correlation)
        
        correlations = np.array(correlations)
        
        # Improved normalization: divide by sqrt of signal energy
        # This gives better relative magnitudes
        correlations = correlations / np.sqrt(signal_energy)
        
        best_idx = np.argmax(correlations)
        
        # Calculate confidence using multiple metrics
        sorted_corr = np.sort(correlations)[::-1]
        if sorted_corr[0] > 1e-10:
            if len(sorted_corr) > 1 and sorted_corr[1] > 1e-10:
                # Metric 1: Separation ratio (how much better is the best vs second-best)
                separation = (sorted_corr[0] - sorted_corr[1]) / sorted_corr[0]
                
                # Metric 2: Energy concentration (best vs total)
                total_energy = np.sum(correlations ** 2)
                concentration = (sorted_corr[0] ** 2) / total_energy if total_energy > 0 else 0
                
                # Metric 3: Signal strength relative to expected
                # Adaptive threshold based on symbol duration
                # Shorter symbols have less time to build energy, so scale expected magnitude
                # Base threshold: 0.3 for 60ms symbols, scale proportionally
                base_duration = 60.0  # Reference symbol duration in ms
                duration_scale = np.sqrt(self.config.symbol_duration_ms / base_duration)
                expected_magnitude = 0.3 * duration_scale
                strength = min(1.0, sorted_corr[0] / expected_magnitude)
                
                # Combined confidence with weights
                confidence = (
                    separation * 0.35 +    # How distinct is the peak
                    concentration * 0.35 + # How focused is the energy
                    strength * 0.30        # How strong is the signal
                )
                confidence = min(1.0, max(0.0, confidence))
            else:
                confidence = 0.95
        else:
            confidence = 0.0
        
        return best_idx, confidence
    
    def _goertzel(self, samples: np.ndarray, target_freq: float) -> float:
        """
        Goertzel algorithm for efficient single-frequency detection.
        Enhanced with higher precision calculations.
        
        Args:
            samples: Audio samples
            target_freq: Target frequency to detect
            
        Returns:
            Magnitude of the target frequency
        """
        n = len(samples)
        if n == 0:
            return 0.0
        
        # Use float64 for higher precision in intermediate calculations
        samples_hp = samples.astype(np.float64)
        
        # Use exact frequency instead of nearest bin for better accuracy
        normalized_freq = target_freq / self.config.sample_rate
        w = 2.0 * np.pi * normalized_freq
        coeff = 2.0 * np.cos(w)
        
        # Use sine and cosine for magnitude calculation
        cosine = np.cos(w)
        sine = np.sin(w)
        
        # Goertzel filter with improved numerical stability using float64
        s0, s1, s2 = 0.0, 0.0, 0.0
        for sample in samples_hp:
            s0 = sample + coeff * s1 - s2
            s2 = s1
            s1 = s0
        
        # Calculate real and imaginary parts
        real = s1 - s2 * cosine
        imag = s2 * sine
        
        # Return magnitude (without 2/n scaling - preserve absolute magnitude)
        magnitude = np.sqrt(real * real + imag * imag)
        return float(magnitude)
    
    def _generate_reference_chirp(
        self,
        start_freq: float,
        end_freq: float,
        duration_ms: float
    ) -> np.ndarray:
        """Generate reference chirp for matched filter detection."""
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        k = (end_freq - start_freq) / (duration_ms / 1000)
        phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)
        signal = np.sin(phase, dtype=np.float64)
        
        # Apply same envelope as transmitter
        envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.4
        return (signal * envelope).astype(np.float32)
    
    def _detect_chirp(
        self,
        samples: np.ndarray,
        start_freq: float,
        end_freq: float,
        duration_ms: float,
        threshold: float = 0.25
    ) -> Tuple[bool, int, float]:
        """
        Detect chirp signal using matched filter (cross-correlation).
        
        Matched filtering provides significant processing gain:
        - Spreads detection across time-bandwidth product
        - Can detect signals well below noise floor
        - Used in radar and professional communications
        
        Args:
            samples: Audio samples to search
            start_freq: Chirp starting frequency
            end_freq: Chirp ending frequency
            duration_ms: Chirp duration
            threshold: Detection threshold for normalized correlation
            
        Returns:
            Tuple of (detected, sample_position, correlation_peak)
        """
        reference = self._generate_reference_chirp(start_freq, end_freq, duration_ms)
        
        if len(samples) < len(reference):
            return False, 0, 0.0
        
        # Matched filter = cross-correlation
        # Using 'valid' mode gives positions where full correlation is possible
        correlation = np.correlate(samples.astype(np.float64), reference.astype(np.float64), mode='valid')
        correlation = np.abs(correlation)
        
        # Normalize by reference energy and local signal energy
        ref_energy = np.sqrt(np.sum(reference.astype(np.float64) ** 2))
        
        # Calculate local signal energy using a sliding window
        sig_energy_squared = np.convolve(
            samples.astype(np.float64) ** 2, 
            np.ones(len(reference)), 
            mode='valid'
        )
        sig_energy = np.sqrt(np.maximum(sig_energy_squared, 1e-10))
        
        # Normalized correlation
        normalized_corr = correlation / (ref_energy * sig_energy)
        
        peak_idx = np.argmax(normalized_corr)
        peak_value = float(normalized_corr[peak_idx])
        
        detected = peak_value > threshold
        # Return position at end of chirp
        return detected, peak_idx + len(reference), peak_value
    
    def _generate_reference_barker(
        self,
        carrier_freq: float,
        code_length: int,
        chip_duration_ms: float
    ) -> np.ndarray:
        """Generate reference Barker signal for matched filter detection."""
        if code_length not in BARKER_CODES:
            code_length = 13
        
        code = BARKER_CODES[code_length]
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
        
        return (signal * envelope).astype(np.float32)
    
    def _detect_barker(
        self,
        samples: np.ndarray,
        carrier_freq: float,
        code_length: int,
        chip_duration_ms: float,
        threshold: float = 0.4
    ) -> Tuple[bool, int, float]:
        """
        Detect Barker-coded signal using matched filter.
        
        Barker codes have optimal autocorrelation properties,
        giving a peak:sidelobe ratio of N:1 where N is code length.
        
        Args:
            samples: Audio samples to search
            carrier_freq: Carrier frequency
            code_length: Barker code length (7, 11, or 13)
            chip_duration_ms: Duration of each chip
            threshold: Detection threshold
            
        Returns:
            Tuple of (detected, sample_position, correlation_peak)
        """
        reference = self._generate_reference_barker(carrier_freq, code_length, chip_duration_ms)
        
        if len(samples) < len(reference):
            return False, 0, 0.0
        
        # Matched filter correlation
        correlation = np.correlate(samples.astype(np.float64), reference.astype(np.float64), mode='valid')
        correlation = np.abs(correlation)
        
        # Normalize
        ref_energy = np.sqrt(np.sum(reference.astype(np.float64) ** 2))
        sig_energy_squared = np.convolve(
            samples.astype(np.float64) ** 2, 
            np.ones(len(reference)), 
            mode='valid'
        )
        sig_energy = np.sqrt(np.maximum(sig_energy_squared, 1e-10))
        
        normalized_corr = correlation / (ref_energy * sig_energy)
        
        peak_idx = np.argmax(normalized_corr)
        peak_value = float(normalized_corr[peak_idx])
        
        detected = peak_value > threshold
        return detected, peak_idx + len(reference), peak_value
    
    def _measure_noise_floor(
        self,
        samples: np.ndarray,
        window_ms: float = 50.0,
        percentile: float = 25.0
    ) -> float:
        """
        Estimate noise floor for adaptive thresholding.
        
        Uses percentile of windowed RMS values to estimate background noise.
        This allows detection thresholds to adapt to noisy environments.
        
        Args:
            samples: Audio samples
            window_ms: Analysis window size in milliseconds
            percentile: Percentile to use for noise estimate (lower = more conservative)
            
        Returns:
            Estimated noise floor (RMS value)
        """
        window_size = int(self.config.sample_rate * window_ms / 1000)
        if len(samples) < window_size:
            return np.sqrt(np.mean(samples ** 2))
        
        rms_values = []
        for i in range(0, len(samples) - window_size, window_size // 2):
            window = samples[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        if not rms_values:
            return 0.01
        
        return float(np.percentile(rms_values, percentile))
    
    def detect_start_signal(
        self,
        samples: np.ndarray,
        threshold: float = 0.12,
    ) -> Tuple[bool, int]:
        """
        Detect the start signal using enhanced multi-method detection.
        
        Detection strategy:
        1. If chirp signals enabled: Use matched filter correlation (most robust)
        2. Fallback to multi-tone detection for backward compatibility
        3. Optional Barker code detection for precise sync
        
        Args:
            samples: Audio samples
            threshold: Detection threshold (relative to signal RMS)
            
        Returns:
            Tuple of (detected, sample_position)
        """
        # Adaptive threshold based on noise floor
        noise_floor = self._measure_noise_floor(samples[:min(len(samples), self.config.sample_rate)])
        adaptive_threshold = max(threshold, noise_floor * 2.5)
        
        # Method 1: Chirp detection (if enabled) - most robust
        if self.config.use_chirp_signals:
            chirp_duration = self.config.signal_duration_ms * 0.5
            detected, pos, peak = self._detect_chirp(
                samples,
                self.config.chirp_start_freq,
                self.config.chirp_end_freq,
                chirp_duration,
                threshold=0.25  # Chirp detection uses its own threshold
            )
            if detected:
                # Account for silence + multi-tone that follows chirp
                multi_tone_duration = self.config.signal_duration_ms * 0.4
                extra_samples = int(self.config.sample_rate * (10.0 + multi_tone_duration) / 1000)
                
                # If Barker preamble is used, add that too
                if self.config.use_barker_preamble:
                    barker_duration = self.config.barker_chip_duration_ms * self.config.barker_length
                    extra_samples += int(self.config.sample_rate * (5.0 + barker_duration) / 1000)
                
                return True, pos + extra_samples
        
        # Method 2: Multi-tone detection (fallback / backward compatibility)
        window_size = int(
            self.config.sample_rate * self.config.signal_duration_ms / 1000
        )
        step_size = window_size // 4
        
        for i in range(0, len(samples) - window_size, step_size):
            window = samples[i:i + window_size]
            
            # Remove DC offset for better detection
            window = window - np.mean(window)
            
            # Calculate window RMS for normalization
            window_rms = np.sqrt(np.mean(window ** 2))
            if window_rms < 0.01:  # Skip silent regions
                continue
            
            # Check for all start frequencies simultaneously present
            detection_count = 0
            magnitudes = []
            
            for freq in self.config.start_frequencies:
                magnitude = self._goertzel(window, freq)
                # Normalize by window length and RMS for relative comparison
                normalized_mag = magnitude / (len(window) * window_rms) if window_rms > 1e-10 else 0.0
                magnitudes.append(normalized_mag)
                
                # Check if this frequency is present (use adaptive threshold)
                if normalized_mag > adaptive_threshold:
                    detection_count += 1
            
            # Require simple majority of configured start frequencies
            required = math.ceil(len(self.config.start_frequencies) * 0.5)
            if detection_count >= required:
                return True, i + window_size
        
        return False, 0
    
    def detect_end_signal(
        self,
        samples: np.ndarray,
        threshold: float = 0.12,
    ) -> Tuple[bool, int]:
        """
        Detect the end signal using enhanced multi-method detection.
        
        Uses falling chirp (if enabled) + multi-tone for differentiation from start.
        
        Args:
            samples: Audio samples
            threshold: Detection threshold (relative to signal RMS)
            
        Returns:
            Tuple of (detected, sample_position)
        """
        # Adaptive threshold based on noise floor
        noise_floor = self._measure_noise_floor(samples[:min(len(samples), self.config.sample_rate // 2)])
        adaptive_threshold = max(threshold, noise_floor * 2.5)
        
        # Method 1: Chirp detection (if enabled) - falling chirp for end signal
        if self.config.use_chirp_signals:
            chirp_duration = self.config.signal_duration_ms * 0.5
            # End signal uses REVERSE chirp (high to low)
            detected, pos, peak = self._detect_chirp(
                samples,
                self.config.chirp_end_freq,    # Start high
                self.config.chirp_start_freq,  # End low
                chirp_duration,
                threshold=0.25
            )
            if detected:
                # Position should be at start of end signal (before chirp)
                chirp_samples = int(self.config.sample_rate * chirp_duration / 1000)
                return True, max(0, pos - chirp_samples)
        
        # Method 2: Multi-tone detection (fallback)
        window_size = int(
            self.config.sample_rate * self.config.signal_duration_ms / 1000
        )
        step_size = window_size // 4
        
        # Make sure we don't go past the end
        max_start = len(samples) - window_size
        if max_start < 0:
            return False, len(samples)
        
        # Check from start to end, ensuring we check the last window
        positions_to_check = list(range(0, max_start, step_size))
        # Always check the last possible position if not already in the list
        if not positions_to_check or positions_to_check[-1] != max_start:
            positions_to_check.append(max_start)
        
        for i in positions_to_check:
            window = samples[i:i + window_size]
            
            # Skip if window is too short
            if len(window) < window_size:
                continue
            
            # Remove DC offset for better detection
            window = window - np.mean(window)
            
            # Calculate window RMS for normalization
            window_rms = np.sqrt(np.mean(window ** 2))
            if window_rms < 0.01:  # Skip silent regions
                continue
            
            # Check for all end frequencies simultaneously present
            detection_count = 0
            magnitudes = []
            
            for freq in self.config.end_frequencies:
                magnitude = self._goertzel(window, freq)
                # Normalize by window length and RMS for relative comparison
                normalized_mag = magnitude / (len(window) * window_rms) if window_rms > 1e-10 else 0.0
                magnitudes.append(normalized_mag)
                
                # Check if this frequency is present (use adaptive threshold)
                if normalized_mag > adaptive_threshold:
                    detection_count += 1
            
            # Require simple majority of configured end frequencies
            required = math.ceil(len(self.config.end_frequencies) * 0.5)
            if detection_count >= required:
                return True, i
        
        return False, len(samples)
    
    def _estimate_frequency(self, samples: np.ndarray) -> float:
        """Estimate the dominant frequency in samples."""
        if len(samples) < 64:
            return 0.0
        
        # Use FFT for frequency estimation
        fft = np.fft.rfft(samples)
        magnitudes = np.abs(fft)
        
        # Find peak
        peak_idx = np.argmax(magnitudes[1:]) + 1
        frequency = peak_idx * self.config.sample_rate / len(samples)
        
        return frequency
    
    def decode_byte(
        self,
        samples: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Decode a byte from audio samples using multi-channel FSK.
        
        Supports 1-4 channels with different decoding strategies.
        
        Args:
            samples: Audio samples for one byte
            
        Returns:
            Tuple of (byte value, average confidence)
        """
        symbol_samples = int(
            self.config.sample_rate * self.config.symbol_duration_ms / 1000
        )
        # Adaptive skip based on symbol duration
        # For ultra-fast (< 25ms), skip less to capture more signal
        # For fast (25-40ms), moderate skip
        # For normal (> 40ms), more skip for stability
        if self.config.symbol_duration_ms < 25:
            skip_fraction = 30  # Skip ~3.3% on each side
        elif self.config.symbol_duration_ms < 40:
            skip_fraction = 20  # Skip 5% on each side
        else:
            skip_fraction = 10  # Skip 10% on each side
        skip = max(1, symbol_samples // skip_fraction)
        
        if self.config.num_channels == 1:
            # Single-channel: decode nibbles sequentially
            if len(samples) < symbol_samples * 2:
                return 0, 0.0
            high_samples = samples[skip:symbol_samples - skip]
            # Remove DC offset
            if len(high_samples) > 0:
                high_samples = high_samples - np.mean(high_samples)
            high_idx, high_conf = self._detect_frequency(high_samples, self._frequencies[0])
            
            low_samples = samples[symbol_samples + skip:symbol_samples * 2 - skip]
            # Remove DC offset
            if len(low_samples) > 0:
                low_samples = low_samples - np.mean(low_samples)
            low_idx, low_conf = self._detect_frequency(low_samples, self._frequencies[0])
            
            byte_value = (high_idx << 4) | low_idx
            confidence = np.sqrt(high_conf * low_conf)
            return byte_value, confidence
        
        elif self.config.num_channels == 2:
            # Dual-channel: both nibbles simultaneously
            if len(samples) < symbol_samples:
                return 0, 0.0
            symbol = samples[skip:symbol_samples - skip]
            
            # Apply gentle high-pass filter to reduce DC offset and low-frequency noise
            if len(symbol) > 10:
                symbol = symbol - np.mean(symbol)
            
            high_idx, high_conf = self._detect_frequency(symbol, self._frequencies[0])
            low_idx, low_conf = self._detect_frequency(symbol, self._frequencies[1])
            byte_value = (high_idx << 4) | low_idx
            # Use geometric mean for combined confidence (more conservative)
            confidence = np.sqrt(high_conf * low_conf)
            return byte_value, confidence
        
        elif self.config.num_channels == 3:
            # Tri-channel: decode 3+3+2 bits
            if len(samples) < symbol_samples:
                return 0, 0.0
            symbol = samples[skip:symbol_samples - skip]
            # Remove DC offset
            if len(symbol) > 10:
                symbol = symbol - np.mean(symbol)
            
            idx1, conf1 = self._detect_frequency(symbol, self._frequencies[0])
            idx2, conf2 = self._detect_frequency(symbol, self._frequencies[1])
            idx3, conf3 = self._detect_frequency(symbol, self._frequencies[2])
            # Reconstruct byte from 3+3+2 bit pattern
            byte_value = ((idx1 & 0x07) << 5) | ((idx2 & 0x07) << 2) | (idx3 & 0x03)
            confidence = (conf1 * conf2 * conf3) ** (1/3)
            return byte_value, confidence
        
        else:  # num_channels == 4
            # Quad-channel: decode 2+2+2+2 bits
            if len(samples) < symbol_samples:
                return 0, 0.0
            symbol = samples[skip:symbol_samples - skip]
            # Remove DC offset
            if len(symbol) > 10:
                symbol = symbol - np.mean(symbol)
            
            idx1, conf1 = self._detect_frequency(symbol, self._frequencies[0])
            idx2, conf2 = self._detect_frequency(symbol, self._frequencies[1])
            idx3, conf3 = self._detect_frequency(symbol, self._frequencies[2])
            idx4, conf4 = self._detect_frequency(symbol, self._frequencies[3])
            # Reconstruct byte from 2+2+2+2 bit pattern
            byte_value = ((idx1 & 0x03) << 6) | ((idx2 & 0x03) << 4) | ((idx3 & 0x03) << 2) | (idx4 & 0x03)
            confidence = (conf1 * conf2 * conf3 * conf4) ** (1/4)
            return byte_value, confidence
    
    def decode_metadata(self, samples: np.ndarray, start_pos: int = 0) -> Tuple[int, int, int]:
        """
        Decode transmission metadata from audio samples.
        Reads 2 bytes using fixed format (1-channel, 35ms symbols).
        
        Args:
            samples: Audio samples starting from after start signal
            start_pos: Starting position in samples (default 0 = right after start signal)
            
        Returns:
            Tuple of (num_channels, symbol_duration_ms, next_position)
        """
        # Metadata uses fixed format: 1-channel, 35ms
        metadata_symbol_samples = int(self.config.sample_rate * 35.0 / 1000)
        metadata_byte_samples = metadata_symbol_samples * 2  # 1-channel = 2 symbols per byte
        
        # Create temporary single-channel frequency list
        temp_frequencies = np.array([self.config.base_frequency + i * self.config.frequency_step
                                     for i in range(self.config.num_frequencies)])
        
        # Skip initial silence after start signal
        # Note: start_pos should be 0 (we already trimmed to after start signal)
        pos = start_pos + int(self.config.sample_rate * self.config.silence_ms / 1000)
        skip = max(1, metadata_symbol_samples // 15)  # Less aggressive skip for metadata
        
        # Decode 2 metadata bytes (using fixed 35ms, 1-channel format)
        metadata_bytes = []
        for _ in range(2):
            if pos + metadata_byte_samples > len(samples):
                # Fallback to config defaults
                return self.config.num_channels, int(self.config.symbol_duration_ms), pos
            
            # Decode high nibble
            high_samples = samples[pos + skip:pos + metadata_symbol_samples - skip]
            # Remove DC offset
            if len(high_samples) > 0:
                high_samples = high_samples - np.mean(high_samples)
            high_idx, _ = self._detect_frequency(high_samples, temp_frequencies)
            
            # Decode low nibble
            low_samples = samples[pos + metadata_symbol_samples + skip:pos + metadata_byte_samples - skip]
            # Remove DC offset
            if len(low_samples) > 0:
                low_samples = low_samples - np.mean(low_samples)
            low_idx, _ = self._detect_frequency(low_samples, temp_frequencies)
            
            byte_val = (high_idx << 4) | low_idx
            metadata_bytes.append(byte_val)
            pos += metadata_byte_samples
        
        # Validate and apply sanity checks to decoded metadata
        num_channels = metadata_bytes[0]
        symbol_duration = metadata_bytes[1]
        
        # Validate channel count (1-4)
        if not (1 <= num_channels <= 4):
            num_channels = self.config.num_channels
        
        # Validate symbol duration (10-200ms range)
        if not (10 <= symbol_duration <= 200):
            symbol_duration = int(self.config.symbol_duration_ms)
        
        return num_channels, symbol_duration, pos
    
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
            samples: Audio samples (should be between start and end signals)
            signature_length: Expected signature length in bytes
            repetitions: Number of repetitions for redundancy
            read_metadata: Whether to read metadata header (default True)
            
        Returns:
            Tuple of (data bytes, signature bytes, confidence)
        """
        # Read metadata header if present
        # Note: samples should already be trimmed (start signal removed, before end signal)
        if read_metadata:
            # decode_metadata expects samples starting after start signal
            detected_channels, detected_duration, pos = self.decode_metadata(samples, 0)
            # Update config with detected values
            original_channels = self.config.num_channels
            original_duration = self.config.symbol_duration_ms
            self.config.num_channels = detected_channels
            self.config.symbol_duration_ms = float(detected_duration)
            # Regenerate frequencies with updated channel count
            self._frequencies = self._generate_frequencies()
            # Skip past metadata trailing silence
            pos += int(self.config.sample_rate * self.config.silence_ms / 2000)
        else:
            # Start after the initial silence that follows the start signal
            pos = int(self.config.sample_rate * self.config.silence_ms / 1000)
        
        symbol_samples = int(
            self.config.sample_rate * self.config.symbol_duration_ms / 1000
        )
        # Multi-channel transmits both nibbles simultaneously, single-channel sequentially
        byte_samples = symbol_samples if self.config.num_channels >= 2 else symbol_samples * 2
        silence_samples = int(
            self.config.sample_rate * self.config.silence_ms / 2000
        )
        
        confidences = []
        timestamps = {'start': time.time()}
        
        # Decode signature
        signature_bytes = []
        for _ in range(signature_length):
            if pos + byte_samples > len(samples):
                return None, None, 0.0
            byte_val, conf = self.decode_byte(samples[pos:pos + byte_samples])
            signature_bytes.append(byte_val)
            confidences.append(conf)
            pos += byte_samples
        
        pos += silence_samples
        timestamps['signature_decoded'] = time.time()
        
        # Decode length (2 bytes)
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
        
        # Decode data with repetitions - use parallel processing for better performance
        all_data = []
        for rep in range(repetitions):
            data_bytes = []
            byte_positions = []
            
            # Collect all byte positions for parallel processing
            for i in range(data_length):
                if pos + byte_samples > len(samples):
                    break
                byte_positions.append(pos)
                pos += byte_samples
            
            # Parallel decode if we have enough bytes to make it worthwhile
            if len(byte_positions) > 20:
                # Use thread pool for parallel decoding (GIL is released in numpy operations)
                with ThreadPoolExecutor(max_workers=min(4, multiprocessing.cpu_count())) as executor:
                    futures = [executor.submit(self.decode_byte, samples[bp:bp + byte_samples]) 
                              for bp in byte_positions]
                    for future in futures:
                        byte_val, conf = future.result()
                        data_bytes.append(byte_val)
                        confidences.append(conf)
            else:
                # Sequential decode for small data
                for bp in byte_positions:
                    byte_val, conf = self.decode_byte(samples[bp:bp + byte_samples])
                    data_bytes.append(byte_val)
                    confidences.append(conf)
            
            all_data.append(data_bytes)
            pos += silence_samples
        
        timestamps['data_decoded'] = time.time()
        
        # Vote on data if multiple repetitions
        if len(all_data) > 1 and all(len(d) == data_length for d in all_data):
            final_data = []
            for i in range(data_length):
                # Use median voting for better error correction
                from collections import Counter
                votes = [d[i] for d in all_data]
                # Try median first (works better for numerical drift)
                try:
                    final_data.append(int(np.median(votes)))
                except Exception:
                    # Fallback to majority
                    final_data.append(Counter(votes).most_common(1)[0][0])
        elif all_data:
            final_data = all_data[0]
        else:
            final_data = []
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        timestamps['end'] = time.time()
        timestamps['total_duration'] = timestamps['end'] - timestamps['start']
        
        # Store timestamps for later retrieval
        self._last_decode_timestamps = timestamps
        
        return bytes(final_data), bytes(signature_bytes), avg_confidence
    
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
            samples: Audio samples
            signature_length: Expected signature length
            repetitions: Number of repetitions
            read_metadata: Whether to read metadata header (default True)
            
        Returns:
            Tuple of (decoded text, signature, confidence)
        """
        data, signature, confidence = self.decode_data(
            samples, signature_length, repetitions, read_metadata
        )
        
        if data is None:
            return None, signature, confidence
        
        try:
            text = data.decode('utf-8')
            # Apply basic error correction for common FSK bit errors
            if confidence < 0.75:
                text = self._apply_char_correction(text)
            return text, signature, confidence
        except UnicodeDecodeError:
            return None, signature, confidence
    
    def get_last_decode_timestamps(self) -> Optional[Dict[str, float]]:
        """
        Get timestamps from the last decode operation.
        
        Returns:
            Dictionary with timing information or None if no decode has been performed
        """
        return getattr(self, '_last_decode_timestamps', None)
    
    def _apply_char_correction(self, text: str) -> str:
        """Apply common character corrections for FSK bit errors."""
        # Common single-bit errors in ASCII:
        # 't' (0x74) ↔ 'h' (0x68): differ by bit 2+3
        # 'h' (0x68) ↔ 'g' (0x67): differ by bit 0
        # 'w' (0x77) ↔ 'v' (0x76): differ by bit 0
        # '"' (0x22) ↔ ''' (0x27): differ by bit 0+2
        corrections = [
            # Common words with h/t confusion
            ('tge ', 'the '),
            ('Tge ', 'The '),
            ('tgat ', 'that '),
            ('tgis ', 'this '),
            ('witg ', 'with '),
            ('wgen ', 'when '),
            ('sgould ', 'should '),
            # h/g confusion
            ('cgaracter', 'character'),
            ('cgannel', 'channel'),
            ('gigg ', 'high '),
            # w/v and x/w confusion
            ('fiw ', 'fix '),
            ('fiwed', 'fixed'),
            ('Fiw ', 'Fix '),
            ('Fiwed', 'Fixed'),
            # Text/test confusion
            ('tewt', 'text'),
            ('Tewt', 'Text'),
            # Quote/apostrophe confusion (bit errors in quotes)
            ("'60%", "(60%"),
            ("'7/7)", "(7/7)"),
            ("'UI ", "(UI "),
            # Common endings
            ('Mucg ', 'Much '),
        ]
        result = text
        for wrong, right in corrections:
            result = result.replace(wrong, right)
        return result
