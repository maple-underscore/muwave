"""
FSK (Frequency-Shift Keying) modulation for muwave.
Provides the core audio encoding/decoding functionality.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


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
        Generate a tone at the given frequency.
        
        Args:
            frequency: Frequency in Hz
            duration_ms: Duration in milliseconds
            fade_ms: Fade in/out duration in milliseconds
            
        Returns:
            Audio samples as numpy array
        """
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        # Generate sine wave
        signal = np.sin(2 * np.pi * frequency * t) * self.config.volume
        
        # Apply fade in/out to reduce clicks
        fade_samples = int(self.config.sample_rate * fade_ms / 1000)
        if fade_samples > 0 and fade_samples < num_samples // 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            signal[:fade_samples] *= fade_in
            signal[-fade_samples:] *= fade_out
        
        return signal.astype(np.float32)
    
    def _generate_silence(self, duration_ms: float) -> np.ndarray:
        """Generate silence."""
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        return np.zeros(num_samples, dtype=np.float32)
    
    def generate_start_signal(self) -> np.ndarray:
        """Generate the distinctive start signal using multiple simultaneous frequencies."""
        duration_ms = self.config.signal_duration_ms
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        # Generate multiple simultaneous tones for better detection
        signal = np.zeros(num_samples, dtype=np.float32)
        for freq in self.config.start_frequencies:
            tone = np.sin(2 * np.pi * freq * t)
            signal += tone
        
        # Normalize by number of frequencies and apply volume
        signal = (signal / len(self.config.start_frequencies)) * self.config.volume
        
        # Apply envelope to reduce clicks
        envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.5
        signal *= envelope
        
        return signal.astype(np.float32)
    
    def generate_end_signal(self) -> np.ndarray:
        """Generate the distinctive end signal using multiple simultaneous frequencies."""
        duration_ms = self.config.signal_duration_ms
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        # Generate multiple simultaneous tones for better detection
        signal = np.zeros(num_samples, dtype=np.float32)
        for freq in self.config.end_frequencies:
            tone = np.sin(2 * np.pi * freq * t)
            signal += tone
        
        # Normalize by number of frequencies and apply volume
        signal = (signal / len(self.config.end_frequencies)) * self.config.volume
        
        # Apply envelope to reduce clicks
        envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.5
        signal *= envelope
        
        return signal.astype(np.float32)
    
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
            return (tone1 + tone2) / 2.0
        
        elif self.config.num_channels == 3:
            # Tri-channel: split 8 bits as 3+3+2 bits
            bits_765 = (byte >> 5) & 0x07  # Top 3 bits
            bits_432 = (byte >> 2) & 0x07  # Middle 3 bits
            bits_10 = byte & 0x03          # Bottom 2 bits
            tone1 = self._generate_tone(self._frequencies[0][bits_765], self.config.symbol_duration_ms)
            tone2 = self._generate_tone(self._frequencies[1][bits_432], self.config.symbol_duration_ms)
            tone3 = self._generate_tone(self._frequencies[2][bits_10], self.config.symbol_duration_ms)
            return (tone1 + tone2 + tone3) / 3.0
        
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
            return (tone1 + tone2 + tone3 + tone4) / 4.0
    
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
    ) -> np.ndarray:
        """
        Encode data into audio with start/end signals.
        
        Args:
            data: Data bytes to encode
            signature: Optional party signature
            repetitions: Number of times to repeat the data
            
        Returns:
            Complete audio signal
        """
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
        
        # End signal
        samples.append(self.generate_end_signal())
        
        return np.concatenate(samples)
    
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
            Audio signal
        """
        data = text.encode('utf-8')
        return self.encode_data(data, signature, repetitions)


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
        
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(samples))
        windowed_samples = samples * window
        
        # Use Goertzel algorithm for efficient frequency detection
        correlations = []
        for freq in target_frequencies:
            correlation = self._goertzel(windowed_samples, freq)
            correlations.append(correlation)
        
        correlations = np.array(correlations)
        
        # Normalize correlations
        signal_rms = np.sqrt(np.mean(samples ** 2))
        if signal_rms > 1e-10:
            correlations = correlations / (signal_rms * len(samples))
        
        best_idx = np.argmax(correlations)
        
        # Calculate confidence using ratio of best to second-best
        sorted_corr = np.sort(correlations)[::-1]
        if sorted_corr[0] > 1e-10:
            # Ratio-based confidence is more robust
            if len(sorted_corr) > 1 and sorted_corr[1] > 1e-10:
                # For multi-channel, we expect lower relative power per frequency
                # Adjust confidence threshold accordingly
                ratio = sorted_corr[0] / (sorted_corr[0] + sorted_corr[1])
                # Boost confidence for multi-channel signals
                confidence = min(1.0, ratio * 1.15)
            else:
                confidence = 0.95
        else:
            confidence = 0.0
        
        return best_idx, confidence
    
    def _goertzel(self, samples: np.ndarray, target_freq: float) -> float:
        """
        Goertzel algorithm for efficient single-frequency detection.
        
        Args:
            samples: Audio samples
            target_freq: Target frequency to detect
            
        Returns:
            Magnitude of the target frequency
        """
        n = len(samples)
        if n == 0:
            return 0.0
        
        # More accurate frequency bin calculation
        k = int(0.5 + n * target_freq / self.config.sample_rate)
        w = 2 * np.pi * k / n
        coeff = 2 * np.cos(w)
        
        # Use sine and cosine for better magnitude calculation
        cosine = np.cos(w)
        sine = np.sin(w)
        
        s0, s1, s2 = 0.0, 0.0, 0.0
        for sample in samples:
            s0 = sample + coeff * s1 - s2
            s2 = s1
            s1 = s0
        
        # Calculate real and imaginary parts
        real = s1 - s2 * cosine
        imag = s2 * sine
        
        # Return magnitude
        magnitude = np.sqrt(real * real + imag * imag)
        return magnitude
    
    def detect_start_signal(
        self,
        samples: np.ndarray,
        threshold: float = 0.3,
    ) -> Tuple[bool, int]:
        """
        Detect the start signal in audio using multiple simultaneous frequencies.
        
        Args:
            samples: Audio samples
            threshold: Detection threshold (relative to signal RMS)
            
        Returns:
            Tuple of (detected, sample_position)
        """
        window_size = int(
            self.config.sample_rate * self.config.signal_duration_ms / 1000
        )
        step_size = window_size // 4
        
        for i in range(0, len(samples) - window_size, step_size):
            window = samples[i:i + window_size]
            
            # Calculate window RMS for normalization
            window_rms = np.sqrt(np.mean(window ** 2))
            if window_rms < 0.01:  # Skip silent regions
                continue
            
            # Check for all start frequencies simultaneously present
            detection_count = 0
            magnitudes = []
            
            for freq in self.config.start_frequencies:
                magnitude = self._goertzel(window, freq)
                # Normalize by window length and RMS
                if window_rms > 1e-10:
                    normalized_mag = magnitude / (len(window) * window_rms)
                else:
                    normalized_mag = 0.0
                magnitudes.append(normalized_mag)
                
                # Check if this frequency is present (lower threshold for multi-freq)
                if normalized_mag > threshold:
                    detection_count += 1
            
            # Require majority of frequencies to be detected
            if detection_count >= len(self.config.start_frequencies) * 0.67:
                return True, i + window_size
        
        return False, 0
    
    def detect_end_signal(
        self,
        samples: np.ndarray,
        threshold: float = 0.3,
    ) -> Tuple[bool, int]:
        """
        Detect the end signal in audio using multiple simultaneous frequencies.
        
        Args:
            samples: Audio samples
            threshold: Detection threshold (relative to signal RMS)
            
        Returns:
            Tuple of (detected, sample_position)
        """
        window_size = int(
            self.config.sample_rate * self.config.signal_duration_ms / 1000
        )
        step_size = window_size // 4
        
        for i in range(0, len(samples) - window_size, step_size):
            window = samples[i:i + window_size]
            
            # Calculate window RMS for normalization
            window_rms = np.sqrt(np.mean(window ** 2))
            if window_rms < 0.01:  # Skip silent regions
                continue
            
            # Check for all end frequencies simultaneously present
            detection_count = 0
            magnitudes = []
            
            for freq in self.config.end_frequencies:
                magnitude = self._goertzel(window, freq)
                # Normalize by window length and RMS
                if window_rms > 1e-10:
                    normalized_mag = magnitude / (len(window) * window_rms)
                else:
                    normalized_mag = 0.0
                magnitudes.append(normalized_mag)
                
                # Check if this frequency is present (lower threshold for multi-freq)
                if normalized_mag > threshold:
                    detection_count += 1
            
            # Require majority of frequencies to be detected
            if detection_count >= len(self.config.end_frequencies) * 0.67:
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
        skip = max(1, symbol_samples // 10)
        
        if self.config.num_channels == 1:
            # Single-channel: decode nibbles sequentially
            if len(samples) < symbol_samples * 2:
                return 0, 0.0
            high_samples = samples[skip:symbol_samples - skip]
            high_idx, high_conf = self._detect_frequency(high_samples, self._frequencies[0])
            low_samples = samples[symbol_samples + skip:symbol_samples * 2 - skip]
            low_idx, low_conf = self._detect_frequency(low_samples, self._frequencies[0])
            byte_value = (high_idx << 4) | low_idx
            confidence = np.sqrt(high_conf * low_conf)
            return byte_value, confidence
        
        elif self.config.num_channels == 2:
            # Dual-channel: both nibbles simultaneously
            if len(samples) < symbol_samples:
                return 0, 0.0
            symbol = samples[skip:symbol_samples - skip]
            high_idx, high_conf = self._detect_frequency(symbol, self._frequencies[0])
            low_idx, low_conf = self._detect_frequency(symbol, self._frequencies[1])
            byte_value = (high_idx << 4) | low_idx
            confidence = np.sqrt(high_conf * low_conf)
            return byte_value, confidence
        
        elif self.config.num_channels == 3:
            # Tri-channel: decode 3+3+2 bits
            if len(samples) < symbol_samples:
                return 0, 0.0
            symbol = samples[skip:symbol_samples - skip]
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
        pos = start_pos + int(self.config.sample_rate * self.config.silence_ms / 1000)
        skip = max(1, metadata_symbol_samples // 10)
        
        # Decode 2 metadata bytes
        metadata_bytes = []
        for _ in range(2):
            if pos + metadata_byte_samples > len(samples):
                # Fallback to config defaults
                return self.config.num_channels, int(self.config.symbol_duration_ms), pos
            
            # Decode high nibble
            high_samples = samples[pos + skip:pos + metadata_symbol_samples - skip]
            high_idx, _ = self._detect_frequency(high_samples, temp_frequencies)
            
            # Decode low nibble
            low_samples = samples[pos + metadata_symbol_samples + skip:pos + metadata_byte_samples - skip]
            low_idx, _ = self._detect_frequency(low_samples, temp_frequencies)
            
            byte_val = (high_idx << 4) | low_idx
            metadata_bytes.append(byte_val)
            pos += metadata_byte_samples
        
        num_channels = metadata_bytes[0] if 1 <= metadata_bytes[0] <= 4 else self.config.num_channels
        symbol_duration = metadata_bytes[1] if metadata_bytes[1] > 0 else int(self.config.symbol_duration_ms)
        
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
        if read_metadata:
            # decode_metadata handles skipping initial silence and returns position after metadata
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
        
        # Decode data with repetitions
        all_data = []
        for rep in range(repetitions):
            data_bytes = []
            for _ in range(data_length):
                if pos + byte_samples > len(samples):
                    break
                byte_val, conf = self.decode_byte(samples[pos:pos + byte_samples])
                data_bytes.append(byte_val)
                confidences.append(conf)
                pos += byte_samples
            all_data.append(data_bytes)
            pos += silence_samples
        
        # Vote on data if multiple repetitions
        if len(all_data) > 1 and all(len(d) == data_length for d in all_data):
            final_data = []
            for i in range(data_length):
                # Majority voting
                from collections import Counter
                votes = [d[i] for d in all_data]
                final_data.append(Counter(votes).most_common(1)[0][0])
        elif all_data:
            final_data = all_data[0]
        else:
            final_data = []
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
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
            return text, signature, confidence
        except UnicodeDecodeError:
            return None, signature, confidence
