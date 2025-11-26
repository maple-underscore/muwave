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
    start_frequency: float = 800.0
    end_frequency: float = 900.0
    signal_duration_ms: float = 200.0
    silence_ms: float = 50.0
    volume: float = 0.8


class FSKModulator:
    """
    FSK modulator for encoding data into audio.
    
    Uses multiple frequencies to encode data, similar to ggwave/gibberlink.
    Each symbol is represented by a specific frequency.
    """
    
    def __init__(self, config: Optional[FSKConfig] = None):
        """
        Initialize the FSK modulator.
        
        Args:
            config: FSK configuration. Uses defaults if None.
        """
        self.config = config or FSKConfig()
        self._frequencies = self._generate_frequencies()
    
    def _generate_frequencies(self) -> np.ndarray:
        """Generate the frequency table for symbols."""
        return np.array([
            self.config.base_frequency + i * self.config.frequency_step
            for i in range(self.config.num_frequencies)
        ])
    
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
        """Generate the distinctive start signal."""
        # Rising chirp for start
        duration_ms = self.config.signal_duration_ms
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        # Chirp from start_frequency up
        start_freq = self.config.start_frequency
        end_freq = start_freq + 300
        freq = start_freq + (end_freq - start_freq) * (t / (duration_ms / 1000))
        signal = np.sin(2 * np.pi * freq * t) * self.config.volume
        
        # Apply envelope
        envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.5
        signal *= envelope
        
        return signal.astype(np.float32)
    
    def generate_end_signal(self) -> np.ndarray:
        """Generate the distinctive end signal."""
        # Falling chirp for end
        duration_ms = self.config.signal_duration_ms
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        # Chirp from end_frequency down
        start_freq = self.config.end_frequency + 300
        end_freq = self.config.end_frequency
        freq = start_freq + (end_freq - start_freq) * (t / (duration_ms / 1000))
        signal = np.sin(2 * np.pi * freq * t) * self.config.volume
        
        # Apply envelope
        envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.5
        signal *= envelope
        
        return signal.astype(np.float32)
    
    def encode_byte(self, byte: int) -> np.ndarray:
        """
        Encode a single byte into audio.
        
        Args:
            byte: Byte value (0-255)
            
        Returns:
            Audio samples for the byte
        """
        # Split byte into two nibbles (4 bits each)
        high_nibble = (byte >> 4) & 0x0F
        low_nibble = byte & 0x0F
        
        # Generate tones for each nibble
        high_tone = self._generate_tone(
            self._frequencies[high_nibble],
            self.config.symbol_duration_ms,
        )
        low_tone = self._generate_tone(
            self._frequencies[low_nibble],
            self.config.symbol_duration_ms,
        )
        
        return np.concatenate([high_tone, low_tone])
    
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
    
    Uses frequency detection to decode symbols from audio.
    """
    
    def __init__(self, config: Optional[FSKConfig] = None):
        """
        Initialize the FSK demodulator.
        
        Args:
            config: FSK configuration. Uses defaults if None.
        """
        self.config = config or FSKConfig()
        self._frequencies = self._generate_frequencies()
    
    def _generate_frequencies(self) -> np.ndarray:
        """Generate the frequency table for symbols."""
        return np.array([
            self.config.base_frequency + i * self.config.frequency_step
            for i in range(self.config.num_frequencies)
        ])
    
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
        
        # Normalize correlations by RMS of input signal
        signal_rms = np.sqrt(np.mean(samples ** 2))
        if signal_rms > 1e-10:
            correlations = correlations / (signal_rms * len(samples))
        
        best_idx = np.argmax(correlations)
        
        # Calculate confidence using ratio of best to second-best
        sorted_corr = np.sort(correlations)[::-1]
        if sorted_corr[0] > 1e-10:
            # Ratio-based confidence is more robust
            if len(sorted_corr) > 1 and sorted_corr[1] > 1e-10:
                confidence = min(1.0, sorted_corr[0] / (sorted_corr[0] + sorted_corr[1]))
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
        threshold: float = 0.5,
    ) -> Tuple[bool, int]:
        """
        Detect the start signal in audio.
        
        Args:
            samples: Audio samples
            threshold: Detection threshold
            
        Returns:
            Tuple of (detected, sample_position)
        """
        window_size = int(
            self.config.sample_rate * self.config.signal_duration_ms / 1000
        )
        step_size = window_size // 4
        
        for i in range(0, len(samples) - window_size, step_size):
            window = samples[i:i + window_size]
            
            # Check for rising chirp pattern
            half = len(window) // 2
            first_half = window[:half]
            second_half = window[half:]
            
            # Detect frequency in each half
            freq1 = self._estimate_frequency(first_half)
            freq2 = self._estimate_frequency(second_half)
            
            # Rising chirp should have increasing frequency
            if freq1 > 0 and freq2 > freq1:
                start_range = (
                    self.config.start_frequency - 100,
                    self.config.start_frequency + 400,
                )
                if start_range[0] < freq1 < start_range[1]:
                    return True, i + window_size
        
        return False, 0
    
    def detect_end_signal(
        self,
        samples: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[bool, int]:
        """
        Detect the end signal in audio.
        
        Args:
            samples: Audio samples
            threshold: Detection threshold
            
        Returns:
            Tuple of (detected, sample_position)
        """
        window_size = int(
            self.config.sample_rate * self.config.signal_duration_ms / 1000
        )
        step_size = window_size // 4
        
        for i in range(0, len(samples) - window_size, step_size):
            window = samples[i:i + window_size]
            
            # Check for falling chirp pattern
            half = len(window) // 2
            first_half = window[:half]
            second_half = window[half:]
            
            freq1 = self._estimate_frequency(first_half)
            freq2 = self._estimate_frequency(second_half)
            
            # Falling chirp should have decreasing frequency
            if freq1 > 0 and freq2 < freq1:
                end_range = (
                    self.config.end_frequency,
                    self.config.end_frequency + 400,
                )
                if end_range[0] < freq2 < end_range[1]:
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
        Decode a byte from audio samples.
        
        Args:
            samples: Audio samples for one byte (two symbols)
            
        Returns:
            Tuple of (byte value, average confidence)
        """
        symbol_samples = int(
            self.config.sample_rate * self.config.symbol_duration_ms / 1000
        )
        
        if len(samples) < symbol_samples * 2:
            return 0, 0.0
        
        # Decode high nibble - use middle portion of symbol for better accuracy
        # Skip first and last 10% to avoid transients
        skip = max(1, symbol_samples // 10)
        high_samples = samples[skip:symbol_samples - skip]
        high_idx, high_conf = self._detect_frequency(high_samples, self._frequencies)
        
        # Decode low nibble
        low_samples = samples[symbol_samples + skip:symbol_samples * 2 - skip]
        low_idx, low_conf = self._detect_frequency(low_samples, self._frequencies)
        
        byte_value = (high_idx << 4) | low_idx
        # Use geometric mean for confidence (more conservative)
        confidence = np.sqrt(high_conf * low_conf)
        
        return byte_value, confidence
    
    def decode_data(
        self,
        samples: np.ndarray,
        signature_length: int = 8,
        repetitions: int = 1,
    ) -> Tuple[Optional[bytes], Optional[bytes], float]:
        """
        Decode data from audio samples.
        
        Args:
            samples: Audio samples (should be between start and end signals)
            signature_length: Expected signature length in bytes
            repetitions: Number of repetitions for redundancy
            
        Returns:
            Tuple of (data bytes, signature bytes, confidence)
        """
        symbol_samples = int(
            self.config.sample_rate * self.config.symbol_duration_ms / 1000
        )
        byte_samples = symbol_samples * 2
        silence_samples = int(
            self.config.sample_rate * self.config.silence_ms / 2000
        )
        
        # Start after the initial silence that follows the start signal
        pos = int(self.config.sample_rate * self.config.silence_ms / 1000)
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
    ) -> Tuple[Optional[str], Optional[bytes], float]:
        """
        Decode text from audio samples.
        
        Args:
            samples: Audio samples
            signature_length: Expected signature length
            repetitions: Number of repetitions
            
        Returns:
            Tuple of (decoded text, signature, confidence)
        """
        data, signature, confidence = self.decode_data(
            samples, signature_length, repetitions
        )
        
        if data is None:
            return None, signature, confidence
        
        try:
            text = data.decode('utf-8')
            return text, signature, confidence
        except UnicodeDecodeError:
            return None, signature, confidence
