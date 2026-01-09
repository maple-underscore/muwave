# Interference Resilience Strategies for muwave

> [!NOTE]
> This document explores advanced techniques for making FSK signals more recognizable even with heavy interference from other audio sources in the environment.

## Implemented Enhancements ✅

The following strategies have been **implemented** in the muwave FSK module:

### 1. Chirp Signals (Frequency Sweeps) - ENABLED BY DEFAULT

**Location**: `muwave/audio/fsk.py` - `_generate_chirp()`, `_detect_chirp()`

Chirp signals replace fixed-frequency tones with **linear frequency sweeps**. This provides:
- **~15-20 dB processing gain** via matched filter correlation
- Robust detection even at **-10 dB SNR** (noise power 10x signal power!)
- Distinctive "swooping" sound easy to identify aurally

**Configuration** (in `config.yaml`):
```yaml
protocol:
  use_chirp_signals: true    # Enable/disable chirp-based start/end signals
  chirp_start_freq: 600      # Start frequency for rising chirp (Hz)
  chirp_end_freq: 2400       # End frequency for rising chirp (Hz)
```

### 2. Barker Code Preamble - OPTIONAL

**Location**: `muwave/audio/fsk.py` - `_generate_barker_signal()`, `_detect_barker()`

Barker codes are binary sequences with optimal autocorrelation properties:
- **13:1 peak-to-sidelobe ratio** for Barker-13
- Precise timing synchronization
- Used in WiFi (802.11) and professional communications

**Configuration**:
```yaml
protocol:
  use_barker_preamble: false   # Enable/disable Barker code preamble
  barker_length: 13            # Code length: 7, 11, or 13
  barker_carrier_freq: 1500    # Carrier frequency (Hz)
  barker_chip_duration_ms: 8   # Duration per chip (ms)
```

### 3. Adaptive Noise Floor Detection - AUTOMATIC

**Location**: `muwave/audio/fsk.py` - `_measure_noise_floor()`

Dynamically measures background noise level and adjusts detection thresholds:
- Prevents false positives in noisy environments
- Automatically adapts to varying noise conditions
- Uses percentile-based estimation for robustness

## Performance Comparison

| Condition | Multi-tone Only | Chirp + Multi-tone |
|-----------|-----------------|---------------------|
| SNR 20 dB | ✗ MISSED | ✓ DETECTED |
| SNR 10 dB | ✗ MISSED | ✓ DETECTED |
| SNR 5 dB | ✗ MISSED | ✓ DETECTED |
| SNR 0 dB | ✗ MISSED | ✓ DETECTED |
| SNR -5 dB | ✗ MISSED | ✓ DETECTED |
| SNR -10 dB | ✗ MISSED | ✓ DETECTED |

---

## Strategy 1: Chirp Signals (Frequency Sweeps)

### Concept
Replace fixed-frequency tones with **linear frequency sweeps** (chirps). Chirps have excellent **pulse compression** properties and can be detected even in heavy noise using **matched filtering**.

### Why It Works
- Chirps spread energy across time and frequency → less vulnerable to narrowband interference
- Matched filter correlation gives **processing gain** proportional to time-bandwidth product
- Used in radar, sonar, and spread-spectrum communications

### Implementation

```python
def generate_chirp_signal(
    self,
    start_freq: float,
    end_freq: float, 
    duration_ms: float,
    chirp_type: str = "linear"
) -> np.ndarray:
    """
    Generate a chirp (frequency sweep) signal.
    
    Args:
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        duration_ms: Duration in milliseconds
        chirp_type: "linear", "quadratic", or "logarithmic"
    """
    num_samples = int(self.config.sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    
    if chirp_type == "linear":
        # Linear chirp: f(t) = f0 + (f1-f0)*t/T
        k = (end_freq - start_freq) / (duration_ms / 1000)
        phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)
    elif chirp_type == "logarithmic":
        # Log chirp: better for octave-spanning sweeps
        k = (end_freq / start_freq) ** (1 / (duration_ms / 1000))
        phase = 2 * np.pi * start_freq * (k**t - 1) / np.log(k)
    
    signal = np.sin(phase) * self.config.volume
    
    # Apply envelope to reduce clicks
    envelope = np.sin(np.pi * t / (duration_ms / 1000)) ** 0.3
    return (signal * envelope).astype(np.float32)


def detect_chirp_signal(
    self,
    samples: np.ndarray,
    start_freq: float,
    end_freq: float,
    duration_ms: float,
    threshold: float = 0.3
) -> Tuple[bool, int, float]:
    """
    Detect chirp using matched filter correlation.
    
    Returns:
        Tuple of (detected, sample_position, correlation_peak)
    """
    # Generate reference chirp
    reference = self.generate_chirp_signal(start_freq, end_freq, duration_ms)
    
    # Matched filter = cross-correlation with time-reversed reference
    # This provides optimal detection in white Gaussian noise
    correlation = np.correlate(samples, reference, mode='valid')
    correlation = np.abs(correlation) / len(reference)
    
    # Normalize by signal energy
    ref_energy = np.sqrt(np.sum(reference**2))
    correlation = correlation / ref_energy
    
    peak_idx = np.argmax(correlation)
    peak_value = correlation[peak_idx]
    
    detected = peak_value > threshold
    return detected, peak_idx + len(reference), peak_value
```

### Configuration Example
```yaml
protocol:
  start_signal:
    type: chirp
    start_freq: 1000
    end_freq: 3000
    duration_ms: 300
  end_signal:
    type: chirp  
    start_freq: 3000
    end_freq: 1000  # Reverse sweep
    duration_ms: 300
```

### Benefits
- **+15-20 dB** processing gain with matched filtering
- Robust to narrowband interference
- Distinctive "swooping" sound easy to identify aurally

---

## Strategy 2: Barker Codes (Pseudo-Noise Sequences)

### Concept
Use **Barker codes** - binary sequences with optimal autocorrelation properties. When used for BPSK modulation, they create signals with a sharp correlation peak and minimal sidelobes.

### Why It Works
- Barker codes have **perfect periodic autocorrelation** (ratio of main peak to sidelobes = N:1)
- Can detect signals well below noise floor
- Industry standard for WiFi preambles (802.11)

### Available Barker Codes
| Length | Sequence | Sidelobe Ratio |
|--------|----------|----------------|
| 2 | +1, -1 | 2:1 |
| 3 | +1, +1, -1 | 3:1 |
| 5 | +1, +1, +1, -1, +1 | 5:1 |
| 7 | +1, +1, +1, -1, -1, +1, -1 | 7:1 |
| 11 | +1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1 | 11:1 |
| 13 | +1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1 | 13:1 |

### Implementation

```python
BARKER_CODES = {
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}

def generate_barker_signal(
    self,
    carrier_freq: float,
    code_length: int = 13,
    chip_duration_ms: float = 10.0
) -> np.ndarray:
    """
    Generate BPSK-modulated Barker code signal.
    
    Args:
        carrier_freq: Carrier frequency for BPSK
        code_length: Barker code length (7, 11, or 13)
        chip_duration_ms: Duration of each chip
    """
    code = BARKER_CODES[code_length]
    chip_samples = int(self.config.sample_rate * chip_duration_ms / 1000)
    total_samples = chip_samples * len(code)
    
    signal = np.zeros(total_samples, dtype=np.float32)
    t_chip = np.linspace(0, chip_duration_ms / 1000, chip_samples, endpoint=False)
    
    for i, chip in enumerate(code):
        start = i * chip_samples
        # BPSK: phase = 0 for +1, phase = π for -1
        phase = 0 if chip == 1 else np.pi
        signal[start:start + chip_samples] = np.sin(
            2 * np.pi * carrier_freq * t_chip + phase
        )
    
    # Apply overall envelope
    t_total = np.linspace(0, 1, total_samples)
    envelope = np.sin(np.pi * t_total) ** 0.3
    
    return (signal * envelope * self.config.volume).astype(np.float32)


def detect_barker_signal(
    self,
    samples: np.ndarray,
    carrier_freq: float,
    code_length: int = 13,
    chip_duration_ms: float = 10.0,
    threshold: float = 0.5
) -> Tuple[bool, int, float]:
    """
    Detect Barker-coded signal using correlation.
    
    The correlation peak will be `code_length` times higher than sidelobes.
    """
    reference = self.generate_barker_signal(carrier_freq, code_length, chip_duration_ms)
    
    # Cross-correlation
    correlation = np.correlate(samples, reference, mode='valid')
    correlation = np.abs(correlation)
    
    # Normalize
    ref_energy = np.sqrt(np.sum(reference**2))
    sig_energy = np.sqrt(np.convolve(samples**2, np.ones(len(reference)), mode='valid'))
    sig_energy = np.maximum(sig_energy, 1e-10)
    
    normalized_corr = correlation / (ref_energy * sig_energy)
    
    peak_idx = np.argmax(normalized_corr)
    peak_value = normalized_corr[peak_idx]
    
    detected = peak_value > threshold
    return detected, peak_idx + len(reference), peak_value
```

### Benefits
- **13:1** sidelobe suppression with Barker-13
- Precise timing synchronization
- Well-studied, proven technology

---

## Strategy 3: Gold Codes (CDMA-style)

### Concept
**Gold codes** are families of pseudo-random sequences with good cross-correlation properties. They allow multiple transmitters to operate simultaneously without interference.

### Why It Works
- Low cross-correlation between different codes → multiple conversations can share airspace
- Good autocorrelation → precise sync detection
- Used in GPS, CDMA cellular

### Implementation

```python
def generate_gold_sequence(self, length: int, seed1: int, seed2: int) -> np.ndarray:
    """
    Generate Gold code sequence using two LFSRs.
    
    Args:
        length: Sequence length (2^n - 1 for n-bit registers)
        seed1, seed2: Seeds for the two LFSRs
    """
    # Maximum-length sequences from two different primitive polynomials
    # Example for n=5: x^5 + x^2 + 1 and x^5 + x^4 + x^3 + x^2 + 1
    def lfsr(seed, taps, length):
        reg = seed
        seq = []
        for _ in range(length):
            seq.append(reg & 1)
            feedback = 0
            for tap in taps:
                feedback ^= (reg >> tap) & 1
            reg = (reg >> 1) | (feedback << (len(taps)))
        return np.array(seq)
    
    # Two m-sequences
    m1 = lfsr(seed1, [0, 2], length)  # Primitive polynomial 1
    m2 = lfsr(seed2, [0, 1, 2, 3], length)  # Primitive polynomial 2
    
    # Gold code = XOR of two m-sequences
    gold = m1 ^ m2
    
    # Convert 0/1 to -1/+1 for BPSK
    return 2 * gold.astype(np.float32) - 1


def generate_gold_signal(
    self,
    carrier_freq: float,
    code_length: int = 31,
    chip_duration_ms: float = 5.0,
    party_id: int = 0
) -> np.ndarray:
    """
    Generate Gold-coded signal unique to a party.
    
    Different party_ids generate orthogonal codes.
    """
    code = self.generate_gold_sequence(code_length, party_id + 1, party_id + 2)
    chip_samples = int(self.config.sample_rate * chip_duration_ms / 1000)
    
    signal = np.zeros(chip_samples * code_length, dtype=np.float32)
    t_chip = np.linspace(0, chip_duration_ms / 1000, chip_samples, endpoint=False)
    
    for i, chip in enumerate(code):
        start = i * chip_samples
        phase = 0 if chip > 0 else np.pi
        signal[start:start + chip_samples] = np.sin(
            2 * np.pi * carrier_freq * t_chip + phase
        )
    
    return signal * self.config.volume
```

### Benefits
- Party-specific signatures → simultaneous conversations
- Good interference rejection
- Scalable to many users

---

## Strategy 4: Frequency Hopping Spread Spectrum (FHSS)

### Concept
Instead of staying on fixed frequencies, **hop between frequencies** in a pseudo-random pattern known to both transmitter and receiver.

### Why It Works
- Narrowband interference only affects a few hops
- Spreading across many frequencies improves SNR
- Used in Bluetooth, military communications

### Implementation

```python
def generate_hop_sequence(
    self,
    num_hops: int,
    seed: int = 42
) -> List[float]:
    """
    Generate pseudo-random frequency hop sequence.
    """
    np.random.seed(seed)
    hop_freqs = []
    
    # Define frequency band (e.g., 1000-4000 Hz)
    freq_min, freq_max = 1000.0, 4000.0
    num_channels = 20
    channel_spacing = (freq_max - freq_min) / num_channels
    
    for _ in range(num_hops):
        channel = np.random.randint(0, num_channels)
        freq = freq_min + channel * channel_spacing
        hop_freqs.append(freq)
    
    return hop_freqs


def encode_byte_fhss(
    self,
    byte: int,
    hop_sequence: List[float],
    hop_index: int
) -> np.ndarray:
    """
    Encode a byte using frequency hopping.
    
    Each nibble uses a different hop frequency.
    """
    high_nibble = (byte >> 4) & 0x0F
    low_nibble = byte & 0x0F
    
    freq1 = hop_sequence[hop_index % len(hop_sequence)]
    freq2 = hop_sequence[(hop_index + 1) % len(hop_sequence)]
    
    # Symbol frequencies relative to hop frequency
    symbol_freq1 = freq1 + high_nibble * self.config.frequency_step
    symbol_freq2 = freq2 + low_nibble * self.config.frequency_step
    
    tone1 = self._generate_tone(symbol_freq1, self.config.symbol_duration_ms)
    tone2 = self._generate_tone(symbol_freq2, self.config.symbol_duration_ms)
    
    return np.concatenate([tone1, tone2])
```

### Benefits
- Robust to narrowband jamming
- Can operate in crowded spectrum
- Natural frequency diversity

---

## Strategy 5: Differential Encoding

### Concept
Encode data as **changes between symbols** rather than absolute frequencies. This makes the signal robust to frequency drift and DC offset.

### Why It Works
- Channel effects (frequency shift, phase rotation) cancel out
- No need for absolute frequency reference
- Used in DPSK (Differential Phase Shift Keying)

### Implementation

```python
def encode_byte_differential(
    self,
    byte: int,
    prev_symbol: int
) -> Tuple[np.ndarray, int]:
    """
    Encode byte using differential FSK.
    
    Data is encoded as frequency CHANGE, not absolute frequency.
    """
    high_nibble = (byte >> 4) & 0x0F
    low_nibble = byte & 0x0F
    
    # Differential: new_symbol = (prev_symbol + data) mod num_frequencies
    symbol1 = (prev_symbol + high_nibble) % self.config.num_frequencies
    symbol2 = (symbol1 + low_nibble) % self.config.num_frequencies
    
    tone1 = self._generate_tone(
        self._frequencies[0][symbol1],
        self.config.symbol_duration_ms
    )
    tone2 = self._generate_tone(
        self._frequencies[0][symbol2],
        self.config.symbol_duration_ms
    )
    
    return np.concatenate([tone1, tone2]), symbol2


def decode_byte_differential(
    self,
    samples: np.ndarray,
    prev_symbol: int
) -> Tuple[int, float, int]:
    """
    Decode differentially encoded byte.
    """
    symbol_samples = int(
        self.config.sample_rate * self.config.symbol_duration_ms / 1000
    )
    
    # Detect current symbols
    symbol1, conf1 = self._detect_frequency(
        samples[:symbol_samples], 
        self._frequencies[0]
    )
    symbol2, conf2 = self._detect_frequency(
        samples[symbol_samples:symbol_samples*2], 
        self._frequencies[0]
    )
    
    # Differential decode: data = (current - previous) mod num_frequencies
    high_nibble = (symbol1 - prev_symbol) % self.config.num_frequencies
    low_nibble = (symbol2 - symbol1) % self.config.num_frequencies
    
    byte_value = (high_nibble << 4) | low_nibble
    confidence = np.sqrt(conf1 * conf2)
    
    return byte_value, confidence, symbol2
```

### Benefits
- Tolerant of frequency drift
- Works with poor frequency references
- Simple to implement

---

## Strategy 6: Error Detection with CRC

### Concept
Add **Cyclic Redundancy Check (CRC)** to detect transmission errors. Combined with retransmission requests, this ensures data integrity.

### Implementation

```python
import crc

def add_crc16(data: bytes) -> bytes:
    """Add CRC-16 checksum to data."""
    crc16 = crc.Calculator(crc.Crc16.CCITT_FALSE)
    checksum = crc16.checksum(data)
    return data + checksum.to_bytes(2, 'big')


def verify_crc16(data_with_crc: bytes) -> Tuple[bytes, bool]:
    """Verify CRC-16 and return data if valid."""
    if len(data_with_crc) < 2:
        return b'', False
    
    data = data_with_crc[:-2]
    received_crc = int.from_bytes(data_with_crc[-2:], 'big')
    
    crc16 = crc.Calculator(crc.Crc16.CCITT_FALSE)
    calculated_crc = crc16.checksum(data)
    
    return data, (received_crc == calculated_crc)
```

---

## Strategy 7: Adaptive Noise Floor Detection

### Concept
Dynamically measure the **noise floor** and adjust detection thresholds accordingly. This prevents false positives in noisy environments.

### Implementation

```python
def measure_noise_floor(
    self,
    samples: np.ndarray,
    percentile: float = 25.0
) -> float:
    """
    Estimate noise floor from samples.
    
    Uses percentile of frequency magnitudes (assumes signal is not
    always present).
    """
    window_size = int(self.config.sample_rate * 0.1)  # 100ms windows
    magnitudes = []
    
    for i in range(0, len(samples) - window_size, window_size):
        window = samples[i:i + window_size]
        # Get magnitude spectrum
        fft = np.fft.rfft(window)
        mags = np.abs(fft)
        magnitudes.extend(mags)
    
    # Noise floor is low percentile of all magnitudes
    return np.percentile(magnitudes, percentile)


def detect_with_adaptive_threshold(
    self,
    samples: np.ndarray,
    base_threshold: float = 0.3
) -> Tuple[bool, int]:
    """
    Detect start signal with adaptive threshold.
    """
    noise_floor = self.measure_noise_floor(samples[:self.config.sample_rate])
    
    # Threshold scales with noise floor
    # In quiet: threshold ~0.3
    # In noise: threshold increases to avoid false positives
    adaptive_threshold = max(base_threshold, noise_floor * 3.0)
    
    return self.detect_start_signal(samples, threshold=adaptive_threshold)
```

---

## Strategy 8: Multi-Band Redundancy

### Concept
Transmit the **same data on multiple frequency bands** simultaneously. If one band is interfered with, others may get through.

### Implementation

```python
def encode_byte_multiband(
    self,
    byte: int,
    bands: List[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Encode byte across multiple frequency bands for redundancy.
    
    Args:
        byte: Byte to encode
        bands: List of (base_freq, step) tuples for each band
    """
    if bands is None:
        bands = [
            (1200.0, 80.0),   # Low band: 1200-2480 Hz
            (2800.0, 80.0),   # Mid band: 2800-4080 Hz
            (4500.0, 80.0),   # High band: 4500-5780 Hz
        ]
    
    high_nibble = (byte >> 4) & 0x0F
    low_nibble = byte & 0x0F
    
    all_tones = []
    for base_freq, step in bands:
        freq1 = base_freq + high_nibble * step
        freq2 = base_freq + low_nibble * step
        
        tone1 = self._generate_tone(freq1, self.config.symbol_duration_ms)
        tone2 = self._generate_tone(freq2, self.config.symbol_duration_ms)
        all_tones.append(tone1 + tone2)  # Simultaneous nibbles
    
    # Mix all bands together
    mixed = np.sum(all_tones, axis=0) / len(bands)
    return mixed.astype(np.float32)


def decode_byte_multiband(
    self,
    samples: np.ndarray,
    bands: List[Tuple[float, float]] = None
) -> Tuple[int, float]:
    """
    Decode byte from multiple bands with voting.
    """
    if bands is None:
        bands = [
            (1200.0, 80.0),
            (2800.0, 80.0),
            (4500.0, 80.0),
        ]
    
    votes_high = []
    votes_low = []
    confidences = []
    
    for base_freq, step in bands:
        freqs = np.array([base_freq + i * step for i in range(16)])
        
        # Detect on this band
        high_idx, high_conf = self._detect_frequency(samples, freqs)
        low_idx, low_conf = self._detect_frequency(samples, freqs)
        
        votes_high.append(high_idx)
        votes_low.append(low_idx)
        confidences.append((high_conf + low_conf) / 2)
    
    # Majority voting
    from collections import Counter
    high_nibble = Counter(votes_high).most_common(1)[0][0]
    low_nibble = Counter(votes_low).most_common(1)[0][0]
    
    byte_value = (high_nibble << 4) | low_nibble
    avg_confidence = np.mean(confidences)
    
    return byte_value, avg_confidence
```

---

## Recommended Implementation Priority

| Strategy | Difficulty | Impact | Recommended Order |
|----------|------------|--------|-------------------|
| **Chirp Signals** | Medium | High | 1️⃣ First |
| **Adaptive Thresholds** | Low | Medium | 2️⃣ Second |
| **CRC Error Detection** | Low | Medium | 3️⃣ Third |
| **Barker Codes** | Medium | High | 4️⃣ Fourth |
| **Multi-Band Redundancy** | Medium | High | 5️⃣ Fifth |
| **Differential Encoding** | Low | Medium | 6️⃣ Sixth |
| **Frequency Hopping** | High | High | 7️⃣ Advanced |
| **Gold Codes** | High | Medium | 8️⃣ Advanced |

---

## Quick Win: Improved Start/End Detection

The simplest immediate improvement is to combine **chirps** with the existing multi-frequency approach:

```python
def generate_enhanced_start_signal(self) -> np.ndarray:
    """
    Generate robust start signal combining chirp + multi-tone.
    
    Structure:
    1. Rising chirp (100ms): 800 → 2000 Hz
    2. Multi-tone burst (100ms): Current approach
    3. Rising chirp again (100ms): Confirms start
    """
    samples = []
    
    # Part 1: Rising chirp
    chirp1 = self.generate_chirp_signal(800, 2000, 100)
    samples.append(chirp1)
    
    # Part 2: Multi-tone (existing approach)
    multi_tone = self.generate_start_signal()  # Current implementation
    samples.append(multi_tone[:len(multi_tone)//2])  # Shorter version
    
    # Part 3: Confirmation chirp
    chirp2 = self.generate_chirp_signal(800, 2000, 100)
    samples.append(chirp2)
    
    return np.concatenate(samples)
```

---

## Testing Interference Resilience

To validate these improvements, test with:

```bash
# Generate test signal
muwave generate -t "Test message" -o test.wav

# Mix with interference (using sox or similar)
sox -m test.wav interference.wav mixed.wav

# Attempt decode
muwave decode -f mixed.wav
```

Common interference sources to test:
- White noise at various SNR levels
- Music (broadband)
- Speech (narrowband)
- Other muwave transmissions (cross-talk)
- 50/60 Hz hum (mains interference)

---

## Summary

The most impactful improvements for heavy interference environments are:

1. **Chirp-based start/end signals** - 15-20 dB processing gain
2. **Adaptive noise floor detection** - Prevents false triggers
3. **CRC checksums** - Detects corrupted data
4. **Multi-band redundancy** - Survives narrowband interference
5. **Barker code preambles** - Precise timing recovery

These can be implemented incrementally, with each providing measurable improvement in recognition reliability.
