# muwave Configuration Guide

> [!NOTE]
> All muwave settings are now centralized in `config.yaml` and properly propagated throughout the system. This ensures consistent behavior across all commands and features.

## Configuration Structure

### Audio Settings

```yaml
audio:
  sample_rate: 44100        # Standard audio sample rate (Hz)
  buffer_size: 1024         # Audio buffer size for recording/playback
  volume: 0.8               # Transmission volume (0.0 to 1.0)
  input_device: null        # Input device (null = system default)
  output_device: null       # Output device (null = system default)
```

### Speed Modes

Define transmission speed characteristics:

```yaml
speed:
  mode: medium              # Current mode: slow, medium, fast, ultra-fast
  modes:
    slow:
      symbol_duration_ms: 120
      bandwidth_hz: 200
      description: "Most reliable, lowest speed (8.3 symbols/sec)"
    medium:
      symbol_duration_ms: 60
      bandwidth_hz: 400
      description: "Balanced reliability and speed (16.7 symbols/sec)"
    fast:
      symbol_duration_ms: 35
      bandwidth_hz: 800
      description: "Highest speed, requires good audio (28.6 symbols/sec)"
    ultra-fast:
      symbol_duration_ms: 20
      bandwidth_hz: 1200
      description: "Maximum speed, requires excellent audio (50 symbols/sec)"
```

### Redundancy Modes

Control error correction and retry behavior:

```yaml
redundancy:
  mode: medium              # Current mode: low, medium, high
  modes:
    low:
      repetitions: 1
      error_correction: false
      description: "No redundancy, fastest but least reliable"
    medium:
      repetitions: 2
      error_correction: true
      description: "Basic error correction with 2x repetition"
    high:
      repetitions: 3
      error_correction: true
      description: "Maximum reliability with 3x repetition"
```

### Protocol Settings

Core FSK modulation parameters:

```yaml
protocol:
  # Data encoding frequencies
  base_frequency: 1800            # Base frequency for FSK modulation (Hz)
  frequency_step: 120             # Frequency step between symbols (Hz)
  num_frequencies: 16             # Number of frequency bins (alphabet size)
  
  # Multi-frequency start/end signals
  start_frequencies: [800, 850, 900]     # Multiple simultaneous tones
  end_frequencies: [900, 950, 1000]      # Multiple simultaneous tones
  
  # Timing parameters
  signal_duration_ms: 200         # Start/end signal duration
  silence_ms: 50                  # Silence between signals
  
  # Multi-channel configuration
  channel_spacing: 2400           # Spacing between channel base frequencies (Hz)
  
  # Party identification
  party_id: null                  # Unique party ID (auto-generated if null)
  self_recognition: true          # Enable recognition of own transmissions
```

## Key Features

> [!TIP]
> The configuration system supports hot-reloading for most settings. Changes to `config.yaml` are picked up automatically without restarting muwave.

### 1. Multi-Frequency Start/End Signals

> [!IMPORTANT]
> **Previous**: Single chirp frequency for start/end detection  
> **Current**: Multiple simultaneous frequencies for robust detection

```yaml
start_frequencies: [800, 850, 900]    # 3 simultaneous tones
end_frequencies: [900, 950, 1000]     # 3 simultaneous tones
```

**Benefits**:
- More reliable signal detection in noisy environments
- Better resistance to frequency-selective interference
- Reduced false positives from environmental sounds
- Improved detection with majority voting (67% of frequencies must be present)

### 2. Party ID Configuration

**Location**: `protocol.party_id`

**Behavior**:
- If `null`: Auto-generates unique ID based on UUID + process ID + timestamp
- If specified: Uses the provided ID consistently across sessions

**Usage**:
```yaml
# Auto-generate (default)
protocol:
  party_id: null

# Fixed ID for consistent identity
protocol:
  party_id: "my-device-12345"
```

**Where it's used**:
- Message signature generation
- Self-recognition filtering
- Web interface party identification
- Conversation logging

### 3. Centralized Configuration

All `FSKConfig` instances are now created via `Config.create_fsk_config()`:

```python
# In your code
from muwave.core.config import Config

cfg = Config()
fsk_config = cfg.create_fsk_config(
    symbol_duration_ms=35,   # Optional override
    num_channels=2,          # Optional override
)
```

**Benefits**:
- Single source of truth for all FSK parameters
- Easy testing with different configurations
- Consistent behavior across generate/decode/transmit
- No hardcoded values scattered in codebase

## Configuration Priority

Settings are resolved in the following order:

1. **Command-line arguments** (highest priority)
   ```bash
   muwave generate "text" --speed fast --channels 4 --volume 0.9
   ```

2. **config.yaml values**
   ```yaml
   speed:
     mode: medium
   audio:
     volume: 0.8
   ```

3. **Built-in defaults** (lowest priority)
   - Defined in `Config._get_defaults()`

## Advanced Configuration

### Custom Frequency Bands

For different acoustic environments:

```yaml
protocol:
  # Higher frequencies (better for speakers)
  base_frequency: 2000
  frequency_step: 150
  start_frequencies: [1000, 1050, 1100]
  end_frequencies: [1100, 1150, 1200]
  
  # Lower frequencies (better for long-range)
  base_frequency: 1200
  frequency_step: 80
  start_frequencies: [600, 650, 700]
  end_frequencies: [700, 750, 800]
```

### Multi-Channel Spacing

Adjust channel separation to reduce inter-channel interference:

```yaml
protocol:
  channel_spacing: 3000  # Wider spacing (less interference, more bandwidth)
  channel_spacing: 1800  # Tighter spacing (less bandwidth, potential crosstalk)
```

**Recommendation**: Use 2400 Hz spacing for optimal balance.

## Configuration Files

### Search Order

muwave searches for configuration files in this order:

1. `./config.yaml` (current directory)
2. `./muwave.yaml`
3. `~/.config/muwave/config.yaml` (user config)
4. `/etc/muwave/config.yaml` (system config)

### Specifying Config File

```bash
muwave generate "text" --config /path/to/custom-config.yaml
muwave decode audio.wav --config /path/to/custom-config.yaml
```

## Validation

Configuration values are validated at creation:

- `num_channels`: Must be 1-4
- `volume`: Must be 0.0-1.0
- `start_frequencies`, `end_frequencies`: Must be non-empty lists
- All frequencies: Must be positive values

Invalid configurations will raise descriptive errors.

## Migration Notes

### From Pre-1.0

If upgrading from versions before multi-frequency signals:

**Old config**:
```yaml
protocol:
  start_frequency: 800
  end_frequency: 900
```

**New config**:
```yaml
protocol:
  start_frequencies: [800, 850, 900]
  end_frequencies: [900, 950, 1000]
```

Legacy files with single-frequency signals can still be decoded (backward compatible detection falls back to FFT estimation).

## Best Practices

1. **Start with defaults**: The default configuration is tuned for general use
2. **Test changes**: Use `demo_channels.py` to verify custom configurations
3. **Document modifications**: Add comments explaining why you changed values
4. **Version control**: Track `config.yaml` in your repository
5. **Environment-specific**: Consider separate configs for different deployment environments

## Troubleshooting

### Signal Detection Failures

If start/end signals aren't detected:

1. Check frequency ranges don't overlap with data bands
2. Increase `signal_duration_ms` (more robust detection)
3. Verify `start_frequencies` and `end_frequencies` are well-separated from `base_frequency`

### Low Decode Confidence

If decoding has low confidence:

1. Reduce `frequency_step` (wider symbol separation)
2. Increase `symbol_duration_ms` (more detection time)
3. Enable higher redundancy mode
4. Reduce `num_channels` (less interference)

### Audio Quality Issues

If audio sounds distorted or clipped:

1. Reduce `volume` (< 0.8)
2. Check `sample_rate` matches your audio device
3. Verify frequency ranges are within device capabilities

## See Also

- [README.md](README.md) - General usage guide
- [PARALLEL_DECODE.md](PARALLEL_DECODE.md) - Parallel decoding feature
- [config.yaml](config.yaml) - Default configuration file
