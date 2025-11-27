# Parallel Decoding Enhancement

## Overview
The decode command now supports multi-threaded speed auto-detection for improved performance.

## Feature Details

### New Option: `--threads`
```bash
muwave decode <file> --threads <N>
```

- **Default**: Uses all available CPU cores
- **Purpose**: Controls the number of parallel workers during speed auto-detection
- **Behavior**: When metadata is present in the audio file, parallel testing is bypassed (fast path)

### Implementation

#### Key Changes
1. **Added `--threads` parameter** to decode command in `muwave/cli.py`
2. **Parallel speed evaluation** using `concurrent.futures.ThreadPoolExecutor`
3. **Preserved metadata detection** - when metadata provides symbol duration, parallel testing is skipped
4. **Thread-safe decoding** - each worker creates independent FSKDemodulator instances

#### Code Structure
```python
# Parallel speed testing block
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_map = {executor.submit(_test_speed, ts): ts for ts in test_speeds}
    for future in concurrent.futures.as_completed(future_map):
        # Process results as they complete
        speed_name, symbol_dur, text, signature, confidence, reps = future.result()
        # Select highest confidence result
```

### Performance Characteristics

#### Metadata Present (Fast Path)
- **Behavior**: Reads metadata header, extracts symbol duration and channel count
- **Speed**: ~2.7 seconds (single decode)
- **Threading**: Not used (bypassed)
- **Use Case**: Modern generated files with standardized metadata header

#### Metadata Absent (Parallel Path)
- **Behavior**: Tests all speed modes (ultra-fast, fast, medium, slow) concurrently
- **Speed**: Reduced from sequential testing (4× evaluations in parallel)
- **Threading**: Utilizes all CPU cores by default
- **Use Case**: Legacy files or files from external sources

### Usage Examples

#### Basic decode (uses CPU cores automatically)
```bash
muwave decode output.wav
```

#### Specify thread count
```bash
muwave decode output.wav --threads 2
```

#### Single-threaded (for comparison)
```bash
muwave decode output.wav --threads 1
```

#### With explicit speed (no auto-detection)
```bash
muwave decode output.wav --speed fast --channels 4
```

### Technical Notes

1. **Independence**: Each speed test is completely independent, making this an ideal parallel workload
2. **Thread Safety**: Each worker creates its own FSKConfig and FSKDemodulator instance
3. **Result Ordering**: Results are processed as they complete (unordered), but highest confidence wins
4. **Error Handling**: Failed decodes return low confidence; best result is still selected

### Testing

You can benchmark parallel performance using the CLI:
- Compare metadata vs explicit speed paths
- Test various thread counts with `--threads`
- Validate correctness with test suite: `pytest tests/`

### Future Enhancements

Potential improvements:
- Early termination when 100% confidence reached
- Adaptive thread pool sizing based on file length
- Caching of partial decode results
- GPU acceleration for Goertzel computation

## Compatibility

- **Backward Compatible**: Works with all existing audio files
- **Default Behavior**: Optimal for most use cases (auto thread count)
- **Override Available**: `--threads` allows manual control
