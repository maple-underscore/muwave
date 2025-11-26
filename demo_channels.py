#!/usr/bin/env python3
"""
Demonstration script comparing 1, 2, 3, and 4 channel FSK performance.
Shows bitrate improvements and confidence trade-offs.
"""

import time
import os
import tempfile
from muwave.audio.fsk import FSKModulator, FSKDemodulator, FSKConfig
from muwave.core.party import Party

def test_channel_mode(num_channels: int, test_message: str, party_sig: bytes) -> dict:
    """Test a specific channel mode and return performance metrics."""
    print(f"\n{'='*70}")
    print(f"Testing {num_channels}-Channel Mode")
    print(f"{'='*70}")
    
    # Create FSK configuration
    config = FSKConfig(
        sample_rate=44100,
        base_frequency=1800,
        frequency_step=120,
        num_frequencies=16,
        symbol_duration_ms=60,  # medium speed
        start_frequency=800,
        end_frequency=900,
        signal_duration_ms=200,
        silence_ms=50,
        volume=0.8,
        num_channels=num_channels,
    )
    
    # Encode
    print(f"Encoding message: '{test_message}'")
    print(f"Message length: {len(test_message)} characters, {len(test_message.encode('utf-8'))} bytes")
    
    modulator = FSKModulator(config)
    start_time = time.time()
    audio_samples = modulator.encode_text(test_message, signature=party_sig, repetitions=1)
    encode_time = time.time() - start_time
    
    # Calculate audio duration
    audio_duration = len(audio_samples) / config.sample_rate
    
    # Calculate effective bitrate
    data_bits = len(test_message.encode('utf-8')) * 8
    bitrate = data_bits / audio_duration
    
    print(f"\nEncoding completed in {encode_time:.3f}s")
    print(f"Audio samples: {len(audio_samples):,}")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Effective bitrate: {bitrate:.1f} bits/second")
    
    # Decode
    print(f"\nDecoding...")
    demodulator = FSKDemodulator(config)
    
    # Extract data region starting from start signal
    # Note: Not using end signal detection as it may prematurely cut off data
    start_detected, start_pos = demodulator.detect_start_signal(audio_samples)
    if not start_detected:
        print("ERROR: Start signal not detected!")
        return None
    
    # Use all samples after start signal
    message_samples = audio_samples[start_pos:]
    
    start_time = time.time()
    decoded_text, detected_sig, confidence = demodulator.decode_text(message_samples)
    decode_time = time.time() - start_time
    
    # Verify
    success = decoded_text == test_message if decoded_text else False
    sig_match = detected_sig == party_sig if detected_sig else False
    
    print(f"Decoding completed in {decode_time:.3f}s")
    print(f"Decoded text: '{decoded_text}'")
    print(f"Confidence: {confidence:.2%}")
    print(f"Signature match: {sig_match}")
    print(f"Text match: {success}")
    
    # Calculate throughput
    if success:
        throughput = len(test_message) / audio_duration
        print(f"Throughput: {throughput:.1f} characters/second")
    
    return {
        'channels': num_channels,
        'encode_time': encode_time,
        'decode_time': decode_time,
        'audio_duration': audio_duration,
        'audio_samples': len(audio_samples),
        'bitrate': bitrate,
        'confidence': confidence,
        'success': success,
        'signature_match': sig_match,
        'throughput': throughput if success else 0,
    }

def main():
    """Run the demonstration comparing all channel modes."""
    print("="*70)
    print("Multi-Channel FSK Performance Demonstration")
    print("="*70)
    print("\nThis demonstration compares 1, 2, 3, and 4 channel FSK modes.")
    print("Expected results:")
    print("  - Higher channel counts = faster bitrate")
    print("  - Potential trade-off in accuracy/confidence")
    print("  - 2-channel should be optimal balance")
    
    # Test message
    test_message = "Hello World! This is a test of multi-channel FSK transmission."
    
    # Create a party signature
    party = Party("TestParty")
    party_sig = party.signature
    
    # Test all channel modes
    results = []
    for num_channels in [1, 2, 3, 4]:
        try:
            result = test_channel_mode(num_channels, test_message, party_sig)
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {num_channels}-channel mode: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Channels':<10} {'Duration':<12} {'Bitrate':<15} {'Confidence':<12} {'Success':<10}")
    print("-"*70)
    
    for result in results:
        print(f"{result['channels']:<10} "
              f"{result['audio_duration']:<12.2f} "
              f"{result['bitrate']:<15.1f} "
              f"{result['confidence']:<12.2%} "
              f"{'✓' if result['success'] else '✗':<10}")
    
    # Calculate speedup
    if results:
        base_duration = results[0]['audio_duration']
        base_bitrate = results[0]['bitrate']
        
        print("\n" + "="*70)
        print("PERFORMANCE GAINS vs 1-Channel")
        print("="*70)
        print(f"\n{'Channels':<10} {'Speedup':<15} {'Bitrate Gain':<20} {'Confidence':<12}")
        print("-"*70)
        
        for result in results:
            speedup = base_duration / result['audio_duration']
            bitrate_gain = result['bitrate'] / base_bitrate
            
            print(f"{result['channels']:<10} "
                  f"{speedup:<15.2f}x "
                  f"{bitrate_gain:<20.2f}x "
                  f"{result['confidence']:<12.2%}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("• 1-channel: Slowest but most reliable (baseline)")
    print("• 2-channel: Best balance - 2x faster with high confidence")
    print("• 3-channel: 3x faster but may have reduced accuracy")
    print("• 4-channel: 4x faster but accuracy may degrade further")
    print("\nFor most applications, 2-channel mode is recommended.")
    print("="*70)

if __name__ == "__main__":
    main()
