#!/usr/bin/env python3
"""Test script to diagnose audio detection issues."""

import time
import logging
from muwave.protocol.receiver import Receiver
from muwave.protocol.transmitter import Transmitter
from muwave.core.party import Party
from muwave.audio.device import AudioDevice

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_loopback():
    """Test transmission and reception using loopback."""
    logger.info("=== Testing Loopback Audio ===")
    
    # Create parties
    sender = Party(name="sender")
    receiver_party = Party(name="receiver")
    
    # Create transmitter and receiver
    audio_device_tx = AudioDevice(sample_rate=44100)
    audio_device_rx = AudioDevice(sample_rate=44100)
    
    transmitter = Transmitter(
        party=sender,
        config={},
        audio_device=audio_device_tx,
    )
    
    receiver = Receiver(
        party=receiver_party,
        config={},
        audio_device=audio_device_rx,
    )
    
    # Set up callbacks
    def on_receive_progress(progress):
        logger.info(f"[RX] State: {progress.state}, Confidence: {progress.confidence:.2f}")
    
    def on_message(message, is_own):
        logger.info(f"[RX] Received message: '{message.content}' (own={is_own})")
    
    receiver.set_progress_callback(on_receive_progress)
    receiver.set_message_callback(on_message)
    
    # Start listening
    logger.info("Starting receiver...")
    receiver.start_listening()
    
    # Wait a bit for receiver to be ready
    time.sleep(1)
    
    # Transmit
    test_message = "hello world"
    logger.info(f"Transmitting: '{test_message}'")
    transmitter.transmit_text(test_message, blocking=True)
    
    # Wait for reception
    logger.info("Waiting for reception (10 seconds)...")
    time.sleep(10)
    
    # Stop listening
    receiver.stop_listening()
    
    logger.info("Test complete")

def test_live_monitoring():
    """Monitor live audio input and show detection attempts."""
    logger.info("=== Testing Live Audio Monitoring ===")
    logger.info("Play a muwave audio file through your speakers...")
    logger.info("Press Ctrl+C to stop")
    
    party = Party(name="monitor")
    audio_device = AudioDevice(sample_rate=44100)
    
    receiver = Receiver(
        party=party,
        config={},
        audio_device=audio_device,
    )
    
    message_count = 0
    
    def on_receive_progress(progress):
        logger.info(f"[Monitor] State: {progress.state}")
        if progress.partial_content:
            logger.info(f"[Monitor] Content: '{progress.partial_content}' (confidence: {progress.confidence:.2f})")
    
    def on_message(message, is_own):
        nonlocal message_count
        message_count += 1
        logger.info(f"[Monitor] Message #{message_count}: '{message.content}'")
    
    receiver.set_progress_callback(on_receive_progress)
    receiver.set_message_callback(on_message)
    
    try:
        receiver.start_listening()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        receiver.stop_listening()
        logger.info(f"Total messages received: {message_count}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        test_live_monitoring()
    else:
        test_loopback()
