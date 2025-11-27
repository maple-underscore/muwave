"""
Receiver for muwave protocol.
Handles listening for and decoding audio messages.
"""

import time
import threading
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal

from muwave.audio.fsk import FSKDemodulator, FSKConfig
from muwave.audio.device import AudioDevice, AudioBuffer
from muwave.core.party import Party, Message


class ReceiverState(Enum):
    """Receiver state."""
    IDLE = "idle"
    LISTENING = "listening"
    RECEIVING = "receiving"
    DECODING = "decoding"
    ERROR = "error"


@dataclass
class ReceptionProgress:
    """Tracks reception progress for UI updates."""
    state: ReceiverState = ReceiverState.IDLE
    received_bytes: int = 0
    expected_bytes: int = 0
    partial_content: str = ""
    confidence: float = 0.0
    sender_signature: Optional[str] = None
    is_own_transmission: bool = False
    start_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.expected_bytes == 0:
            return 0.0
        return (self.received_bytes / self.expected_bytes) * 100


class Receiver:
    """
    Audio receiver for muwave protocol.
    
    Listens for FSK-encoded messages and decodes them.
    """
    
    def __init__(
        self,
        party: Party,
        config: Optional[Dict[str, Any]] = None,
        audio_device: Optional[AudioDevice] = None,
    ):
        """
        Initialize receiver.
        
        Args:
            party: The party receiving messages
            config: Configuration dictionary
            audio_device: Audio device to use (creates new if None)
        """
        self.party = party
        self._config = config or {}
        
        # Set up FSK configuration
        fsk_config = FSKConfig(
            sample_rate=self._config.get("sample_rate", 44100),
            base_frequency=self._config.get("base_frequency", 1800),
            frequency_step=self._config.get("frequency_step", 120),
            num_frequencies=self._config.get("num_frequencies", 16),
            symbol_duration_ms=self._config.get("symbol_duration_ms", 60),
            start_frequencies=self._config.get("start_frequencies", [800, 1000, 1200, 1400, 1600, 1800, 2000]),
            end_frequencies=self._config.get("end_frequencies", [900, 1100, 1300, 1500, 1700, 1900, 2100]),
            signal_duration_ms=self._config.get("signal_duration_ms", 500),
            silence_ms=self._config.get("silence_ms", 50),
            volume=self._config.get("volume", 0.8),
            num_channels=self._config.get("num_channels", 2),
            channel_spacing=self._config.get("channel_spacing", 1600),
        )
        
        self._demodulator = FSKDemodulator(fsk_config)
        self._audio_device = audio_device or AudioDevice(
            sample_rate=fsk_config.sample_rate,
            buffer_size=self._config.get("buffer_size", 1024),
            input_device=self._config.get("input_device"),
            output_device=self._config.get("output_device"),
        )
        
        self._repetitions = self._config.get("repetitions", 1)
        self._self_recognition = self._config.get("self_recognition", True)
        self._input_gain = float(self._config.get("input_gain", 1.0))
        
        self._buffer = AudioBuffer(
            max_duration_seconds=60.0,
            sample_rate=fsk_config.sample_rate,
        )
        
        self._progress = ReceptionProgress()
        self._progress_callback: Optional[Callable[[ReceptionProgress], None]] = None
        self._message_callback: Optional[Callable[[Message, bool], None]] = None
        
        self._listening = False
        self._stop_event = threading.Event()
        self._listen_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        
        self._received_messages: List[Message] = []
        self._lock = threading.Lock()
    
    def set_progress_callback(
        self,
        callback: Callable[[ReceptionProgress], None],
    ) -> None:
        """Set callback for reception progress updates."""
        self._progress_callback = callback
    
    def set_message_callback(
        self,
        callback: Callable[[Message, bool], None],
    ) -> None:
        """
        Set callback for received messages.
        
        Args:
            callback: Function(message, is_own_transmission)
        """
        self._message_callback = callback
    
    def _update_progress(
        self,
        state: ReceiverState,
        **kwargs,
    ) -> None:
        """Update and report progress."""
        self._progress.state = state
        for key, value in kwargs.items():
            if hasattr(self._progress, key):
                setattr(self._progress, key, value)
        
        if state == ReceiverState.RECEIVING and self._progress.start_time == 0:
            self._progress.start_time = time.time()
        
        if self._progress_callback:
            self._progress_callback(self._progress)
    
    def _on_audio_buffer(self, samples: np.ndarray) -> None:
        """Handle incoming audio buffer."""
        try:
            if self._input_gain != 1.0:
                samples = np.asarray(samples, dtype=np.float32) * self._input_gain
        except Exception:
            pass
        self._buffer.add(samples)

    def inject_samples(self, samples: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """Inject samples directly into the receiver buffer, resampling if needed."""
        arr = np.asarray(samples, dtype=np.float32).flatten()
        target_sr = int(self._demodulator.config.sample_rate)
        if sample_rate is not None and int(sample_rate) != target_sr:
            try:
                # Polyphase resampling for quality
                from math import gcd
                g = gcd(int(sample_rate), target_sr)
                up = target_sr // g
                down = int(sample_rate) // g
                arr = scipy_signal.resample_poly(arr, up, down).astype(np.float32)
            except Exception:
                # Fallback to linear interpolation
                old_n = len(arr)
                new_n = int(old_n * (target_sr / float(sample_rate)))
                x_old = np.linspace(0.0, 1.0, num=old_n, endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=new_n, endpoint=False)
                arr = np.interp(x_new, x_old, arr).astype(np.float32)
        # Apply input gain similar to live path
        try:
            if self._input_gain != 1.0:
                arr = arr * self._input_gain
        except Exception:
            pass
        self._buffer.add(arr)

    def inject_wav_file(self, path: str) -> None:
        """Read a WAV file and inject its samples into the buffer for decoding."""
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            arr = data.astype(np.float32) / 32767.0
        elif data.dtype == np.int32:
            arr = data.astype(np.float32) / 2147483647.0
        elif data.dtype == np.uint8:
            arr = (data.astype(np.float32) - 128) / 128.0
        else:
            arr = data.astype(np.float32)
        if arr.ndim > 1:
            arr = arr[:, 0]
        self.inject_samples(arr, sample_rate=sr)
    
    def _process_buffer(self) -> None:
        """Process audio buffer to detect and decode messages."""
        import logging
        logger = logging.getLogger(__name__)
        
        while not self._stop_event.is_set():
            if self._buffer.num_samples < 1000:
                time.sleep(0.05)
                continue
            
            samples = self._buffer.get_all()
            
            # Log buffer stats periodically
            if int(time.time() * 2) % 10 == 0:  # Every ~5 seconds
                max_amplitude = np.max(np.abs(samples)) if len(samples) > 0 else 0
                logger.debug(f"[Receiver] Buffer: {len(samples)} samples, max amplitude: {max_amplitude:.4f}")
            
            # Try to detect start signal
            detected, start_pos = self._demodulator.detect_start_signal(samples)
            
            if not detected:
                # Trim old samples if buffer is getting large
                if self._buffer.duration_seconds > 5.0:
                    self._buffer.clear_before(
                        int(self._buffer.num_samples * 0.5)
                    )
                time.sleep(0.05)
                continue
            
            self._update_progress(ReceiverState.RECEIVING)
            
            # Wait for more data or end signal
            max_wait = 30.0  # Maximum 30 seconds for a transmission
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait:
                samples = self._buffer.get_all()
                if len(samples) <= start_pos:
                    time.sleep(0.05)
                    continue
                
                data_samples = samples[start_pos:]
                
                # Try to detect end signal
                end_detected, end_pos = self._demodulator.detect_end_signal(
                    data_samples
                )
                
                if end_detected:
                    # Extract the message data
                    message_samples = data_samples[:end_pos]
                    self._update_progress(ReceiverState.DECODING)
                    
                    # Decode the message (always read metadata for auto-detection)
                    text, signature, confidence = self._demodulator.decode_text(
                        message_samples,
                        signature_length=8,
                        repetitions=self._repetitions,
                        read_metadata=True,
                    )
                    
                    logger.info(f"[Receiver] Decoded: text={text}, confidence={confidence:.2f}")
                    
                    if text is not None and confidence > 0.2:
                        # Check if it's our own transmission
                        is_own = False
                        sender_sig = None
                        if signature:
                            sender_sig = signature.hex()
                            if self._self_recognition:
                                is_own = self.party.is_own_transmission(signature)
                        
                        self._update_progress(
                            ReceiverState.LISTENING,
                            partial_content=text,
                            confidence=confidence,
                            sender_signature=sender_sig,
                            is_own_transmission=is_own,
                        )
                        
                        # Create message
                        message = Message(
                            content=text,
                            sender_id=sender_sig or "unknown",
                            message_type="audio",
                            received=True,
                        )
                        
                        with self._lock:
                            self._received_messages.append(message)
                        
                        if not is_own:
                            self.party.receive_message(message)
                        
                        if self._message_callback:
                            self._message_callback(message, is_own)
                    
                    # Clear processed data from buffer
                    self._buffer.clear_before(start_pos + end_pos)
                    break
                
                time.sleep(0.05)
            
            self._update_progress(ReceiverState.LISTENING)
    
    def start_listening(self) -> None:
        """Start listening for messages."""
        if self._listening:
            return
        
        self._listening = True
        self._stop_event.clear()
        self._update_progress(ReceiverState.LISTENING)
        
        # Start audio recording
        self._audio_device.start_continuous_recording(self._on_audio_buffer)
        
        # Start processing thread
        self._process_thread = threading.Thread(
            target=self._process_buffer,
            daemon=True,
        )
        self._process_thread.start()
    
    def stop_listening(self) -> None:
        """Stop listening for messages."""
        self._listening = False
        self._stop_event.set()
        
        self._audio_device.stop_continuous_recording()
        
        if self._process_thread:
            self._process_thread.join(timeout=1.0)
        
        self._update_progress(ReceiverState.IDLE)
    
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._listening
    
    def get_progress(self) -> ReceptionProgress:
        """Get current reception progress."""
        return self._progress
    
    def get_received_messages(self) -> List[Message]:
        """Get all received messages."""
        with self._lock:
            return self._received_messages.copy()
    
    def clear_received(self) -> None:
        """Clear received messages list."""
        with self._lock:
            self._received_messages.clear()
    
    def receive_once(
        self,
        timeout_seconds: float = 30.0,
    ) -> Optional[Tuple[Message, bool]]:
        """
        Listen for a single message.
        
        Args:
            timeout_seconds: Maximum time to wait
            
        Returns:
            Tuple of (message, is_own_transmission) or None if timeout
        """
        result = None
        event = threading.Event()
        
        def on_message(message: Message, is_own: bool):
            nonlocal result
            result = (message, is_own)
            event.set()
        
        old_callback = self._message_callback
        self.set_message_callback(on_message)
        
        self.start_listening()
        event.wait(timeout=timeout_seconds)
        self.stop_listening()
        
        self._message_callback = old_callback
        return result
