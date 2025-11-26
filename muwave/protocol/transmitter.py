"""
Transmitter for muwave protocol.
Handles encoding and sending messages over audio.
"""

import time
import threading
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field

import numpy as np

from muwave.audio.fsk import FSKModulator, FSKConfig
from muwave.audio.device import AudioDevice
from muwave.core.party import Party, Message


@dataclass
class TransmissionProgress:
    """Tracks transmission progress for UI updates."""
    total_bytes: int = 0
    sent_bytes: int = 0
    status: str = "waiting"  # waiting, sending, sent, error
    start_time: float = 0.0
    end_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.sent_bytes / self.total_bytes) * 100
    
    @property
    def duration_ms(self) -> float:
        """Get transmission duration in milliseconds."""
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        elif self.start_time > 0:
            return (time.time() - self.start_time) * 1000
        return 0.0


class Transmitter:
    """
    Audio transmitter for muwave protocol.
    
    Encodes messages using FSK modulation and plays them through
    the audio device.
    """
    
    def __init__(
        self,
        party: Party,
        config: Optional[Dict[str, Any]] = None,
        audio_device: Optional[AudioDevice] = None,
    ):
        """
        Initialize transmitter.
        
        Args:
            party: The party sending messages
            config: Configuration dictionary
            audio_device: Audio device to use (creates new if None)
        """
        self.party = party
        self._config = config or {}
        
        # Set up FSK configuration
        fsk_config = FSKConfig(
            sample_rate=self._config.get("sample_rate", 44100),
            base_frequency=self._config.get("base_frequency", 1000),
            frequency_step=self._config.get("frequency_step", 100),
            num_frequencies=self._config.get("num_frequencies", 16),
            symbol_duration_ms=self._config.get("symbol_duration_ms", 50),
            start_frequency=self._config.get("start_frequency", 500),
            end_frequency=self._config.get("end_frequency", 600),
            signal_duration_ms=self._config.get("signal_duration_ms", 200),
            silence_ms=self._config.get("silence_ms", 50),
            volume=self._config.get("volume", 0.8),
        )
        
        self._modulator = FSKModulator(fsk_config)
        self._audio_device = audio_device or AudioDevice(
            sample_rate=fsk_config.sample_rate,
            buffer_size=self._config.get("buffer_size", 1024),
        )
        
        self._repetitions = self._config.get("repetitions", 1)
        self._progress = TransmissionProgress()
        self._progress_callback: Optional[Callable[[TransmissionProgress], None]] = None
        self._transmitting = False
        self._lock = threading.Lock()
    
    def set_progress_callback(
        self,
        callback: Callable[[TransmissionProgress], None],
    ) -> None:
        """Set callback for transmission progress updates."""
        self._progress_callback = callback
    
    def _update_progress(
        self,
        status: str,
        sent_bytes: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Update and report progress."""
        self._progress.status = status
        self._progress.sent_bytes = sent_bytes
        self._progress.error = error
        
        if status == "sending" and self._progress.start_time == 0:
            self._progress.start_time = time.time()
        elif status in ("sent", "error"):
            self._progress.end_time = time.time()
        
        if self._progress_callback:
            self._progress_callback(self._progress)
    
    def transmit(
        self,
        message: Message,
        blocking: bool = True,
    ) -> bool:
        """
        Transmit a message.
        
        Args:
            message: Message to transmit
            blocking: Whether to block until transmission completes
            
        Returns:
            True if transmission successful
        """
        with self._lock:
            if self._transmitting:
                return False
            self._transmitting = True
        
        try:
            # Prepare progress tracking
            content_bytes = message.content.encode('utf-8')
            self._progress = TransmissionProgress(total_bytes=len(content_bytes))
            self._update_progress("waiting")
            
            # Encode message
            audio_samples = self._modulator.encode_text(
                message.content,
                signature=self.party.signature,
                repetitions=self._repetitions,
            )
            
            self._update_progress("sending")
            
            def on_complete():
                self._update_progress("sent", len(content_bytes))
                self.party.mark_transmitted(message)
                self._transmitting = False
            
            # Play audio
            if blocking:
                self._audio_device.play(audio_samples, blocking=True)
                on_complete()
            else:
                self._audio_device.play(
                    audio_samples,
                    blocking=False,
                    callback=on_complete,
                )
            
            return True
            
        except Exception as e:
            self._update_progress("error", error=str(e))
            self._transmitting = False
            return False
    
    def transmit_text(
        self,
        text: str,
        blocking: bool = True,
    ) -> bool:
        """
        Transmit text directly.
        
        Args:
            text: Text to transmit
            blocking: Whether to block until complete
            
        Returns:
            True if successful
        """
        message = self.party.create_message(text)
        return self.transmit(message, blocking)
    
    def is_transmitting(self) -> bool:
        """Check if currently transmitting."""
        return self._transmitting
    
    def get_progress(self) -> TransmissionProgress:
        """Get current transmission progress."""
        return self._progress
    
    def estimate_duration(self, text: str) -> float:
        """
        Estimate transmission duration for text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated duration in seconds
        """
        content_bytes = text.encode('utf-8')
        
        # Calculate based on FSK parameters
        symbol_duration_s = self._modulator.config.symbol_duration_ms / 1000
        bytes_duration = len(content_bytes) * 2 * symbol_duration_s  # 2 symbols per byte
        
        # Add overhead for signature, length, start/end signals
        overhead_duration = (
            self._modulator.config.signal_duration_ms * 2 / 1000 +  # start/end
            8 * 2 * symbol_duration_s +  # signature
            2 * 2 * symbol_duration_s +  # length
            self._modulator.config.silence_ms * 3 / 1000  # silences
        )
        
        # Account for repetitions
        total = (bytes_duration * self._repetitions) + overhead_duration
        
        return total
    
    def stop(self) -> None:
        """Stop current transmission."""
        self._audio_device.stop_playback()
        self._transmitting = False
        self._update_progress("error", error="Transmission stopped")
