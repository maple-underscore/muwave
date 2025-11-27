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
from muwave.utils.formats import FormatEncoder, FormatDetector, FormatMetadata


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
        
        self._modulator = FSKModulator(fsk_config)
        self._audio_device = audio_device or AudioDevice(
            sample_rate=fsk_config.sample_rate,
            buffer_size=self._config.get("buffer_size", 1024),
            input_device=self._config.get("input_device"),
            output_device=self._config.get("output_device"),
        )
        
        self._repetitions = self._config.get("repetitions", 1)
        self._progress = TransmissionProgress()
        self._progress_callback: Optional[Callable[[TransmissionProgress], None]] = None
        self._transmitting = False
        self._lock = threading.Lock()
        self._progress_thread: Optional[threading.Thread] = None
        self._progress_stop = threading.Event()
        self._expected_duration_s: float = 0.0
    
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
            # Detect format if not specified
            format_meta = None
            if message.content_format:
                # Use message's format metadata
                from muwave.utils.formats import ContentFormat
                format_type = None
                for fmt in ContentFormat:
                    if fmt.value == message.content_format or fmt.name.lower() == message.content_format.lower():
                        format_type = fmt
                        break
                if format_type:
                    format_meta = FormatMetadata(format_type, language=message.format_language)
            
            # Encode content with format metadata
            encoded_data = FormatEncoder.encode(message.content, format_meta)
            
            # Prepare progress tracking
            self._progress = TransmissionProgress(total_bytes=len(encoded_data))
            self._update_progress("waiting")
            
            # Encode message (use encode_data instead of encode_text to preserve binary format info)
            audio_samples = self._modulator.encode_data(
                encoded_data,
                signature=self.party.signature,
                repetitions=self._repetitions,
            )
            
            # Estimate expected duration for progress interpolation
            try:
                # Account for format metadata overhead
                effective_content = message.content if not format_meta else message.content + "XX"  # ~2 byte overhead
                self._expected_duration_s = self.estimate_duration(effective_content)
            except Exception:
                self._expected_duration_s = max(0.1, len(audio_samples) / self._modulator.config.sample_rate)

            self._update_progress("sending")

            # Start background progress updater to interpolate during playback
            def progress_updater():
                total_bytes = max(1, self._progress.total_bytes)
                start_time = time.time()
                while not self._progress_stop.is_set() and self._transmitting:
                    elapsed = time.time() - start_time
                    if self._expected_duration_s > 0:
                        frac = min(0.99, max(0.0, elapsed / self._expected_duration_s))
                    else:
                        frac = 0.0
                    sent_est = int(total_bytes * frac)
                    if sent_est != self._progress.sent_bytes:
                        self._update_progress("sending", sent_bytes=sent_est)
                    time.sleep(0.1)
            
            def on_complete():
                # Stop progress updater and finalize
                self._progress_stop.set()
                if self._progress_thread and self._progress_thread.is_alive():
                    try:
                        self._progress_thread.join(timeout=0.5)
                    except Exception:
                        pass
                self._update_progress("sent", len(content_bytes))
                self.party.mark_transmitted(message)
                self._transmitting = False
            
            # Play audio
            # Launch progress updater thread after playback starts
            self._progress_stop.clear()
            self._progress_thread = threading.Thread(target=progress_updater, daemon=True)
            self._progress_thread.start()

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
            self._progress_stop.set()
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
