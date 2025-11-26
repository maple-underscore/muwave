"""
Audio device management for muwave.
Handles audio input/output across Linux and macOS.
"""

import threading
import queue
import time
from typing import Callable, Optional, List, Tuple
import numpy as np

# Try to import sounddevice, handle gracefully if not available
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    sd = None


class AudioDevice:
    """
    Cross-platform audio device manager.
    
    Provides unified interface for audio I/O on Linux and macOS.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 1024,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ):
        """
        Initialize audio device.
        
        Args:
            sample_rate: Audio sample rate in Hz
            buffer_size: Buffer size for audio processing
            input_device: Input device index (None for default)
            output_device: Output device index (None for default)
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not available. "
                "Please install it with: pip install sounddevice"
            )
        
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.input_device = input_device
        self.output_device = output_device
        
        self._recording = False
        self._playing = False
        self._record_thread: Optional[threading.Thread] = None
        self._play_thread: Optional[threading.Thread] = None
        self._record_queue: queue.Queue = queue.Queue()
        self._record_callback: Optional[Callable[[np.ndarray], None]] = None
        self._stop_event = threading.Event()
    
    @staticmethod
    def list_devices() -> List[dict]:
        """List available audio devices."""
        if not SOUNDDEVICE_AVAILABLE:
            return []
        
        devices = sd.query_devices()
        return [
            {
                "index": i,
                "name": d["name"],
                "inputs": d["max_input_channels"],
                "outputs": d["max_output_channels"],
                "default_samplerate": d["default_samplerate"],
            }
            for i, d in enumerate(devices)
        ]
    
    @staticmethod
    def get_default_input() -> Optional[int]:
        """Get default input device index."""
        if not SOUNDDEVICE_AVAILABLE:
            return None
        try:
            return sd.default.device[0]
        except Exception:
            return None
    
    @staticmethod
    def get_default_output() -> Optional[int]:
        """Get default output device index."""
        if not SOUNDDEVICE_AVAILABLE:
            return None
        try:
            return sd.default.device[1]
        except Exception:
            return None
    
    def play(
        self,
        samples: np.ndarray,
        blocking: bool = True,
        callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Play audio samples.
        
        Args:
            samples: Audio samples to play (float32, mono)
            blocking: Whether to block until playback completes
            callback: Optional callback when playback completes
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice not available")
        
        # Ensure samples are float32
        samples = np.asarray(samples, dtype=np.float32)
        
        if blocking:
            sd.play(samples, self.sample_rate, device=self.output_device)
            sd.wait()
            if callback:
                callback()
        else:
            def play_thread():
                sd.play(samples, self.sample_rate, device=self.output_device)
                sd.wait()
                self._playing = False
                if callback:
                    callback()
            
            self._playing = True
            self._play_thread = threading.Thread(target=play_thread, daemon=True)
            self._play_thread.start()
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._playing
    
    def stop_playback(self) -> None:
        """Stop current playback."""
        if SOUNDDEVICE_AVAILABLE:
            sd.stop()
        self._playing = False
    
    def record(
        self,
        duration_seconds: float,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> Optional[np.ndarray]:
        """
        Record audio for a specified duration.
        
        Args:
            duration_seconds: Recording duration in seconds
            callback: Optional callback with recorded samples
            
        Returns:
            Recorded samples if callback not provided
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice not available")
        
        num_samples = int(self.sample_rate * duration_seconds)
        samples = sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            device=self.input_device,
        )
        sd.wait()
        
        samples = samples.flatten()
        
        if callback:
            callback(samples)
            return None
        return samples
    
    def start_continuous_recording(
        self,
        callback: Callable[[np.ndarray], None],
    ) -> None:
        """
        Start continuous recording with callback for each buffer.
        
        Args:
            callback: Function called with each audio buffer
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice not available")
        
        if self._recording:
            return
        
        self._recording = True
        self._record_callback = callback
        self._stop_event.clear()
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                pass  # Handle status if needed
            if self._record_callback and self._recording:
                self._record_callback(indata.copy().flatten())
        
        self._input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.buffer_size,
            device=self.input_device,
            callback=audio_callback,
        )
        self._input_stream.start()
    
    def stop_continuous_recording(self) -> None:
        """Stop continuous recording."""
        self._recording = False
        self._stop_event.set()
        if hasattr(self, '_input_stream'):
            self._input_stream.stop()
            self._input_stream.close()
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
    
    def play_and_record(
        self,
        samples: np.ndarray,
        extra_duration: float = 0.5,
    ) -> np.ndarray:
        """
        Play audio and record simultaneously.
        
        Useful for testing and loopback scenarios.
        
        Args:
            samples: Samples to play
            extra_duration: Extra recording time after playback
            
        Returns:
            Recorded samples
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice not available")
        
        play_duration = len(samples) / self.sample_rate
        record_duration = play_duration + extra_duration
        
        recorded = []
        record_done = threading.Event()
        
        def record_thread():
            rec = self.record(record_duration)
            recorded.append(rec)
            record_done.set()
        
        # Start recording first
        rec_thread = threading.Thread(target=record_thread, daemon=True)
        rec_thread.start()
        
        # Small delay to ensure recording starts
        time.sleep(0.05)
        
        # Play audio
        self.play(samples, blocking=True)
        
        # Wait for recording to complete
        record_done.wait()
        
        return recorded[0] if recorded else np.array([], dtype=np.float32)


class AudioBuffer:
    """
    Thread-safe audio buffer for continuous recording.
    
    Collects audio samples and provides methods to detect
    and extract complete transmissions.
    """
    
    def __init__(
        self,
        max_duration_seconds: float = 60.0,
        sample_rate: int = 44100,
    ):
        """
        Initialize audio buffer.
        
        Args:
            max_duration_seconds: Maximum buffer duration
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self._buffer: List[np.ndarray] = []
        self._total_samples = 0
        self._lock = threading.Lock()
    
    def add(self, samples: np.ndarray) -> None:
        """Add samples to the buffer."""
        with self._lock:
            self._buffer.append(samples)
            self._total_samples += len(samples)
            
            # Trim if exceeding max
            while self._total_samples > self.max_samples and self._buffer:
                removed = self._buffer.pop(0)
                self._total_samples -= len(removed)
    
    def get_all(self) -> np.ndarray:
        """Get all samples in the buffer."""
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._buffer)
    
    def get_recent(self, duration_seconds: float) -> np.ndarray:
        """Get recent samples for a given duration."""
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            
            all_samples = np.concatenate(self._buffer)
            num_samples = int(duration_seconds * self.sample_rate)
            
            if len(all_samples) <= num_samples:
                return all_samples
            return all_samples[-num_samples:]
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._total_samples = 0
    
    def clear_before(self, num_samples: int) -> None:
        """Clear samples before a given position."""
        with self._lock:
            if not self._buffer:
                return
            
            all_samples = np.concatenate(self._buffer)
            if num_samples >= len(all_samples):
                self._buffer.clear()
                self._total_samples = 0
            else:
                remaining = all_samples[num_samples:]
                self._buffer = [remaining]
                self._total_samples = len(remaining)
    
    @property
    def duration_seconds(self) -> float:
        """Get current buffer duration in seconds."""
        return self._total_samples / self.sample_rate
    
    @property
    def num_samples(self) -> int:
        """Get number of samples in buffer."""
        return self._total_samples
