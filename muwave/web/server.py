"""
Web server for muwave.
Provides a browser-based interface with real-time audio visualization.
"""

import asyncio
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import psutil

try:
    from flask import Flask, render_template, request, jsonify, Response
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

import numpy as np

from muwave.core.config import Config
from muwave.core.party import Party, Message
from muwave.core.logger import ConversationLogger
from muwave.audio.fsk import FSKModulator, FSKDemodulator, FSKConfig
from muwave.ollama.client import OllamaClient, create_ollama_client, OllamaError


@dataclass
class PartyState:
    """State of a party in the web interface."""
    party: Party
    is_speaking: bool = False
    is_listening: bool = True
    audio_queue: queue.Queue = field(default_factory=queue.Queue)
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    base_frequency: float = 1000.0  # Different tones for different parties
    color: str = "#4CAF50"


@dataclass
class SystemStats:
    """System resource statistics."""
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemMonitor:
    """Monitor system resources (CPU, RAM, GPU)."""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize system monitor.
        
        Args:
            update_interval: Seconds between updates
        """
        self.update_interval = update_interval
        self._stats = SystemStats()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._gpu_available = self._check_gpu()
    
    def _check_gpu(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _update_stats(self) -> None:
        """Update system statistics."""
        # CPU
        self._stats.cpu_percent = psutil.cpu_percent(interval=None)
        
        # RAM
        memory = psutil.virtual_memory()
        self._stats.ram_percent = memory.percent
        self._stats.ram_used_gb = memory.used / (1024 ** 3)
        self._stats.ram_total_gb = memory.total / (1024 ** 3)
        
        # GPU (if available)
        if self._gpu_available:
            try:
                import subprocess
                # Get GPU utilization
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 3:
                        self._stats.gpu_percent = float(parts[0].strip())
                        self._stats.gpu_memory_used_gb = float(parts[1].strip()) / 1024
                        self._stats.gpu_memory_total_gb = float(parts[2].strip()) / 1024
            except (subprocess.TimeoutExpired, ValueError, IndexError):
                pass
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            self._update_stats()
            time.sleep(self.update_interval)
    
    def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def get_stats(self) -> SystemStats:
        """Get current statistics."""
        return SystemStats(
            cpu_percent=self._stats.cpu_percent,
            ram_percent=self._stats.ram_percent,
            ram_used_gb=self._stats.ram_used_gb,
            ram_total_gb=self._stats.ram_total_gb,
            gpu_percent=self._stats.gpu_percent,
            gpu_memory_used_gb=self._stats.gpu_memory_used_gb,
            gpu_memory_total_gb=self._stats.gpu_memory_total_gb,
        )


class AudioProcessor:
    """Process audio for web interface with queueing and filtering."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 2048,
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate
            buffer_size: Size of audio buffers
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Audio queues
        self._transmit_queue: queue.Queue = queue.Queue()
        self._receive_buffer: List[np.ndarray] = []
        self._receive_lock = threading.Lock()
        
        # Spectrogram history
        self._spectrogram_history: List[np.ndarray] = []
        self._max_spectrogram_frames = 100
        
        # FSK modulators for different party tones
        self._modulators: Dict[str, FSKModulator] = {}
        self._demodulator = FSKDemodulator()
        
        # State
        self._processing = False
        self._listen_while_speaking = True
        
    def create_modulator_for_party(
        self,
        party_id: str,
        base_frequency: float,
    ) -> FSKModulator:
        """Create a modulator with a specific base frequency for a party."""
        config = FSKConfig(
            sample_rate=self.sample_rate,
            base_frequency=base_frequency,
        )
        modulator = FSKModulator(config)
        self._modulators[party_id] = modulator
        return modulator
    
    def queue_transmission(
        self,
        text: str,
        party_id: str,
        signature: Optional[bytes] = None,
    ) -> None:
        """
        Queue text for transmission.
        
        Args:
            text: Text to transmit
            party_id: ID of the transmitting party
            signature: Party signature
        """
        self._transmit_queue.put({
            "text": text,
            "party_id": party_id,
            "signature": signature,
            "timestamp": time.time(),
        })
    
    def get_next_transmission(self) -> Optional[Dict[str, Any]]:
        """Get the next queued transmission."""
        try:
            return self._transmit_queue.get_nowait()
        except queue.Empty:
            return None
    
    def has_pending_transmissions(self) -> bool:
        """Check if there are pending transmissions."""
        return not self._transmit_queue.empty()
    
    def generate_audio(
        self,
        text: str,
        party_id: str,
        signature: Optional[bytes] = None,
    ) -> np.ndarray:
        """
        Generate audio for text.
        
        Args:
            text: Text to encode
            party_id: ID of the transmitting party
            signature: Party signature
            
        Returns:
            Audio samples
        """
        modulator = self._modulators.get(party_id)
        if modulator is None:
            modulator = FSKModulator()
        
        return modulator.encode_text(text, signature=signature)
    
    def add_received_audio(self, samples: np.ndarray) -> None:
        """Add received audio samples to buffer."""
        with self._receive_lock:
            self._receive_buffer.append(samples)
            
            # Compute spectrogram frame
            if len(samples) >= 256:
                spectrum = self._compute_spectrum(samples)
                self._spectrogram_history.append(spectrum)
                
                # Trim history
                if len(self._spectrogram_history) > self._max_spectrogram_frames:
                    self._spectrogram_history.pop(0)
    
    def _compute_spectrum(self, samples: np.ndarray) -> np.ndarray:
        """Compute frequency spectrum for samples."""
        # Use FFT with windowing
        window = np.hanning(len(samples))
        windowed = samples * window
        
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Normalize to 0-1 range
        magnitude_db = np.clip((magnitude_db + 60) / 60, 0, 1)
        
        return magnitude_db
    
    def get_waveform_data(self, num_samples: int = 512) -> List[float]:
        """Get recent waveform data for visualization."""
        with self._receive_lock:
            if not self._receive_buffer:
                return [0.0] * num_samples
            
            # Concatenate recent buffers
            all_samples = np.concatenate(self._receive_buffer[-10:])
            
            # Return most recent samples
            if len(all_samples) >= num_samples:
                return all_samples[-num_samples:].tolist()
            else:
                # Pad with zeros
                padded = np.zeros(num_samples)
                padded[-len(all_samples):] = all_samples
                return padded.tolist()
    
    def get_spectrogram_data(self) -> List[List[float]]:
        """Get spectrogram history for visualization."""
        with self._receive_lock:
            if not self._spectrogram_history:
                return []
            return [s.tolist() for s in self._spectrogram_history]
    
    def apply_self_filter(
        self,
        samples: np.ndarray,
        own_frequency: float,
        bandwidth: float = 200.0,
    ) -> np.ndarray:
        """
        Filter out own transmission frequency while listening.
        
        Args:
            samples: Input samples
            own_frequency: Base frequency to filter out
            bandwidth: Bandwidth to filter
            
        Returns:
            Filtered samples
        """
        # Simple notch filter
        from scipy import signal
        
        # Normalize frequency
        nyquist = self.sample_rate / 2
        low = (own_frequency - bandwidth / 2) / nyquist
        high = (own_frequency + bandwidth / 2) / nyquist
        
        # Ensure valid range
        low = max(0.01, min(0.99, low))
        high = max(0.01, min(0.99, high))
        
        if low >= high:
            return samples
        
        # Design notch filter
        b, a = signal.butter(4, [low, high], btype='bandstop')
        
        try:
            filtered = signal.filtfilt(b, a, samples)
            return filtered.astype(np.float32)
        except Exception:
            return samples
    
    def clear_buffers(self) -> None:
        """Clear all audio buffers."""
        with self._receive_lock:
            self._receive_buffer.clear()
            self._spectrogram_history.clear()
        
        # Clear transmit queue
        while not self._transmit_queue.empty():
            try:
                self._transmit_queue.get_nowait()
            except queue.Empty:
                break


class MuwaveWebServer:
    """
    Web server for muwave.
    
    Provides:
    - Real-time audio waveform visualization
    - Spectrogram history display
    - Text input with audio generation and queueing
    - Multi-party support with different tones
    - System resource monitoring
    - Conversation logging
    """
    
    # Default party colors
    PARTY_COLORS = [
        "#4CAF50",  # Green
        "#2196F3",  # Blue
        "#FF9800",  # Orange
        "#9C27B0",  # Purple
        "#F44336",  # Red
        "#00BCD4",  # Cyan
    ]
    
    # Base frequencies for different parties (Hz)
    PARTY_FREQUENCIES = [
        1000.0,
        1500.0,
        2000.0,
        2500.0,
        3000.0,
        3500.0,
    ]
    
    def __init__(
        self,
        config: Optional[Config] = None,
        host: str = "127.0.0.1",
        port: int = 5000,
    ):
        """
        Initialize web server.
        
        Args:
            config: muwave configuration
            host: Host to bind to
            port: Port to listen on
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError(
                "Flask and flask-socketio are required for web interface. "
                "Install with: pip install flask flask-socketio"
            )
        
        self.config = config or Config()
        self.host = host
        self.port = port
        
        # Flask app
        template_dir = Path(__file__).parent / "templates"
        static_dir = Path(__file__).parent / "static"
        
        self.app = Flask(
            __name__,
            template_folder=str(template_dir),
            static_folder=str(static_dir),
        )
        self.app.config['SECRET_KEY'] = os.urandom(24).hex()
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Components
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.audio.get("sample_rate", 44100),
        )
        self.system_monitor = SystemMonitor(update_interval=1.0)
        self.logger = ConversationLogger(
            log_file=self.config.logging_config.get("file", "muwave_conversation.log"),
        )
        
        # Parties
        self._parties: Dict[str, PartyState] = {}
        self._party_lock = threading.Lock()
        
        # Ollama client
        self._ollama: Optional[OllamaClient] = None
        
        # State
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        
        # Register routes and socket events
        self._register_routes()
        self._register_socket_events()
    
    def _register_routes(self) -> None:
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            return render_template('index.html', parties=self._get_party_list())
        
        @self.app.route('/party/<party_id>')
        def party_page(party_id: str):
            """Party-specific page."""
            party_state = self._parties.get(party_id)
            if not party_state:
                return "Party not found", 404
            return render_template(
                'party.html',
                party=party_state.party.to_dict(),
                color=party_state.color,
            )
        
        @self.app.route('/user')
        def user_page():
            """User input page."""
            return render_template('user.html', parties=self._get_party_list())
        
        @self.app.route('/api/parties')
        def api_parties():
            """Get list of parties."""
            return jsonify(self._get_party_list())
        
        @self.app.route('/api/party/<party_id>')
        def api_party(party_id: str):
            """Get party details."""
            party_state = self._parties.get(party_id)
            if not party_state:
                return jsonify({"error": "Party not found"}), 404
            return jsonify({
                "party": party_state.party.to_dict(),
                "is_speaking": party_state.is_speaking,
                "is_listening": party_state.is_listening,
                "color": party_state.color,
                "base_frequency": party_state.base_frequency,
            })
        
        @self.app.route('/api/stats')
        def api_stats():
            """Get system statistics."""
            return jsonify(self.system_monitor.get_stats().to_dict())
        
        @self.app.route('/api/waveform')
        def api_waveform():
            """Get waveform data."""
            return jsonify({
                "waveform": self.audio_processor.get_waveform_data(),
            })
        
        @self.app.route('/api/spectrogram')
        def api_spectrogram():
            """Get spectrogram data."""
            return jsonify({
                "spectrogram": self.audio_processor.get_spectrogram_data(),
            })
        
        @self.app.route('/api/conversation/<party_id>')
        def api_conversation(party_id: str):
            """Get conversation history for a party."""
            party_state = self._parties.get(party_id)
            if not party_state:
                return jsonify({"error": "Party not found"}), 404
            return jsonify({
                "messages": party_state.message_history[-50:],
            })
        
        @self.app.route('/api/send', methods=['POST'])
        def api_send():
            """Send a message."""
            data = request.json
            text = data.get('text', '')
            party_id = data.get('party_id')
            
            if not text:
                return jsonify({"error": "No text provided"}), 400
            
            if party_id and party_id in self._parties:
                party_state = self._parties[party_id]
                self.audio_processor.queue_transmission(
                    text,
                    party_id,
                    party_state.party.signature,
                )
                
                # Add to history
                party_state.message_history.append({
                    "role": "user",
                    "content": text,
                    "timestamp": time.time(),
                    "party_id": party_id,
                })
                
                # Emit via socket
                self.socketio.emit('message_queued', {
                    "text": text,
                    "party_id": party_id,
                })
            
            return jsonify({"status": "queued"})
        
        @self.app.route('/api/ai', methods=['POST'])
        def api_ai():
            """Send a message to AI."""
            data = request.json
            prompt = data.get('prompt', '')
            party_id = data.get('party_id')
            transmit = data.get('transmit', True)
            
            if not prompt:
                return jsonify({"error": "No prompt provided"}), 400
            
            try:
                ollama = self._get_ollama()
                if not ollama.is_available():
                    return jsonify({"error": "Ollama not available"}), 503
                
                response = ollama.chat(prompt)
                
                # Add to party history if specified
                if party_id and party_id in self._parties:
                    party_state = self._parties[party_id]
                    party_state.message_history.append({
                        "role": "user",
                        "content": prompt,
                        "timestamp": time.time(),
                    })
                    party_state.message_history.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": time.time(),
                    })
                    
                    # Queue for transmission if requested
                    if transmit:
                        self.audio_processor.queue_transmission(
                            response,
                            party_id,
                            party_state.party.signature,
                        )
                
                return jsonify({
                    "response": response,
                    "transmitted": transmit,
                })
                
            except OllamaError as e:
                return jsonify({"error": str(e)}), 500
    
    def _register_socket_events(self) -> None:
        """Register Socket.IO events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            emit('connected', {'status': 'ok'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            pass
        
        @self.socketio.on('join_party')
        def handle_join_party(data):
            """Handle party join request."""
            party_id = data.get('party_id')
            if party_id and party_id in self._parties:
                emit('party_joined', {
                    'party_id': party_id,
                    'party': self._parties[party_id].party.to_dict(),
                })
        
        @self.socketio.on('send_message')
        def handle_send_message(data):
            """Handle message send request."""
            text = data.get('text', '')
            party_id = data.get('party_id')
            
            if text and party_id and party_id in self._parties:
                party_state = self._parties[party_id]
                self.audio_processor.queue_transmission(
                    text,
                    party_id,
                    party_state.party.signature,
                )
                emit('message_queued', {'text': text, 'party_id': party_id})
        
        @self.socketio.on('request_stats')
        def handle_request_stats():
            """Handle stats request."""
            emit('stats_update', self.system_monitor.get_stats().to_dict())
        
        @self.socketio.on('request_waveform')
        def handle_request_waveform():
            """Handle waveform data request."""
            emit('waveform_update', {
                'waveform': self.audio_processor.get_waveform_data(),
            })
    
    def _get_party_list(self) -> List[Dict[str, Any]]:
        """Get list of parties."""
        with self._party_lock:
            return [
                {
                    "party_id": party_id,
                    "name": state.party.name,
                    "color": state.color,
                    "is_speaking": state.is_speaking,
                    "is_listening": state.is_listening,
                }
                for party_id, state in self._parties.items()
            ]
    
    def _get_ollama(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._ollama is None:
            self._ollama = create_ollama_client(self.config.ollama)
        return self._ollama
    
    def add_party(
        self,
        party: Party,
        color: Optional[str] = None,
        base_frequency: Optional[float] = None,
    ) -> str:
        """
        Add a party to the web server.
        
        Args:
            party: Party to add
            color: Color for the party (auto-assigned if None)
            base_frequency: Base frequency for the party's tone
            
        Returns:
            Party ID
        """
        with self._party_lock:
            idx = len(self._parties)
            
            if color is None:
                color = self.PARTY_COLORS[idx % len(self.PARTY_COLORS)]
            
            if base_frequency is None:
                base_frequency = self.PARTY_FREQUENCIES[idx % len(self.PARTY_FREQUENCIES)]
            
            state = PartyState(
                party=party,
                base_frequency=base_frequency,
                color=color,
            )
            
            self._parties[party.party_id] = state
            
            # Create modulator for this party
            self.audio_processor.create_modulator_for_party(
                party.party_id,
                base_frequency,
            )
            
            return party.party_id
    
    def remove_party(self, party_id: str) -> None:
        """Remove a party."""
        with self._party_lock:
            if party_id in self._parties:
                del self._parties[party_id]
    
    def _broadcast_updates(self) -> None:
        """Broadcast periodic updates to all clients."""
        while self._running:
            try:
                # Broadcast stats
                self.socketio.emit('stats_update', self.system_monitor.get_stats().to_dict())
                
                # Broadcast waveform
                self.socketio.emit('waveform_update', {
                    'waveform': self.audio_processor.get_waveform_data(),
                })
                
                # Broadcast spectrogram
                self.socketio.emit('spectrogram_update', {
                    'spectrogram': self.audio_processor.get_spectrogram_data()[-20:],
                })
                
                # Broadcast party states
                self.socketio.emit('parties_update', self._get_party_list())
                
                time.sleep(0.1)  # 10 Hz update rate
                
            except Exception:
                pass
    
    def start(self, debug: bool = False) -> None:
        """
        Start the web server.
        
        Args:
            debug: Enable debug mode
        """
        self._running = True
        
        # Start system monitor
        self.system_monitor.start()
        
        # Start update thread
        self._update_thread = threading.Thread(
            target=self._broadcast_updates,
            daemon=True,
        )
        self._update_thread.start()
        
        # Run Flask app
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            use_reloader=False,
        )
    
    def stop(self) -> None:
        """Stop the web server."""
        self._running = False
        self.system_monitor.stop()
        if self._update_thread:
            self._update_thread.join(timeout=2.0)


def create_web_server(
    config: Optional[Config] = None,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> MuwaveWebServer:
    """
    Create a web server instance.
    
    Args:
        config: muwave configuration
        host: Host to bind to
        port: Port to listen on
        
    Returns:
        MuwaveWebServer instance
    """
    return MuwaveWebServer(config=config, host=host, port=port)
