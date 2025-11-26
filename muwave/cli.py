"""
Command-line interface for muwave.
"""

import sys
import time
import threading
import signal
from pathlib import Path
from typing import Optional

import click

from muwave.core.config import Config
from muwave.core.party import Party, Message
from muwave.core.logger import ConversationLogger
from muwave.protocol.transmitter import Transmitter
from muwave.protocol.receiver import Receiver, ReceiverState
from muwave.protocol.sync import ProcessSync, SyncMethod
from muwave.ollama.client import OllamaClient, OllamaError, create_ollama_client
from muwave.ui.interface import (
    RichInterface,
    TransmitStatus,
    ReceiveStatus,
    create_interface,
)
from muwave.utils.helpers import (
    get_platform,
    is_docker_available,
    is_ollama_container_running,
    start_ollama_container,
    format_duration,
)


class MuwaveApp:
    """Main muwave application."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        party_name: Optional[str] = None,
        is_ai: bool = False,
    ):
        """
        Initialize the muwave application.
        
        Args:
            config_path: Path to configuration file
            party_name: Name for this party
            is_ai: Whether this party is an AI agent
        """
        # Load configuration
        try:
            self.config = Config(config_path)
        except FileNotFoundError:
            self.config = Config()
        
        # Create party
        self.party = Party(
            party_id=self.config.protocol.get("party_id"),
            name=party_name,
            is_ai=is_ai,
        )
        
        # Set up system prompt if AI
        if is_ai and self.config.ollama.get("system_prompt"):
            self.party.set_system_prompt(self.config.ollama["system_prompt"])
        
        # Initialize components
        self._init_components()
        
        # Running state
        self._running = False
        self._text_input_pending = False
    
    def _init_components(self) -> None:
        """Initialize application components."""
        # UI
        self.ui = create_interface(self.config.ui)
        
        # Logger
        log_config = self.config.logging_config
        self.logger = ConversationLogger(
            log_file=log_config.get("file", "muwave_conversation.log"),
            log_format=log_config.get("format", "text"),
            include_timestamps=log_config.get("timestamps", True),
            log_level=log_config.get("level", "info"),
        )
        
        # Get protocol settings based on speed and redundancy modes
        speed_settings = self.config.get_speed_mode_settings()
        redundancy_settings = self.config.get_redundancy_mode_settings()
        
        protocol_config = {
            **self.config.audio,
            **self.config.protocol,
            "symbol_duration_ms": speed_settings.get("symbol_duration_ms", 50),
            "repetitions": redundancy_settings.get("repetitions", 1),
        }
        
        # Transmitter and Receiver - lazy init (requires audio device)
        self._transmitter: Optional[Transmitter] = None
        self._receiver: Optional[Receiver] = None
        self._protocol_config = protocol_config
        
        # Process sync
        sync_config = self.config.sync
        self.sync = ProcessSync(
            party_id=self.party.party_id,
            method=SyncMethod(sync_config.get("ipc_method", "file")),
            lock_file=sync_config.get("ipc_file", "/tmp/muwave_sync.lock"),
            timeout_seconds=sync_config.get("timeout_seconds", 30),
        )
        
        # Ollama client
        self._ollama: Optional[OllamaClient] = None
    
    def _get_transmitter(self) -> Transmitter:
        """Get or create transmitter."""
        if self._transmitter is None:
            self._transmitter = Transmitter(
                self.party,
                config=self._protocol_config,
            )
            self._transmitter.set_progress_callback(self._on_transmit_progress)
        return self._transmitter
    
    def _get_receiver(self) -> Receiver:
        """Get or create receiver."""
        if self._receiver is None:
            self._receiver = Receiver(
                self.party,
                config=self._protocol_config,
            )
            self._receiver.set_progress_callback(self._on_receive_progress)
            self._receiver.set_message_callback(self._on_message_received)
        return self._receiver
    
    def _get_ollama(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._ollama is None:
            self._ollama = create_ollama_client(self.config.ollama)
        return self._ollama
    
    def _on_transmit_progress(self, progress) -> None:
        """Handle transmission progress updates."""
        status_map = {
            "waiting": TransmitStatus.WAITING,
            "sending": TransmitStatus.SENDING,
            "sent": TransmitStatus.SENT,
            "error": TransmitStatus.ERROR,
        }
        status = status_map.get(progress.status, TransmitStatus.WAITING)
        
        if hasattr(self, '_current_transmit_text'):
            self.ui.show_transmission_progress(
                self._current_transmit_text,
                progress.progress_percent / 100,
                status,
            )
    
    def _on_receive_progress(self, progress) -> None:
        """Handle reception progress updates."""
        state_map = {
            ReceiverState.IDLE: ReceiveStatus.IDLE,
            ReceiverState.LISTENING: ReceiveStatus.LISTENING,
            ReceiverState.RECEIVING: ReceiveStatus.RECEIVING,
            ReceiverState.DECODING: ReceiveStatus.RECEIVING,
            ReceiverState.ERROR: ReceiveStatus.ERROR,
        }
        status = state_map.get(progress.state, ReceiveStatus.IDLE)
        
        sender = progress.sender_signature[:8] if progress.sender_signature else None
        
        if not progress.is_own_transmission:
            self.ui.show_receiving_output(
                progress.partial_content,
                status,
                sender,
            )
    
    def _on_message_received(self, message: Message, is_own: bool) -> None:
        """Handle received message."""
        if is_own:
            return  # Ignore own transmissions
        
        self.logger.log_message_received(
            self.party.party_id,
            message.content,
            message.sender_id,
        )
        
        self.ui.show_receiving_output(
            message.content,
            ReceiveStatus.COMPLETE,
            message.sender_id[:8] if message.sender_id else None,
        )
    
    def send_message(self, text: str, wait_for_audio: bool = True) -> bool:
        """
        Send a message.
        
        Args:
            text: Message text
            wait_for_audio: Whether to wait for other audio to complete
            
        Returns:
            True if sent successfully
        """
        if wait_for_audio and self.config.sync.get("wait_for_audio", True):
            # Wait for any active audio to complete
            if not self.sync.wait_for_audio_complete():
                self.ui.print_error("Timeout waiting for audio")
                return False
        
        # Acquire audio lock
        with self.sync.audio_lock():
            self._current_transmit_text = text
            
            self.logger.log_transmission_start(self.party.party_id, len(text))
            
            transmitter = self._get_transmitter()
            success = transmitter.transmit_text(text, blocking=True)
            
            if success:
                self.logger.log_message_sent(self.party.party_id, text)
                self.ui.print_success("Message sent")
            else:
                self.ui.print_error("Failed to send message")
            
            return success
    
    def send_to_ai(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to AI and get response.
        
        Args:
            prompt: User prompt
            
        Returns:
            AI response or None if error
        """
        try:
            ollama = self._get_ollama()
            
            if not ollama.is_available():
                self.ui.print_error("Ollama is not available")
                return None
            
            self.logger.log_ai_request(self.party.party_id, prompt)
            self.ui.print_info(f"Sending to AI ({ollama.config.model})...")
            
            response = ollama.chat(prompt)
            
            self.logger.log_ai_response(self.party.party_id, response)
            
            return response
            
        except OllamaError as e:
            self.ui.print_error(f"AI error: {e}")
            return None
    
    def start_listening(self) -> None:
        """Start listening for messages."""
        receiver = self._get_receiver()
        receiver.start_listening()
        self.ui.print_info("Listening for messages...")
    
    def stop_listening(self) -> None:
        """Stop listening for messages."""
        if self._receiver:
            self._receiver.stop_listening()
        self.ui.print_info("Stopped listening")
    
    def run_interactive(self) -> None:
        """Run interactive mode."""
        self._running = True
        
        # Handle Ctrl+C
        def signal_handler(sig, frame):
            self._running = False
            self.ui.print_info("\nExiting...")
        
        signal.signal(signal.SIGINT, signal_handler)
        
        self.ui.clear()
        self.ui.print_header()
        self.ui.print_info(f"Party: {self.party.name} ({self.party.party_id[:8]})")
        self.ui.print_info(f"Platform: {get_platform()}")
        self.ui.print_info(f"Speed mode: {self.config.speed.get('mode', 'medium')}")
        self.ui.print_info(f"Redundancy mode: {self.config.redundancy.get('mode', 'medium')}")
        self.ui.console.print()
        self.ui.show_help()
        self.ui.console.print()
        
        self.logger.log_party_join(self.party.party_id, self.party.name)
        
        while self._running:
            try:
                user_input = self.ui.prompt_input()
                self._handle_command(user_input)
            except EOFError:
                break
            except KeyboardInterrupt:
                break
        
        self.cleanup()
    
    def _handle_command(self, user_input: str) -> None:
        """Handle user command."""
        if not user_input.strip():
            return
        
        parts = user_input.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "/send":
            if args:
                self.send_message(args)
            else:
                self.ui.print_error("Usage: /send <message>")
        
        elif command == "/ai":
            if args:
                response = self.send_to_ai(args)
                if response:
                    self.ui.console.print(f"\n[bold cyan]AI:[/] {response}\n")
                    # Optionally transmit the AI response
                    if self.ui.prompt_input("Transmit response? (y/n): ").lower() == 'y':
                        self.send_message(response)
            else:
                self.ui.print_error("Usage: /ai <prompt>")
        
        elif command == "/listen":
            self.start_listening()
        
        elif command == "/stop":
            self.stop_listening()
        
        elif command == "/clear":
            if self._ollama:
                self._ollama.clear_context()
            self.ui.print_success("Conversation cleared")
        
        elif command == "/status":
            self._show_status()
        
        elif command == "/config":
            self._show_config()
        
        elif command == "/help":
            self.ui.show_help()
        
        elif command == "/quit" or command == "/exit":
            self._running = False
        
        elif command.startswith("/"):
            self.ui.print_error(f"Unknown command: {command}")
            self.ui.print_info("Type /help for available commands")
        
        else:
            # Treat as message to send
            self.send_message(user_input)
    
    def _show_status(self) -> None:
        """Show current status."""
        self.ui.console.print()
        self.ui.console.print("[bold]Status:[/]")
        self.ui.console.print(f"  Party: {self.party.name}")
        self.ui.console.print(f"  ID: {self.party.party_id[:16]}...")
        self.ui.console.print(f"  Listening: {self._receiver.is_listening() if self._receiver else False}")
        self.ui.console.print(f"  Audio active: {self.sync.is_audio_active()}")
        
        if self._ollama:
            self.ui.console.print(f"  AI available: {self._ollama.is_available()}")
            self.ui.console.print(f"  AI model: {self._ollama.config.model}")
            self.ui.console.print(f"  Context length: {self._ollama.get_context_length()} chars")
        
        self.ui.console.print()
    
    def _show_config(self) -> None:
        """Show current configuration."""
        self.ui.console.print()
        self.ui.console.print("[bold]Configuration:[/]")
        self.ui.console.print(f"  Speed mode: {self.config.speed.get('mode')}")
        self.ui.console.print(f"  Redundancy mode: {self.config.redundancy.get('mode')}")
        self.ui.console.print(f"  Sample rate: {self.config.audio.get('sample_rate')} Hz")
        self.ui.console.print(f"  Base frequency: {self.config.protocol.get('base_frequency')} Hz")
        self.ui.console.print(f"  Ollama mode: {self.config.ollama.get('mode')}")
        self.ui.console.print(f"  Log file: {self.config.logging_config.get('file')}")
        self.ui.console.print()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_listening()
        self.sync.cleanup()
        self.logger.log_party_leave(self.party.party_id, self.party.name)
        self.ui.print_info("Goodbye!")


@click.group()
@click.version_option(version="0.1.0")
def main():
    """muwave - Sound-based communication protocol for AI agents."""
    pass


@main.command()
@click.option('--config', '-c', type=str, default=None, help='Path to configuration file')
@click.option('--name', '-n', type=str, default=None, help='Party name')
@click.option('--ai', is_flag=True, help='Enable AI agent mode')
def run(config: Optional[str], name: Optional[str], ai: bool):
    """Run muwave in interactive mode."""
    app = MuwaveApp(config_path=config, party_name=name, is_ai=ai)
    app.run_interactive()


@main.command()
@click.argument('message')
@click.option('--config', '-c', type=str, default=None, help='Path to configuration file')
@click.option('--name', '-n', type=str, default=None, help='Party name')
def send(message: str, config: Optional[str], name: Optional[str]):
    """Send a message and exit."""
    app = MuwaveApp(config_path=config, party_name=name)
    app.send_message(message)
    app.cleanup()


@main.command()
@click.option('--config', '-c', type=str, default=None, help='Path to configuration file')
@click.option('--name', '-n', type=str, default=None, help='Party name')
@click.option('--timeout', '-t', type=float, default=30.0, help='Timeout in seconds')
def listen(config: Optional[str], name: Optional[str], timeout: float):
    """Listen for messages."""
    app = MuwaveApp(config_path=config, party_name=name)
    
    app.ui.print_header()
    app.ui.print_info(f"Listening for {timeout} seconds...")
    
    receiver = app._get_receiver()
    result = receiver.receive_once(timeout_seconds=timeout)
    
    if result:
        message, is_own = result
        if not is_own:
            app.ui.print_success(f"Received: {message.content}")
    else:
        app.ui.print_warning("No message received")
    
    app.cleanup()


@main.command()
@click.argument('prompt')
@click.option('--config', '-c', type=str, default=None, help='Path to configuration file')
@click.option('--model', '-m', type=str, default=None, help='Ollama model to use')
@click.option('--transmit', '-t', is_flag=True, help='Transmit the AI response')
def ai(prompt: str, config: Optional[str], model: Optional[str], transmit: bool):
    """Send a prompt to AI."""
    app = MuwaveApp(config_path=config)
    
    if model:
        app.config.set("ollama.model.name", model)
        app._ollama = None  # Force recreation with new model
    
    response = app.send_to_ai(prompt)
    
    if response:
        app.ui.console.print(f"\n[bold cyan]AI:[/] {response}\n")
        
        if transmit:
            app.send_message(response)
    
    app.cleanup()


@main.command()
def devices():
    """List available audio devices."""
    from muwave.audio.device import AudioDevice
    
    console = click.get_current_context().obj or None
    
    devices = AudioDevice.list_devices()
    
    if not devices:
        click.echo("No audio devices found (sounddevice may not be available)")
        return
    
    click.echo("\nAvailable audio devices:")
    click.echo("-" * 60)
    
    for d in devices:
        inputs = f"{d['inputs']} in" if d['inputs'] > 0 else ""
        outputs = f"{d['outputs']} out" if d['outputs'] > 0 else ""
        channels = ", ".join(filter(None, [inputs, outputs]))
        
        click.echo(f"  [{d['index']}] {d['name']}")
        click.echo(f"      {channels}, {int(d['default_samplerate'])} Hz")
    
    click.echo()
    click.echo(f"Default input: {AudioDevice.get_default_input()}")
    click.echo(f"Default output: {AudioDevice.get_default_output()}")


@main.command()
@click.option('--container', '-c', type=str, default='ollama', help='Container name')
@click.option('--port', '-p', type=int, default=11434, help='Port to expose')
def docker_start(container: str, port: int):
    """Start Ollama Docker container."""
    click.echo(f"Starting Ollama container '{container}'...")
    
    if not is_docker_available():
        click.echo("Error: Docker is not available")
        return
    
    if is_ollama_container_running(container):
        click.echo(f"Container '{container}' is already running")
        return
    
    success, message = start_ollama_container(container, port)
    
    if success:
        click.echo(f"Success: {message}")
        click.echo(f"Ollama API available at http://localhost:{port}")
    else:
        click.echo(f"Error: {message}")


@main.command()
@click.option('--output', '-o', type=str, default='config.yaml', help='Output file path')
def init(output: str):
    """Initialize a new configuration file."""
    from muwave.core.config import Config
    
    config = Config()
    config.save(output)
    
    click.echo(f"Configuration saved to {output}")
    click.echo("Edit this file to customize muwave settings.")


@main.command()
@click.option('--config', '-c', type=str, default=None, help='Path to configuration file')
@click.option('--host', '-h', type=str, default='127.0.0.1', help='Host to bind to')
@click.option('--port', '-p', type=int, default=5000, help='Port to listen on')
@click.option('--party', type=str, multiple=True, help='Party names to create')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def web(config: Optional[str], host: str, port: int, party: tuple, debug: bool):
    """Start the web interface."""
    try:
        from muwave.web.server import MuwaveWebServer
        from muwave.core.party import Party
    except ImportError as e:
        click.echo(f"Error: Web dependencies not installed. Run: pip install flask flask-socketio psutil")
        click.echo(f"Details: {e}")
        return
    
    click.echo(f"Starting muwave web interface on http://{host}:{port}")
    
    # Load config
    try:
        cfg = Config(config) if config else Config()
    except FileNotFoundError:
        cfg = Config()
    
    # Create server
    server = MuwaveWebServer(config=cfg, host=host, port=port)
    
    # Create parties
    if party:
        for name in party:
            p = Party(name=name)
            server.add_party(p)
            click.echo(f"  Added party: {name} ({p.party_id[:8]}...)")
    else:
        # Create default parties
        p1 = Party(name="Party-1")
        p2 = Party(name="Party-2")
        server.add_party(p1)
        server.add_party(p2)
        click.echo(f"  Added party: Party-1 ({p1.party_id[:8]}...)")
        click.echo(f"  Added party: Party-2 ({p2.party_id[:8]}...)")
    
    click.echo()
    click.echo("Available endpoints:")
    click.echo(f"  Dashboard: http://{host}:{port}/")
    click.echo(f"  User Input: http://{host}:{port}/user")
    for pid, state in server._parties.items():
        click.echo(f"  {state.party.name}: http://{host}:{port}/party/{pid}")
    click.echo()
    click.echo("Press Ctrl+C to stop the server")
    
    try:
        server.start(debug=debug)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        server.stop()


@main.command()
@click.argument('prompt')
@click.option('--output', '-o', type=str, default='output.wav', help='Output WAV file path')
@click.option('--config', '-c', type=str, default=None, help='Path to configuration file')
@click.option('--name', '-n', type=str, default=None, help='Party name')
def generate(prompt: str, output: str, config: Optional[str], name: Optional[str]):
    """Generate a sound wave from a prompt and save it to a WAV file.
    
    This command takes a text prompt, converts it to an audio signal using
    FSK modulation, and saves it as a WAV file that can be downloaded and played.
    
    Example:
        muwave generate "Hello, World!" -o hello.wav
    """
    from scipy.io import wavfile
    from muwave.audio.fsk import FSKModulator, FSKConfig
    from muwave.ui.interface import create_interface
    
    # Load configuration
    try:
        cfg = Config(config) if config else Config()
    except FileNotFoundError:
        cfg = Config()
    
    # Create party for signature
    party = Party(
        party_id=cfg.protocol.get("party_id"),
        name=name,
        is_ai=False,
    )
    
    # Create UI for output
    ui = create_interface(cfg.ui)
    ui.print_header()
    ui.print_info(f"Generating sound wave for: {prompt}")
    
    # Get protocol settings based on speed and redundancy modes
    speed_settings = cfg.get_speed_mode_settings()
    redundancy_settings = cfg.get_redundancy_mode_settings()
    
    # Set up FSK configuration
    fsk_config = FSKConfig(
        sample_rate=cfg.audio.get("sample_rate", 44100),
        base_frequency=cfg.protocol.get("base_frequency", 1000),
        frequency_step=cfg.protocol.get("frequency_step", 100),
        num_frequencies=cfg.protocol.get("num_frequencies", 16),
        symbol_duration_ms=speed_settings.get("symbol_duration_ms", 50),
        start_frequency=cfg.protocol.get("start_frequency", 500),
        end_frequency=cfg.protocol.get("end_frequency", 600),
        signal_duration_ms=cfg.protocol.get("signal_duration_ms", 200),
        silence_ms=cfg.protocol.get("silence_ms", 50),
        volume=cfg.audio.get("volume", 0.8),
    )
    
    repetitions = redundancy_settings.get("repetitions", 1)
    
    # Create modulator and encode the text
    modulator = FSKModulator(fsk_config)
    audio_samples = modulator.encode_text(
        prompt,
        signature=party.signature,
        repetitions=repetitions,
    )
    
    # Get sample rate from config
    sample_rate = fsk_config.sample_rate
    
    # Convert float32 samples to int16 for WAV file
    # Scale from [-1.0, 1.0] to [-32767, 32767]
    import numpy as np
    audio_int16 = (audio_samples * 32767).astype(np.int16)
    
    # Save to WAV file
    wavfile.write(output, sample_rate, audio_int16)
    
    # Calculate duration
    duration_seconds = len(audio_samples) / sample_rate
    
    ui.print_success(f"Sound wave saved to: {output}")
    ui.print_info(f"Duration: {duration_seconds:.2f} seconds")
    ui.print_info(f"Sample rate: {sample_rate} Hz")
    ui.print_info(f"Samples: {len(audio_samples)}")


if __name__ == "__main__":
    main()
