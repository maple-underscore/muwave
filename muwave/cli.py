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
import numpy as np
from scipy.io import wavfile

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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from muwave.utils.helpers import (
    get_platform,
    is_docker_available,
    is_ollama_container_running,
    start_ollama_container,
    format_duration,
)
from muwave.audio.fsk import FSKModulator, FSKDemodulator, FSKConfig


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
        from muwave.utils.formats import format_content_for_display, FormatMetadata, ContentFormat
        
        # If it's our own signature while we're currently transmitting, ignore
        # to prevent echo/duplicates. But allow showing "self" messages when
        # not actively transmitting (e.g., playing back a test file locally).
        if is_own:
            try:
                if self._get_transmitter().is_transmitting():
                    return
            except Exception:
                return
        
        self.logger.log_message_received(
            self.party.party_id,
            message.content,
            message.sender_id,
        )
        
        # Format content for display if it has format metadata
        display_content = message.content
        if message.content_format:
            try:
                # Reconstruct format metadata
                format_type = None
                for fmt in ContentFormat:
                    if fmt.value == message.content_format or fmt.name.lower() == message.content_format.lower():
                        format_type = fmt
                        break
                if format_type:
                    format_meta = FormatMetadata(format_type, language=message.format_language)
                    display_content = format_content_for_display(message.content, format_meta)
            except Exception:
                pass  # Fall back to plain display
        
        self.ui.show_receiving_output(
            display_content,
            ReceiveStatus.COMPLETE,
            message.sender_id[:8] if message.sender_id else None,
        )
    
    def send_message(self, text: str, wait_for_audio: bool = True, 
                     format_meta: Optional['FormatMetadata'] = None) -> bool:
        """
        Send a message.
        
        Args:
            text: Message text
            wait_for_audio: Whether to wait for other audio to complete
            format_meta: Optional format metadata for the content
            
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
            
            # Create message with format metadata if provided
            if format_meta:
                message = self.party.create_message(text)
                message.content_format = format_meta.format_type.value
                message.format_language = format_meta.language
                success = transmitter.transmit(message, blocking=True)
            else:
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

    def monitor_input(self, duration: float = 5.0) -> None:
        """Monitor input RMS levels for a short duration to verify device selection."""
        import numpy as np
        receiver = self._get_receiver()
        device = receiver._audio_device
        self.ui.print_info(f"Monitoring input for {duration:.1f}s (device: {device.input_device})…")
        samples = device.record(duration_seconds=duration)  # blocking
        if samples is None or len(samples) == 0:
            self.ui.print_warning("No samples captured. Check input device.")
            return
        arr = np.asarray(samples, dtype=np.float32)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        peak = float(np.max(np.abs(arr)))
        eff_rms = rms * float(getattr(receiver, "_input_gain", 1.0))
        self.ui.print_info(f"Input RMS: {rms:.4f}, Peak: {peak:.4f} (effective RMS with gain: {eff_rms:.4f})")
        if rms < 0.01:
            self.ui.print_warning("Very low level. Consider selecting a loopback device and/or increasing input_gain.")

    def set_input_gain(self, gain: float) -> None:
        """Set input gain at runtime for the receiver."""
        try:
            receiver = self._get_receiver()
            receiver._input_gain = float(gain)
            self.ui.print_success(f"Set input_gain to {gain}")
        except Exception as e:
            self.ui.print_error(f"Failed to set input_gain: {e}")
    
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
        
        elif command == "/monitor":
            # Monitor input for 5 seconds
            self.monitor_input(5.0)
        
        elif command == "/devices":
            try:
                from muwave.audio.device import AudioDevice
                devices = AudioDevice.list_devices()
                if not devices:
                    self.ui.print_warning("No audio devices found or sounddevice unavailable.")
                else:
                    self.ui.console.print("\n[bold]Available audio devices:[/]")
                    for d in devices:
                        ins = d['inputs']
                        outs = d['outputs']
                        sr = int(d['default_samplerate'])
                        self.ui.console.print(f"  [{d['index']}] {d['name']}  ({ins} in, {outs} out, {sr} Hz)")
                    self.ui.console.print()
                    from muwave.audio.device import AudioDevice as AD
                    self.ui.print_info(f"Default input: {AD.get_default_input()}  Default output: {AD.get_default_output()}")
            except Exception as e:
                self.ui.print_error(f"Failed to list devices: {e}")
        
        elif command == "/gain":
            try:
                val = float(args)
            except (TypeError, ValueError):
                self.ui.print_error("Usage: /gain <float>, e.g., /gain 2.0")
                return
            self.set_input_gain(val)

        elif command == "/usein":
            try:
                idx = int(args)
            except (TypeError, ValueError):
                self.ui.print_error("Usage: /usein <device_index>")
                return
            try:
                receiver = self._get_receiver()
                was_listening = receiver.is_listening()
                if was_listening:
                    receiver.stop_listening()
                receiver._audio_device.input_device = idx
                self.ui.print_success(f"Set input_device to {idx}")
                if was_listening:
                    receiver.start_listening()
            except Exception as e:
                self.ui.print_error(f"Failed to set input device: {e}")

        elif command == "/inject":
            path = args.strip()
            if not path:
                self.ui.print_error("Usage: /inject <wav_path>")
                return
            try:
                receiver = self._get_receiver()
                self.ui.print_info(f"Injecting WAV file: {path}")
                receiver.inject_wav_file(path)
                self.ui.print_success("Injected samples. If listening, decoding will trigger shortly.")
            except Exception as e:
                self.ui.print_error(f"Failed to inject: {e}")
        
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
@click.version_option(version="0.1.4")
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
@click.option('--speed', '-s', type=str, default=None, help='Speed mode (overrides config, see config.yaml for available modes)')
@click.option('--redundancy', '-r', type=str, default=None, help='Redundancy mode (overrides config, see config.yaml for available modes)')
@click.option('--symbol-duration', type=float, default=None, help='Symbol duration in milliseconds (overrides speed mode)')
@click.option('--repetitions', type=int, default=None, help='Number of repetitions (overrides redundancy mode)')
@click.option('--volume', '-v', type=float, default=None, help='Audio volume (0.0 to 1.0)')
@click.option('--channels', type=click.Choice(['1', '2', '3', '4'], case_sensitive=False), default=None, help='Number of frequency channels (1=mono, 2=dual, 3=tri, 4=quad). Higher=faster but less accurate.')
@click.option('--file', '-f', is_flag=True, help='Treat MESSAGE as a file path and encode its contents')
@click.option('--format', type=click.Choice(['plain', 'markdown', 'html', 'json', 'xml', 'code', 'yaml', 'auto'], case_sensitive=False), default='auto', help='Content format (auto=detect automatically)')
@click.option('--language', '-l', type=str, default=None, help='Programming language for code blocks')
def send(message: str, config: Optional[str], name: Optional[str],
         speed: Optional[str], redundancy: Optional[str],
         symbol_duration: Optional[float], repetitions: Optional[int],
         volume: Optional[float], channels: Optional[str], file: bool,
         format: Optional[str], language: Optional[str]):
    """Send a message and exit.
    
    This command takes a text message (or file with --file), converts it to an audio 
    signal using FSK modulation, and transmits it through the audio device.
    
    Examples:
        muwave send "Hello, World!"
        muwave send message.txt --file
        muwave send "Test message" --speed fast --redundancy high
        muwave send script.txt -f --format code --language python
    """
    # Load configuration
    try:
        cfg = Config(config) if config else Config()
    except FileNotFoundError:
        cfg = Config()
    
    # Validate speed and redundancy modes if provided
    if speed:
        available_speeds = list(cfg.speed.get('modes', {}).keys())
        if speed not in available_speeds:
            click.echo(f"Error: Invalid speed mode '{speed}'. Available modes: {', '.join(available_speeds)}")
            return
    
    if redundancy:
        available_redundancy = list(cfg.redundancy.get('modes', {}).keys())
        if redundancy not in available_redundancy:
            click.echo(f"Error: Invalid redundancy mode '{redundancy}'. Available modes: {', '.join(available_redundancy)}")
            return
    
    # Read from file if --file flag is set
    text_to_send = message
    if file:
        try:
            with open(message, 'r', encoding='utf-8') as f:
                text_to_send = f.read()
        except FileNotFoundError:
            click.echo(f"Error: File not found: {message}")
            return
        except Exception as e:
            click.echo(f"Error reading file: {e}")
            return
    
    # Apply configuration overrides
    if speed:
        cfg.set("speed.mode", speed)
    
    if redundancy:
        cfg.set("redundancy.mode", redundancy)
    
    # Get protocol settings based on speed and redundancy modes
    speed_settings = cfg.get_speed_mode_settings()
    redundancy_settings = cfg.get_redundancy_mode_settings()
    
    # Apply specific overrides
    if symbol_duration is not None:
        speed_settings["symbol_duration_ms"] = symbol_duration
    
    if repetitions is not None:
        redundancy_settings["repetitions"] = repetitions
    
    if volume is not None:
        if not 0.0 <= volume <= 1.0:
            click.echo("Error: Volume must be between 0.0 and 1.0")
            return
        cfg.set("audio.volume", volume)
    
    # Set num_channels
    if channels:
        cfg.set("audio.num_channels", int(channels))
    
    # Update protocol config with new settings
    protocol_config = {
        **cfg.audio,
        **cfg.protocol,
        "symbol_duration_ms": speed_settings.get("symbol_duration_ms", 50),
        "repetitions": redundancy_settings.get("repetitions", 1),
    }
    
    # Create app with updated config
    app = MuwaveApp(config_path=config, party_name=name)
    app._protocol_config = protocol_config
    app._transmitter = None  # Force recreation with new config
    
    # Detect or set format
    from muwave.utils.formats import FormatDetector, FormatMetadata, ContentFormat
    
    format_meta = None
    if format and format != 'auto':
        # Map format string to ContentFormat enum
        format_map = {
            'plain': ContentFormat.PLAIN_TEXT,
            'markdown': ContentFormat.MARKDOWN,
            'html': ContentFormat.HTML,
            'json': ContentFormat.JSON,
            'xml': ContentFormat.XML,
            'code': ContentFormat.CODE,
            'yaml': ContentFormat.YAML,
        }
        if format.lower() in format_map:
            format_meta = FormatMetadata(format_map[format.lower()], language=language)
    else:
        # Auto-detect format
        format_meta = FormatDetector.detect(text_to_send)
    
    # Send the message
    app.send_message(text_to_send, format_meta=format_meta)
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
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--config', '-c', type=str, default=None, help='Path to configuration file')
@click.option('--speed', '-s', type=str, default=None, help='Speed mode used during encoding (see config.yaml for available modes)')
@click.option('--redundancy', '-r', type=str, default=None, help='Redundancy mode used during encoding (see config.yaml for available modes)')
@click.option('--symbol-duration', type=float, default=None, help='Symbol duration in milliseconds (overrides speed mode)')
@click.option('--repetitions', type=int, default=None, help='Number of repetitions (overrides redundancy mode)')
@click.option('--channels', type=click.Choice(['1', '2', '3', '4'], case_sensitive=False), default=None, help='Number of frequency channels used during encoding (1=mono, 2=dual, 3=tri, 4=quad)')
@click.option('--threads', type=int, default=None, help='Number of threads for parallel speed auto-detection (default: CPU cores)')
def decode(input_file: str, config: Optional[str], speed: Optional[str], 
           redundancy: Optional[str], symbol_duration: Optional[float], 
           repetitions: Optional[int], channels: Optional[str], threads: Optional[int]):
    """Decode a WAV file and extract the message.
    
    This command takes a WAV file containing an FSK-encoded message,
    decodes it, and prints the decoded text.
    
    By default, uses the settings from config.yaml. If the file was encoded
    with different settings, specify them with --speed and --redundancy options.
    
    Examples:
        muwave decode input.wav
        muwave decode input.wav --speed fast --redundancy high
        muwave decode input.wav --symbol-duration 25 --repetitions 3
    """
    # Load configuration
    try:
        cfg = Config(config) if config else Config()
    except FileNotFoundError:
        cfg = Config()
    
    # Validate speed and redundancy modes if provided
    if speed:
        available_speeds = list(cfg.speed.get('modes', {}).keys())
        if speed not in available_speeds:
            click.echo(f"Error: Invalid speed mode '{speed}'. Available modes: {', '.join(available_speeds)}")
            return
    
    if redundancy:
        available_redundancy = list(cfg.redundancy.get('modes', {}).keys())
        if redundancy not in available_redundancy:
            click.echo(f"Error: Invalid redundancy mode '{redundancy}'. Available modes: {', '.join(available_redundancy)}")
            return
    
    # Create UI for output
    ui = create_interface(cfg.ui)
    ui.print_header()
    ui.print_info(f"Decoding WAV file: {input_file}")
    
    # Read WAV file
    try:
        sample_rate, audio_data = wavfile.read(input_file)
    except Exception as e:
        ui.print_error(f"Failed to read WAV file: {e}")
        return
    
    # Convert to float32 [-1.0, 1.0]
    if audio_data.dtype == np.int16:
        audio_samples = audio_data.astype(np.float32) / 32767.0
    elif audio_data.dtype == np.int32:
        audio_samples = audio_data.astype(np.float32) / 2147483647.0
    elif audio_data.dtype == np.uint8:
        audio_samples = (audio_data.astype(np.float32) - 128) / 128.0
    else:
        audio_samples = audio_data.astype(np.float32)
    
    # Handle stereo by taking first channel
    if len(audio_samples.shape) > 1:
        audio_samples = audio_samples[:, 0]
    
    ui.print_info(f"Sample rate: {sample_rate} Hz")
    ui.print_info(f"Duration: {len(audio_samples) / sample_rate:.2f} seconds")
    ui.print_info(f"Samples: {len(audio_samples)}")
    
    # Detect start/end signals first (works with any speed)
    # Use a temporary demodulator with default settings for signal detection
    temp_config = cfg.create_fsk_config(num_channels=1)
    temp_demod = FSKDemodulator(temp_config)
    
    ui.print_info("Detecting start signal...")
    detected, start_pos = temp_demod.detect_start_signal(audio_samples)
    
    if not detected:
        ui.print_error("No start signal detected")
        return
    
    ui.print_success(f"Start signal detected at position {start_pos}")
    
    # Extract data after start signal
    data_samples = audio_samples[start_pos:]
    
    ui.print_info("Detecting end signal...")
    end_detected, end_pos = temp_demod.detect_end_signal(data_samples)
    
    if not end_detected:
        ui.print_warning("No end signal detected, attempting to decode available data")
        message_samples = data_samples
        absolute_end_pos = len(audio_samples)
    else:
        ui.print_success(f"End signal detected at position {end_pos}")
        message_samples = data_samples[:end_pos]
        absolute_end_pos = start_pos + end_pos
    
    # Add debug information about signal positions
    ui.console.print()
    ui.console.print("[bold cyan]Debug Info:[/]")
    ui.console.print(f"  Start signal ends at: {start_pos}")
    ui.console.print(f"  End signal starts at: {absolute_end_pos}")
    ui.console.print(f"  Data region length: {len(message_samples)} samples")
    
    # Try to read metadata header if no manual settings provided
    detected_channels = None
    detected_duration = None
    detected_base_freq = None
    detected_freq_step = None
    detected_channel_spacing = None
    metadata_valid = False
    
    if channels is None and (speed is None and symbol_duration is None):
        ui.print_info("Reading metadata header...")
        # Create a temporary demodulator to read metadata (uses standardized format)
        temp_config = cfg.create_fsk_config()
        temp_demod = FSKDemodulator(temp_config)
        metadata, _ = temp_demod.decode_metadata(message_samples)
        
        if metadata['valid']:
            detected_channels = metadata['num_channels']
            detected_duration = int(metadata['symbol_duration_ms'])
            detected_base_freq = metadata['base_frequency']
            detected_freq_step = metadata['frequency_step']
            detected_channel_spacing = metadata['channel_spacing']
            detected_sig_length = metadata.get('signature_length', 8)
            metadata_valid = True
            ui.print_success(f"Metadata valid (checksum OK, version {metadata.get('version', 1)})")
            ui.print_success(f"  Channels: {detected_channels}")
            ui.print_success(f"  Symbol duration: {detected_duration}ms")
            ui.print_success(f"  Base frequency: {detected_base_freq}Hz")
            ui.print_success(f"  Frequency step: {detected_freq_step}Hz")
            ui.print_success(f"  Channel spacing: {detected_channel_spacing}Hz")
            ui.print_success(f"  Signature length: {detected_sig_length} bytes")
        else:
            detected_sig_length = 8  # Default for invalid/missing metadata
            ui.print_warning("Metadata header invalid or not found, will auto-detect settings")
    
    # Set num_channels
    num_channels = int(channels) if channels else (detected_channels if detected_channels else 2)
    
    # Auto-detect speed if not specified and not detected from metadata
    if speed is None and symbol_duration is None:
        if detected_duration:
            # Use detected duration from metadata
            symbol_dur = detected_duration
            speed_name = "detected"
        else:
            ui.print_info("Auto-detecting optimal speed settings...")
            
            # Test all speed modes
            test_speeds = [
                ("ultra-fast", 20),
                ("fast", 35),
                ("medium", 60),
                ("slow", 120),
            ]
        
        if not detected_duration:
            import concurrent.futures, os
            best_result = None
            best_confidence = 0.0
            max_workers = threads if threads and threads > 0 else os.cpu_count() or 4
            ui.print_info(f"Parallel speed evaluation using {max_workers} threads...")
            
            def _test_speed(args):
                speed_name, symbol_dur = args
                # Local config copy per thread
                local_cfg = cfg.create_fsk_config(
                    symbol_duration_ms=symbol_dur,
                    num_channels=num_channels,
                )
                redundancy_settings_local = cfg.get_redundancy_mode_settings()
                reps_local = repetitions if repetitions is not None else redundancy_settings_local.get("repetitions", 1)
                demod = FSKDemodulator(local_cfg)
                data, signature, confidence = demod.decode_data(
                    message_samples,
                    signature_length=8,
                    repetitions=reps_local,
                    read_metadata=False,
                )
                # Decode formatted content
                from muwave.utils.formats import FormatEncoder
                if data:
                    text, _ = FormatEncoder.decode(data)
                else:
                    text = None
                return (speed_name, symbol_dur, text, signature, confidence, reps_local)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=ui.console,
            ) as progress:
                task = progress.add_task("[cyan]Testing speed modes...", total=len(test_speeds))
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {executor.submit(_test_speed, ts): ts for ts in test_speeds}
                    for future in concurrent.futures.as_completed(future_map):
                        speed_name, symbol_dur, text, signature, confidence, reps_local = future.result()
                        if text is not None:
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_result = (speed_name, symbol_dur, text, signature, confidence, reps_local)
                                progress.update(task, description=f"[cyan]Testing speeds... [green]✓ {speed_name} ({confidence:.0%})")
                        progress.advance(task)
            
            if best_result is None:
                ui.print_error("Failed to decode with any speed setting")
                return
            
            speed_name, symbol_dur, text, signature, confidence, reps = best_result
            ui.console.print()
            ui.print_success(f"Best match: {speed_name} speed ({symbol_dur}ms) with {confidence:.2%} confidence")
        else:
            # Use detected settings from metadata
            if redundancy:
                cfg.set("redundancy.mode", redundancy)
            redundancy_settings = cfg.get_redundancy_mode_settings()
            reps = repetitions if repetitions is not None else redundancy_settings.get("repetitions", 1)
            
            # Create FSK config with all detected metadata values
            fsk_config_kwargs = {
                'symbol_duration_ms': symbol_dur,
                'num_channels': num_channels,
            }
            if metadata_valid:
                # Use all values from standardized metadata header
                if detected_base_freq:
                    fsk_config_kwargs['base_frequency'] = detected_base_freq
                if detected_freq_step:
                    fsk_config_kwargs['frequency_step'] = detected_freq_step
                if detected_channel_spacing:
                    fsk_config_kwargs['channel_spacing'] = detected_channel_spacing
            
            fsk_config = cfg.create_fsk_config(**fsk_config_kwargs)
            
            ui.print_info(f"Decoding with detected settings: {num_channels} channels, {symbol_dur}ms symbols...")
            
            # Show progress bar while decoding
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=ui.console,
            ) as progress:
                task = progress.add_task("[cyan]Decoding message...", total=100)
                
                # Decode in thread to show progress
            result = [None, None, None, None]  # data, signature, confidence, demodulator
            def decode_thread():
                demodulator = FSKDemodulator(fsk_config)
                result[0], result[1], result[2] = demodulator.decode_data(
                    message_samples,
                    signature_length=8,
                    repetitions=reps,
                    read_metadata=True,
                )
                result[3] = demodulator
            
            thread = threading.Thread(target=decode_thread)
            thread.start()
            
            while thread.is_alive():
                thread.join(timeout=0.05)
                if progress.tasks[task].percentage < 90:
                    progress.update(task, advance=3)
            
            thread.join()
            progress.update(task, completed=100)
            data, signature, confidence, demodulator = result[0], result[1], result[2], result[3]
        
        # Decode formatted content
        from muwave.utils.formats import FormatEncoder
        if data:
            text, format_meta = FormatEncoder.decode(data)
        else:
            text = None
            format_meta = None
        
        ui.console.print()
        ui.print_success(f"Decoded using metadata: {num_channels} channels, {symbol_dur}ms ({confidence:.2%} confidence)")
        
        # Display timing information
        if demodulator:
            timestamps = demodulator.get_last_decode_timestamps()
            if timestamps:
                total_time = timestamps.get('total_duration', 0)
                ui.console.print(f"[dim]⏱  Decode time: {total_time:.3f}s[/dim]")
    else:
        # Use specified settings
        if speed:
            cfg.set("speed.mode", speed)
        if redundancy:
            cfg.set("redundancy.mode", redundancy)
        
        speed_settings = cfg.get_speed_mode_settings()
        redundancy_settings = cfg.get_redundancy_mode_settings()
        
        if symbol_duration is not None:
            speed_settings["symbol_duration_ms"] = symbol_duration
        if repetitions is not None:
            redundancy_settings["repetitions"] = repetitions
        
        symbol_dur = speed_settings.get("symbol_duration_ms", 60)
        reps = redundancy_settings.get("repetitions", 1)
        
        fsk_config = cfg.create_fsk_config(
            symbol_duration_ms=symbol_dur,
            num_channels=num_channels,
        )
        
        ui.console.print()
        ui.print_info(f"Decode settings: {cfg.speed.get('mode', 'medium')} speed (symbol: {symbol_dur}ms), repetitions: {reps}")
        ui.print_info("Decoding message...")
        
        # Show progress bar while decoding
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=ui.console,
        ) as progress:
            task = progress.add_task("[cyan]Decoding message...", total=100)
            
            # Decode in thread to show progress
            result = [None, None, None, None]  # data, signature, confidence, demodulator
            def decode_thread():
                demodulator = FSKDemodulator(fsk_config)
                result[0], result[1], result[2] = demodulator.decode_data(
                    message_samples,
                    signature_length=8,
                    repetitions=reps,
                )
                result[3] = demodulator
            
            thread = threading.Thread(target=decode_thread)
            thread.start()
            
            while thread.is_alive():
                thread.join(timeout=0.05)
                if progress.tasks[task].percentage < 90:
                    progress.update(task, advance=3)
            
            thread.join()
            progress.update(task, completed=100)
            data, signature, confidence, demodulator = result[0], result[1], result[2], result[3]
        
        # Decode formatted content
        from muwave.utils.formats import FormatEncoder
        if data:
            text, format_meta = FormatEncoder.decode(data)
        else:
            text = None
            format_meta = None
        
        if text is None:
            ui.print_error(f"Failed to decode message (confidence: {confidence:.2%})")
            ui.console.print()
            ui.console.print("[bold yellow]Tip:[/] Try removing --speed option to auto-detect the correct settings")
            return
        
        ui.print_success(f"Decoded successfully (confidence: {confidence:.2%})")
        
        # Display timing information
        if demodulator:
            timestamps = demodulator.get_last_decode_timestamps()
            if timestamps:
                total_time = timestamps.get('total_duration', 0)
                ui.console.print(f"[dim]⏱  Decode time: {total_time:.3f}s[/dim]")
    
    # Adaptive confidence threshold based on transmission parameters
    # Fast symbol rates and high channel counts inherently have more interference
    # Adjust warning threshold accordingly
    base_threshold = 0.5
    if symbol_dur and num_channels:
        # Lower threshold for faster symbols (less time for clean frequency separation)
        # Lower threshold for more channels (more inter-channel interference)
        duration_factor = max(0.7, min(1.0, symbol_dur / 40.0))  # 40ms is baseline
        channel_factor = max(0.7, 1.0 - (num_channels - 1) * 0.1)  # Each channel adds 10% interference
        adaptive_threshold = base_threshold * duration_factor * channel_factor
    else:
        adaptive_threshold = base_threshold
    
    if confidence < adaptive_threshold:
        ui.print_warning(f"Low confidence ({confidence:.2%}) - decoding may be inaccurate")
    
    # Format content for display using already-decoded format metadata
    from muwave.utils.formats import format_content_for_display
    
    display_text = format_content_for_display(text, format_meta)
    
    ui.console.print()
    ui.console.print("[bold cyan]Message:[/]")
    ui.console.print(display_text)
    
    if signature:
        ui.console.print()
        ui.console.print("[bold cyan]Sender signature:[/]")
        ui.console.print(f"  {signature.hex()}")
    
    ui.console.print()


@main.command()
@click.argument('prompt')
@click.option('--output', '-o', type=str, default='output.wav', help='Output WAV file path')
@click.option('--config', '-c', type=str, default=None, help='Path to configuration file')
@click.option('--name', '-n', type=str, default=None, help='Party name')
@click.option('--speed', '-s', type=str, default=None, help='Speed mode (overrides config, see config.yaml for available modes)')
@click.option('--redundancy', '-r', type=str, default=None, help='Redundancy mode (overrides config, see config.yaml for available modes)')
@click.option('--symbol-duration', type=float, default=None, help='Symbol duration in milliseconds (overrides speed mode)')
@click.option('--repetitions', type=int, default=None, help='Number of repetitions (overrides redundancy mode)')
@click.option('--volume', '-v', type=float, default=None, help='Audio volume (0.0 to 1.0)')
@click.option('--channels', type=click.Choice(['1', '2', '3', '4'], case_sensitive=False), default=None, help='Number of frequency channels (1=mono, 2=dual, 3=tri, 4=quad). Higher=faster but less accurate.')
@click.option('--file', '-f', is_flag=True, help='Treat PROMPT as a file path and encode its contents')
@click.option('--format', type=click.Choice(['plain', 'markdown', 'html', 'json', 'xml', 'code', 'yaml', 'auto'], case_sensitive=False), default='auto', help='Content format (auto=detect automatically)')
@click.option('--language', '-l', type=str, default=None, help='Programming language for code blocks')
def generate(prompt: str, output: str, config: Optional[str], name: Optional[str], 
             speed: Optional[str], redundancy: Optional[str], 
             symbol_duration: Optional[float], repetitions: Optional[int],
             volume: Optional[float], channels: Optional[str], file: bool,
             format: Optional[str], language: Optional[str]):
    """Generate a sound wave from a prompt or file and save it to a WAV file.
    
    This command takes a text prompt (or file with --file), converts it to an audio 
    signal using FSK modulation, and saves it as a WAV file.
    
    Examples:
        muwave generate "Hello, World!" -o hello.wav
        muwave generate message.txt --file -o message.wav
        muwave generate "Test message" -o test.wav --speed fast --redundancy high
        muwave generate script.txt -f --symbol-duration 25 --repetitions 3
    """
    # Load configuration
    try:
        cfg = Config(config) if config else Config()
    except FileNotFoundError:
        cfg = Config()
    
    # Validate speed and redundancy modes if provided
    if speed:
        available_speeds = list(cfg.speed.get('modes', {}).keys())
        if speed not in available_speeds:
            click.echo(f"Error: Invalid speed mode '{speed}'. Available modes: {', '.join(available_speeds)}")
            return
    
    if redundancy:
        available_redundancy = list(cfg.redundancy.get('modes', {}).keys())
        if redundancy not in available_redundancy:
            click.echo(f"Error: Invalid redundancy mode '{redundancy}'. Available modes: {', '.join(available_redundancy)}")
            return
    
    # Read from file if --file flag is set
    text_to_encode = prompt
    if file:
        try:
            with open(prompt, 'r', encoding='utf-8') as f:
                text_to_encode = f.read()
        except FileNotFoundError:
            click.echo(f"Error: File not found: {prompt}")
            return
        except Exception as e:
            click.echo(f"Error reading file: {e}")
            return
    
    # Create party for signature
    party = Party(
        party_id=cfg.protocol.get("party_id"),
        name=name,
        is_ai=False,
    )
    
    # Create UI for output
    ui = create_interface(cfg.ui)
    ui.print_header()
    if file:
        ui.print_info(f"Encoding file: {prompt} ({len(text_to_encode)} chars)")
    else:
        ui.print_info(f"Generating sound wave for: {text_to_encode}")
    
    # Override speed mode if specified
    if speed:
        cfg.set("speed.mode", speed)
    
    # Override redundancy mode if specified
    if redundancy:
        cfg.set("redundancy.mode", redundancy)
    
    # Get protocol settings based on speed and redundancy modes
    speed_settings = cfg.get_speed_mode_settings()
    redundancy_settings = cfg.get_redundancy_mode_settings()
    
    # Apply specific overrides
    if symbol_duration is not None:
        speed_settings["symbol_duration_ms"] = symbol_duration
    
    if repetitions is not None:
        redundancy_settings["repetitions"] = repetitions
    
    if volume is not None:
        if not 0.0 <= volume <= 1.0:
            ui.print_error("Volume must be between 0.0 and 1.0")
            return
        cfg.set("audio.volume", volume)
    
    # Set num_channels
    num_channels = int(channels) if channels else 2
    
    # Set up FSK configuration
    fsk_config = cfg.create_fsk_config(
        symbol_duration_ms=speed_settings.get("symbol_duration_ms"),
        num_channels=num_channels,
    )
    
    reps = redundancy_settings.get("repetitions", 1)
    
    # Display settings
    ui.print_info(f"Speed: {cfg.speed.get('mode', 'medium')} (symbol: {fsk_config.symbol_duration_ms}ms)")
    ui.print_info(f"Redundancy: {cfg.redundancy.get('mode', 'medium')} (repetitions: {reps})")
    ui.print_info(f"Volume: {fsk_config.volume:.1f}")
    ui.print_info(f"Channels: {num_channels} {'(mono)' if num_channels == 1 else '(dual)' if num_channels == 2 else '(tri)' if num_channels == 3 else '(quad)'}")
    
    # Detect or set format
    from muwave.utils.formats import FormatEncoder, FormatDetector, FormatMetadata, ContentFormat
    
    format_meta = None
    if format and format != 'auto':
        # Map format string to ContentFormat enum
        format_map = {
            'plain': ContentFormat.PLAIN_TEXT,
            'markdown': ContentFormat.MARKDOWN,
            'html': ContentFormat.HTML,
            'json': ContentFormat.JSON,
            'xml': ContentFormat.XML,
            'code': ContentFormat.CODE,
            'yaml': ContentFormat.YAML,
        }
        if format.lower() in format_map:
            format_meta = FormatMetadata(format_map[format.lower()], language=language)
            ui.print_info(f"Format: {format.upper()}" + (f" ({language})" if language else ""))
    else:
        # Auto-detect format
        format_meta = FormatDetector.detect(text_to_encode)
        ui.print_info(f"Detected format: {format_meta.format_type.name}" + 
                     (f" ({format_meta.language})" if format_meta.language else "") +
                     f" (confidence: {format_meta.confidence:.0%})")
    
    # Encode content with format metadata
    encoded_data = FormatEncoder.encode(text_to_encode, format_meta)
    
    # Create modulator and encode the data
    modulator = FSKModulator(fsk_config)
    
    # Show progress bar while encoding
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=ui.console,
    ) as progress:
        task = progress.add_task("[cyan]Encoding audio...", total=100)
        
        # Start encoding in a separate thread to allow progress updates
        import threading
        result = [None, None]  # audio, timestamps
        def encode_thread():
            audio, timestamps = modulator.encode_data(
                encoded_data,
                signature=party.signature,
                repetitions=reps,
            )
            result[0] = audio
            result[1] = timestamps
        
        thread = threading.Thread(target=encode_thread)
        thread.start()
        
        # Update progress while encoding
        while thread.is_alive():
            thread.join(timeout=0.05)
            if progress.tasks[task].percentage < 90:
                progress.update(task, advance=5)
        
        thread.join()
        progress.update(task, completed=100)
        audio_samples = result[0]
        encode_timestamps = result[1]
    
    # Display encoding timing
    if encode_timestamps:
        encode_time = encode_timestamps.get('total_duration', 0)
        ui.console.print(f"[dim]⏱  Encode time: {encode_time:.3f}s[/dim]")
        ui.console.print()
    
    # Get sample rate from config
    sample_rate = fsk_config.sample_rate
    
    # Calculate signal positions for debug info
    start_signal_samples = int(sample_rate * fsk_config.signal_duration_ms / 1000)
    silence_samples = int(sample_rate * fsk_config.silence_ms / 1000)
    
    # Multi-channel uses 1 symbol per byte, single-channel uses 2
    symbols_per_byte = 1 if fsk_config.num_channels >= 2 else 2
    
    signature_samples = 8 * symbols_per_byte * int(sample_rate * fsk_config.symbol_duration_ms / 1000)  # 8 bytes
    length_samples = 2 * symbols_per_byte * int(sample_rate * fsk_config.symbol_duration_ms / 1000)  # 2 bytes
    
    start_pos = start_signal_samples + silence_samples
    data_start_pos = start_pos + signature_samples + (silence_samples // 2) + length_samples
    
    # Calculate data length
    data_length = len(prompt.encode('utf-8'))
    data_samples = data_length * symbols_per_byte * int(sample_rate * fsk_config.symbol_duration_ms / 1000) * reps
    if reps > 1:
        data_samples += (reps - 1) * (silence_samples // 2)
    
    end_start_pos = data_start_pos + data_samples + silence_samples
    
    # Convert float32 samples to int16 for WAV file
    # Scale from [-1.0, 1.0] to [-32767, 32767]
    audio_int16 = (audio_samples * 32767).astype(np.int16)
    
    # Save to WAV file
    wavfile.write(output, sample_rate, audio_int16)
    
    # Calculate duration
    duration_seconds = len(audio_samples) / sample_rate
    
    ui.print_success(f"Sound wave saved to: {output}")
    ui.print_info(f"Duration: {duration_seconds:.2f} seconds")
    ui.print_info(f"Sample rate: {sample_rate} Hz")
    ui.print_info(f"Samples: {len(audio_samples)}")
    ui.console.print()
    ui.console.print("[bold cyan]Debug Info:[/]")
    ui.console.print(f"  Signature: {party.signature.hex()}")
    ui.console.print(f"  Text length: {len(text_to_encode)} chars, {len(text_to_encode.encode('utf-8'))} bytes")
    ui.console.print(f"  Start signal ends at: {start_pos}")
    ui.console.print(f"  Data region: {data_start_pos} - {end_start_pos}")
    ui.console.print(f"  End signal starts at: {end_start_pos}")


if __name__ == "__main__":
    main()
