"""
Rich text interface for muwave.
Provides color-coded transmission status and real-time output.
"""

import time
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.style import Style


class TransmitStatus(Enum):
    """Transmission status for text coloring."""
    WAITING = "waiting"
    SENDING = "sending"
    SENT = "sent"
    ERROR = "error"


class ReceiveStatus(Enum):
    """Reception status for text coloring."""
    IDLE = "idle"
    LISTENING = "listening"
    RECEIVING = "receiving"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ColorScheme:
    """Color scheme for the interface."""
    waiting: str = "yellow"
    sending: str = "blue"
    sent: str = "green"
    receiving: str = "cyan"
    error: str = "red"
    info: str = "white"
    highlight: str = "magenta"
    
    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "ColorScheme":
        """Create ColorScheme from dictionary."""
        return cls(
            waiting=d.get("waiting", "yellow"),
            sending=d.get("sending", "blue"),
            sent=d.get("sent", "green"),
            receiving=d.get("receiving", "cyan"),
            error=d.get("error", "red"),
        )


class RichInterface:
    """
    Rich text interface for muwave.
    
    Provides:
    - Color-coded transmission status (yellow=waiting, blue=sending, green=sent)
    - Real-time receiving output display
    - Progress bars for transmission
    - Conversation display
    """
    
    def __init__(
        self,
        colors: Optional[ColorScheme] = None,
        show_progress: bool = True,
        show_receiving: bool = True,
    ):
        """
        Initialize the rich interface.
        
        Args:
            colors: Color scheme to use
            show_progress: Whether to show progress bars
            show_receiving: Whether to show real-time receiving output
        """
        self.colors = colors or ColorScheme()
        self.show_progress = show_progress
        self.show_receiving = show_receiving
        
        self.console = Console()
        self._live: Optional[Live] = None
        self._lock = threading.Lock()
        
        self._transmit_text = ""
        self._transmit_status = TransmitStatus.WAITING
        self._transmit_progress = 0.0
        
        self._receive_text = ""
        self._receive_status = ReceiveStatus.IDLE
        
        self._messages: List[Dict[str, Any]] = []
    
    def _get_status_color(self, status: TransmitStatus) -> str:
        """Get color for transmit status."""
        if status == TransmitStatus.WAITING:
            return self.colors.waiting
        elif status == TransmitStatus.SENDING:
            return self.colors.sending
        elif status == TransmitStatus.SENT:
            return self.colors.sent
        else:
            return self.colors.error
    
    def _get_receive_color(self, status: ReceiveStatus) -> str:
        """Get color for receive status."""
        if status == ReceiveStatus.RECEIVING:
            return self.colors.receiving
        elif status == ReceiveStatus.COMPLETE:
            return self.colors.sent
        elif status == ReceiveStatus.ERROR:
            return self.colors.error
        else:
            return self.colors.info
    
    def print_header(self, text: str = "muwave - Sound Communication Protocol") -> None:
        """Print application header."""
        self.console.print(Panel(
            Text(text, style="bold white"),
            style="blue",
        ))
    
    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[{self.colors.info}]ℹ {message}[/]")
    
    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[{self.colors.sent}]✓ {message}[/]")
    
    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[{self.colors.error}]✗ {message}[/]")
    
    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[{self.colors.waiting}]⚠ {message}[/]")
    
    def create_transmit_text(
        self,
        text: str,
        sent_chars: int = 0,
        sending_chars: int = 0,
    ) -> Text:
        """
        Create color-coded transmit text.
        
        Args:
            text: Full text being transmitted
            sent_chars: Number of characters already sent (green)
            sending_chars: Number of characters currently sending (blue)
            
        Returns:
            Rich Text object with color coding
        """
        result = Text()
        
        if sent_chars > 0:
            result.append(text[:sent_chars], style=self.colors.sent)
        
        if sending_chars > 0:
            end_sending = min(sent_chars + sending_chars, len(text))
            result.append(text[sent_chars:end_sending], style=self.colors.sending)
        
        remaining_start = sent_chars + sending_chars
        if remaining_start < len(text):
            result.append(text[remaining_start:], style=self.colors.waiting)
        
        return result
    
    def show_transmission_progress(
        self,
        text: str,
        progress: float,
        status: TransmitStatus = TransmitStatus.SENDING,
    ) -> None:
        """
        Show transmission progress with color-coded text.
        
        Args:
            text: Text being transmitted
            progress: Progress from 0.0 to 1.0
            status: Current status
        """
        total_chars = len(text)
        sent_chars = int(total_chars * progress)
        
        # Calculate sending chunk (about 10% of text or min 1 char)
        sending_chars = max(1, int(total_chars * 0.1))
        if sent_chars + sending_chars > total_chars:
            sending_chars = total_chars - sent_chars
        
        if status == TransmitStatus.SENT:
            sent_chars = total_chars
            sending_chars = 0
        
        colored_text = self.create_transmit_text(text, sent_chars, sending_chars)
        
        self.console.clear()
        self.print_header()
        self.console.print()
        
        status_label = {
            TransmitStatus.WAITING: "⏳ Waiting",
            TransmitStatus.SENDING: "📡 Sending",
            TransmitStatus.SENT: "✓ Sent",
            TransmitStatus.ERROR: "✗ Error",
        }.get(status, "Unknown")
        
        self.console.print(Panel(
            colored_text,
            title=f"[{self._get_status_color(status)}]{status_label}[/]",
            border_style=self._get_status_color(status),
        ))
        
        if self.show_progress and status == TransmitStatus.SENDING:
            self.console.print()
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress_bar:
                task = progress_bar.add_task("Transmitting...", total=100)
                progress_bar.update(task, completed=progress * 100)
    
    def show_receiving_output(
        self,
        text: str,
        status: ReceiveStatus = ReceiveStatus.RECEIVING,
        sender: Optional[str] = None,
    ) -> None:
        """
        Show real-time receiving output.
        
        Args:
            text: Received text so far
            status: Reception status
            sender: Sender identifier if known
        """
        if not self.show_receiving:
            return
        
        color = self._get_receive_color(status)
        
        status_label = {
            ReceiveStatus.IDLE: "💤 Idle",
            ReceiveStatus.LISTENING: "👂 Listening",
            ReceiveStatus.RECEIVING: "📥 Receiving",
            ReceiveStatus.COMPLETE: "✓ Received",
            ReceiveStatus.ERROR: "✗ Error",
        }.get(status, "Unknown")
        
        title = f"[{color}]{status_label}[/]"
        if sender:
            title += f" from {sender[:8]}"
        
        panel = Panel(
            Text(text or "...", style=color),
            title=title,
            border_style=color,
        )
        
        self.console.print(panel)
    
    def show_conversation(
        self,
        messages: List[Dict[str, Any]],
        title: str = "Conversation",
    ) -> None:
        """
        Display the conversation history.
        
        Args:
            messages: List of message dictionaries
            title: Panel title
        """
        table = Table(show_header=True, header_style="bold")
        table.add_column("Sender", style="cyan", width=12)
        table.add_column("Message", style="white")
        table.add_column("Status", style="dim", width=10)
        
        for msg in messages[-10:]:  # Show last 10 messages
            sender = msg.get("sender", "Unknown")[:10]
            content = msg.get("content", "")[:80]
            if len(msg.get("content", "")) > 80:
                content += "..."
            
            status = ""
            if msg.get("transmitted"):
                status = f"[{self.colors.sent}]✓[/]"
            elif msg.get("sending"):
                status = f"[{self.colors.sending}]→[/]"
            elif msg.get("received"):
                status = f"[{self.colors.receiving}]←[/]"
            
            table.add_row(sender, content, status)
        
        self.console.print(Panel(table, title=title))
    
    def show_status_bar(
        self,
        party_name: str,
        listening: bool = False,
        transmitting: bool = False,
        ai_enabled: bool = False,
    ) -> None:
        """
        Show a status bar with current state.
        
        Args:
            party_name: Name of the local party
            listening: Whether listening for audio
            transmitting: Whether transmitting audio
            ai_enabled: Whether AI is enabled
        """
        status_parts = [f"[bold]{party_name}[/]"]
        
        if listening:
            status_parts.append(f"[{self.colors.receiving}]👂 Listening[/]")
        if transmitting:
            status_parts.append(f"[{self.colors.sending}]📡 Transmitting[/]")
        if ai_enabled:
            status_parts.append(f"[{self.colors.highlight}]🤖 AI[/]")
        
        self.console.print(" | ".join(status_parts))
    
    def prompt_input(self, prompt: str = "Enter message: ") -> str:
        """
        Prompt for user input.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            User input string
        """
        self.console.print()
        return self.console.input(f"[{self.colors.highlight}]{prompt}[/]")
    
    def show_help(self) -> None:
        """Show help information."""
        help_text = """
[bold]Commands:[/]
  /send <text>     Send a message
  /ai <prompt>     Send prompt to AI
  /listen          Start listening for messages
  /stop            Stop listening
  /clear           Clear conversation
  /status          Show current status
  /config          Show configuration
  /help            Show this help
  /quit            Exit the application

[bold]Color Legend:[/]
  [yellow]Yellow[/] - Waiting to send
  [blue]Blue[/]   - Currently sending
  [green]Green[/]  - Sent successfully
  [cyan]Cyan[/]   - Receiving
  [red]Red[/]    - Error
        """
        self.console.print(Panel(help_text, title="Help"))
    
    def clear(self) -> None:
        """Clear the console."""
        self.console.clear()


def create_interface(config_dict: Dict[str, Any]) -> RichInterface:
    """
    Create a RichInterface from configuration.
    
    Args:
        config_dict: UI configuration dictionary
        
    Returns:
        Configured RichInterface
    """
    colors = ColorScheme.from_dict(config_dict.get("colors", {}))
    
    return RichInterface(
        colors=colors,
        show_progress=config_dict.get("show_progress", True),
        show_receiving=config_dict.get("show_receiving", True),
    )
