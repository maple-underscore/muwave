"""
Rich text interface for muwave.
Provides color-coded transmission status and real-time output.
"""

import time
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from rich.console import Console, Group
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
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
        # Persistent progress widget and transmission tracking
        self._progress_widget: Optional[Progress] = None
        self._progress_task_id: Optional[int] = None
        self._tx_prev_time: Optional[float] = None
        self._tx_prev_sent: int = 0
        self._tx_rate_ema: float = 0.0
        self._tx_active_text: Optional[str] = None
        self._tx_start_time: Optional[float] = None

    def _fmt_seconds(self, seconds: Optional[float]) -> str:
        """Format seconds as MM:SS; returns --:-- if None or invalid."""
        if seconds is None or seconds != seconds or seconds < 0:
            return "--:--"
        total = int(round(seconds))
        m, s = divmod(total, 60)
        return f"{m:02d}:{s:02d}"

    def _make_progress_bar(self, progress: float, width: int = 40) -> Text:
        """Create a simple textual progress bar (no new lines printed)."""
        pct = max(0.0, min(1.0, float(progress)))
        done = int(pct * width)
        remaining = width - done
        bar = f"[{'#' * done}{'-' * remaining}] {int(pct * 100):3d}%"
        return Text(bar, style=self.colors.sending)
    
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
        
        # Calculate sending chunk (smaller window ~3% of text or min 1 char)
        sending_chars = max(1, int(total_chars * 0.03))
        if sent_chars + sending_chars > total_chars:
            sending_chars = total_chars - sent_chars
        
        if status == TransmitStatus.SENT:
            sent_chars = total_chars
            sending_chars = 0
        
        colored_text = self.create_transmit_text(text, sent_chars, sending_chars)

        # Reset tracking if new transmission text or terminal state
        if self._tx_active_text != text or status in (TransmitStatus.SENT, TransmitStatus.ERROR, TransmitStatus.WAITING and sent_chars == 0):
            self._tx_prev_time = None
            self._tx_prev_sent = 0
            self._tx_rate_ema = 0.0
            self._tx_start_time = None
            if status in (TransmitStatus.SENT, TransmitStatus.ERROR):
                self._tx_active_text = None
            else:
                self._tx_active_text = text

        # Update EMA rate for ETA calculation
        now = time.time()
        if status == TransmitStatus.SENDING:
            if self._tx_start_time is None:
                self._tx_start_time = now
            if self._tx_prev_time is not None and sent_chars >= self._tx_prev_sent:
                dt = max(1e-6, now - self._tx_prev_time)
                dchars = sent_chars - self._tx_prev_sent
                inst_rate = dchars / dt
                alpha = 0.3  # smoothing factor
                self._tx_rate_ema = (1 - alpha) * self._tx_rate_ema + alpha * inst_rate if self._tx_rate_ema > 0 else inst_rate
            self._tx_prev_time = now
            self._tx_prev_sent = sent_chars
        elif status in (TransmitStatus.SENT, TransmitStatus.ERROR):
            self._tx_rate_ema = 0.0
            self._tx_prev_time = None
        elapsed_s = 0.0 if self._tx_start_time is None else max(0.0, now - self._tx_start_time)

        status_label = {
            TransmitStatus.WAITING: "⏳ Waiting",
            TransmitStatus.SENDING: "📡 Sending",
            TransmitStatus.SENT: "✓ Sent",
            TransmitStatus.ERROR: "✗ Error",
        }.get(status, "Unknown")

        # Ensure persistent rich Progress with desired columns
        if self.show_progress:
            if self._progress_widget is None:
                self._progress_widget = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.percentage:>3.0f}%"),
                    TextColumn(" {task.completed}/{task.total}"),
                    TextColumn(" {task.fields[time_pair]}"),
                )
                self._progress_task_id = self._progress_widget.add_task("Transmitting...", total=max(1, total_chars))
            else:
                # If total changes (new text), recreate task
                if self._tx_active_text != text and self._progress_task_id is not None:
                    try:
                        # remove old task by recreating the Progress
                        self._progress_widget = Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            BarColumn(),
                            TextColumn("{task.percentage:>3.0f}%"),
                            TextColumn(" {task.completed}/{task.total}"),
                            TextColumn(" {task.fields[time_pair]}"),
                        )
                        self._progress_task_id = self._progress_widget.add_task("Transmitting...", total=max(1, total_chars))
                    except Exception:
                        pass
            # Update task progress
            if self._progress_widget is not None and self._progress_task_id is not None:
                try:
                    # Ensure total matches current text length
                    # Compute times for first line: elapsed / total_time
                    left_chars = max(0, total_chars - sent_chars)
                    time_left_s = (left_chars / self._tx_rate_ema) if (self._tx_rate_ema > 0 and status == TransmitStatus.SENDING) else None
                    if status == TransmitStatus.SENT:
                        time_left_s = 0.0
                    total_time_s = (elapsed_s + time_left_s) if time_left_s is not None else None
                    time_pair = f"{self._fmt_seconds(elapsed_s)} / {self._fmt_seconds(total_time_s)}"
                    self._progress_widget.update(
                        self._progress_task_id,
                        total=max(1, total_chars),
                        completed=max(0, min(sent_chars, total_chars)),
                        time_pair=time_pair,
                    )
                except Exception:
                    pass

        # Build a single renderable and update via Live to avoid scrolling
        renderables: List[Any] = [colored_text]
        if self.show_progress and self._progress_widget is not None:
            renderables.append(self._progress_widget)

        # Additional stats line: sent/left/total and ETA
        left_chars = max(0, total_chars - sent_chars)
        time_left_s = (left_chars / self._tx_rate_ema) if (self._tx_rate_ema > 0 and status == TransmitStatus.SENDING) else None
        if status == TransmitStatus.SENT:
            time_left_s = 0.0
        total_time_s = (elapsed_s + time_left_s) if time_left_s is not None else None
        stats_line = Text(
            f"{sent_chars}/{left_chars}/{total_chars}  {self._fmt_seconds(elapsed_s)}/{self._fmt_seconds(time_left_s)}/{self._fmt_seconds(total_time_s)}",
            style=self.colors.info,
        )
        renderables.append(stats_line)

        panel = Panel(
            Group(*renderables),
            title=f"[{self._get_status_color(status)}]{status_label}[/]",
            border_style=self._get_status_color(status),
        )

        # Start or update a persistent live view
        if self._live is None:
            self._live = Live(panel, refresh_per_second=12, console=self.console)
            try:
                self._live.start()
            except Exception:
                # Fallback: print once if Live cannot start in this context
                self.console.print(panel)
                return
        else:
            try:
                self._live.update(panel)
            except Exception:
                # Fallback update
                self.console.print(panel)

        # On terminal states, stop Live and leave final panel rendered
        if status in (TransmitStatus.SENT, TransmitStatus.ERROR):
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
            # Reset progress widget for next transmission
            self._progress_widget = None
            self._progress_task_id = None
    
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
  /devices         List audio devices
  /usein <idx>     Set input device
  /gain <float>    Set input gain (e.g., /gain 2.0)
  /monitor         Monitor input RMS for 5s
  /inject <wav>    Inject WAV file for decoding (no hardware needed)
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
