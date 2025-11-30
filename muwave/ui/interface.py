"""
Rich text interface for muwave.
Provides color-coded transmission status and real-time output.
"""

import time
import threading
from typing import Optional, Dict, Any, List, Tuple
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
        self._tx_final_elapsed: Optional[float] = None
        self._tx_final_total: Optional[float] = None

    def _fmt_seconds(self, seconds: Optional[float]) -> str:
        """Format seconds as MM:SS; returns --:-- if None or invalid."""
        if seconds is None or seconds != seconds or seconds < 0:
            return "--:--"
        total = int(round(seconds))
        m, s = divmod(total, 60)
        return f"{m:02d}:{s:02d}"

    def _truncate_text_for_display(self, text: str, max_lines: int = 10, max_chars_per_line: int = 100,
                                   current_pos: int = 0) -> Tuple[str, bool, int, int, List[str]]:
        """
        Truncate long text to fit within terminal bounds, showing start and end.
        
        Args:
            text: Original text to display
            max_lines: Maximum number of lines to show
            max_chars_per_line: Maximum characters per line
            current_pos: Current character position in transmission (for scrolling preview)
            
        Returns:
            Tuple of (truncated_text, was_truncated, truncation_start_pos, truncation_end_pos, hidden_lines)
            where positions indicate where the truncated content starts and ends in original text
            and hidden_lines are the lines in the truncated section
        """
        lines = text.split('\n')
        
        # If text is short enough, return as-is
        if len(lines) <= max_lines and all(len(line) <= max_chars_per_line for line in lines):
            return text, False, 0, 0, []
        
        # Special case: single very long line – create synthetic hidden segments inside the line
        if len(lines) == 1 and len(lines[0]) > max_chars_per_line:
            line = lines[0]
            # We will keep a head and tail segment and treat middle as hidden
            head_len = max_chars_per_line // 2
            tail_len = max_chars_per_line // 2
            if head_len + tail_len >= len(line):  # Fallback if sizing odd
                return line, False, 0, 0, []
            truncation_start_pos = head_len
            truncation_end_pos = len(line) - tail_len
            hidden_mid = line[truncation_start_pos:truncation_end_pos]
            hidden_lines = [hidden_mid]  # treat middle as single hidden line
            display = (
                line[:head_len] + '\n' + '<<TRUNCATION_START>>' + '\n' + '<<TRUNCATION_PREVIEW>>' + '\n' + '<<TRUNCATION_END>>' + '\n' + line[-tail_len:]
            )
            return display, True, truncation_start_pos, truncation_end_pos, hidden_lines

        # Calculate how many lines to show from start and end (multi-line case)
        if len(lines) > max_lines:
            lines_to_show = max(2, max_lines - 3)  # Reserve 3 lines for truncation markers and preview
            start_lines = lines_to_show // 2
            end_lines = lines_to_show - start_lines
            
            # Calculate character positions in original text
            truncation_start_pos = sum(len(line) + 1 for line in lines[:start_lines])  # +1 for newline
            truncation_end_pos = sum(len(line) + 1 for line in lines[:-end_lines]) if end_lines > 0 else len(text)
            
            # Get the hidden lines
            hidden_lines = lines[start_lines:-end_lines] if end_lines > 0 else lines[start_lines:]
            
            truncated_lines = (
                lines[:start_lines] +
                ['<<TRUNCATION_START>>'] +
                ['<<TRUNCATION_PREVIEW>>'] +
                ['<<TRUNCATION_END>>'] +
                lines[-end_lines:]
            )
        else:
            truncated_lines = lines
            truncation_start_pos = 0
            truncation_end_pos = 0
            hidden_lines = []
        
        # Truncate individual lines if they're too long
        result_lines = []
        for line in truncated_lines:
            if len(line) > max_chars_per_line and not line.startswith('<<TRUNCATION'):
                # Show start and end of long lines
                show_chars = max_chars_per_line - 7  # Reserve space for " ... "
                start_chars = show_chars // 2
                end_chars = show_chars - start_chars
                result_lines.append(line[:start_chars] + ' ... ' + line[-end_chars:])
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines), True, truncation_start_pos, truncation_end_pos, hidden_lines

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
        truncation_start: int = 0,
        truncation_end: int = 0,
        original_sent_chars: int = 0,
        original_sending_chars: int = 0,
        hidden_lines: Optional[List[str]] = None,
    ) -> Text:
        """
        Create color-coded transmit text with scrolling preview of hidden content.
        
        Args:
            text: Text being displayed (may be truncated)
            sent_chars: Number of characters already sent in display text (green)
            sending_chars: Number of characters currently sending in display text (blue)
            truncation_start: Start position of truncated content in original text
            truncation_end: End position of truncated content in original text
            original_sent_chars: Actual sent chars in original text (for truncation marker coloring)
            original_sending_chars: Actual sending chars in original text
            hidden_lines: Lines that are hidden in the truncated section
            
        Returns:
            Rich Text object with color coding
        """
        result = Text()
        hidden_lines = hidden_lines or []
        
        # Find the truncation markers
        start_marker = '<<TRUNCATION_START>>'
        preview_marker = '<<TRUNCATION_PREVIEW>>'
        end_marker = '<<TRUNCATION_END>>'
        
        start_pos = text.find(start_marker)
        preview_pos = text.find(preview_marker)
        end_pos = text.find(end_marker)
        
        if start_pos >= 0 and preview_pos >= 0 and end_pos >= 0 and truncation_start > 0 and truncation_end > truncation_start:
            # Split text around the markers
            before_truncation = text[:start_pos]
            after_truncation = text[end_pos + len(end_marker):]
            
            # Color the text before truncation
            before_len = len(before_truncation)
            if original_sent_chars >= truncation_start:
                result.append(before_truncation, style=self.colors.sent)
            elif original_sent_chars > 0:
                result.append(before_truncation[:original_sent_chars], style=self.colors.sent)
                if original_sent_chars + original_sending_chars >= truncation_start:
                    result.append(before_truncation[original_sent_chars:], style=self.colors.sending)
                else:
                    if original_sending_chars > 0:
                        result.append(before_truncation[original_sent_chars:original_sent_chars + original_sending_chars], 
                                    style=self.colors.sending)
                    if original_sent_chars + original_sending_chars < before_len:
                        result.append(before_truncation[original_sent_chars + original_sending_chars:], 
                                    style=self.colors.waiting)
            else:
                result.append(before_truncation, style=self.colors.waiting)
            
            # Add first truncation marker with progress bar
            truncated_length = truncation_end - truncation_start
            hidden_sent = max(0, min(truncated_length, original_sent_chars - truncation_start))
            hidden_progress = min(1.0, hidden_sent / truncated_length) if truncated_length > 0 else 0.0
            
            marker_text = '⋯⋯⋯ content truncated ⋯⋯⋯'
            # When fully sent, make whole marker green
            marker_sent_len = len(marker_text) if hidden_progress >= 1.0 else int(len(marker_text) * hidden_progress)
            
            if marker_sent_len > 0:
                result.append(marker_text[:marker_sent_len], style=self.colors.sent)
            hidden_sending_end = max(0, min(truncated_length, original_sent_chars + original_sending_chars - truncation_start))
            if marker_sent_len < len(marker_text) and hidden_progress < 1.0:
                if hidden_sending_end > hidden_sent:
                    sending_progress = min(1.0, hidden_sending_end / truncated_length)
                    marker_sending_len = int(len(marker_text) * sending_progress) - marker_sent_len
                    if marker_sending_len > 0:
                        result.append(marker_text[marker_sent_len:marker_sent_len + marker_sending_len], style=self.colors.sending)
                        remainder = marker_text[marker_sent_len + marker_sending_len:]
                        if remainder:
                            result.append(remainder, style=self.colors.waiting)
                    else:
                        result.append(marker_text[marker_sent_len:], style=self.colors.waiting)
                else:
                    result.append(marker_text[marker_sent_len:], style=self.colors.waiting)
            
            result.append('\n')
            
            # Show dynamic preview based on current transmission position
            if hidden_lines:
                # Calculate which line we're currently on in the hidden section
                hidden_char_pos = max(0, original_sent_chars - truncation_start)
                
                # Find which hidden line corresponds to this position
                cumulative_chars = 0
                current_line_idx = 0
                for idx, line in enumerate(hidden_lines):
                    if cumulative_chars + len(line) + 1 > hidden_char_pos:  # +1 for newline
                        current_line_idx = idx
                        break
                    cumulative_chars += len(line) + 1
                
                # Determine what to show
                if hidden_progress >= 1.0:
                    # Done transmitting, show last 5 lines of hidden content
                    show_lines = hidden_lines[-5:] if len(hidden_lines) >= 5 else hidden_lines
                    for line in show_lines:
                        result.append(line, style=self.colors.sent)
                        result.append('\n')
                elif hidden_progress > 0:
                    # Currently transmitting, show fixed 8-line window (preview) plus markers outside
                    preview_limit = 8
                    half = preview_limit // 2
                    start_idx = max(0, current_line_idx - half)
                    end_idx = start_idx + preview_limit
                    if end_idx > len(hidden_lines):
                        end_idx = len(hidden_lines)
                        start_idx = max(0, end_idx - preview_limit)
                    show_lines = hidden_lines[start_idx:end_idx]
                    # Recompute cumulative chars for each line for accurate char_in_line
                    cumulative_list = []
                    acc = 0
                    for l in hidden_lines:
                        cumulative_list.append(acc)
                        acc += len(l) + 1
                    for line_idx, line in enumerate(hidden_lines):
                        if line_idx == current_line_idx:
                            line_start_pos = cumulative_list[line_idx]
                            char_in_line = max(0, hidden_char_pos - line_start_pos)
                            break
                    for idx, line in enumerate(show_lines):
                        actual_idx = start_idx + idx
                        if actual_idx < current_line_idx:
                            result.append(line, style=self.colors.sent)
                        elif actual_idx == current_line_idx:
                            line_start_pos = cumulative_list[actual_idx]
                            char_in_line = max(0, hidden_char_pos - line_start_pos)
                            if char_in_line >= len(line):
                                # Fully sent line
                                result.append(line, style=self.colors.sent)
                            else:
                                if char_in_line > 0:
                                    result.append(line[:char_in_line], style=self.colors.sent)
                                sending_in_line = min(len(line) - char_in_line, max(1, len(line) // 25))
                                # If this chunk completes the line, color it green instead of leaving tail yellow
                                if char_in_line + sending_in_line >= len(line):
                                    result.append(line[char_in_line:], style=self.colors.sending)
                                else:
                                    result.append(line[char_in_line:char_in_line + sending_in_line], style=self.colors.sending)
                                    result.append(line[char_in_line + sending_in_line:], style=self.colors.waiting)
                        else:
                            result.append(line, style=self.colors.waiting)
                        result.append('\n')
                else:
                    # Haven't started hidden section, show first 5 lines
                    show_lines = hidden_lines[:5] if len(hidden_lines) >= 5 else hidden_lines
                    for line in show_lines:
                        result.append(line, style=self.colors.waiting)
                        result.append('\n')
            
            # Add second truncation marker (mirror first) fully green when done
            if hidden_progress >= 1.0:
                result.append(marker_text, style=self.colors.sent)
            else:
                if marker_sent_len > 0:
                    result.append(marker_text[:marker_sent_len], style=self.colors.sent)
                if marker_sent_len < len(marker_text):
                    if hidden_sending_end > hidden_sent:
                        sending_progress = min(1.0, hidden_sending_end / truncated_length)
                        marker_sending_len = int(len(marker_text) * sending_progress) - marker_sent_len
                        if marker_sending_len > 0:
                            result.append(marker_text[marker_sent_len:marker_sent_len + marker_sending_len], style=self.colors.sending)
                            remainder = marker_text[marker_sent_len + marker_sending_len:]
                            if remainder:
                                result.append(remainder, style=self.colors.waiting)
                        else:
                            result.append(marker_text[marker_sent_len:], style=self.colors.waiting)
                    else:
                        result.append(marker_text[marker_sent_len:], style=self.colors.waiting)
            
            # Color the text after truncation
            chars_before_and_hidden = truncation_end
            
            if original_sent_chars >= chars_before_and_hidden:
                after_sent = min(len(after_truncation), original_sent_chars - chars_before_and_hidden)
                after_sending = min(len(after_truncation) - after_sent, original_sending_chars)
                
                if after_sent > 0:
                    result.append(after_truncation[:after_sent], style=self.colors.sent)
                if after_sending > 0:
                    result.append(after_truncation[after_sent:after_sent + after_sending], style=self.colors.sending)
                if after_sent + after_sending < len(after_truncation):
                    result.append(after_truncation[after_sent + after_sending:], style=self.colors.waiting)
            else:
                result.append(after_truncation, style=self.colors.waiting)
        else:
            # No truncation, use simple coloring
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
        
        # Truncate text for display if it's too long
        display_text, was_truncated, truncation_start, truncation_end, hidden_lines = self._truncate_text_for_display(
            text, max_lines=8, max_chars_per_line=120, current_pos=sent_chars
        )
        
        # Create colored text with proper progress tracking
        if was_truncated:
            colored_text = self.create_transmit_text(
                display_text,
                sent_chars=sent_chars,  # Not used when truncated, but passed for compatibility
                sending_chars=sending_chars,  # Not used when truncated
                truncation_start=truncation_start,
                truncation_end=truncation_end,
                original_sent_chars=sent_chars,
                original_sending_chars=sending_chars,
                hidden_lines=hidden_lines,
            )
        else:
            colored_text = self.create_transmit_text(display_text, sent_chars, sending_chars)

        # Reset tracking if new transmission text or terminal state
        if self._tx_active_text != text or status in (TransmitStatus.SENT, TransmitStatus.ERROR, TransmitStatus.WAITING and sent_chars == 0):
            self._tx_prev_time = None
            self._tx_prev_sent = 0
            self._tx_rate_ema = 0.0
            self._tx_start_time = None
            self._tx_final_elapsed = None
            self._tx_final_total = None
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
            # Capture final times before resetting
            if self._tx_start_time is not None and self._tx_final_elapsed is None:
                self._tx_final_elapsed = max(0.0, now - self._tx_start_time)
                # Calculate final total time if we have rate data
                if self._tx_rate_ema > 0:
                    self._tx_final_total = self._tx_final_elapsed
                else:
                    self._tx_final_total = self._tx_final_elapsed
            self._tx_rate_ema = 0.0
            self._tx_prev_time = None
        
        # Use frozen time for completed transmissions, otherwise calculate current elapsed
        if status in (TransmitStatus.SENT, TransmitStatus.ERROR) and self._tx_final_elapsed is not None:
            elapsed_s = self._tx_final_elapsed
        else:
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
                    # Compute times for progress bar
                    left_chars = max(0, total_chars - sent_chars)
                    
                    if status == TransmitStatus.SENT:
                        # Use frozen final times
                        time_left_s = 0.0
                        total_time_s = self._tx_final_total if self._tx_final_total is not None else elapsed_s
                        description = "Transmitted!"
                    else:
                        time_left_s = (left_chars / self._tx_rate_ema) if (self._tx_rate_ema > 0 and status == TransmitStatus.SENDING) else None
                        total_time_s = (elapsed_s + time_left_s) if time_left_s is not None else None
                        description = "Transmitting..."
                    
                    time_pair = f"{self._fmt_seconds(elapsed_s)} / {self._fmt_seconds(total_time_s)}"
                    self._progress_widget.update(
                        self._progress_task_id,
                        description=description,
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
        
        if status == TransmitStatus.SENT:
            # Use frozen final times
            time_left_s = 0.0
            total_time_s = self._tx_final_total if self._tx_final_total is not None else elapsed_s
        else:
            time_left_s = (left_chars / self._tx_rate_ema) if (self._tx_rate_ema > 0 and status == TransmitStatus.SENDING) else None
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
