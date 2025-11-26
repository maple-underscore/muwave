"""
Conversation logger for muwave.
Saves conversation history to log files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading


class ConversationLogger:
    """Logger for muwave conversations."""
    
    def __init__(
        self,
        log_file: str = "muwave_conversation.log",
        log_format: str = "text",
        include_timestamps: bool = True,
        log_level: str = "info",
    ):
        """
        Initialize the conversation logger.
        
        Args:
            log_file: Path to the log file
            log_format: Log format (text or json)
            include_timestamps: Whether to include timestamps
            log_level: Logging level (debug, info, warning, error)
        """
        self.log_file = Path(log_file)
        self.log_format = log_format
        self.include_timestamps = include_timestamps
        self.log_level = log_level
        self._lock = threading.Lock()
        self._ensure_log_file()
    
    def _ensure_log_file(self) -> None:
        """Ensure log file directory exists."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.touch()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def _format_text(
        self,
        level: str,
        event: str,
        party_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a log entry as text."""
        parts = []
        if self.include_timestamps:
            parts.append(f"[{self._get_timestamp()}]")
        parts.append(f"[{level.upper()}]")
        parts.append(f"[{party_id[:8]}]")
        parts.append(f"{event}:")
        parts.append(content)
        if metadata:
            parts.append(f"| {metadata}")
        return " ".join(parts)
    
    def _format_json(
        self,
        level: str,
        event: str,
        party_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a log entry as JSON."""
        entry = {
            "level": level,
            "event": event,
            "party_id": party_id,
            "content": content,
        }
        if self.include_timestamps:
            entry["timestamp"] = self._get_timestamp()
        if metadata:
            entry["metadata"] = metadata
        return json.dumps(entry)
    
    def _write(self, entry: str) -> None:
        """Write an entry to the log file."""
        with self._lock:
            with open(self.log_file, "a") as f:
                f.write(entry + "\n")
    
    def _should_log(self, level: str) -> bool:
        """Check if the level should be logged."""
        levels = ["debug", "info", "warning", "error"]
        return levels.index(level) >= levels.index(self.log_level)
    
    def log(
        self,
        event: str,
        party_id: str,
        content: str,
        level: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an event.
        
        Args:
            event: Event type (e.g., "message_sent", "message_received")
            party_id: ID of the party involved
            content: Content of the log entry
            level: Log level
            metadata: Optional additional metadata
        """
        if not self._should_log(level):
            return
        
        if self.log_format == "json":
            entry = self._format_json(level, event, party_id, content, metadata)
        else:
            entry = self._format_text(level, event, party_id, content, metadata)
        
        self._write(entry)
    
    def log_message_sent(
        self,
        party_id: str,
        content: str,
        transmission_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a sent message."""
        self.log("message_sent", party_id, content, "info", transmission_info)
    
    def log_message_received(
        self,
        party_id: str,
        content: str,
        sender_id: str,
    ) -> None:
        """Log a received message."""
        self.log(
            "message_received",
            party_id,
            content,
            "info",
            {"sender_id": sender_id},
        )
    
    def log_transmission_start(self, party_id: str, content_length: int) -> None:
        """Log start of transmission."""
        self.log(
            "transmission_start",
            party_id,
            f"Starting transmission of {content_length} characters",
            "debug",
        )
    
    def log_transmission_complete(
        self,
        party_id: str,
        duration_ms: float,
    ) -> None:
        """Log completion of transmission."""
        self.log(
            "transmission_complete",
            party_id,
            f"Transmission completed in {duration_ms:.1f}ms",
            "debug",
        )
    
    def log_error(self, party_id: str, error: str) -> None:
        """Log an error."""
        self.log("error", party_id, error, "error")
    
    def log_ai_request(self, party_id: str, prompt: str) -> None:
        """Log an AI request."""
        self.log("ai_request", party_id, prompt, "info")
    
    def log_ai_response(self, party_id: str, response: str) -> None:
        """Log an AI response."""
        self.log("ai_response", party_id, response, "info")
    
    def log_party_join(self, party_id: str, party_name: str) -> None:
        """Log a party joining."""
        self.log("party_join", party_id, f"Party '{party_name}' joined", "info")
    
    def log_party_leave(self, party_id: str, party_name: str) -> None:
        """Log a party leaving."""
        self.log("party_leave", party_id, f"Party '{party_name}' left", "info")
    
    def get_conversation_history(self) -> List[str]:
        """Read and return all log entries."""
        if not self.log_file.exists():
            return []
        with open(self.log_file, "r") as f:
            return f.readlines()
    
    def clear_log(self) -> None:
        """Clear the log file."""
        with self._lock:
            self.log_file.write_text("")
