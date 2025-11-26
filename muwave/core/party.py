"""
Party management for muwave.
Represents a participant in the audio communication protocol.
"""

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class Message:
    """Represents a message in the conversation."""
    content: str
    sender_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "text"  # text, audio, system
    transmitted: bool = False
    received: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "content": self.content,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "transmitted": self.transmitted,
            "received": self.received,
        }


@dataclass
class ConversationContext:
    """Maintains conversation context for AI interactions."""
    messages: List[Message] = field(default_factory=list)
    system_prompt: str = ""
    max_context_length: int = 4096
    
    def add_message(self, message: Message) -> None:
        """Add a message to the context."""
        self.messages.append(message)
        self._trim_context()
    
    def _trim_context(self) -> None:
        """Trim context to fit within max length."""
        total_length = sum(len(m.content) for m in self.messages)
        while total_length > self.max_context_length and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total_length -= len(removed.content)
    
    def get_context_string(self) -> str:
        """Get context as a formatted string for the AI."""
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        for msg in self.messages:
            role = "User" if msg.sender_id != "ai" else "Assistant"
            parts.append(f"{role}: {msg.content}")
        return "\n\n".join(parts)
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for Ollama API."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            role = "user" if msg.sender_id != "ai" else "assistant"
            messages.append({"role": role, "content": msg.content})
        return messages
    
    def clear(self) -> None:
        """Clear the conversation context."""
        self.messages.clear()


class Party:
    """
    Represents a party in the muwave communication protocol.
    
    A party can:
    - Send and receive audio-encoded messages
    - Maintain conversation context
    - Recognize its own transmissions (for same-machine scenarios)
    - Interact with AI (Ollama) if configured
    """
    
    def __init__(
        self,
        party_id: Optional[str] = None,
        name: Optional[str] = None,
        is_ai: bool = False,
    ):
        """
        Initialize a party.
        
        Args:
            party_id: Unique identifier for this party. Auto-generated if None.
            name: Human-readable name for the party.
            is_ai: Whether this party is an AI agent.
        """
        self.party_id = party_id or self._generate_party_id()
        self.name = name or f"Party-{self.party_id[:8]}"
        self.is_ai = is_ai
        self.context = ConversationContext()
        self._signature = self._generate_signature()
        self._active = True
        self._last_activity = time.time()
        self._pending_messages: List[Message] = []
        self._received_messages: List[Message] = []
    
    def _generate_party_id(self) -> str:
        """Generate a unique party ID."""
        # Combine UUID with process ID and timestamp for uniqueness
        unique_data = f"{uuid.uuid4()}-{os.getpid()}-{time.time()}"
        return hashlib.sha256(unique_data.encode()).hexdigest()[:16]
    
    def _generate_signature(self) -> bytes:
        """Generate a unique signature for this party's transmissions."""
        # Used to identify own transmissions
        sig_data = f"{self.party_id}-{os.getpid()}"
        return hashlib.sha256(sig_data.encode()).digest()[:8]
    
    @property
    def signature(self) -> bytes:
        """Get the party's transmission signature."""
        return self._signature
    
    @property
    def signature_hex(self) -> str:
        """Get the party's signature as hex string."""
        return self._signature.hex()
    
    def is_own_transmission(self, signature: bytes) -> bool:
        """
        Check if a transmission signature belongs to this party.
        
        Args:
            signature: The signature from a received transmission
            
        Returns:
            True if the signature matches this party's signature
        """
        return signature == self._signature
    
    def create_message(self, content: str, message_type: str = "text") -> Message:
        """
        Create a new message from this party.
        
        Args:
            content: Message content
            message_type: Type of message (text, audio, system)
            
        Returns:
            Created Message object
        """
        message = Message(
            content=content,
            sender_id=self.party_id,
            message_type=message_type,
        )
        self._pending_messages.append(message)
        self.context.add_message(message)
        self._last_activity = time.time()
        return message
    
    def receive_message(self, message: Message) -> None:
        """
        Process a received message.
        
        Args:
            message: The received message
        """
        message.received = True
        self._received_messages.append(message)
        self.context.add_message(message)
        self._last_activity = time.time()
    
    def mark_transmitted(self, message: Message) -> None:
        """Mark a message as transmitted."""
        message.transmitted = True
        if message in self._pending_messages:
            self._pending_messages.remove(message)
    
    def get_pending_messages(self) -> List[Message]:
        """Get all pending (not yet transmitted) messages."""
        return self._pending_messages.copy()
    
    def get_received_messages(self) -> List[Message]:
        """Get all received messages."""
        return self._received_messages.copy()
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for AI context."""
        self.context.system_prompt = prompt
    
    def set_max_context_length(self, length: int) -> None:
        """Set maximum context length."""
        self.context.max_context_length = length
    
    def deactivate(self) -> None:
        """Deactivate this party."""
        self._active = False
    
    def is_active(self) -> bool:
        """Check if party is active."""
        return self._active
    
    def get_idle_time(self) -> float:
        """Get seconds since last activity."""
        return time.time() - self._last_activity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert party to dictionary."""
        return {
            "party_id": self.party_id,
            "name": self.name,
            "is_ai": self.is_ai,
            "signature": self.signature_hex,
            "active": self._active,
            "pending_count": len(self._pending_messages),
            "received_count": len(self._received_messages),
        }
    
    def __repr__(self) -> str:
        return f"Party(id={self.party_id[:8]}, name={self.name}, ai={self.is_ai})"
