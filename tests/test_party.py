"""Tests for the party module."""

import pytest

from muwave.core.party import Party, Message, ConversationContext


class TestMessage:
    """Tests for Message class."""
    
    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(
            content="Hello, world!",
            sender_id="test_sender",
        )
        
        assert msg.content == "Hello, world!"
        assert msg.sender_id == "test_sender"
        assert msg.message_type == "text"
        assert not msg.transmitted
        assert not msg.received
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(
            content="Test",
            sender_id="sender",
            message_type="audio",
        )
        
        d = msg.to_dict()
        
        assert d["content"] == "Test"
        assert d["sender_id"] == "sender"
        assert d["message_type"] == "audio"
        assert "timestamp" in d


class TestConversationContext:
    """Tests for ConversationContext class."""
    
    def test_add_message(self):
        """Test adding messages to context."""
        ctx = ConversationContext()
        
        msg1 = Message(content="Hello", sender_id="user1")
        msg2 = Message(content="Hi there", sender_id="ai")
        
        ctx.add_message(msg1)
        ctx.add_message(msg2)
        
        assert len(ctx.messages) == 2
    
    def test_context_trimming(self):
        """Test that context is trimmed when exceeding max length."""
        ctx = ConversationContext(max_context_length=100)
        
        # Add messages that exceed the limit
        for i in range(10):
            msg = Message(content="A" * 50, sender_id=f"user{i}")
            ctx.add_message(msg)
        
        # Context should be trimmed
        total_length = sum(len(m.content) for m in ctx.messages)
        assert total_length <= 100
    
    def test_get_context_string(self):
        """Test getting context as string."""
        ctx = ConversationContext(system_prompt="You are helpful.")
        
        ctx.add_message(Message(content="Hello", sender_id="user"))
        ctx.add_message(Message(content="Hi!", sender_id="ai"))
        
        context_str = ctx.get_context_string()
        
        assert "System: You are helpful." in context_str
        assert "User: Hello" in context_str
        assert "Assistant: Hi!" in context_str
    
    def test_get_messages_for_api(self):
        """Test getting messages in API format."""
        ctx = ConversationContext(system_prompt="Be helpful")
        
        ctx.add_message(Message(content="Hi", sender_id="user"))
        ctx.add_message(Message(content="Hello", sender_id="ai"))
        
        messages = ctx.get_messages_for_api()
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
    
    def test_clear(self):
        """Test clearing context."""
        ctx = ConversationContext()
        ctx.add_message(Message(content="Test", sender_id="user"))
        
        ctx.clear()
        
        assert len(ctx.messages) == 0


class TestParty:
    """Tests for Party class."""
    
    def test_party_creation(self):
        """Test creating a party."""
        party = Party(name="TestParty")
        
        assert party.name == "TestParty"
        assert party.party_id is not None
        assert len(party.party_id) == 16
        assert not party.is_ai
    
    def test_party_id_generation(self):
        """Test that party IDs are unique."""
        party1 = Party()
        party2 = Party()
        
        assert party1.party_id != party2.party_id
    
    def test_signature_generation(self):
        """Test signature generation."""
        party = Party()
        
        assert party.signature is not None
        assert len(party.signature) == 8
        assert party.signature_hex == party.signature.hex()
    
    def test_own_transmission_detection(self):
        """Test detecting own transmissions."""
        party = Party()
        
        # Own signature should match
        assert party.is_own_transmission(party.signature)
        
        # Different signature should not match
        other_party = Party()
        assert not party.is_own_transmission(other_party.signature)
    
    def test_create_message(self):
        """Test creating messages."""
        party = Party()
        
        msg = party.create_message("Hello")
        
        assert msg.content == "Hello"
        assert msg.sender_id == party.party_id
        assert msg in party.get_pending_messages()
    
    def test_receive_message(self):
        """Test receiving messages."""
        party = Party()
        
        msg = Message(content="Incoming", sender_id="other")
        party.receive_message(msg)
        
        assert msg.received
        assert msg in party.get_received_messages()
    
    def test_mark_transmitted(self):
        """Test marking messages as transmitted."""
        party = Party()
        
        msg = party.create_message("Test")
        assert msg in party.get_pending_messages()
        
        party.mark_transmitted(msg)
        
        assert msg.transmitted
        assert msg not in party.get_pending_messages()
    
    def test_party_as_ai(self):
        """Test party with AI flag."""
        party = Party(is_ai=True)
        
        assert party.is_ai
    
    def test_system_prompt(self):
        """Test setting system prompt."""
        party = Party()
        party.set_system_prompt("Be helpful")
        
        assert party.context.system_prompt == "Be helpful"
    
    def test_party_to_dict(self):
        """Test converting party to dictionary."""
        party = Party(name="Test", is_ai=True)
        
        d = party.to_dict()
        
        assert d["name"] == "Test"
        assert d["is_ai"] is True
        assert "party_id" in d
        assert "signature" in d
