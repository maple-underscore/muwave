"""Tests for the logger module."""

import os
import json
import tempfile
from pathlib import Path

import pytest

from muwave.core.logger import ConversationLogger


class TestConversationLogger:
    """Tests for ConversationLogger class."""
    
    def test_logger_creation(self):
        """Test creating a logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(log_file=str(log_file))
            
            assert logger.log_file == log_file
    
    def test_log_text_format(self):
        """Test logging in text format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(
                log_file=str(log_file),
                log_format="text",
            )
            
            logger.log("test_event", "party123", "Test content")
            
            content = log_file.read_text()
            assert "test_event" in content
            assert "party123" in content.lower() or "party12" in content
            assert "Test content" in content
    
    def test_log_json_format(self):
        """Test logging in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(
                log_file=str(log_file),
                log_format="json",
            )
            
            logger.log("test_event", "party123", "Test content")
            
            content = log_file.read_text().strip()
            data = json.loads(content)
            
            assert data["event"] == "test_event"
            assert data["party_id"] == "party123"
            assert data["content"] == "Test content"
    
    def test_log_with_metadata(self):
        """Test logging with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(
                log_file=str(log_file),
                log_format="json",
            )
            
            logger.log(
                "test_event",
                "party123",
                "Content",
                metadata={"key": "value"},
            )
            
            content = log_file.read_text().strip()
            data = json.loads(content)
            
            assert data["metadata"]["key"] == "value"
    
    def test_log_message_sent(self):
        """Test logging sent messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(log_file=str(log_file))
            
            logger.log_message_sent("party1", "Hello world")
            
            content = log_file.read_text()
            assert "message_sent" in content
            assert "Hello world" in content
    
    def test_log_message_received(self):
        """Test logging received messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(log_file=str(log_file))
            
            logger.log_message_received("party1", "Incoming", "sender1")
            
            content = log_file.read_text()
            assert "message_received" in content
    
    def test_log_level_filtering(self):
        """Test that log level filtering works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(
                log_file=str(log_file),
                log_level="warning",
            )
            
            # Debug and info should be filtered
            logger.log("event", "party", "debug msg", level="debug")
            logger.log("event", "party", "info msg", level="info")
            logger.log("event", "party", "warning msg", level="warning")
            
            content = log_file.read_text()
            assert "debug msg" not in content
            assert "info msg" not in content
            assert "warning msg" in content
    
    def test_get_conversation_history(self):
        """Test retrieving conversation history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(log_file=str(log_file))
            
            logger.log("event1", "party", "Content 1")
            logger.log("event2", "party", "Content 2")
            
            history = logger.get_conversation_history()
            
            assert len(history) == 2
    
    def test_clear_log(self):
        """Test clearing the log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(log_file=str(log_file))
            
            logger.log("event", "party", "Content")
            logger.clear_log()
            
            assert log_file.read_text() == ""
    
    def test_timestamps(self):
        """Test that timestamps are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = ConversationLogger(
                log_file=str(log_file),
                include_timestamps=True,
            )
            
            logger.log("event", "party", "Content")
            
            content = log_file.read_text()
            # Should contain date-like pattern
            assert "-" in content and ":" in content
