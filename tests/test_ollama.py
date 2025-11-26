"""Tests for the Ollama client module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from muwave.ollama.client import (
    OllamaClient,
    OllamaConfig,
    OllamaMode,
    OllamaError,
    ChatMessage,
    create_ollama_client,
)


class TestChatMessage:
    """Tests for ChatMessage class."""
    
    def test_message_creation(self):
        """Test creating a chat message."""
        msg = ChatMessage(role="user", content="Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        msg = ChatMessage(role="assistant", content="Hi there")
        
        d = msg.to_dict()
        
        assert d == {"role": "assistant", "content": "Hi there"}


class TestOllamaConfig:
    """Tests for OllamaConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = OllamaConfig()
        
        assert config.mode == OllamaMode.DOCKER
        assert config.model == "llama3.2"
        assert config.host == "localhost"
        assert config.port == 11434
        assert config.keep_context is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = OllamaConfig(
            mode=OllamaMode.TERMINAL,
            model="mistral",
            timeout_seconds=60,
        )
        
        assert config.mode == OllamaMode.TERMINAL
        assert config.model == "mistral"
        assert config.timeout_seconds == 60


class TestOllamaClient:
    """Tests for OllamaClient class."""
    
    def test_client_creation(self):
        """Test creating a client."""
        client = OllamaClient()
        
        assert client.config is not None
        assert client.base_url == "http://localhost:11434"
    
    def test_client_with_system_prompt(self):
        """Test client initializes with system prompt."""
        config = OllamaConfig(system_prompt="Be helpful")
        client = OllamaClient(config)
        
        messages = client._get_context_messages()
        
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
    
    def test_add_message(self):
        """Test adding messages to context."""
        config = OllamaConfig(system_prompt="")  # No system prompt
        client = OllamaClient(config)
        
        client._add_message("user", "Hello")
        client._add_message("assistant", "Hi!")
        
        messages = client._get_context_messages()
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_context_trimming(self):
        """Test that context is trimmed when exceeding max length."""
        config = OllamaConfig(max_context_length=100)
        client = OllamaClient(config)
        
        # Add messages that exceed the limit
        for i in range(10):
            client._add_message("user", "A" * 50)
        
        total_length = client.get_context_length()
        assert total_length <= 100
    
    def test_clear_context(self):
        """Test clearing context."""
        config = OllamaConfig(system_prompt="Be helpful")
        client = OllamaClient(config)
        
        client._add_message("user", "Hello")
        client._add_message("assistant", "Hi!")
        
        client.clear_context()
        
        messages = client._get_context_messages()
        
        # Only system prompt should remain
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
    
    def test_set_system_prompt(self):
        """Test setting system prompt."""
        client = OllamaClient()
        
        client.set_system_prompt("New prompt")
        
        messages = client._get_context_messages()
        
        assert len(messages) == 1
        assert messages[0]["content"] == "New prompt"
    
    def test_build_context_prompt(self):
        """Test building context prompt for terminal mode."""
        config = OllamaConfig(system_prompt="Be helpful")
        client = OllamaClient(config)
        
        client._add_message("user", "What is 2+2?")
        client._add_message("assistant", "4")
        
        prompt = client._build_context_prompt("What is 3+3?")
        
        assert "System: Be helpful" in prompt
        assert "User: What is 2+2?" in prompt
        assert "Assistant: 4" in prompt
        assert "User: What is 3+3?" in prompt
        assert prompt.endswith("Assistant:")
    
    @patch('muwave.ollama.client.requests.post')
    def test_chat_http_success(self, mock_post):
        """Test successful HTTP chat."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Hello! How can I help?"}
        }
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        response = client.chat("Hi")
        
        assert response == "Hello! How can I help?"
        
        # Check that message was added to context
        messages = client._get_context_messages()
        assert any(m["content"] == "Hi" for m in messages)
        assert any(m["content"] == "Hello! How can I help?" for m in messages)
    
    @patch('muwave.ollama.client.requests.post')
    def test_chat_http_failure(self, mock_post):
        """Test HTTP chat failure."""
        from requests.exceptions import RequestException
        mock_post.side_effect = RequestException("Connection failed")
        
        config = OllamaConfig(system_prompt="")
        client = OllamaClient(config)
        
        with pytest.raises(OllamaError):
            client.chat("Hi")
    
    @patch('muwave.ollama.client.subprocess.run')
    def test_chat_terminal_success(self, mock_run):
        """Test successful terminal chat."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Hello from terminal!",
            stderr="",
        )
        
        config = OllamaConfig(mode=OllamaMode.TERMINAL)
        client = OllamaClient(config)
        
        response = client.chat("Hi")
        
        assert response == "Hello from terminal!"
        mock_run.assert_called_once()
    
    @patch('muwave.ollama.client.subprocess.run')
    def test_chat_terminal_not_found(self, mock_run):
        """Test terminal chat when ollama not found."""
        mock_run.side_effect = FileNotFoundError()
        
        config = OllamaConfig(mode=OllamaMode.TERMINAL)
        client = OllamaClient(config)
        
        with pytest.raises(OllamaError, match="not found"):
            client.chat("Hi")
    
    @patch('muwave.ollama.client.requests.get')
    def test_is_available_http(self, mock_get):
        """Test checking HTTP availability."""
        mock_get.return_value = Mock(status_code=200)
        
        client = OllamaClient()
        
        assert client.is_available() is True
    
    @patch('muwave.ollama.client.requests.get')
    def test_is_available_http_not_running(self, mock_get):
        """Test HTTP not available."""
        from requests.exceptions import RequestException
        mock_get.side_effect = RequestException("Connection refused")
        
        config = OllamaConfig(system_prompt="")
        client = OllamaClient(config)
        
        assert client.is_available() is False
    
    @patch('muwave.ollama.client.subprocess.run')
    def test_is_available_terminal(self, mock_run):
        """Test checking terminal availability."""
        mock_run.return_value = Mock(returncode=0)
        
        config = OllamaConfig(mode=OllamaMode.TERMINAL)
        client = OllamaClient(config)
        
        assert client.is_available() is True
    
    def test_get_conversation_history(self):
        """Test getting conversation history."""
        config = OllamaConfig(system_prompt="")  # No system prompt
        client = OllamaClient(config)
        
        client._add_message("user", "Question 1")
        client._add_message("assistant", "Answer 1")
        client._add_message("user", "Question 2")
        
        history = client.get_conversation_history()
        
        assert len(history) == 3
        assert history[0]["content"] == "Question 1"


class TestCreateOllamaClient:
    """Tests for create_ollama_client function."""
    
    def test_create_docker_client(self):
        """Test creating Docker-mode client."""
        config_dict = {
            "mode": "docker",
            "docker": {
                "host": "localhost",
                "port": 11434,
                "container_name": "ollama",
            },
            "model": {
                "name": "llama3.2",
                "keep_context": True,
            },
        }
        
        client = create_ollama_client(config_dict)
        
        assert client.config.mode == OllamaMode.DOCKER
        assert client.config.host == "localhost"
        assert client.config.port == 11434
    
    def test_create_terminal_client(self):
        """Test creating terminal-mode client."""
        config_dict = {
            "mode": "terminal",
            "terminal": {
                "timeout_seconds": 60,
            },
            "model": {
                "name": "mistral",
            },
        }
        
        client = create_ollama_client(config_dict)
        
        assert client.config.mode == OllamaMode.TERMINAL
        assert client.config.model == "mistral"
    
    def test_create_with_system_prompt(self):
        """Test creating client with system prompt."""
        config_dict = {
            "mode": "docker",
            "docker": {},
            "model": {"name": "llama3.2"},
            "system_prompt": "You are a helpful assistant.",
        }
        
        client = create_ollama_client(config_dict)
        
        messages = client._get_context_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "You are a helpful assistant."
