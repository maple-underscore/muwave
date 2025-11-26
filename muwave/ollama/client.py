"""
Ollama integration for muwave.
Provides AI agent capabilities using Ollama models.
"""

import json
import subprocess
import threading
import time
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
from enum import Enum

import requests


class OllamaMode(Enum):
    """Ollama connection mode."""
    DOCKER = "docker"
    TERMINAL = "terminal"
    HTTP = "http"


@dataclass
class ChatMessage:
    """A message in the chat history."""
    role: str  # "system", "user", or "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to API format."""
        return {"role": self.role, "content": self.content}


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    mode: OllamaMode = OllamaMode.DOCKER
    model: str = "llama3.2"
    host: str = "localhost"
    port: int = 11434
    container_name: str = "ollama"
    timeout_seconds: float = 120.0
    keep_context: bool = True
    max_context_length: int = 4096
    system_prompt: str = "You are a helpful AI assistant."


class OllamaClient:
    """
    Client for interacting with Ollama AI.
    
    Supports multiple connection modes:
    - Docker: Connect to Ollama running in a Docker container
    - Terminal: Run Ollama commands directly via terminal
    - HTTP: Connect to a running Ollama HTTP API
    
    Maintains conversation context across messages.
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama client.
        
        Args:
            config: Ollama configuration. Uses defaults if None.
        """
        self.config = config or OllamaConfig()
        self._messages: List[ChatMessage] = []
        self._lock = threading.Lock()
        
        # Add system prompt if configured
        if self.config.system_prompt:
            self._messages.append(ChatMessage(
                role="system",
                content=self.config.system_prompt,
            ))
    
    @property
    def base_url(self) -> str:
        """Get the base URL for Ollama API."""
        return f"http://{self.config.host}:{self.config.port}"
    
    def _get_context_messages(self) -> List[Dict[str, str]]:
        """Get messages formatted for the API."""
        with self._lock:
            return [msg.to_dict() for msg in self._messages]
    
    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the context."""
        with self._lock:
            self._messages.append(ChatMessage(role=role, content=content))
            self._trim_context()
    
    def _trim_context(self) -> None:
        """Trim context to fit within max length."""
        # Keep system message, trim oldest user/assistant messages
        total_length = sum(len(m.content) for m in self._messages)
        
        while total_length > self.config.max_context_length and len(self._messages) > 2:
            # Find first non-system message to remove
            for i, msg in enumerate(self._messages):
                if msg.role != "system":
                    removed = self._messages.pop(i)
                    total_length -= len(removed.content)
                    break
    
    def clear_context(self) -> None:
        """Clear conversation context (keeps system prompt)."""
        with self._lock:
            self._messages = [m for m in self._messages if m.role == "system"]
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt."""
        with self._lock:
            # Remove existing system prompt
            self._messages = [m for m in self._messages if m.role != "system"]
            # Add new system prompt at the beginning
            self._messages.insert(0, ChatMessage(role="system", content=prompt))
    
    def chat(self, prompt: str, stream: bool = False) -> str:
        """
        Send a chat message and get a response.
        
        Uses the appropriate connection mode based on configuration.
        Maintains conversation context.
        
        Args:
            prompt: User message
            stream: Whether to stream the response
            
        Returns:
            AI response text
        """
        if self.config.mode == OllamaMode.TERMINAL:
            return self._chat_terminal(prompt)
        else:
            return self._chat_http(prompt, stream)
    
    def _chat_http(self, prompt: str, stream: bool = False) -> str:
        """Send chat via HTTP API."""
        # Add user message to context
        self._add_message("user", prompt)
        
        # Prepare request
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": self._get_context_messages(),
            "stream": stream,
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_seconds,
                stream=stream,
            )
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            full_response += data["message"]["content"]
                        if data.get("done", False):
                            break
                assistant_response = full_response
            else:
                # Handle non-streaming response
                data = response.json()
                if "message" not in data or "content" not in data.get("message", {}):
                    raise OllamaError(
                        f"Unexpected API response format. Expected 'message.content' but got: {list(data.keys())}"
                    )
                assistant_response = data["message"]["content"]
            
            # Add assistant response to context
            if self.config.keep_context:
                self._add_message("assistant", assistant_response)
            
            return assistant_response
            
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"HTTP request failed: {e}")
    
    def _chat_terminal(self, prompt: str) -> str:
        """
        Send chat via terminal command.
        
        Uses: ollama run {model} {prompt}
        
        For context, we build a prompt that includes conversation history.
        """
        # Build context-aware prompt
        context_prompt = self._build_context_prompt(prompt)
        
        try:
            # Run ollama command
            result = subprocess.run(
                ["ollama", "run", self.config.model, context_prompt],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            
            if result.returncode != 0:
                raise OllamaError(f"Ollama command failed: {result.stderr}")
            
            assistant_response = result.stdout.strip()
            
            # Add to context
            self._add_message("user", prompt)
            if self.config.keep_context:
                self._add_message("assistant", assistant_response)
            
            return assistant_response
            
        except subprocess.TimeoutExpired:
            raise OllamaError("Ollama command timed out")
        except FileNotFoundError:
            raise OllamaError("Ollama command not found. Is Ollama installed?")
    
    def _build_context_prompt(self, new_prompt: str) -> str:
        """Build a prompt that includes conversation context for terminal mode."""
        parts = []
        
        with self._lock:
            for msg in self._messages:
                if msg.role == "system":
                    parts.append(f"System: {msg.content}")
                elif msg.role == "user":
                    parts.append(f"User: {msg.content}")
                elif msg.role == "assistant":
                    parts.append(f"Assistant: {msg.content}")
        
        parts.append(f"User: {new_prompt}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    def chat_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream a chat response.
        
        Args:
            prompt: User message
            
        Yields:
            Response chunks as they arrive
        """
        # Add user message to context
        self._add_message("user", prompt)
        
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": self._get_context_messages(),
            "stream": True,
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_seconds,
                stream=True,
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        chunk = data["message"]["content"]
                        full_response += chunk
                        yield chunk
                    if data.get("done", False):
                        break
            
            # Add complete response to context
            if self.config.keep_context:
                self._add_message("assistant", full_response)
                
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"HTTP request failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self.config.mode == OllamaMode.TERMINAL:
            return self._check_terminal_available()
        else:
            return self._check_http_available()
    
    def _check_http_available(self) -> bool:
        """Check if Ollama HTTP API is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _check_terminal_available(self) -> bool:
        """Check if Ollama terminal command is available."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                timeout=5.0,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        if self.config.mode == OllamaMode.TERMINAL:
            return self._list_models_terminal()
        else:
            return self._list_models_http()
    
    def _list_models_http(self) -> List[str]:
        """List models via HTTP API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.exceptions.RequestException:
            return []
    
    def _list_models_terminal(self) -> List[str]:
        """List models via terminal command."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            if result.returncode != 0:
                return []
            
            # Parse output (skip header line)
            lines = result.stdout.strip().split('\n')[1:]
            return [line.split()[0] for line in lines if line.strip()]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history."""
        return self._get_context_messages()
    
    def get_context_length(self) -> int:
        """Get current context length in characters."""
        with self._lock:
            return sum(len(m.content) for m in self._messages)


class OllamaError(Exception):
    """Exception raised for Ollama-related errors."""
    pass


def create_ollama_client(config_dict: Dict[str, Any]) -> OllamaClient:
    """
    Create an OllamaClient from a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary (from Config.ollama)
        
    Returns:
        Configured OllamaClient
    """
    mode_str = config_dict.get("mode", "docker")
    mode = OllamaMode(mode_str)
    
    docker_config = config_dict.get("docker", {})
    terminal_config = config_dict.get("terminal", {})
    model_config = config_dict.get("model", {})
    
    if mode == OllamaMode.DOCKER:
        host = docker_config.get("host", "localhost")
        port = docker_config.get("port", 11434)
        container_name = docker_config.get("container_name", "ollama")
    elif mode == OllamaMode.TERMINAL:
        host = "localhost"
        port = 11434
        container_name = "ollama"
    else:
        http_config = config_dict.get("http", {})
        # Parse base_url if provided
        base_url = http_config.get("base_url", "http://localhost:11434")
        if "://" in base_url:
            parts = base_url.split("://")[1].split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 11434
        else:
            host = "localhost"
            port = 11434
        container_name = "ollama"
    
    ollama_config = OllamaConfig(
        mode=mode,
        model=model_config.get("name", "llama3.2"),
        host=host,
        port=port,
        container_name=container_name,
        timeout_seconds=terminal_config.get("timeout_seconds", 120.0),
        keep_context=model_config.get("keep_context", True),
        max_context_length=model_config.get("max_context_length", 4096),
        system_prompt=config_dict.get("system_prompt", ""),
    )
    
    return OllamaClient(ollama_config)
