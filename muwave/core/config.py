"""
Configuration management for muwave.
Provides easy configuration loading and editing.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager for muwave."""
    
    DEFAULT_CONFIG_PATHS = [
        Path("config.yaml"),
        Path("muwave.yaml"),
        Path.home() / ".config" / "muwave" / "config.yaml",
        Path("/etc/muwave/config.yaml"),
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, searches default paths.
        """
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from file."""
        if config_path:
            path = Path(config_path)
            if path.exists():
                self._config_path = path
                with open(path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
                return
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Search default paths
        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                self._config_path = path
                with open(path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
                return
        
        # Use default configuration if no file found
        self._config = self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "audio": {
                "sample_rate": 44100,
                "buffer_size": 1024,
                "volume": 0.8,
                "input_device": None,
                "output_device": None,
            },
            "speed": {
                "mode": "medium",
                "modes": {
                    "slow": {"symbol_duration_ms": 120, "bandwidth_hz": 200},
                    "medium": {"symbol_duration_ms": 60, "bandwidth_hz": 400},
                    "fast": {"symbol_duration_ms": 35, "bandwidth_hz": 800},
                },
            },
            "redundancy": {
                "mode": "medium",
                "modes": {
                    "low": {"repetitions": 1, "error_correction": False},
                    "medium": {"repetitions": 2, "error_correction": True},
                    "high": {"repetitions": 3, "error_correction": True},
                },
            },
            "protocol": {
                "base_frequency": 1800,
                "frequency_step": 120,
                "num_frequencies": 16,
                "start_frequency": 800,
                "end_frequency": 900,
                "signal_duration_ms": 200,
                "silence_ms": 50,
                "party_id": None,
                "self_recognition": True,
            },
            "ollama": {
                "mode": "docker",
                "docker": {
                    "container_name": "ollama",
                    "host": "localhost",
                    "port": 11434,
                },
                "terminal": {
                    "command": "ollama",
                    "timeout_seconds": 120,
                },
                "http": {
                    "base_url": "http://localhost:11434",
                    "timeout_seconds": 120,
                },
                "model": {
                    "name": "llama3.2",
                    "keep_context": True,
                    "max_context_length": 4096,
                },
                "system_prompt": "You are a helpful AI assistant.",
            },
            "ui": {
                "rich_interface": True,
                "colors": {
                    "waiting": "yellow",
                    "sending": "blue",
                    "sent": "green",
                    "receiving": "cyan",
                    "error": "red",
                },
                "show_receiving": True,
                "show_progress": True,
            },
            "logging": {
                "enabled": True,
                "file": "muwave_conversation.log",
                "format": "text",
                "timestamps": True,
                "level": "info",
            },
            "sync": {
                "wait_for_audio": True,
                "timeout_seconds": 30,
                "ipc_method": "file",
                "ipc_file": "/tmp/muwave_sync.lock",
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "audio.sample_rate")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "audio.sample_rate")
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save to. If None, uses the loaded path or config.yaml
        """
        save_path = Path(path) if path else (self._config_path or Path("config.yaml"))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    @property
    def audio(self) -> Dict[str, Any]:
        """Get audio configuration."""
        return self._config.get("audio", {})
    
    @property
    def speed(self) -> Dict[str, Any]:
        """Get speed configuration."""
        return self._config.get("speed", {})
    
    @property
    def redundancy(self) -> Dict[str, Any]:
        """Get redundancy configuration."""
        return self._config.get("redundancy", {})
    
    @property
    def protocol(self) -> Dict[str, Any]:
        """Get protocol configuration."""
        return self._config.get("protocol", {})
    
    @property
    def ollama(self) -> Dict[str, Any]:
        """Get Ollama configuration."""
        return self._config.get("ollama", {})
    
    @property
    def ui(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self._config.get("ui", {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get("logging", {})
    
    @property
    def sync(self) -> Dict[str, Any]:
        """Get sync configuration."""
        return self._config.get("sync", {})
    
    def get_speed_mode_settings(self) -> Dict[str, Any]:
        """Get settings for current speed mode."""
        mode = self.speed.get("mode", "medium")
        return self.speed.get("modes", {}).get(mode, {})
    
    def get_redundancy_mode_settings(self) -> Dict[str, Any]:
        """Get settings for current redundancy mode."""
        mode = self.redundancy.get("mode", "medium")
        return self.redundancy.get("modes", {}).get(mode, {})
    
    def __repr__(self) -> str:
        return f"Config(path={self._config_path})"
