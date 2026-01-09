"""Tests for the configuration module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from muwave.core.config import Config


class TestConfig:
    """Tests for Config class."""
    
    def test_default_config(self):
        """Test loading default configuration."""
        config = Config()
        
        assert config.audio is not None
        assert config.speed is not None
        assert config.redundancy is not None
        assert config.protocol is not None
        assert config.ollama is not None
        assert config.ui is not None
    
    def test_get_with_dot_notation(self):
        """Test getting values with dot notation."""
        config = Config()
        
        sample_rate = config.get("audio.sample_rate")
        assert sample_rate == 44100
        
        mode = config.get("speed.mode")
        # Speed modes are named like s40, s60, s90, etc.
        assert mode.startswith("s") and mode[1:].isdigit()
    
    def test_get_with_default(self):
        """Test getting non-existent values returns default."""
        config = Config()
        
        value = config.get("nonexistent.key", "default_value")
        assert value == "default_value"
    
    def test_set_value(self):
        """Test setting configuration values."""
        config = Config()
        
        config.set("audio.sample_rate", 48000)
        assert config.get("audio.sample_rate") == 48000
    
    def test_speed_mode_settings(self):
        """Test getting speed mode settings."""
        config = Config()
        
        settings = config.get_speed_mode_settings()
        assert "symbol_duration_ms" in settings
        assert "bandwidth_hz" in settings
    
    def test_redundancy_mode_settings(self):
        """Test getting redundancy mode settings."""
        config = Config()
        
        # First set a valid mode that exists in the modes dict
        config.set("redundancy.mode", "r2")
        settings = config.get_redundancy_mode_settings()
        assert "repetitions" in settings
        assert "error_correction" in settings
    
    def test_save_and_load(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create and save config
            config1 = Config()
            config1.set("audio.sample_rate", 48000)
            config1.save(temp_path)
            
            # Load config
            config2 = Config(temp_path)
            assert config2.get("audio.sample_rate") == 48000
            
        finally:
            os.unlink(temp_path)
    
    def test_load_from_yaml_file(self):
        """Test loading from a YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "audio": {"sample_rate": 22050},
                "speed": {"mode": "slow"},
            }, f)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            assert config.get("audio.sample_rate") == 22050
            assert config.get("speed.mode") == "slow"
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_config_file_raises(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path/to/config.yaml")
