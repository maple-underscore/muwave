"""Tests for the CLI generate command."""

import os
import tempfile
import numpy as np
import pytest

from click.testing import CliRunner
from muwave.cli import main
from muwave.audio.fsk import FSKModulator, FSKConfig


class TestGenerateCommand:
    """Tests for the generate CLI command."""
    
    def test_generate_creates_wav_file(self):
        """Test that generate command creates a WAV file."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.wav')
            result = runner.invoke(main, ['generate', 'Hello', '-o', output_path])
            
            assert result.exit_code == 0
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
    
    def test_generate_with_default_output(self):
        """Test that generate uses output.wav as default."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['generate', 'Hello'])
            
            assert result.exit_code == 0
            assert os.path.exists('output.wav')
    
    def test_generate_shows_success_message(self):
        """Test that generate shows success message."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.wav')
            result = runner.invoke(main, ['generate', 'Hello', '-o', output_path])
            
            assert 'Sound wave saved to' in result.output
            assert 'Duration' in result.output
            assert 'Sample rate' in result.output
    
    def test_generate_wav_file_format(self):
        """Test that generated WAV file has correct format."""
        from scipy.io import wavfile
        
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.wav')
            result = runner.invoke(main, ['generate', 'Hello', '-o', output_path])
            
            assert result.exit_code == 0
            
            # Read the WAV file and check properties
            sample_rate, data = wavfile.read(output_path)
            
            assert sample_rate == 44100  # Default sample rate
            assert data.dtype == np.int16  # 16-bit audio
            assert len(data) > 0
    
    def test_generate_longer_message(self):
        """Test generating a longer message."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.wav')
            long_message = "This is a longer message to test the encoding functionality."
            result = runner.invoke(main, ['generate', long_message, '-o', output_path])
            
            assert result.exit_code == 0
            assert os.path.exists(output_path)
    
    def test_generate_with_party_name(self):
        """Test generating with a specific party name."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.wav')
            result = runner.invoke(main, ['generate', 'Hello', '-o', output_path, '-n', 'Alice'])
            
            assert result.exit_code == 0
            assert os.path.exists(output_path)
    
    def test_generate_unicode_message(self):
        """Test generating a message with unicode characters."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.wav')
            unicode_message = "Hello, 世界! 🌍"
            result = runner.invoke(main, ['generate', unicode_message, '-o', output_path])
            
            assert result.exit_code == 0
            assert os.path.exists(output_path)
    
    def test_generate_empty_message(self):
        """Test generating an empty message."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.wav')
            result = runner.invoke(main, ['generate', '', '-o', output_path])
            
            # Empty message should still create a WAV with start/end signals
            assert result.exit_code == 0
            assert os.path.exists(output_path)


class TestFSKWaveGeneration:
    """Tests for the FSK wave generation functionality."""
    
    def test_encoded_audio_is_valid_for_wav(self):
        """Test that encoded audio can be saved as WAV."""
        from scipy.io import wavfile
        
        modulator = FSKModulator()
        samples = modulator.encode_text("Test")
        
        # Check samples are valid
        assert samples.dtype == np.float32
        assert np.all(np.abs(samples) <= 1.0)
        
        # Convert to int16 and save
        audio_int16 = (samples * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wavfile.write(f.name, 44100, audio_int16)
            assert os.path.exists(f.name)
            assert os.path.getsize(f.name) > 0
            os.unlink(f.name)
    
    def test_wav_roundtrip(self):
        """Test that WAV file can be read back correctly."""
        from scipy.io import wavfile
        
        modulator = FSKModulator()
        original_samples = modulator.encode_text("Test")
        audio_int16 = (original_samples * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wavfile.write(f.name, 44100, audio_int16)
            
            # Read back
            sample_rate, read_data = wavfile.read(f.name)
            
            assert sample_rate == 44100
            assert len(read_data) == len(audio_int16)
            np.testing.assert_array_equal(read_data, audio_int16)
            
            os.unlink(f.name)
